"""FastAPI UCI Wine model serving app with Prometheus metrics."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow.pyfunc
import pandas as pd
import psutil
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR.parent / "Membangun_model/wine_preprocessing/processed_wine.csv"
DEFAULT_MODEL_PATHS = [
    BASE_DIR / "folder tambahan/model/latest_model.joblib",
    BASE_DIR.parent / "Membangun_model/model/latest_model.joblib",
]
DEFAULT_METRIC_INFO_PATHS = [
    BASE_DIR / "folder tambahan/metrics/metric_info.json",
    BASE_DIR.parent / "Membangun_model/artifacts/metric_info.json",
]
TARGET_COLUMN = "wine_class"


REQUEST_COUNTER = Counter(
    "ml_prediction_requests_total",
    "Total prediction requests received by the model API.",
    ["status"],
)
ERROR_COUNTER = Counter(
    "ml_prediction_errors_total",
    "Total prediction errors raised by the model API.",
    ["error_type"],
)
LATENCY_SECONDS = Histogram(
    "ml_prediction_latency_seconds",
    "Prediction latency in seconds.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
PAYLOAD_ROWS = Histogram(
    "ml_prediction_payload_rows",
    "Number of rows per prediction payload.",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
)
PREDICTION_CLASS_COUNTER = Counter(
    "ml_prediction_class_total",
    "Prediction count by class.",
    ["class_name"],
)
CPU_USAGE = Gauge("ml_cpu_usage_percent", "Host CPU usage percentage.")
MEMORY_USAGE = Gauge("ml_memory_usage_percent", "Host memory usage percentage.")
PROCESS_MEMORY = Gauge("ml_process_memory_mb", "Serving process memory usage in MB.")
MODEL_LOADED = Gauge("ml_model_loaded", "Whether a trained model was loaded successfully.")
MODEL_ACCURACY = Gauge("ml_model_accuracy", "Latest validation accuracy logged during training.")
MODEL_PRECISION = Gauge("ml_model_precision", "Latest validation precision logged during training.")
MODEL_RECALL = Gauge("ml_model_recall", "Latest validation recall logged during training.")
MODEL_F1 = Gauge("ml_model_f1_score", "Latest validation F1 score logged during training.")
MODEL_ROC_AUC = Gauge("ml_model_roc_auc", "Latest validation ROC AUC logged during training.")


class PredictionPayload(BaseModel):
    instances: list[dict[str, Any]] = Field(default_factory=list)


class RuleBasedFallbackModel:
    """Fallback to keep the exporter demonstrable before model training runs."""

    def predict(self, frame: pd.DataFrame) -> list[int]:
        score = (
            0.35 * frame.get("flavanoids", 0)
            + 0.25 * frame.get("proline", 0)
            - 0.20 * frame.get("color_intensity", 0)
            + 0.15 * frame.get("alcohol", 0)
        )
        return pd.cut(score, bins=[-float("inf"), -0.4, 0.4, float("inf")], labels=[1, 2, 0]).astype(int).tolist()


def load_feature_columns() -> list[str]:
    data_path = Path(os.getenv("DATA_PATH", DEFAULT_DATA_PATH))
    if data_path.exists():
        dataset = pd.read_csv(data_path, nrows=5)
        return [column for column in dataset.columns if column != TARGET_COLUMN]

    feature_path = BASE_DIR.parent / "Membangun_model/model/feature_columns.json"
    if feature_path.exists():
        return json.loads(feature_path.read_text(encoding="utf-8"))

    return []


def load_model() -> Any:
    model_uri = os.getenv("MODEL_URI")
    if model_uri:
        try:
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            MODEL_LOADED.set(1)
            print(f"Loaded MLflow model from MODEL_URI={model_uri}")
            return loaded_model
        except Exception as exc:  # pragma: no cover - operational fallback
            print(f"Unable to load MLflow model: {exc}")

    env_model_path = os.getenv("MODEL_PATH")
    candidate_paths = [Path(env_model_path)] if env_model_path else DEFAULT_MODEL_PATHS
    model_path = next((path for path in candidate_paths if path.exists()), candidate_paths[0])
    if model_path.exists():
        loaded_model = joblib.load(model_path)
        MODEL_LOADED.set(1)
        print(f"Loaded joblib model from {model_path}")
        return loaded_model

    MODEL_LOADED.set(0)
    print("No trained model found; using rule-based fallback model.")
    return RuleBasedFallbackModel()


def load_training_metrics() -> None:
    env_metric_path = os.getenv("METRIC_INFO_PATH")
    candidate_paths = [Path(env_metric_path)] if env_metric_path else DEFAULT_METRIC_INFO_PATHS
    metric_path = next((path for path in candidate_paths if path.exists()), candidate_paths[0])
    if not metric_path.exists():
        return

    metric_info = json.loads(metric_path.read_text(encoding="utf-8"))
    metrics = metric_info.get("metrics", {})
    MODEL_ACCURACY.set(float(metrics.get("accuracy", 0)))
    MODEL_PRECISION.set(float(metrics.get("precision", 0)))
    MODEL_RECALL.set(float(metrics.get("recall", 0)))
    MODEL_F1.set(float(metrics.get("f1_score", 0)))
    MODEL_ROC_AUC.set(float(metrics.get("roc_auc", 0)))


def update_system_metrics() -> None:
    process = psutil.Process(os.getpid())
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    PROCESS_MEMORY.set(process.memory_info().rss / (1024 * 1024))


feature_columns = load_feature_columns()
model = load_model()
load_training_metrics()

app = FastAPI(title="UCI Wine Model API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": bool(MODEL_LOADED._value.get()),
        "feature_count": len(feature_columns),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "target": TARGET_COLUMN,
        "features": feature_columns,
        "model_loaded": bool(MODEL_LOADED._value.get()),
    }


@app.post("/predict")
def predict(payload: PredictionPayload) -> dict[str, Any]:
    start_time = time.perf_counter()
    try:
        if not payload.instances:
            raise HTTPException(status_code=400, detail="Payload must contain non-empty instances.")

        frame = pd.DataFrame(payload.instances)
        if feature_columns:
            frame = frame.reindex(columns=feature_columns, fill_value=0)

        predictions = model.predict(frame)
        predictions = [int(value) for value in predictions]

        for value in predictions:
            class_name = f"class_{value}"
            PREDICTION_CLASS_COUNTER.labels(class_name=class_name).inc()

        REQUEST_COUNTER.labels(status="success").inc()
        PAYLOAD_ROWS.observe(len(frame))
        return {
            "predictions": predictions,
            "labels": [f"class_{value}" for value in predictions],
            "row_count": len(frame),
        }
    except HTTPException:
        REQUEST_COUNTER.labels(status="failed").inc()
        ERROR_COUNTER.labels(error_type="bad_request").inc()
        raise
    except Exception as exc:  # pragma: no cover - runtime protection
        REQUEST_COUNTER.labels(status="failed").inc()
        ERROR_COUNTER.labels(error_type=exc.__class__.__name__).inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        LATENCY_SECONDS.observe(time.perf_counter() - start_time)
        update_system_metrics()


@app.get("/metrics")
def metrics() -> Response:
    update_system_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
