"""Baseline model training with MLflow autologging for the UCI Wine dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
TARGET_COLUMN = "wine_class"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline UCI Wine classifier.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=BASE_DIR / "wine_preprocessing/processed_wine.csv",
    )
    parser.add_argument("--experiment-name", default="uci-wine-baseline")
    parser.add_argument("--tracking-uri", default=f"file:{BASE_DIR / 'mlruns'}")
    return parser.parse_args()


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(data_path)
    features = dataset.drop(columns=[TARGET_COLUMN]).astype(float)
    target = dataset[TARGET_COLUMN].astype(int)
    return features, target


def main() -> None:
    args = parse_args()
    features, target = load_dataset(args.data_path)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
    )

    with mlflow.start_run(run_name="baseline-random-forest-autolog"):
        mlflow.log_param("dataset_path", str(args.data_path))
        mlflow.log_param("train_rows", len(x_train))
        mlflow.log_param("test_rows", len(x_test))

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, predictions, average="weighted", zero_division=0),
        }
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

    model_dir = BASE_DIR / "model"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(model, model_dir / "baseline_model.joblib")


if __name__ == "__main__":
    main()
