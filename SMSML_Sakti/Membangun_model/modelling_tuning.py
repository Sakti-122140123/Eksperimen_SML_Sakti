"""Advanced UCI Wine model training with tuning, DagsHub, and manual MLflow logging."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.utils import estimator_html_repr


BASE_DIR = Path(__file__).resolve().parent
TARGET_COLUMN = "wine_class"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tuned UCI Wine classifier.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=BASE_DIR / "wine_preprocessing/processed_wine.csv",
    )
    parser.add_argument("--experiment-name", default="uci-wine-advanced")
    parser.add_argument("--tracking-uri", default=f"file:{BASE_DIR / 'mlruns'}")
    parser.add_argument(
        "--use-dagshub",
        action="store_true",
        help="Initialize DagsHub MLflow tracking from DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME.",
    )
    return parser.parse_args()


def configure_tracking(args: argparse.Namespace) -> None:
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")

    if args.use_dagshub and repo_owner and repo_name:
        import dagshub

        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        print(f"Using DagsHub tracking for {repo_owner}/{repo_name}")
    else:
        mlflow.set_tracking_uri(args.tracking_uri)
        print(f"Using MLflow tracking URI: {args.tracking_uri}")

    mlflow.set_experiment(args.experiment_name)


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(data_path)
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found.")

    features = dataset.drop(columns=[TARGET_COLUMN]).astype(float)
    target = dataset[TARGET_COLUMN].astype(int)
    return features, target


def tune_model(x_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    estimator = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    parameter_grid = {
        "n_estimators": [120, 200],
        "max_depth": [4, 8, None],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=parameter_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    search.fit(x_train, y_train)
    return search


def build_artifacts(
    model: RandomForestClassifier,
    search: GridSearchCV,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: pd.Series,
    probabilities: pd.Series,
    metrics: dict[str, float],
) -> Path:
    artifact_dir = BASE_DIR / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    for existing_file in artifact_dir.iterdir():
        if existing_file.is_file():
            existing_file.unlink()

    report = classification_report(y_test, predictions, output_dict=True)
    (artifact_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    metric_info = {
        "metrics": metrics,
        "notes": {
            "accuracy": "overall correct predictions divided by total predictions",
            "precision": "weighted positive predictive value across wine classes",
            "recall": "weighted sensitivity across wine classes",
            "f1_score": "harmonic mean of precision and recall",
            "roc_auc": "weighted multiclass one-vs-rest ROC AUC",
        },
    }
    (artifact_dir / "metric_info.json").write_text(
        json.dumps(metric_info, indent=2),
        encoding="utf-8",
    )

    class_labels = sorted(y_test.unique().tolist())
    cm = confusion_matrix(y_test, predictions, labels=class_labels)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    display.plot(cmap="Blues", values_format="d")
    plt.title("Training Confusion Matrix")
    plt.tight_layout()
    plt.savefig(artifact_dir / "training_confusion_matrix.png", dpi=160)
    plt.close()

    feature_importance = pd.Series(model.feature_importances_, index=x_test.columns)
    feature_importance = feature_importance.sort_values(ascending=True).tail(15)
    plt.figure(figsize=(9, 6))
    feature_importance.plot(kind="barh", color="#2f6f73")
    plt.title("Top Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(artifact_dir / "feature_importance.png", dpi=160)
    plt.close()

    (artifact_dir / "estimator.html").write_text(estimator_html_repr(model), encoding="utf-8")

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(artifact_dir / "grid_search_cv_results.csv", index=False)

    model_card = f"""# UCI Wine Classifier

Algorithm: RandomForestClassifier

Best parameters:

```json
{json.dumps(search.best_params_, indent=2)}
```

Validation metrics:

```json
{json.dumps(metrics, indent=2)}
```
"""
    (artifact_dir / "model_card.md").write_text(model_card, encoding="utf-8")

    schema = {
        "target": TARGET_COLUMN,
        "feature_columns": x_test.columns.tolist(),
        "input_example": x_test.head(3).to_dict(orient="records"),
    }
    (artifact_dir / "input_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    return artifact_dir


def main() -> None:
    args = parse_args()
    configure_tracking(args)

    features, target = load_dataset(args.data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    search = tune_model(x_train, y_train)
    best_model: RandomForestClassifier = search.best_estimator_
    predictions = best_model.predict(x_test)
    probabilities = best_model.predict_proba(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities, multi_class="ovr", average="weighted")),
        "best_cv_f1": float(search.best_score_),
    }
    artifact_dir = build_artifacts(
        best_model,
        search,
        x_test,
        y_test,
        predictions,
        probabilities,
        metrics,
    )

    with mlflow.start_run(run_name="advanced-random-forest-grid-search") as run:
        mlflow.log_param("dataset_path", str(args.data_path))
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("train_rows", len(x_train))
        mlflow.log_param("test_rows", len(x_test))
        mlflow.log_param("tuning_scoring", "f1_weighted")
        mlflow.log_params({f"best_{key}": value for key, value in search.best_params_.items()})
        mlflow.log_metrics(metrics)

        signature = infer_signature(x_train, best_model.predict(x_train))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=x_train.head(5),
            registered_model_name=None,
        )
        mlflow.log_artifacts(str(artifact_dir), artifact_path="training_artifacts")

        model_dir = BASE_DIR / "model"
        model_dir.mkdir(exist_ok=True)
        joblib.dump(best_model, model_dir / "latest_model.joblib")
        (model_dir / "feature_columns.json").write_text(
            json.dumps(features.columns.tolist(), indent=2),
            encoding="utf-8",
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Artifact URI: {mlflow.get_artifact_uri()}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
