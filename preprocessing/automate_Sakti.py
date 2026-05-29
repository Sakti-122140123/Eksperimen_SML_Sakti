import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "wine_class"


def load_data(raw_path: Path) -> pd.DataFrame:
    return pd.read_csv(raw_path)


def preprocess_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    data = data.copy()
    numeric_columns = [column for column in data.columns if column != TARGET_COLUMN]

    missing_before = data.isna().sum().to_dict()
    for column in numeric_columns:
        data[column] = data[column].fillna(data[column].median())
    if data[TARGET_COLUMN].isna().any():
        data[TARGET_COLUMN] = data[TARGET_COLUMN].fillna(data[TARGET_COLUMN].mode()[0])

    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)
    summary = {
        "dataset": "UCI Wine Dataset",
        "source": "https://archive.ics.uci.edu/dataset/109/wine",
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "target_column": TARGET_COLUMN,
        "target_distribution": data[TARGET_COLUMN].value_counts().sort_index().astype(int).to_dict(),
        "numeric_columns": numeric_columns,
        "missing_before": {key: int(value) for key, value in missing_before.items()},
        "missing_after": {key: int(value) for key, value in data.isna().sum().to_dict().items()},
    }
    return data, summary


def save_outputs(processed_data: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_data.to_csv(output_dir / "processed_wine.csv", index=False)
    with (output_dir / "preprocessing_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def run_preprocessing(raw_path: str, output_dir: str) -> pd.DataFrame:
    raw_data = load_data(Path(raw_path))
    processed_data, summary = preprocess_data(raw_data)
    save_outputs(processed_data, summary, Path(output_dir))
    return processed_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate preprocessing for UCI Wine dataset.")
    parser.add_argument("--raw-path", default="wine_raw/wine.csv")
    parser.add_argument("--output-dir", default="preprocessing/wine_preprocessing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preprocessing(args.raw_path, args.output_dir)
