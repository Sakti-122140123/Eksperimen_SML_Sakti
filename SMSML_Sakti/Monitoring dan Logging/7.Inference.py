"""Send sample inference traffic to the local model API."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR.parent / "Membangun_model/wine_preprocessing/processed_wine.csv"
TARGET_COLUMN = "wine_class"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate inference traffic for monitoring.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = pd.read_csv(args.data_path)
    features = dataset.drop(columns=[TARGET_COLUMN])

    for index in range(args.iterations):
        batch = features.sample(n=args.batch_size, replace=True, random_state=index)
        payload = {"instances": batch.to_dict(orient="records")}
        response = requests.post(args.url, json=payload, timeout=15)
        print(f"{index + 1:03d} status={response.status_code} response={response.text[:160]}")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
