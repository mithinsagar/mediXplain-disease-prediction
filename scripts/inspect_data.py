"""
Quick inspection utility for the raw Kaggle dataset.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import CONFIG
from utils import get_logger

log = get_logger(__name__)


def _default_csv() -> Path:
    candidates = sorted(CONFIG.paths.raw.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV files in {CONFIG.paths.raw}. "
            "Run `python scripts/download_dataset.py` or place the CSV manually."
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the raw dataset.")
    parser.add_argument("--path", type=Path, default=None, help="Optional CSV path.")
    parser.add_argument("--top", type=int, default=20, help="Rows of class distribution.")
    args = parser.parse_args()

    path = args.path or _default_csv()
    df = pd.read_csv(path)
    print(f"Path: {path}")
    print(f"Shape: {df.shape}")
    print("Columns preview:", df.columns.tolist()[:10], "...")
    print(f"Missing values (total): {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    disease_col = df.columns[0]
    print(f"Detected label column: {disease_col}")
    print(f"Number of classes: {df[disease_col].nunique()}")
    print(f"Top-{args.top} classes by frequency:")
    print(df[disease_col].value_counts().head(args.top).to_string())


if __name__ == "__main__":
    main()
