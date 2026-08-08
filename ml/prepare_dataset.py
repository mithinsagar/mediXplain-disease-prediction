"""
Prepare the raw Kaggle CSV for training.

Steps:
1. Locate the raw CSV inside `data/raw/`.
2. Normalise column names to snake_case.
3. Detect the target column heuristically.
4. Drop duplicate rows and coerce every feature column to strict {0, 1}.
5. Emit a `symptom_text` column for the deep-learning branch and the RAG
   knowledge-base builder.
6. Persist the cleaned dataset to `data/processed/disease_dataset_clean.csv`
   and a metadata sidecar `data/processed/metadata.json`.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG
from utils import get_logger, write_json
from utils.constants import BOOL_MAP, TARGET_COLUMN_CANDIDATES

log = get_logger(__name__)


def _normalise_column(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    return col.strip("_")


def _find_target_column(df: pd.DataFrame) -> str:
    for column in df.columns:
        if column in TARGET_COLUMN_CANDIDATES:
            return column
    # Fall back to the first column, which is the layout used by the Kaggle CSV.
    return df.columns[0]


def _load_raw_csv(raw_dir: Path) -> pd.DataFrame:
    candidates = sorted(raw_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Place the Kaggle "
            "'Final_Augmented_dataset_Diseases_and_Symptoms.csv' file there, "
            "or run `python scripts/download_dataset.py`."
        )
    csv_path = candidates[0]
    log.info("Loading raw dataset: %s", csv_path)
    return pd.read_csv(csv_path)


def _coerce_binary(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        mapped = series.astype(str).str.strip().str.lower().map(BOOL_MAP)
        if mapped.notna().mean() > 0.9:
            return mapped.fillna(0).astype(int)
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def prepare_dataset() -> Path:
    """Prepare the dataset and return the path to the cleaned CSV."""
    paths = CONFIG.paths
    df = _load_raw_csv(paths.raw)
    original_shape = df.shape
    log.info("Original shape: %s", original_shape)

    df.columns = [_normalise_column(c) for c in df.columns]
    target_col = _find_target_column(df)
    log.info("Detected target column: %s", target_col)

    df = df.drop_duplicates().copy()
    df = df.replace(["", " ", "None", "nan", "NaN"], np.nan)

    feature_cols = [c for c in df.columns if c != target_col]
    df[feature_cols] = df[feature_cols].fillna(0)
    for col in feature_cols:
        df[col] = _coerce_binary(df[col])

    df[target_col] = df[target_col].astype(str).str.strip()
    df = df[df[target_col] != ""]

    # Build the symptom_text auxiliary column.
    symptom_texts: list[str] = []
    for _, row in df.iterrows():
        present = [c.replace("_", " ") for c in feature_cols if int(row[c]) == 1]
        symptom_texts.append(", ".join(present))
    df["symptom_text"] = symptom_texts

    processed_csv = paths.processed / "disease_dataset_clean.csv"
    df.to_csv(processed_csv, index=False)
    log.info("Wrote cleaned dataset: %s (%s rows)", processed_csv, len(df))

    metadata = {
        "author": "Mithin Sagar S",
        "github": "https://github.com/mithinsagar",
        "project": "MediXplain - Explainable Disease Prediction System",
        "original_shape": list(original_shape),
        "processed_shape": list(df.shape),
        "target_col": target_col,
        "n_features": len(feature_cols),
        "n_classes": int(df[target_col].nunique()),
        "feature_columns_preview": feature_cols[:25],
        "processed_csv": str(processed_csv),
    }
    write_json(metadata, paths.processed / "metadata.json")
    log.info("Wrote metadata sidecar: metadata.json")

    return processed_csv


def main() -> None:
    prepare_dataset()


if __name__ == "__main__":
    main()
