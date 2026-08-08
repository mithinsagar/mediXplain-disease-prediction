"""
Generate a small text file with a handful of sample symptom lists.

The file lands at `data/processed/sample_symptom_input.txt` and can be piped
into `python -m ml.predict` for smoke testing.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from config import CONFIG
from utils import MODEL_FILES, get_logger, load_pickle

log = get_logger(__name__)


def _sample_disease_row(df: pd.DataFrame, target_col: str, seed: int) -> tuple[str, list[str]]:
    random.seed(seed)
    disease = random.choice(df[target_col].unique().tolist())
    group = df[df[target_col] == disease]
    row = group.sample(1, random_state=seed).iloc[0]
    feature_cols = [c for c in df.columns if c != target_col and c != "symptom_text"]
    active = [c.replace("_", " ") for c in feature_cols if int(row[c]) == 1]
    return disease, active[:12]


def main() -> None:
    processed_csv = CONFIG.paths.processed / "disease_dataset_clean.csv"
    if not processed_csv.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_csv}. "
            "Run `python -m ml.prepare_dataset` first."
        )
    df = pd.read_csv(processed_csv)
    target_col = df.columns[0]

    # Guard for the case where training has not run.
    feature_cols_path = CONFIG.paths.models / MODEL_FILES["feature_columns"]
    if feature_cols_path.exists():
        _ = load_pickle(feature_cols_path)  # sanity load

    lines: list[str] = [
        "# MediXplain sample symptom inputs",
        "# Author:  Mithin Sagar S",
        "# GitHub:  https://github.com/mithinsagar",
        "",
    ]
    for seed in (1, 2, 3, 4, 5):
        disease, symptoms = _sample_disease_row(df, target_col, seed)
        joined = ", ".join(symptoms) if symptoms else "no symptoms found"
        lines.append(f"# expected: {disease}")
        lines.append(joined)
        lines.append("")

    out = CONFIG.paths.processed / "sample_symptom_input.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote sample inputs to %s", out)


if __name__ == "__main__":
    main()
