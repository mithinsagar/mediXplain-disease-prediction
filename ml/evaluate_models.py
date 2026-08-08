"""
Re-evaluate the persisted best model against a fresh stratified split.

Writes a JSON artifact under `models/best_model_report.json` containing the
scikit-learn classification report and confusion matrix.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config import CONFIG
from utils import MODEL_FILES, get_logger, load_pickle, write_json

log = get_logger(__name__)


def main() -> None:
    processed_csv = CONFIG.paths.processed / "disease_dataset_clean.csv"
    if not processed_csv.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_csv}. "
            "Run `python -m ml.prepare_dataset` first."
        )

    df = pd.read_csv(processed_csv)
    target_col = df.columns[0]
    feature_cols = load_pickle(CONFIG.paths.models / MODEL_FILES["feature_columns"])
    label_encoder = load_pickle(CONFIG.paths.models / MODEL_FILES["label_encoder"])
    model = load_pickle(CONFIG.paths.models / MODEL_FILES["best_model"])

    X = df[feature_cols]
    y = label_encoder.transform(df[target_col])

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG.training.test_size,
        random_state=CONFIG.training.random_state,
        stratify=y,
    )

    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds).tolist()

    payload = {
        "classification_report": report,
        "confusion_matrix": cm,
        "meta": {
            "author": "Mithin Sagar S",
            "github": "https://github.com/mithinsagar",
            "project": "MediXplain - Explainable Disease Prediction System",
        },
    }
    write_json(payload, CONFIG.paths.models / "best_model_report.json")
    log.info("Wrote models/best_model_report.json")


if __name__ == "__main__":
    main()
