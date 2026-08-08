"""
Train and benchmark the classical scikit-learn model zoo.

For every model defined in `config/model_config.json` this script:

1. Fits the estimator on the stratified training split.
2. Evaluates accuracy on the hold-out test split.
3. Persists the fitted estimator to `models/<name>.pkl`.
4. Writes a per-model classification report to `models/<name>_report.txt`.
5. Aggregates every result into `models/model_results.csv`.
6. Selects the highest-accuracy model as `models/best_model.pkl` and records
   its name in `models/best_model_name.txt`.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from config import CONFIG, BASE_DIR
from ml.preprocess import SplitData, filter_rare_classes, make_split
from utils import MODEL_FILES, get_logger, save_pickle, write_json

log = get_logger(__name__)

MODEL_CONFIG_PATH: Path = BASE_DIR / "config" / "model_config.json"


def _load_hyperparameters() -> dict[str, dict[str, Any]]:
    with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def build_model_zoo(hyperparameters: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Instantiate every estimator with hyperparameters loaded from JSON."""
    zoo: dict[str, Any] = {
        "logistic_regression": LogisticRegression(**hyperparameters["logistic_regression"]),
        "bernoulli_nb": BernoulliNB(**hyperparameters["bernoulli_nb"]),
        "linear_svm": LinearSVC(**hyperparameters["linear_svm"]),
        "decision_tree": DecisionTreeClassifier(**hyperparameters["decision_tree"]),
        "random_forest": RandomForestClassifier(**hyperparameters["random_forest"]),
        "adaboost": AdaBoostClassifier(**hyperparameters["adaboost"]),
        "mlp": MLPClassifier(**hyperparameters["mlp"]),
    }
    return zoo


def _load_processed_dataset() -> tuple[pd.DataFrame, str]:
    processed_csv = CONFIG.paths.processed / "disease_dataset_clean.csv"
    if not processed_csv.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_csv}. "
            "Run `python -m ml.prepare_dataset` first."
        )
    df = pd.read_csv(processed_csv)
    metadata = CONFIG.paths.processed / "metadata.json"
    target_col = df.columns[0]
    if metadata.exists():
        with metadata.open("r", encoding="utf-8") as handle:
            target_col = json.load(handle).get("target_col", target_col)
    return df, target_col


def train_and_evaluate(split: SplitData) -> tuple[list[dict[str, Any]], str, Any]:
    """Fit every model, save artifacts, and return the winning estimator."""
    hyperparameters = _load_hyperparameters()
    models = build_model_zoo(hyperparameters)

    all_label_ids = np.arange(len(split.label_encoder.classes_))
    results: list[dict[str, Any]] = []
    best_model: Any = None
    best_name: str = ""
    best_acc: float = -1.0

    for name, model in models.items():
        log.info("=" * 60)
        log.info("Training %s...", name)
        start = time.time()
        try:
            model.fit(split.X_train, split.y_train)
            preds = model.predict(split.X_test)
            acc = accuracy_score(split.y_test, preds)
            elapsed = time.time() - start
            log.info("%s -> accuracy=%.4f time=%.2fs", name, acc, elapsed)

            model_path = CONFIG.paths.models / f"{name}.pkl"
            save_pickle(model, model_path)

            report = classification_report(
                split.y_test,
                preds,
                labels=all_label_ids,
                target_names=split.label_encoder.classes_,
                zero_division=0,
            )
            (CONFIG.paths.models / f"{name}_report.txt").write_text(report, encoding="utf-8")

            results.append(
                {
                    "model": name,
                    "accuracy": round(float(acc), 4),
                    "time_seconds": round(elapsed, 2),
                    "status": "SUCCESS",
                }
            )
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_name = name
        except Exception as exc:  # pragma: no cover - training resilience
            elapsed = time.time() - start
            log.exception("%s failed after %.2fs: %s", name, elapsed, exc)
            results.append(
                {
                    "model": name,
                    "accuracy": None,
                    "time_seconds": round(elapsed, 2),
                    "status": f"FAILED: {exc}",
                }
            )

    return results, best_name, best_model


def main() -> None:
    df, target_col = _load_processed_dataset()
    df = filter_rare_classes(df, target_col)
    split = make_split(df, target_col)

    save_pickle(split.label_encoder, CONFIG.paths.models / MODEL_FILES["label_encoder"])
    save_pickle(split.feature_columns, CONFIG.paths.models / MODEL_FILES["feature_columns"])

    filtered_csv = CONFIG.paths.processed / "filtered_disease_dataset.csv"
    df.to_csv(filtered_csv, index=False)

    results, best_name, best_model = train_and_evaluate(split)

    results_df = pd.DataFrame(results)
    results_csv = CONFIG.paths.models / MODEL_FILES["results_csv"]
    results_df.to_csv(results_csv, index=False)
    log.info("Training summary:\n%s", results_df.to_string(index=False))

    if best_model is None:
        log.error("No model was trained successfully.")
        return

    save_pickle(best_model, CONFIG.paths.models / MODEL_FILES["best_model"])
    (CONFIG.paths.models / MODEL_FILES["best_name"]).write_text(best_name, encoding="utf-8")
    write_json(
        {"best_model": best_name, "results": results},
        CONFIG.paths.models / "training_summary.json",
    )
    log.info("Best model: %s (accuracy=%.4f)", best_name, max(r["accuracy"] or 0 for r in results))


if __name__ == "__main__":
    main()
