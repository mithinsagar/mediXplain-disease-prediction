"""
Render explainability plots from persisted training/evaluation artifacts.

Reads `models/model_results.csv` (written by `ml.train_models`) and
`models/best_model_report.json` (written by `ml.evaluate_models`) and saves:

- `models/accuracy_comparison.png`: per-model held-out accuracy bar chart.
- `models/confusion_matrix.png`: confusion matrix heatmap for the best model,
  restricted to the most frequent classes so ~773 diseases don't render as an
  unreadable grid.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import CONFIG
from utils import MODEL_FILES, get_logger, load_pickle

log = get_logger(__name__)

DEFAULT_TOP_N_CLASSES = 15


def plot_accuracy_comparison(results_csv: Path, output_path: Path) -> Path:
    """Bar chart of held-out accuracy per model in the zoo."""
    df = pd.read_csv(results_csv)
    df = df.dropna(subset=["accuracy"]).sort_values("accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["model"], df["accuracy"], color="#2f6f4f")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    ax.set_title("Model zoo accuracy comparison")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_confusion_matrix(
    report_json: Path,
    class_names: list[str],
    output_path: Path,
    top_n: int = DEFAULT_TOP_N_CLASSES,
) -> Path:
    """Heatmap of the best model's confusion matrix, top-N classes by support.

    `classification_report` and `confusion_matrix` are both computed by
    `ml.evaluate_models` from the same sorted set of labels present in the
    test split, so the report's numeric keys line up positionally with the
    matrix's rows and columns. That mapping is recovered here rather than
    assumed, since a class absent from the hold-out split would otherwise
    shift every index after it.
    """
    with Path(report_json).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    cm = np.array(payload["confusion_matrix"])
    report = payload["classification_report"]

    present_label_ids = sorted(int(key) for key in report if key.lstrip("-").isdigit())
    if len(present_label_ids) != cm.shape[0]:
        raise ValueError(
            "classification_report label count does not match confusion_matrix size; "
            "cannot map class names to matrix indices"
        )

    support = cm.sum(axis=1)
    top_order = np.argsort(support)[::-1][: min(top_n, len(support))]
    top_order = np.sort(top_order)

    sub_cm = cm[np.ix_(top_order, top_order)]
    labels = [class_names[present_label_ids[i]] for i in top_order]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(sub_cm, cmap="viridis", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (top {len(top_order)} most frequent classes)")
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    results_csv = CONFIG.paths.models / MODEL_FILES["results_csv"]
    report_json = CONFIG.paths.models / "best_model_report.json"
    label_encoder_path = CONFIG.paths.models / MODEL_FILES["label_encoder"]

    if not results_csv.exists():
        raise FileNotFoundError(f"{results_csv} not found. Run `python -m ml.train_models` first.")
    if not report_json.exists() or not label_encoder_path.exists():
        raise FileNotFoundError(
            f"{report_json} or {label_encoder_path} not found. Run `python -m ml.evaluate_models` first."
        )

    accuracy_path = plot_accuracy_comparison(results_csv, CONFIG.paths.models / "accuracy_comparison.png")
    log.info("Wrote %s", accuracy_path)

    label_encoder = load_pickle(label_encoder_path)
    cm_path = plot_confusion_matrix(
        report_json,
        list(label_encoder.classes_),
        CONFIG.paths.models / "confusion_matrix.png",
    )
    log.info("Wrote %s", cm_path)


if __name__ == "__main__":
    main()
