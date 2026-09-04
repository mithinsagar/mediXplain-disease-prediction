"""
Tests for the training/evaluation visualisation script.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ml.visualize_results import plot_accuracy_comparison, plot_confusion_matrix


@pytest.fixture()
def results_csv(tmp_path):
    path = tmp_path / "model_results.csv"
    pd.DataFrame(
        [
            {"model": "logistic_regression", "accuracy": 0.83, "time_seconds": 1.2, "status": "SUCCESS"},
            {"model": "decision_tree", "accuracy": 0.11, "time_seconds": 0.4, "status": "SUCCESS"},
            {"model": "adaboost", "accuracy": None, "time_seconds": 0.9, "status": "FAILED: oops"},
        ]
    ).to_csv(path, index=False)
    return path


@pytest.fixture()
def report_json(tmp_path):
    """A tiny 3-class confusion matrix matching a classification_report subset."""
    path = tmp_path / "best_model_report.json"
    payload = {
        "classification_report": {
            "0": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 5},
            "1": {"precision": 0.8, "recall": 0.8, "f1-score": 0.8, "support": 3},
            "2": {"precision": 0.5, "recall": 0.5, "f1-score": 0.5, "support": 1},
            "accuracy": 0.8,
            "macro avg": {"precision": 0.77, "recall": 0.77, "f1-score": 0.77, "support": 9},
            "weighted avg": {"precision": 0.83, "recall": 0.83, "f1-score": 0.83, "support": 9},
        },
        "confusion_matrix": [[5, 0, 0], [1, 2, 0], [0, 0, 1]],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_plot_accuracy_comparison_writes_a_png(results_csv, tmp_path):
    output = plot_accuracy_comparison(results_csv, tmp_path / "accuracy_comparison.png")
    assert output.exists()
    assert output.stat().st_size > 0
    assert plt.get_fignums() == []


def test_plot_accuracy_comparison_drops_failed_models(results_csv, tmp_path):
    df = pd.read_csv(results_csv).dropna(subset=["accuracy"])
    assert "adaboost" not in df["model"].values
    assert len(df) == 2


def test_plot_confusion_matrix_writes_a_png(report_json, tmp_path):
    class_names = ["flu", "cold", "migraine"]
    output = plot_confusion_matrix(report_json, class_names, tmp_path / "confusion_matrix.png")
    assert output.exists()
    assert output.stat().st_size > 0
    assert plt.get_fignums() == []


def test_plot_confusion_matrix_respects_top_n(report_json, tmp_path):
    class_names = ["flu", "cold", "migraine"]
    output = plot_confusion_matrix(report_json, class_names, tmp_path / "cm.png", top_n=2)
    assert output.exists()


def test_plot_confusion_matrix_rejects_mismatched_label_count(tmp_path):
    bad_report = tmp_path / "bad_report.json"
    bad_report.write_text(
        json.dumps(
            {
                "classification_report": {"0": {"support": 5}, "1": {"support": 3}},
                "confusion_matrix": [[5, 0, 0], [1, 2, 0], [0, 0, 1]],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        plot_confusion_matrix(bad_report, ["flu", "cold", "migraine"], tmp_path / "cm.png")
