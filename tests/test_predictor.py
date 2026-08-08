"""
Tests for `app.predictor.DiseasePredictor` using a stubbed model set.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


@pytest.fixture()
def stub_predictor(tmp_path, monkeypatch, tiny_dataset):
    """Train a tiny model and persist the artifacts DiseasePredictor expects."""
    from config import CONFIG
    from utils import MODEL_FILES

    monkeypatch.setattr(CONFIG.paths, "models", tmp_path)

    feature_cols = [c for c in tiny_dataset.columns if c != "disease"]
    X = tiny_dataset[feature_cols].values
    y = tiny_dataset["disease"].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    model = LogisticRegression(max_iter=500)
    model.fit(X, y_encoded)

    joblib.dump(model, tmp_path / MODEL_FILES["best_model"])
    joblib.dump(encoder, tmp_path / MODEL_FILES["label_encoder"])
    joblib.dump(feature_cols, tmp_path / MODEL_FILES["feature_columns"])
    (tmp_path / MODEL_FILES["best_name"]).write_text("logistic_regression", encoding="utf-8")

    from app.predictor import DiseasePredictor

    return DiseasePredictor(load_rag=False), feature_cols


def test_predictor_loads_expected_metadata(stub_predictor, tiny_dataset):
    predictor, feature_cols = stub_predictor
    assert predictor.num_features == len(feature_cols)
    assert predictor.num_classes == tiny_dataset["disease"].nunique()
    assert predictor.best_model_name == "logistic_regression"


def test_predictor_returns_top_k(stub_predictor):
    predictor, feature_cols = stub_predictor
    result = predictor.predict(["fever", "cough", "fatigue"], top_k=3)
    assert "prediction" in result
    assert result["confidence"] is None or 0 <= result["confidence"] <= 100
    assert len(result["top_predictions"]) == 3
    assert result["evidence"] == []  # RAG disabled in fixture
    assert "explanation" in result and isinstance(result["explanation"], str)


def test_predictor_handles_unknown_symptoms(stub_predictor):
    predictor, _ = stub_predictor
    result = predictor.predict(["not_a_symptom", "also_missing", "fever"], top_k=2)
    assert isinstance(result["prediction"], str)
    assert isinstance(result["explanation"], str)
