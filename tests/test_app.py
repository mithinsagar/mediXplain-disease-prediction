"""
Tests for the Flask application.

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
def client(tmp_path, monkeypatch, tiny_dataset):
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

    # Force retriever to no-op path for tests.
    monkeypatch.setattr(CONFIG.rag, "backend", "tfidf")

    from app.app import create_app

    app = create_app()
    app.config["TESTING"] = True

    # Reset the routes-level predictor cache so it picks up our stubs.
    import app.routes as routes

    routes._PREDICTOR = None

    with app.test_client() as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["author"] == "Mithin Sagar S"


def test_home_get_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.data.decode("utf-8")
    assert "MediXplain" in body or "Medi" in body


def test_predict_api_requires_min_symptoms(client):
    response = client.post("/api/predict", json={"symptoms": ["fever"]})
    assert response.status_code == 400
    assert "at least" in response.get_json()["error"].lower()
