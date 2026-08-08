"""
Flask blueprints for MediXplain.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.llm_explainer import generate_llm_explanation
from app.predictor import DiseasePredictor
from config import CONFIG
from utils import (
    ValidationError,
    get_logger,
    validate_min_symptoms,
    validate_symptom_list,
)

log = get_logger(__name__)

web_bp = Blueprint("web", __name__)
api_bp = Blueprint("api", __name__, url_prefix="/api")

_PREDICTOR: DiseasePredictor | None = None


def _get_predictor() -> DiseasePredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = DiseasePredictor()
    return _PREDICTOR


@web_bp.route("/", methods=["GET", "POST"])
def home():
    predictor = _get_predictor()
    context: dict[str, object] = {
        "symptoms": predictor.feature_columns,
        "num_diseases": predictor.num_classes,
        "num_features": predictor.num_features,
        "best_model_name": predictor.best_model_name,
        "backend": (predictor.retriever.backend if predictor.retriever else "none"),
        "min_required": CONFIG.rag.top_k,
    }

    if request.method == "GET":
        return render_template("index.html", **context, selected_symptoms=[])

    try:
        raw_symptoms = request.form.getlist("symptoms")
        symptoms = validate_symptom_list(raw_symptoms)
        validate_min_symptoms(symptoms, minimum=3)
    except ValidationError as exc:
        return render_template(
            "index.html",
            error_message=str(exc),
            selected_symptoms=raw_symptoms,
            **context,
        )

    result = predictor.predict(symptoms)
    rag_text = result["evidence"][0]["paragraph"] if result["evidence"] else ""
    llm_text, llm_enabled, llm_error = generate_llm_explanation(
        selected_symptoms=symptoms,
        prediction=result["prediction"],
        confidence=result["confidence"] if result["confidence"] is not None else "Not available",
        rag_text=rag_text,
    )

    return render_template(
        "index.html",
        selected_symptoms=symptoms,
        prediction=result["prediction"],
        confidence=result["confidence"],
        retrieval_score=result["retrieval_score"],
        top_predictions=result["top_predictions"],
        explanation=rag_text or result["explanation"],
        llm_explanation=llm_text,
        llm_enabled=llm_enabled,
        llm_error=llm_error,
        **context,
    )


@web_bp.route("/about", methods=["GET"])
def about():
    predictor = _get_predictor()
    return render_template(
        "about.html",
        num_diseases=predictor.num_classes,
        num_features=predictor.num_features,
        best_model_name=predictor.best_model_name,
        backend=(predictor.retriever.backend if predictor.retriever else "none"),
    )


@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "author": "Mithin Sagar S",
            "github": "https://github.com/mithinsagar",
            "project": "MediXplain",
        }
    )


@api_bp.route("/predict", methods=["POST"])
def predict_api():
    """JSON endpoint accepting `{"symptoms": [...], "top_k": 3}`."""
    payload = request.get_json(silent=True) or {}
    raw = payload.get("symptoms", [])

    try:
        symptoms = validate_symptom_list(raw)
        validate_min_symptoms(symptoms, minimum=3)
    except ValidationError as exc:
        return jsonify({"error": str(exc)}), 400

    top_k = int(payload.get("top_k", CONFIG.rag.top_k))
    predictor = _get_predictor()
    result = predictor.predict(symptoms, top_k=top_k)
    return jsonify(result)
