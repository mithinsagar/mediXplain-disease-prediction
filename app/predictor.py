"""
DiseasePredictor: single entry point for MediXplain inference.

Loads the best classical model, the label encoder, the feature columns, and
(optionally) a RAG retriever. Combines an ML prediction with retrieved
evidence into a single dictionary consumed by both the Flask app and the
CLI predict script.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

from typing import Any, Iterable

from config import CONFIG
from ml.feature_engineering import symptom_vector
from rag.retriever import BaseRetriever, load_retriever
from utils import MODEL_FILES, get_logger, load_pickle, top_k_indices

log = get_logger(__name__)


class DiseasePredictor:
    """High-level inference API used by the Flask app and CLI."""

    def __init__(self, load_rag: bool = True) -> None:
        models_dir = CONFIG.paths.models
        log.info("Loading model artifacts from %s", models_dir)
        self.model = load_pickle(models_dir / MODEL_FILES["best_model"])
        self.label_encoder = load_pickle(models_dir / MODEL_FILES["label_encoder"])
        self.feature_columns: list[str] = list(
            load_pickle(models_dir / MODEL_FILES["feature_columns"])
        )

        best_name_path = models_dir / MODEL_FILES["best_name"]
        self.best_model_name: str = (
            best_name_path.read_text(encoding="utf-8").strip()
            if best_name_path.exists()
            else type(self.model).__name__
        )

        self.retriever: BaseRetriever | None = None
        if load_rag:
            try:
                self.retriever = load_retriever()
            except Exception as exc:  # pragma: no cover - retriever optional
                log.warning("RAG retriever unavailable: %s", exc)
                self.retriever = None

    # ------------------------------------------------------------------ helpers
    @property
    def num_features(self) -> int:
        return len(self.feature_columns)

    @property
    def num_classes(self) -> int:
        return int(len(self.label_encoder.classes_))

    # -------------------------------------------------------------- inference
    def predict(
        self, symptoms: Iterable[str], top_k: int | None = None
    ) -> dict[str, Any]:
        """Run the full pipeline and return a structured response."""
        top_k = top_k or CONFIG.rag.top_k
        symptoms = list(symptoms)
        input_df = symptom_vector(symptoms, self.feature_columns)

        top_predictions: list[dict[str, Any]]
        confidence: float | None = None

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(input_df)[0]
            indices = top_k_indices(probs, top_k)
            top_predictions = [
                {
                    "disease": self.label_encoder.inverse_transform([int(i)])[0],
                    "confidence": round(float(probs[int(i)]) * 100, 2),
                }
                for i in indices
            ]
            confidence = top_predictions[0]["confidence"]
        else:
            pred_encoded = int(self.model.predict(input_df)[0])
            top_predictions = [
                {
                    "disease": self.label_encoder.inverse_transform([pred_encoded])[0],
                    "confidence": None,
                }
            ]

        prediction = top_predictions[0]["disease"]

        evidence: list[dict[str, Any]] = []
        retrieval_score: float | None = None
        if self.retriever is not None:
            try:
                evidence = self.retriever.retrieve(prediction, symptoms, top_k=top_k)
                if evidence:
                    retrieval_score = round(float(evidence[0]["score"]), 4)
            except Exception as exc:  # pragma: no cover - retriever robustness
                log.warning("RAG retrieval failed: %s", exc)

        explanation = self._build_explanation(
            prediction=prediction,
            symptoms=symptoms,
            confidence=confidence,
            evidence=evidence,
        )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "retrieval_score": retrieval_score,
            "top_predictions": top_predictions,
            "evidence": evidence,
            "explanation": explanation,
            "best_model_name": self.best_model_name,
        }

    # ------------------------------------------------------------- narrative
    @staticmethod
    def _build_explanation(
        prediction: str,
        symptoms: list[str],
        confidence: float | None,
        evidence: list[dict[str, Any]],
    ) -> str:
        symptom_text = ", ".join(s.replace("_", " ") for s in symptoms) or "not provided"
        parts: list[str] = [
            f"The model predicts {prediction} based on the symptom pattern: {symptom_text}."
        ]
        if confidence is not None:
            parts.append(f"The calibrated model confidence is {confidence:.2f} percent.")
        if evidence:
            parts.append(
                "The RAG module retrieved the most relevant medical paragraph: "
                + evidence[0]["paragraph"]
            )
        parts.append(
            "MediXplain surfaces both the ML confidence and the retrieved evidence "
            "so the recommendation can be interpreted by a human, not just trusted "
            "blindly. This tool is educational and does not replace a clinician."
        )
        return " ".join(parts)
