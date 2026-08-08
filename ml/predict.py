"""
Command-line inference helper for MediXplain.

Usage:
    python -m ml.predict --symptoms "fever, cough, sore throat"

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import argparse
import json

from app.predictor import DiseasePredictor
from utils import get_logger, validate_min_symptoms, validate_symptom_list

log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a disease from symptoms.")
    parser.add_argument(
        "--symptoms",
        required=True,
        help="Comma-separated list of symptoms, e.g. 'fever, cough, headache'.",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of differential predictions to show."
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Skip the RAG retrieval step (useful for smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw = [s for s in args.symptoms.split(",")]
    symptoms = validate_symptom_list(raw)
    validate_min_symptoms(symptoms, minimum=3)

    predictor = DiseasePredictor(load_rag=not args.no_rag)
    result = predictor.predict(symptoms, top_k=args.top_k)

    # Strip embeddings and any non-JSON-friendly fields before printing.
    printable = {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "retrieval_score": result.get("retrieval_score"),
        "top_predictions": result["top_predictions"],
        "evidence": [
            {"disease": e["disease"], "score": round(e["score"], 4), "paragraph": e["paragraph"]}
            for e in result.get("evidence", [])
        ],
        "explanation": result["explanation"],
    }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
