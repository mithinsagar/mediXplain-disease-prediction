"""
Shared constants for MediXplain.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

TARGET_COLUMN_CANDIDATES: tuple[str, ...] = (
    "prognosis",
    "disease",
    "label",
    "target",
    "diseases",
)

# Files produced by ml.train_models and consumed by app.predictor
MODEL_FILES: dict[str, str] = {
    "best_model": "best_model.pkl",
    "best_name": "best_model_name.txt",
    "label_encoder": "label_encoder.pkl",
    "feature_columns": "feature_columns.pkl",
    "results_csv": "model_results.csv",
}

# Files produced by the RAG index builders
RAG_FILES: dict[str, str] = {
    "faiss_index": "rag_index.faiss",
    "faiss_metadata": "rag_metadata.pkl",
    "tfidf_vectorizer": "rag_vectorizer.pkl",
    "tfidf_documents": "rag_documents.pkl",
    "tfidf_matrix": "rag_matrix.pkl",
}

# Symbolic values used inside the Flask UI
UI_MIN_SYMPTOMS: int = 3
UI_TOP_K: int = 3

# Boolean-like string mapping used when normalising the raw dataset
BOOL_MAP: dict[str, int] = {
    "yes": 1, "present": 1, "true": 1, "1": 1, "positive": 1,
    "no": 0, "absent": 0, "false": 0, "0": 0, "negative": 0,
}
