"""
TF-IDF fallback for the RAG index.

Used when FAISS or sentence-transformers is not available (for example on
constrained CI runners). Produces three artifacts under `models/`:

- `rag_vectorizer.pkl`  - the fitted TfidfVectorizer
- `rag_documents.pkl`   - list of {"disease", "text"} dictionaries
- `rag_matrix.pkl`      - sparse TF-IDF matrix aligned with rag_documents

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config import CONFIG
from rag.knowledge_base import build_paragraphs
from utils import RAG_FILES, get_logger, load_pickle, save_pickle

log = get_logger(__name__)


def _load_dataset() -> tuple[pd.DataFrame, str]:
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


def _feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    feature_columns_path = CONFIG.paths.models / "feature_columns.pkl"
    if feature_columns_path.exists():
        return list(load_pickle(feature_columns_path))
    return [c for c in df.columns if c not in {target_col, "symptom_text"}]


def build_index() -> Path:
    df, target_col = _load_dataset()
    feature_cols = _feature_columns(df, target_col)
    kb = build_paragraphs(df, target_col=target_col, feature_cols=feature_cols)

    documents = [{"disease": row.disease, "text": row.paragraph} for row in kb.itertuples()]
    texts = [doc["text"] for doc in documents]

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    save_pickle(vectorizer, CONFIG.paths.models / RAG_FILES["tfidf_vectorizer"])
    save_pickle(documents, CONFIG.paths.models / RAG_FILES["tfidf_documents"])
    save_pickle(matrix, CONFIG.paths.models / RAG_FILES["tfidf_matrix"])

    log.info("TF-IDF index written (%s documents).", len(documents))
    return CONFIG.paths.models / RAG_FILES["tfidf_vectorizer"]


def main() -> None:
    build_index()


if __name__ == "__main__":
    main()
