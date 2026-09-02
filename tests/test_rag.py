"""
Tests for the RAG package.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import pytest

from rag.knowledge_base import build_paragraphs
from rag.retriever import _boost_for_disease


def test_build_paragraphs_creates_one_row_per_disease(tiny_dataset):
    kb = build_paragraphs(
        tiny_dataset,
        target_col="disease",
        feature_cols=[c for c in tiny_dataset.columns if c != "disease"],
        limit=6,
    )
    assert set(kb["disease"]) == set(tiny_dataset["disease"].unique())
    assert all(isinstance(text, str) and text for text in kb["paragraph"])


def test_paragraph_contains_disease_name(tiny_dataset):
    kb = build_paragraphs(tiny_dataset, target_col="disease")
    for row in kb.itertuples():
        assert row.disease in row.paragraph


def test_tfidf_retriever_smoke(tmp_path, monkeypatch, tiny_dataset):
    """End-to-end smoke test for the TF-IDF retriever using tmp artifacts."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    from config import CONFIG

    monkeypatch.setattr(CONFIG.paths, "models", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    kb = build_paragraphs(tiny_dataset, target_col="disease")
    documents = [{"disease": r.disease, "text": r.paragraph} for r in kb.itertuples()]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([d["text"] for d in documents])

    import joblib

    joblib.dump(vectorizer, tmp_path / "rag_vectorizer.pkl")
    joblib.dump(documents, tmp_path / "rag_documents.pkl")
    joblib.dump(matrix, tmp_path / "rag_matrix.pkl")

    from rag.retriever import TFIDFRetriever

    retriever = TFIDFRetriever()
    results = retriever.retrieve("flu", ["fever", "cough", "fatigue"], top_k=2)
    assert len(results) == 2
    assert all("disease" in r and "paragraph" in r and "score" in r for r in results)
    assert results[0]["score"] >= results[-1]["score"]


def test_boost_for_disease_matches_case_insensitively():
    diseases = ["Flu", "cold", "MIGRAINE"]
    boosts = _boost_for_disease(diseases, "flu", 0.2)
    assert boosts.tolist() == pytest.approx([0.2, 0.0, 0.0])

    boosts = _boost_for_disease(diseases, "Migraine", 0.2)
    assert boosts.tolist() == pytest.approx([0.0, 0.0, 0.2])


def test_tfidf_retriever_boosts_matching_disease(tmp_path, monkeypatch):
    """A same-disease paragraph should outrank an equally-similar off-disease one."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    from config import CONFIG

    monkeypatch.setattr(CONFIG.paths, "models", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    # Both documents share identical wording, so without the disease boost
    # they would tie on cosine similarity and ranking would be arbitrary.
    documents = [
        {"disease": "cold", "text": "fever cough fatigue"},
        {"disease": "flu", "text": "fever cough fatigue"},
    ]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([d["text"] for d in documents])

    import joblib

    joblib.dump(vectorizer, tmp_path / "rag_vectorizer.pkl")
    joblib.dump(documents, tmp_path / "rag_documents.pkl")
    joblib.dump(matrix, tmp_path / "rag_matrix.pkl")

    from rag.retriever import TFIDFRetriever

    retriever = TFIDFRetriever()
    results = retriever.retrieve("flu", ["fever", "cough", "fatigue"], top_k=1)
    assert results[0]["disease"] == "flu"
