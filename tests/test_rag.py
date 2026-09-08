"""
Tests for the RAG package.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import pytest

from rag.knowledge_base import build_paragraphs
from rag.retriever import _boost_for_disease, _symptom_overlap_boost


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


def test_symptom_overlap_boost_scales_with_jaccard_overlap():
    candidate_symptoms = [
        ["fever", "cough", "fatigue"],  # identical to the query -> full overlap
        ["fever", "headache"],  # 1 shared symptom out of 4 in the union
        ["rash", "itching"],  # no shared symptoms
    ]
    boosts = _symptom_overlap_boost(candidate_symptoms, ["fever", "cough", "fatigue"], 0.4)
    assert boosts.tolist() == pytest.approx([0.4, 0.4 * (1 / 4), 0.0])


def test_symptom_overlap_boost_is_zero_with_no_symptoms():
    # An empty query or an empty candidate symptom list can't overlap with
    # anything, so both should fall back to no boost rather than dividing by
    # zero.
    assert _symptom_overlap_boost([["fever"]], [], 0.5).tolist() == pytest.approx([0.0])
    assert _symptom_overlap_boost([[]], ["fever"], 0.5).tolist() == pytest.approx([0.0])


def test_faiss_retriever_uses_symptom_overlap_to_rank_candidates(tmp_path, monkeypatch):
    """When embedding scores tie, the candidate whose known symptoms overlap
    the reported symptoms should be ranked first."""
    import sys
    import types

    import joblib
    import numpy as np

    from config import CONFIG

    monkeypatch.setattr(CONFIG.paths, "models", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    diseases = ["cold", "flu"]
    paragraphs = ["Cold paragraph.", "Flu paragraph."]
    symptoms_by_disease = [
        ["runny nose", "sneezing"],  # cold: no overlap with the reported symptoms
        ["fever", "cough", "fatigue"],  # flu: exactly matches the reported symptoms
    ]

    class FakeIndex:
        ntotal = len(diseases)

        def search(self, vec, k):
            # Both candidates tie on raw embedding similarity, so only the
            # symptom-overlap boost can separate them. Neither disease name
            # matches the (deliberately different) predicted label below, so
            # the disease-name boost stays out of the picture.
            scores = np.array([[0.5, 0.5]], dtype=np.float32)
            indices = np.array([[0, 1]], dtype=np.int64)
            return scores, indices

    class FakeEncoder:
        def __init__(self, name):
            self.name = name

        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            return np.zeros((len(texts), 4), dtype=np.float32)

    monkeypatch.setitem(
        sys.modules, "faiss", types.SimpleNamespace(read_index=lambda path: FakeIndex())
    )
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeEncoder)
    )

    joblib.dump(
        {
            "diseases": diseases,
            "paragraphs": paragraphs,
            "symptoms": symptoms_by_disease,
            "encoder_name": "fake-encoder",
        },
        tmp_path / "rag_metadata.pkl",
    )
    (tmp_path / "rag_index.faiss").write_text("stub")

    from rag.retriever import FAISSRetriever

    retriever = FAISSRetriever()
    results = retriever.retrieve("influenza", ["fever", "cough", "fatigue"], top_k=1)
    assert results[0]["disease"] == "flu"
