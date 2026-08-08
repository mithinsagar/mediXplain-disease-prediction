"""
Retriever classes for MediXplain.

Two implementations are provided:

- FAISSRetriever  : uses sentence-transformers + FAISS `IndexFlatIP` on
                    L2-normalised vectors (cosine similarity).
- TFIDFRetriever  : uses scikit-learn TF-IDF + cosine similarity.

The `load_retriever` helper returns whichever backend the runtime
configuration requests, transparently falling back to TF-IDF if FAISS is not
available.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import CONFIG
from utils import RAG_FILES, get_logger, load_pickle, top_k_indices

log = get_logger(__name__)


@dataclass
class RetrievedParagraph:
    disease: str
    paragraph: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "disease": self.disease,
            "paragraph": self.paragraph,
            "score": float(self.score),
        }


class BaseRetriever:
    """Common interface for all retriever backends."""

    backend: str = "base"

    def retrieve(
        self, predicted_disease: str, symptoms: Iterable[str], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class FAISSRetriever(BaseRetriever):
    backend = "faiss"

    def __init__(self) -> None:
        try:
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "FAISSRetriever requires `faiss-cpu` and `sentence-transformers`."
            ) from exc

        index_path: Path = CONFIG.paths.models / RAG_FILES["faiss_index"]
        metadata_path: Path = CONFIG.paths.models / RAG_FILES["faiss_metadata"]
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "FAISS index or metadata missing. Run "
                "`python -m rag.build_faiss_index` first."
            )

        self._faiss = faiss
        self.index = faiss.read_index(str(index_path))
        metadata = load_pickle(metadata_path)
        self.diseases: list[str] = list(metadata["diseases"])
        self.paragraphs: list[str] = list(metadata["paragraphs"])
        self.encoder_name: str = metadata["encoder_name"]
        self.encoder = SentenceTransformer(self.encoder_name)
        log.info("FAISS retriever ready (%s vectors).", self.index.ntotal)

    def retrieve(
        self, predicted_disease: str, symptoms: Iterable[str], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        top_k = top_k or CONFIG.rag.top_k
        symptom_str = ", ".join(symptoms)
        query = f"Disease: {predicted_disease}. Symptoms: {symptom_str}"
        vec = self.encoder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        scores, indices = self.index.search(vec, top_k)

        out: list[dict[str, Any]] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0:
                continue
            out.append(
                RetrievedParagraph(
                    disease=self.diseases[idx],
                    paragraph=self.paragraphs[idx],
                    score=float(score),
                ).as_dict()
            )
        return out


class TFIDFRetriever(BaseRetriever):
    backend = "tfidf"

    def __init__(self) -> None:
        vec_path = CONFIG.paths.models / RAG_FILES["tfidf_vectorizer"]
        docs_path = CONFIG.paths.models / RAG_FILES["tfidf_documents"]
        matrix_path = CONFIG.paths.models / RAG_FILES["tfidf_matrix"]
        for p in (vec_path, docs_path, matrix_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"TF-IDF artifact missing: {p}. Run "
                    "`python -m rag.build_tfidf_index` first."
                )
        self.vectorizer = load_pickle(vec_path)
        self.documents: list[dict[str, str]] = load_pickle(docs_path)
        self.matrix = load_pickle(matrix_path)
        log.info("TF-IDF retriever ready (%s documents).", len(self.documents))

    def retrieve(
        self, predicted_disease: str, symptoms: Iterable[str], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        top_k = top_k or CONFIG.rag.top_k
        query = f"{predicted_disease} " + " ".join(symptoms)
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.matrix).flatten()
        indices = top_k_indices(similarities, top_k)
        results: list[dict[str, Any]] = []
        for idx in indices:
            doc = self.documents[int(idx)]
            results.append(
                RetrievedParagraph(
                    disease=doc["disease"],
                    paragraph=doc["text"],
                    score=float(similarities[int(idx)]),
                ).as_dict()
            )
        return results


def load_retriever(backend: str | None = None) -> BaseRetriever:
    """Return a retriever instance for the requested backend."""
    backend = (backend or CONFIG.rag.backend).lower()

    if backend == "faiss":
        try:
            return FAISSRetriever()
        except (ImportError, FileNotFoundError) as exc:
            log.warning("Falling back to TF-IDF retriever: %s", exc)
            return TFIDFRetriever()

    if backend == "tfidf":
        return TFIDFRetriever()

    raise ValueError(f"Unknown RAG backend '{backend}'. Use 'faiss' or 'tfidf'.")
