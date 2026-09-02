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


def _boost_for_disease(diseases: Iterable[str], predicted_disease: str, boost: float) -> np.ndarray:
    """Per-candidate score bonus for paragraphs whose disease matches the prediction.

    Pure semantic/TF-IDF similarity can rank a paragraph from a different but
    symptomatically-similar disease above the paragraph that actually explains
    the model's prediction. Nudging same-disease candidates keeps the RAG
    evidence on-topic without hard-filtering out genuinely useful neighbours.
    """
    target = predicted_disease.strip().casefold()
    return np.array(
        [boost if disease.strip().casefold() == target else 0.0 for disease in diseases],
        dtype=np.float32,
    )


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
                "FAISS index or metadata missing. Run " "`python -m rag.build_faiss_index` first."
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

        # Search a wider candidate pool than requested so the disease boost
        # below can pull a same-disease paragraph back above a merely
        # symptom-similar one before we truncate to top_k.
        search_k = min(max(top_k * 4, 20), self.index.ntotal)
        scores, indices = self.index.search(vec, search_k)

        candidates = [(s, i) for s, i in zip(scores[0].tolist(), indices[0].tolist()) if i >= 0]
        if not candidates:
            return []

        candidate_diseases = [self.diseases[idx] for _, idx in candidates]
        boosts = _boost_for_disease(candidate_diseases, predicted_disease, CONFIG.rag.disease_boost)
        boosted = [(score + boost, idx) for (score, idx), boost in zip(candidates, boosts.tolist())]
        boosted.sort(key=lambda pair: pair[0], reverse=True)

        out: list[dict[str, Any]] = []
        for score, idx in boosted[:top_k]:
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
                    f"TF-IDF artifact missing: {p}. Run " "`python -m rag.build_tfidf_index` first."
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
        doc_diseases = [doc["disease"] for doc in self.documents]
        similarities = similarities + _boost_for_disease(
            doc_diseases, predicted_disease, CONFIG.rag.disease_boost
        )
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
