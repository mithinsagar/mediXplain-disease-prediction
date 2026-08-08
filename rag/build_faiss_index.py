"""
Build a FAISS index over the disease paragraph knowledge base.

Steps:
1. Load the cleaned dataset produced by `ml.prepare_dataset`.
2. Generate one paragraph per disease with `rag.knowledge_base.build_paragraphs`.
3. Embed the paragraphs with a sentence-transformers model.
4. Persist an `IndexFlatIP` on L2-normalised vectors (equivalent to cosine
   similarity) alongside a metadata pickle containing the paragraph text.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG
from rag.knowledge_base import build_paragraphs
from utils import RAG_FILES, get_logger, load_pickle, save_pickle, write_json

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
    """Build the FAISS index and persist it. Returns the index path."""
    try:
        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "FAISS or sentence-transformers is not installed. Install runtime "
            "requirements with `pip install -r requirements.txt`."
        ) from exc

    df, target_col = _load_dataset()
    feature_cols = _feature_columns(df, target_col)
    kb = build_paragraphs(df, target_col=target_col, feature_cols=feature_cols)

    log.info("Encoding %s paragraphs with %s", len(kb), CONFIG.rag.embedding_model)
    encoder = SentenceTransformer(CONFIG.rag.embedding_model)
    embeddings = encoder.encode(
        kb["paragraph"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info("FAISS index built: dim=%s ntotal=%s", dim, index.ntotal)

    index_path = CONFIG.paths.models / RAG_FILES["faiss_index"]
    metadata_path = CONFIG.paths.models / RAG_FILES["faiss_metadata"]

    faiss.write_index(index, str(index_path))
    save_pickle(
        {
            "diseases": kb["disease"].tolist(),
            "paragraphs": kb["paragraph"].tolist(),
            "symptoms": kb["symptoms"].tolist(),
            "encoder_name": CONFIG.rag.embedding_model,
            "dim": int(dim),
        },
        metadata_path,
    )

    write_json(
        {
            "author": "Mithin Sagar S",
            "github": "https://github.com/mithinsagar",
            "project": "MediXplain - Explainable Disease Prediction System",
            "index_type": "IndexFlatIP",
            "dim": int(dim),
            "ntotal": int(index.ntotal),
            "encoder_name": CONFIG.rag.embedding_model,
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
        },
        CONFIG.paths.models / "rag_index_info.json",
    )

    log.info("FAISS index written to %s", index_path)
    return index_path


def main() -> None:
    build_index()


if __name__ == "__main__":
    main()
