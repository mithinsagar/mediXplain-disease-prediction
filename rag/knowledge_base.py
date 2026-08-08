"""
Build a paragraph-level knowledge base from the cleaned dataset.

For each disease in the training data we compute the top symptoms by presence
frequency and generate a short natural-language paragraph. These paragraphs
become the corpus for both the FAISS and the TF-IDF retrievers.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from config import CONFIG
from utils import get_logger

log = get_logger(__name__)


def _paragraph_for_disease(
    disease: str, symptoms: Sequence[str], limit: int
) -> str:
    top = list(symptoms)[:limit]
    if not top:
        return (
            f"{disease}: no strongly associated symptoms were found in the training "
            "corpus. Consult a clinician for a proper differential diagnosis."
        )
    return (
        f"{disease} is commonly associated with symptoms such as "
        + ", ".join(top)
        + ". This description is generated from symptom co-occurrence patterns "
        "observed in the training dataset and should be treated as evidence "
        "supporting the model's prediction, not as a clinical diagnosis."
    )


def build_paragraphs(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with columns [disease, paragraph, symptoms]."""
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in {target_col, "symptom_text"}]
    if limit is None:
        limit = CONFIG.rag.max_paragraph_symptoms

    log.info(
        "Building RAG knowledge base: %s diseases, %s feature columns, limit=%s",
        df[target_col].nunique(),
        len(feature_cols),
        limit,
    )

    rows: list[dict[str, object]] = []
    for disease, group in df.groupby(target_col):
        frequency = group[list(feature_cols)].sum().sort_values(ascending=False)
        active = [c.replace("_", " ") for c in frequency.index if frequency[c] > 0]
        paragraph = _paragraph_for_disease(str(disease), active, limit)
        rows.append(
            {
                "disease": str(disease),
                "paragraph": paragraph,
                "symptoms": active[:limit],
            }
        )

    kb = pd.DataFrame(rows)
    log.info("Knowledge base rows: %s", len(kb))
    return kb
