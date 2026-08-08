"""
Feature engineering helpers for MediXplain.

The current dataset is already a wide binary matrix, but this module keeps
the door open for future improvements such as symptom co-occurrence groups,
frequency-based weighting, or PCA compression.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from utils import get_logger, normalise_symptom

log = get_logger(__name__)


def symptom_vector(symptoms: Sequence[str], feature_columns: Sequence[str]) -> pd.DataFrame:
    """Turn a raw list of symptom names into a single-row DataFrame.

    The output uses the training feature order so it can be fed directly into
    a fitted scikit-learn estimator.
    """
    active = {normalise_symptom(s) for s in symptoms if s}
    row = [1 if col in active else 0 for col in feature_columns]
    return pd.DataFrame([row], columns=list(feature_columns))


def most_common_symptoms(
    df: pd.DataFrame,
    disease: str,
    target_col: str,
    feature_cols: Sequence[str],
    limit: int = 15,
) -> list[str]:
    """Return the most common present-symptoms for a given disease."""
    subset = df[df[target_col] == disease]
    if subset.empty:
        return []
    freq = subset[list(feature_cols)].sum().sort_values(ascending=False)
    return [c.replace("_", " ") for c in freq.index if freq[c] > 0][:limit]


def coverage_score(symptoms: Sequence[str], feature_columns: Sequence[str]) -> float:
    """Fraction of supplied symptoms that map to a known feature column."""
    if not symptoms:
        return 0.0
    known = {normalise_symptom(s) for s in symptoms}
    hits = sum(1 for s in known if s in feature_columns)
    return float(hits) / float(len(known))


def entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy of a probability distribution, in nats."""
    p = probabilities[probabilities > 0]
    return float(-np.sum(p * np.log(p)))
