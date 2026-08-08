"""
Post-preparation preprocessing utilities.

Handles rare-class filtering, label encoding, and train/test splitting. The
train_models module imports these helpers so the same logic is reused by the
evaluation script and by unit tests.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import CONFIG
from utils import get_logger

log = get_logger(__name__)


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder
    feature_columns: list[str]
    target_col: str


def filter_rare_classes(
    df: pd.DataFrame, target_col: str, min_samples: int | None = None
) -> pd.DataFrame:
    """Remove classes with fewer than `min_samples` occurrences."""
    if min_samples is None:
        min_samples = CONFIG.training.min_class_samples
    counts = df[target_col].value_counts()
    valid = counts[counts >= min_samples].index
    filtered = df[df[target_col].isin(valid)].copy()
    log.info(
        "Rare-class filter: kept %s of %s classes (min_samples=%s)",
        filtered[target_col].nunique(),
        df[target_col].nunique(),
        min_samples,
    )
    return filtered


def make_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float | None = None,
    random_state: int | None = None,
) -> SplitData:
    """Fit a LabelEncoder and produce a stratified train/test split."""
    test_size = test_size or CONFIG.training.test_size
    random_state = random_state or CONFIG.training.random_state

    feature_cols = [c for c in df.columns if c not in {target_col, "symptom_text"}]
    X = df[feature_cols]
    y = df[target_col]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    log.info(
        "Split shapes -> train=%s test=%s classes=%s",
        X_train.shape,
        X_test.shape,
        len(encoder.classes_),
    )

    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        label_encoder=encoder,
        feature_columns=feature_cols,
        target_col=target_col,
    )
