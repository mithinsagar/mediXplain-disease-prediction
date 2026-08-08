"""
Tests for the preprocessing pipeline.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import pandas as pd

from ml.preprocess import filter_rare_classes, make_split
from utils import normalise_symptom


def test_normalise_symptom_snake_case():
    assert normalise_symptom("Sore Throat") == "sore_throat"
    assert normalise_symptom("  Runny-Nose  ") == "runny_nose"
    assert normalise_symptom("Cough!!!") == "cough"
    assert normalise_symptom("") == ""


def test_filter_rare_classes_drops_singletons():
    df = pd.DataFrame(
        {
            "disease": ["a", "a", "b", "b", "c"],
            "s1": [1, 0, 1, 1, 0],
            "s2": [0, 1, 1, 0, 1],
        }
    )
    kept = filter_rare_classes(df, "disease", min_samples=2)
    assert set(kept["disease"].unique()) == {"a", "b"}
    assert len(kept) == 4


def test_make_split_shapes_and_encoder(tiny_dataset):
    df = filter_rare_classes(tiny_dataset, "disease", min_samples=2)
    split = make_split(df, "disease", test_size=0.25, random_state=0)
    assert set(split.label_encoder.classes_) <= set(df["disease"].unique())
    assert split.X_train.shape[0] + split.X_test.shape[0] == len(df)
    assert split.X_train.shape[1] == split.X_test.shape[1]
    assert "disease" not in split.feature_columns
