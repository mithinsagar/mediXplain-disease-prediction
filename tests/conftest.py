"""
Shared pytest fixtures for MediXplain.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def tiny_dataset() -> pd.DataFrame:
    """A tiny synthetic dataset that mimics the Kaggle layout."""
    rng = np.random.default_rng(seed=42)
    diseases = ["flu", "cold", "migraine", "allergy"]
    symptoms = [
        "fever",
        "cough",
        "sore_throat",
        "headache",
        "runny_nose",
        "sneezing",
        "fatigue",
        "nausea",
        "chills",
        "body_ache",
    ]
    rows = []
    for _ in range(80):
        disease = rng.choice(diseases)
        row = {s: 0 for s in symptoms}
        for s in rng.choice(symptoms, size=rng.integers(3, 6), replace=False):
            row[s] = 1
        row["disease"] = disease
        rows.append(row)
    df = pd.DataFrame(rows)
    columns = ["disease"] + symptoms
    return df[columns]


@pytest.fixture()
def sample_paragraphs() -> list[dict[str, str]]:
    return [
        {"disease": "flu", "text": "Flu is associated with fever, cough, fatigue."},
        {"disease": "cold", "text": "Cold is associated with runny nose, sneezing, cough."},
        {
            "disease": "migraine",
            "text": "Migraine is associated with headache, nausea, sensitivity.",
        },
    ]
