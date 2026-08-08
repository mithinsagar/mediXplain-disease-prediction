"""
Small helper functions used throughout MediXplain.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np


def normalise_symptom(name: str) -> str:
    """Normalise a symptom string to snake_case for feature matching."""
    if not isinstance(name, str):
        return ""
    cleaned = name.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    return cleaned.strip("_")


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k largest values, highest first."""
    k = max(1, min(k, len(scores)))
    return np.argsort(scores)[::-1][:k]


def save_pickle(obj: Any, path: Path) -> None:
    """Persist `obj` to `path` using joblib. Ensures the parent exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path: Path) -> Any:
    """Load a joblib pickle from `path`."""
    return joblib.load(Path(path))


def write_json(payload: Any, path: Path, indent: int = 2) -> None:
    """Write `payload` to `path` as JSON with UTF-8 encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False)


def chunked(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    """Yield successive lists of at most `size` items from `items`."""
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
