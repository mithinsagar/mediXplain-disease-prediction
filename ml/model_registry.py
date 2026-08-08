"""
Lightweight registry that maps model artifact names to filesystem paths.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

from pathlib import Path

from config import CONFIG
from utils import MODEL_FILES, RAG_FILES

REGISTRY: dict[str, Path] = {
    **{name: CONFIG.paths.models / filename for name, filename in MODEL_FILES.items()},
    **{name: CONFIG.paths.models / filename for name, filename in RAG_FILES.items()},
}


def get(name: str) -> Path:
    """Return the absolute path of a registered artifact."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown artifact '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def exists(name: str) -> bool:
    """Whether the named artifact currently exists on disk."""
    return REGISTRY.get(name, Path("/nonexistent")).exists()
