"""
Input validators for MediXplain.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

from typing import Iterable, Sequence


class ValidationError(ValueError):
    """Raised when user or programmatic input fails validation."""


def validate_symptom_list(symptoms: Iterable[str]) -> list[str]:
    """Strip empty entries, collapse whitespace, and preserve input order."""
    if symptoms is None:
        raise ValidationError("Symptom list must not be None.")

    cleaned: list[str] = []
    seen: set[str] = set()
    for symptom in symptoms:
        if not isinstance(symptom, str):
            raise ValidationError("Each symptom must be a string.")
        stripped = symptom.strip()
        if not stripped:
            continue
        key = stripped.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(stripped)
    return cleaned


def validate_min_symptoms(symptoms: Sequence[str], minimum: int = 3) -> None:
    """Raise ValidationError if fewer than `minimum` symptoms were supplied."""
    if len(symptoms) < minimum:
        raise ValidationError(
            f"Please select at least {minimum} symptoms for a reliable prediction."
        )
