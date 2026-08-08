"""
LLM explanation layer for MediXplain.

Supports two providers, selected by the `LLM_PROVIDER` environment variable
(also settable via `config/settings.yaml`):

- openrouter : chat.completions endpoint on https://openrouter.ai/
- gemini     : generateContent endpoint on https://generativelanguage.googleapis.com/

If the configured provider is missing an API key, `generate_llm_explanation`
returns `(None, False, reason)` so the UI can fall back to RAG-only mode.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

from typing import Iterable, Tuple

import requests

from config import CONFIG
from utils import get_logger

log = get_logger(__name__)


PROMPT_TEMPLATE = """You are an assistant explaining the output of a disease
prediction system that combines a machine-learning classifier with
Retrieval-Augmented Generation over a medical knowledge base.

Selected symptoms: {symptoms}
Predicted disease: {prediction}
Model confidence: {confidence}
Supporting evidence retrieved by the RAG module:
{rag_text}

Please produce a short, clear explanation with:
- why this disease was likely predicted from the given symptoms,
- how the retrieved evidence supports (or does not support) the prediction,
- an explicit disclaimer that this is not a medical diagnosis and the user
  should consult a qualified clinician.

Keep the response under 180 words and avoid speculation."""


def _format_symptoms(selected_symptoms: Iterable[str]) -> str:
    joined = ", ".join(s.replace("_", " ") for s in selected_symptoms)
    return joined or "not provided"


def _call_openrouter(prompt: str) -> str:
    if not CONFIG.llm.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {CONFIG.llm.openrouter_api_key}",
        "Content-Type": "application/json",
        "X-Title": "MediXplain",
    }
    payload = {
        "model": CONFIG.llm.openrouter_model,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(
        url, headers=headers, json=payload, timeout=CONFIG.llm.timeout_seconds
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _call_gemini(prompt: str) -> str:
    if not CONFIG.llm.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    model = CONFIG.llm.gemini_model
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={CONFIG.llm.gemini_api_key}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, json=payload, timeout=CONFIG.llm.timeout_seconds)
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def generate_llm_explanation(
    selected_symptoms: Iterable[str],
    prediction: str,
    confidence: object,
    rag_text: str,
) -> Tuple[str | None, bool, str | None]:
    """Return `(text, enabled, error)` where `enabled` reflects success."""
    provider = (CONFIG.llm.provider or "none").lower()
    if provider in {"", "none", "off", "disabled"}:
        return None, False, "LLM explanation disabled by configuration."

    prompt = PROMPT_TEMPLATE.format(
        symptoms=_format_symptoms(selected_symptoms),
        prediction=prediction,
        confidence=confidence,
        rag_text=rag_text or "No evidence retrieved.",
    )

    try:
        if provider == "openrouter":
            text = _call_openrouter(prompt)
        elif provider == "gemini":
            text = _call_gemini(prompt)
        else:
            return None, False, f"Unknown LLM provider '{provider}'."
        return text, True, None
    except Exception as exc:  # noqa: BLE001 - surface any transport error
        log.warning("LLM call failed (%s): %s", provider, exc)
        return None, False, str(exc)
