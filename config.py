"""
Central configuration for MediXplain.

Loads settings from `config/settings.yaml`, merges them with environment
variables (via `.env` when present), and exposes typed accessors used across
`app/`, `ml/`, `rag/`, and `utils/`.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

BASE_DIR: Path = Path(__file__).resolve().parent
SETTINGS_PATH: Path = BASE_DIR / "config" / "settings.yaml"

load_dotenv(BASE_DIR / ".env", override=False)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class Paths:
    """Filesystem locations, all rooted at the repository base directory."""

    base: Path = BASE_DIR
    data: Path = BASE_DIR / "data"
    raw: Path = BASE_DIR / "data" / "raw"
    processed: Path = BASE_DIR / "data" / "processed"
    models: Path = BASE_DIR / "models"
    logs: Path = BASE_DIR / "logs"

    def ensure(self) -> None:
        """Create every managed directory if it does not already exist."""
        for path in (self.data, self.raw, self.processed, self.models, self.logs):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class FlaskConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = True
    secret_key: str = "change-me-in-production"


@dataclass
class LLMConfig:
    provider: str = "openrouter"  # openrouter | gemini | none
    openrouter_api_key: str | None = None
    openrouter_model: str = "openrouter/auto"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-1.5-flash"
    timeout_seconds: int = 30


@dataclass
class RAGConfig:
    backend: str = "faiss"  # faiss | tfidf
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 3
    max_paragraph_symptoms: int = 15
    disease_boost: float = 0.15
    symptom_overlap_boost: float = 0.2


@dataclass
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    min_class_samples: int = 2


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    flask: FlaskConfig = field(default_factory=FlaskConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    log_level: str = "INFO"


def _from_env(default: str | None, key: str) -> str | None:
    value = os.getenv(key)
    return value if value not in (None, "") else default


def load_config() -> Config:
    """Load the Config object from YAML + environment variables."""
    yaml_data = _load_yaml(SETTINGS_PATH)

    flask_yaml = yaml_data.get("flask", {})
    llm_yaml = yaml_data.get("llm", {})
    rag_yaml = yaml_data.get("rag", {})
    train_yaml = yaml_data.get("training", {})

    flask = FlaskConfig(
        host=_from_env(flask_yaml.get("host", "127.0.0.1"), "FLASK_HOST"),
        port=int(_from_env(str(flask_yaml.get("port", 5000)), "FLASK_PORT")),
        debug=str(_from_env(str(flask_yaml.get("debug", True)), "FLASK_DEBUG")).lower()
        in {"1", "true", "yes"},
        secret_key=_from_env(
            flask_yaml.get("secret_key", "change-me-in-production"),
            "FLASK_SECRET_KEY",
        ),
    )

    llm = LLMConfig(
        provider=_from_env(llm_yaml.get("provider", "openrouter"), "LLM_PROVIDER"),
        openrouter_api_key=_from_env(None, "OPENROUTER_API_KEY"),
        openrouter_model=_from_env(
            llm_yaml.get("openrouter_model", "openrouter/auto"), "OPENROUTER_MODEL"
        ),
        gemini_api_key=_from_env(None, "GEMINI_API_KEY"),
        gemini_model=_from_env(llm_yaml.get("gemini_model", "gemini-1.5-flash"), "GEMINI_MODEL"),
        timeout_seconds=int(llm_yaml.get("timeout_seconds", 30)),
    )

    rag = RAGConfig(
        backend=_from_env(rag_yaml.get("backend", "faiss"), "RAG_BACKEND"),
        embedding_model=_from_env(
            rag_yaml.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            "EMBEDDING_MODEL",
        ),
        top_k=int(rag_yaml.get("top_k", 3)),
        max_paragraph_symptoms=int(rag_yaml.get("max_paragraph_symptoms", 15)),
        disease_boost=float(rag_yaml.get("disease_boost", 0.15)),
        symptom_overlap_boost=float(rag_yaml.get("symptom_overlap_boost", 0.2)),
    )

    training = TrainingConfig(
        test_size=float(train_yaml.get("test_size", 0.2)),
        random_state=int(train_yaml.get("random_state", 42)),
        min_class_samples=int(train_yaml.get("min_class_samples", 2)),
    )

    config = Config(
        flask=flask,
        llm=llm,
        rag=rag,
        training=training,
        log_level=_from_env(yaml_data.get("log_level", "INFO"), "LOG_LEVEL"),
    )
    config.paths.ensure()
    return config


# Module-level singleton for convenience.
CONFIG: Config = load_config()
