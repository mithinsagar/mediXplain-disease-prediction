"""
Logging helper for MediXplain.

Every module should acquire its logger via `get_logger(__name__)`. The first
call configures a single stream handler and a rotating file handler under
`logs/medixplain.log`.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def _configure_root_logger(log_dir: Path, level: str) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "medixplain.log"

    root = logging.getLogger("medixplain")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=1_048_576, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the `medixplain` root logger."""
    # Local import to avoid circular dependency at import time.
    from config import CONFIG

    _configure_root_logger(CONFIG.paths.logs, CONFIG.log_level)
    short = name.replace("medixplain.", "")
    return logging.getLogger(f"medixplain.{short}")
