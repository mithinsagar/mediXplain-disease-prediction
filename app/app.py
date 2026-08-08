"""
MediXplain Flask entrypoint.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

from flask import Flask

from app.routes import api_bp, web_bp
from config import CONFIG
from utils import get_logger

log = get_logger(__name__)


def create_app() -> Flask:
    """Application factory used both by `python -m app.app` and by tests."""
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )
    app.config.update(
        SECRET_KEY=CONFIG.flask.secret_key,
        JSON_SORT_KEYS=False,
    )

    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp)

    @app.context_processor
    def _inject_meta() -> dict[str, str]:
        return {
            "brand_name": "MediXplain",
            "brand_author": "Mithin Sagar S",
            "brand_github": "https://github.com/mithinsagar",
        }

    log.info(
        "MediXplain ready on http://%s:%s/ (debug=%s)",
        CONFIG.flask.host,
        CONFIG.flask.port,
        CONFIG.flask.debug,
    )
    return app


def main() -> None:
    app = create_app()
    app.run(
        host=CONFIG.flask.host,
        port=CONFIG.flask.port,
        debug=CONFIG.flask.debug,
    )


if __name__ == "__main__":
    main()
