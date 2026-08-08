"""
MediXplain Flask application package.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from app.app import create_app  # re-exported for convenience

__all__ = ["create_app"]
