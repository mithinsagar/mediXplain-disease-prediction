"""
Retrieval-Augmented Generation for MediXplain.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from rag.knowledge_base import build_paragraphs
from rag.retriever import FAISSRetriever, TFIDFRetriever, load_retriever

__all__ = [
    "build_paragraphs",
    "FAISSRetriever",
    "TFIDFRetriever",
    "load_retriever",
]
