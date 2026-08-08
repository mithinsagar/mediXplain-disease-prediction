# -----------------------------------------------------------------------------
# MediXplain Makefile
# Author:  Mithin Sagar S
# GitHub:  https://github.com/mithinsagar
# Project: MediXplain - Explainable Disease Prediction System
# -----------------------------------------------------------------------------

PYTHON ?= python
PIP    ?= pip

.PHONY: help install install-dev download prepare train train-deep rag run test lint format clean

help:
	@echo "MediXplain - available targets:"
	@echo "  install       Install runtime dependencies"
	@echo "  install-dev   Install development dependencies (tests, linters)"
	@echo "  download      Download the Kaggle dataset via kaggle CLI"
	@echo "  prepare       Clean and preprocess the raw dataset"
	@echo "  train         Train all classical ML models"
	@echo "  train-deep    Train optional LSTM / GRU models"
	@echo "  rag           Build the FAISS RAG index"
	@echo "  run           Start the Flask web app"
	@echo "  test          Run pytest suite"
	@echo "  lint          Run flake8 + black --check"
	@echo "  format        Auto-format with black + isort"
	@echo "  clean         Remove build, cache, and generated artifacts"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

download:
	$(PYTHON) scripts/download_dataset.py

prepare:
	$(PYTHON) -m ml.prepare_dataset

train:
	$(PYTHON) -m ml.train_models

train-deep:
	$(PYTHON) -m ml.train_deep_models

rag:
	$(PYTHON) -m rag.build_faiss_index

run:
	$(PYTHON) -m app.app

test:
	pytest -q --cov=app --cov=ml --cov=rag --cov=utils

lint:
	flake8 app ml rag utils tests
	black --check app ml rag utils tests

format:
	isort app ml rag utils tests
	black app ml rag utils tests

clean:
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov coverage.xml
