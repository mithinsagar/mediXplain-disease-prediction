#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# MediXplain - macOS / Linux launcher
# Author:  Mithin Sagar S
# GitHub:  https://github.com/mithinsagar
# Project: MediXplain - Explainable Disease Prediction System
# -----------------------------------------------------------------------------

set -euo pipefail

banner() {
    echo ""
    echo "================================================================"
    echo "  MediXplain - Explainable Disease Prediction System"
    echo "  Author: Mithin Sagar S | github.com/mithinsagar"
    echo "================================================================"
    echo ""
}

banner

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing requirements..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

DATASET="data/raw/Final_Augmented_dataset_Diseases_and_Symptoms.csv"
if [ ! -f "${DATASET}" ]; then
    echo ""
    echo "Dataset CSV not found at ${DATASET}."
    echo "Attempting to download via scripts/download_dataset.py..."
    python scripts/download_dataset.py || true
fi

echo ""
echo "Preparing dataset..."
python -m ml.prepare_dataset

echo ""
echo "Training classical models..."
python -m ml.train_models

echo ""
echo "Building FAISS RAG index..."
python -m rag.build_faiss_index

echo ""
echo "Starting Flask app on http://127.0.0.1:5000/"
python -m app.app
