# MediXplain — Setup Guide

> Author: Mithin Sagar S · GitHub: https://github.com/mithinsagar

This document walks through a fresh installation from a blank machine to a
running web app.

## 1. System Requirements

- Python 3.10 or newer (3.11 recommended).
- 4 GB RAM minimum, 8 GB comfortable while training.
- Roughly 3 GB of free disk space (Kaggle CSV + models + sentence-transformer
  weights).
- Internet access for the first run so pip and Hugging Face can download
  packages and embedding weights.

## 2. Clone and Environment

```bash
git clone https://github.com/mithinsagar/mediXplain-disease-prediction.git
cd mediXplain-disease-prediction

python -m venv .venv
source .venv/bin/activate     # macOS / Linux
.venv\Scripts\activate        # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Environment Variables

Copy the template and fill in whatever you have. Missing keys degrade the
app gracefully; the RAG-only path always works.

```bash
cp .env.example .env
```

Notable variables:

- `LLM_PROVIDER` — `openrouter` or `gemini` or `none`.
- `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`
- `GEMINI_API_KEY`, `GEMINI_MODEL`
- `RAG_BACKEND` — `faiss` (default) or `tfidf`
- `EMBEDDING_MODEL` — any sentence-transformers ID.

## 4. Dataset

The Kaggle dataset is not checked into git. Bring it in one of these ways:

- Automated: `python scripts/download_dataset.py` (needs the `kaggle` CLI
  and a configured `~/.kaggle/kaggle.json`).
- Manual: download the file from
  https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset
  and place it at
  `data/raw/Final_Augmented_dataset_Diseases_and_Symptoms.csv`.

## 5. Pipeline

```bash
python -m ml.prepare_dataset
python -m ml.train_models
python -m rag.build_faiss_index      # or `python -m rag.build_tfidf_index`
python -m app.app
```

Once the last command reports `MediXplain ready on http://127.0.0.1:5000/`
open that URL in your browser. Select at least three symptoms and click
Analyse and Predict.

## 6. Optional: Deep Models

The LSTM and GRU baselines require TensorFlow. To skip them, comment the
`tensorflow` line in `requirements.txt` before installing. To include them:

```bash
python -m ml.train_deep_models
```

## 7. Development Extras

```bash
pip install -r requirements-dev.txt
pytest -q
make lint      # flake8 + black --check
make format    # isort + black
```

## 8. Troubleshooting

- FAISS wheel fails to install on your platform: switch to the TF-IDF
  backend by setting `RAG_BACKEND=tfidf` and re-running
  `python -m rag.build_tfidf_index`.
- Sentence-transformers download blocked: set `HF_HOME` to a writable
  directory or pre-download the model with `huggingface-cli`.
- LLM call fails with 401 or 429: verify the API key, then either switch
  providers or set `LLM_PROVIDER=none` to use RAG-only explanations.
