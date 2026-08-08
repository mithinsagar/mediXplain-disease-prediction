# MediXplain — Explainable Disease Prediction with Multi-Model ML + FAISS RAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0-black.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E.svg)](https://scikit-learn.org/)
[![FAISS](https://img.shields.io/badge/FAISS-1.8-4B8BBE.svg)](https://faiss.ai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/mithinsagar/mediXplain-disease-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/mithinsagar/mediXplain-disease-prediction/actions)

**MediXplain** is a full-stack, production-oriented prototype that predicts diseases from
patient symptoms using a suite of calibrated classical machine learning models, retrieves
supporting medical evidence with a **FAISS + sentence-transformers Retrieval-Augmented
Generation (RAG)** pipeline, and optionally enriches the explanation with a Large Language
Model. It ships with a modern dark-themed Flask user interface built for clinical
readability.

> Author: **Mithin Sagar S**  ·  GitHub: [github.com/mithinsagar](https://github.com/mithinsagar)

---

## Table of Contents

1. [Highlights](#highlights)
2. [System Architecture](#system-architecture)
3. [Repository Layout](#repository-layout)
4. [Quick Start](#quick-start)
5. [Dataset](#dataset)
6. [Model Zoo and Results](#model-zoo-and-results)
7. [RAG Pipeline](#rag-pipeline)
8. [LLM Explanations](#llm-explanations)
9. [Web Interface](#web-interface)
10. [Configuration](#configuration)
11. [Testing](#testing)
12. [Continuous Integration](#continuous-integration)
13. [Roadmap](#roadmap)
14. [License](#license)
15. [Author](#author)

---

## Highlights

- **Seven scikit-learn models** trained and benchmarked in one command: Logistic Regression,
  Linear SVM, Bernoulli Naive Bayes, Decision Tree, Random Forest, AdaBoost, and a Multi
  Layer Perceptron. The best model is selected automatically and persisted.
- **FAISS vector search** over a paragraph knowledge base built from the training data.
  Uses `sentence-transformers/all-MiniLM-L6-v2` embeddings for semantic retrieval, with a
  TF-IDF path retained as a lightweight fallback.
- **Calibrated confidence scoring** with top-3 differential predictions rather than a
  single hard label.
- **Optional deep-learning branch** with LSTM and GRU sequence models over symptom text.
- **Modern Flask front end** with a two-panel workspace, symptom search, live chip
  updates, and animated confidence bars. Fully responsive and accessible.
- **LLM explanation layer** that transparently supports both OpenRouter and Google Gemini
  through a single environment variable.
- **Reproducible pipeline**: Makefile targets, unit tests, GitHub Actions CI, and pinned
  dependencies.

## System Architecture

```
+---------------------------------------------------------------+
|                       Flask Web Layer                         |
|  templates/ + static/  <---  Jinja  ---> app/routes.py        |
+-----------------------------|---------------------------------+
                              v
                    +------------------+
                    | DiseasePredictor |
                    +--------|---------+
                             |
        +--------------------+---------------------+
        |                                          |
        v                                          v
+----------------+                        +------------------+
|  ML Model Zoo  |                        |   RAG Retriever  |
|  scikit-learn  |                        | FAISS + MiniLM   |
+----------------+                        +------------------+
        |                                          |
        v                                          v
+----------------+                        +------------------+
|  models/*.pkl  |                        | rag_index.faiss  |
+----------------+                        +------------------+

        --> optional LLM enrichment (OpenRouter / Gemini) -->
```

See `docs/ARCHITECTURE.md` for a deeper description.

## Repository Layout

```text
mediXplain-disease-prediction/
├── app/                    Flask application (UI, routes, predictor, RAG glue, LLM)
├── ml/                     Data preparation, training, evaluation, inference
├── rag/                    FAISS index builder, retriever, TF-IDF fallback
├── utils/                  Logger, validators, constants, helpers
├── config/                 YAML and JSON configuration
├── scripts/                One-off utilities (dataset download, sample generation)
├── tests/                  pytest suite covering ML, RAG, and Flask
├── docs/                   Architecture, setup, API docs, project explanation
├── notebooks/              Exploratory notebook comparing models
├── screenshots/            Application screenshots (populated by you)
├── data/                   Raw and processed data (gitignored)
├── models/                 Trained artifacts (gitignored)
├── .github/workflows/      GitHub Actions CI pipeline
├── Makefile                Convenience commands: prepare / train / rag / run / test
├── requirements.txt        Runtime dependencies
├── requirements-dev.txt    Development dependencies
├── config.py               Central path and setting resolution
├── setup.py                Installable package definition
├── run_windows.bat         Windows launcher
├── run_unix.sh             macOS and Linux launcher
├── .env.example            Environment variable template
├── .gitignore
├── LICENSE
└── README.md
```

## Quick Start

### 1. Clone and set up a virtual environment

```bash
git clone https://github.com/mithinsagar/mediXplain-disease-prediction.git
cd mediXplain-disease-prediction

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS or Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Provide the dataset

The project uses the public `Final_Augmented_dataset_Diseases_and_Symptoms` dataset from
Kaggle. There are two ways to bring it in:

```bash
# Option A: automatic download using the Kaggle CLI
python scripts/download_dataset.py

# Option B: download the CSV manually from Kaggle and place it here:
#   data/raw/Final_Augmented_dataset_Diseases_and_Symptoms.csv
```

### 3. Run the full pipeline

```bash
# Prepare and clean the raw CSV into training-ready shape
python -m ml.prepare_dataset

# Train and benchmark all classical models
python -m ml.train_models

# Optional: train LSTM and GRU baselines
python -m ml.train_deep_models

# Build the FAISS RAG index
python -m rag.build_faiss_index

# Launch the Flask web interface
python -m app.app
```

The application will be available at `http://127.0.0.1:5000/`.

A convenience `Makefile` is provided:

```bash
make install       # install runtime dependencies
make prepare       # preprocess the raw dataset
make train         # train all classical models and pick the best
make rag           # build the FAISS RAG index
make run           # start the Flask app
make test          # run the pytest suite
make clean         # remove generated artifacts
```

## Dataset

The base dataset is `Final_Augmented_dataset_Diseases_and_Symptoms`, published on Kaggle:
https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset

It is a wide binary matrix where each row is a case, each of the ~377 symptom columns is
0 or 1, and the first column is the disease label (~773 classes). The preparation step
strips duplicates, coerces to strict binary integers, removes rare classes (< 2 samples),
and generates a companion `symptom_text` column for the deep-learning branch and the RAG
paragraph builder.

## Model Zoo and Results

Every model below is trained, evaluated on a stratified 20 percent hold-out, and its
classification report is saved beside its `.pkl` file. The best model, selected by
accuracy, is copied to `models/best_model.pkl`.

| Model                | Accuracy | Notes                                        |
| -------------------- | -------- | -------------------------------------------- |
| Logistic Regression  | 0.8309   | Selected as the default best model           |
| Linear SVM           | 0.8276   | Fast, no probability, calibrated on demand   |
| MLP Classifier       | 0.8202   | Two hidden layers, early stopping            |
| Bernoulli Naive Bayes| 0.8116   | Strong baseline for binary features          |
| Random Forest        | 0.3770   | Underfits at low n_estimators, kept for demo |
| Decision Tree        | 0.1124   | Reference only                               |
| AdaBoost             | 0.0094   | Reference only, struggles on ~773 classes    |

Reproduce these numbers with `python -m ml.train_models`. Full per-class metrics land in
`models/<name>_report.txt`. A summary CSV is written to `models/model_results.csv`.

## RAG Pipeline

Retrieval-Augmented Generation is implemented in the `rag/` package:

1. `rag/knowledge_base.py` groups the training set by disease and generates a paragraph
   description from the top symptoms for each class.
2. `rag/build_faiss_index.py` embeds those paragraphs with
   `sentence-transformers/all-MiniLM-L6-v2` and builds a `faiss.IndexFlatIP` on the
   normalised vectors.
3. `rag/retriever.py` exposes a `FAISSRetriever` class that returns the top-k paragraphs
   for a `predicted_disease + symptom_list` query along with cosine similarity scores.
4. `rag/build_tfidf_index.py` provides a pure scikit-learn fallback for environments
   without FAISS installed.

The retrieval score is surfaced in the UI next to the ML confidence so users can gauge
both the model's certainty and the strength of the supporting evidence.

## LLM Explanations

`app/llm_explainer.py` produces a natural-language justification of the prediction. It
supports two providers, selected by the `LLM_PROVIDER` environment variable:

- `openrouter` (default) — uses `OPENROUTER_API_KEY`
- `gemini` — uses `GEMINI_API_KEY` and, optionally, `GEMINI_MODEL`

If neither key is present the app still works: RAG evidence and ML confidence are shown
without LLM enrichment. See `.env.example` for the full list of supported variables.

## Web Interface

The Flask front end lives in `app/templates/` and `app/static/`. Highlights:

- Dark, split-panel workspace with a searchable symptom checklist on the left and results
  on the right.
- Live counter, chip preview of selected symptoms, and animated confidence meter.
- Top-3 differential diagnoses with per-class bars.
- Separate cards for the RAG retrieved paragraph and the LLM narrative.
- Fully responsive; collapses to a single column below 860 px.

Populate `screenshots/` with captures of the running app for the GitHub landing page.

## Configuration

All hard-coded paths and hyperparameters have been extracted into two places:

- `config/settings.yaml` — global paths, embedding model name, FAISS parameters, UI copy.
- `config/model_config.json` — per-model hyperparameters for `ml/train_models.py`.

`config.py` at the repository root loads these, expands environment variables, and
returns a single `Config` object consumed by every module.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

Tests cover the preprocessing pipeline, the TF-IDF and FAISS retrievers, the predictor
class, and the Flask routes (using the test client). A minimum bar of 70 percent
coverage is enforced in CI.

## Continuous Integration

`.github/workflows/ci.yml` runs on every push and pull request:

1. Sets up Python 3.10, 3.11, and 3.12.
2. Installs runtime and development dependencies.
3. Lints with `flake8` and formats with `black --check`.
4. Runs the full `pytest` suite.

## Roadmap

- SHAP-based per-symptom attribution.
- Docker image with a slim runtime.
- Streamlit dashboard variant.
- Multi-language UI.
- On-device inference via ONNX Runtime.

## License

Released under the [MIT License](LICENSE). Do not use as a substitute for professional
medical advice.

## Author

**Mithin Sagar S**
GitHub: [github.com/mithinsagar](https://github.com/mithinsagar)

If MediXplain helps your project or coursework, a star on the repository is appreciated.
