# MediXplain — Architecture

> Author: Mithin Sagar S · GitHub: https://github.com/mithinsagar

MediXplain is intentionally organised as a small set of focused packages so
that each responsibility can be reasoned about, tested, and swapped
independently. This document explains how the packages fit together and how
data flows through the system at inference time.

## High-Level Layout

```
                +-----------------------+
                |     Flask Web App     |
                |  templates + static   |
                +-----------|-----------+
                            |
                            v
                +-----------------------+
                |     app.routes        |
                +-----------|-----------+
                            |
                            v
                +-----------------------+
                |   app.predictor       |
                |   DiseasePredictor    |
                +-----|-----------|-----+
                      |           |
              +-------+           +-------+
              v                           v
   +-------------------+       +-----------------------+
   |    ML Model Zoo   |       |   RAG Retriever       |
   |   sklearn (.pkl)  |       |  FAISS + MiniLM       |
   +-------------------+       |  or TF-IDF fallback   |
                               +-----------------------+

                       optional
                       enrichment
                            |
                            v
                +-----------------------+
                |  app.llm_explainer    |
                | OpenRouter or Gemini  |
                +-----------------------+
```

## Package Responsibilities

- `config.py` and `config/`:
  Loads YAML settings, merges with environment variables, and exposes a
  singleton `CONFIG` object used everywhere. Central point for hyperparameter
  and path changes.

- `utils/`:
  Small, dependency-free helpers: a logger factory, joblib IO wrappers,
  validators, and shared constants. Nothing here imports from `app`, `ml`,
  or `rag`.

- `ml/`:
  End-to-end machine-learning pipeline. `prepare_dataset` cleans the raw
  Kaggle CSV, `preprocess` handles rare-class filtering and splitting,
  `train_models` fits every classical model and selects the best, and
  `train_deep_models` optionally trains LSTM and GRU baselines.

- `rag/`:
  Paragraph knowledge base and two retriever backends. `build_faiss_index`
  is the primary path; `build_tfidf_index` is a scikit-learn fallback. The
  retrievers share a common interface.

- `app/`:
  Flask application factory, blueprints, templates, and the
  `DiseasePredictor` class that stitches ML + RAG + LLM together for both
  the web UI and the REST endpoints.

- `scripts/`:
  One-off utilities that don't belong in the core packages: dataset
  download, dataset inspection, sample-input generation.

- `tests/`:
  pytest suite that exercises the pipeline end-to-end using a tiny
  synthetic dataset generated in `conftest.py`.

## Inference Data Flow

1. The browser POSTs a set of symptom checkbox values to `/`.
2. `app.routes.home` validates the input (minimum of three symptoms) and
   forwards to `DiseasePredictor.predict`.
3. `DiseasePredictor` turns the symptoms into a feature vector aligned with
   the training columns, calls `predict_proba` on the best model, and
   builds a top-K differential.
4. The FAISS retriever (or TF-IDF fallback) is queried with the predicted
   disease name and symptom list; the top paragraph is used as evidence.
5. If an LLM key is present, `llm_explainer` produces a short natural-
   language justification using the prediction, confidence, and evidence
   text.
6. All three pieces are rendered in the right-hand panel of the UI.

## Where to Extend

- Add a new model by editing `config/model_config.json` and, if the
  estimator is new to the zoo, importing it in `ml/train_models.py`.
- Switch embedding models by changing `EMBEDDING_MODEL` in `.env`.
- Add a new LLM provider by implementing a `_call_<provider>` function in
  `app/llm_explainer.py` and updating the dispatch block.
- Add a new page by creating a template and registering a route in
  `app/routes.py`.
