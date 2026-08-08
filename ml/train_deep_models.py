"""
Optional deep-learning branch: LSTM and GRU over symptom text sequences.

TensorFlow is imported lazily so the main pipeline still runs on machines
where it is not installed. Trained models are saved under `models/` in the
Keras format and a small JSON summary is written to
`data/processed/deep_model_summary.json`.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import CONFIG
from utils import get_logger, write_json

log = get_logger(__name__)


def _load_dataset() -> tuple[pd.DataFrame, str]:
    processed_csv = CONFIG.paths.processed / "disease_dataset_clean.csv"
    if not processed_csv.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_csv}. "
            "Run `python -m ml.prepare_dataset` first."
        )
    df = pd.read_csv(processed_csv)
    target_col = "prognosis" if "prognosis" in df.columns else df.columns[0]
    if "symptom_text" not in df.columns:
        raise KeyError(
            "'symptom_text' column not found in processed dataset. "
            "Re-run `python -m ml.prepare_dataset`."
        )
    return df, target_col


def main() -> None:
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import (
            GRU,
            LSTM,
            Dense,
            Embedding,
            TextVectorization,
        )
        from tensorflow.keras.models import Sequential
    except ImportError as exc:
        log.warning("TensorFlow is not installed; skipping deep models: %s", exc)
        return

    df, target_col = _load_dataset()
    X = df["symptom_text"].astype(str).values
    y = df[target_col].astype(str).values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_encoded,
        test_size=CONFIG.training.test_size,
        random_state=CONFIG.training.random_state,
        stratify=y_encoded,
    )

    vectorizer = TextVectorization(max_tokens=5000, output_sequence_length=48)
    vectorizer.adapt(X_train)

    def make_ds(texts, labels, batch: int = 32) -> "tf.data.Dataset":
        ds = tf.data.Dataset.from_tensor_slices((texts, labels))
        return ds.shuffle(len(texts)).batch(batch).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(X_train, y_train)
    val_ds = make_ds(X_val, y_val)

    models = {
        "lstm_text_model": Sequential(
            [
                vectorizer,
                Embedding(5000, 64, mask_zero=True),
                LSTM(64),
                Dense(128, activation="relu"),
                Dense(num_classes, activation="softmax"),
            ],
            name="lstm_text_model",
        ),
        "gru_text_model": Sequential(
            [
                vectorizer,
                Embedding(5000, 64, mask_zero=True),
                GRU(64),
                Dense(128, activation="relu"),
                Dense(num_classes, activation="softmax"),
            ],
            name="gru_text_model",
        ),
    }

    summary: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        log.info("Training %s...", name)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)
        model.save(CONFIG.paths.models / f"{name}.keras")
        summary[name] = {
            "final_train_accuracy": float(history.history["accuracy"][-1]),
            "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        }

    write_json(summary, CONFIG.paths.processed / "deep_model_summary.json")
    log.info("Deep-model summary:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
