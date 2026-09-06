"""
Tests for the classical model zoo.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

from ml.preprocess import filter_rare_classes, make_split
from ml.train_models import MODEL_CONFIG_PATH, _load_hyperparameters, build_model_zoo


def test_model_config_lists_knn_hyperparameters():
    hyperparameters = _load_hyperparameters()
    assert "knn" in hyperparameters
    assert hyperparameters["knn"]["n_neighbors"] == 7


def test_build_model_zoo_includes_knn():
    hyperparameters = _load_hyperparameters()
    zoo = build_model_zoo(hyperparameters)
    assert "knn" in zoo
    assert isinstance(zoo["knn"], KNeighborsClassifier)
    assert zoo["knn"].n_neighbors == 7
    assert zoo["knn"].weights == "distance"


def test_knn_fits_and_predicts_on_symptom_data(tiny_dataset):
    df = filter_rare_classes(tiny_dataset, "disease", min_samples=2)
    split = make_split(df, "disease", test_size=0.25, random_state=0)

    hyperparameters = _load_hyperparameters()
    knn = build_model_zoo(hyperparameters)["knn"]
    knn.fit(split.X_train, split.y_train)
    preds = knn.predict(split.X_test)

    assert len(preds) == split.X_test.shape[0]
    assert set(preds).issubset(set(range(len(split.label_encoder.classes_))))


def test_model_config_file_is_valid_json():
    import json

    with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    assert "knn" in data


def test_model_config_lists_extra_trees_hyperparameters():
    hyperparameters = _load_hyperparameters()
    assert "extra_trees" in hyperparameters
    assert hyperparameters["extra_trees"]["n_estimators"] == 200


def test_build_model_zoo_includes_extra_trees():
    hyperparameters = _load_hyperparameters()
    zoo = build_model_zoo(hyperparameters)
    assert "extra_trees" in zoo
    assert isinstance(zoo["extra_trees"], ExtraTreesClassifier)
    assert zoo["extra_trees"].n_estimators == 200
    assert zoo["extra_trees"].max_depth == 20


def test_extra_trees_fits_and_predicts_on_symptom_data(tiny_dataset):
    df = filter_rare_classes(tiny_dataset, "disease", min_samples=2)
    split = make_split(df, "disease", test_size=0.25, random_state=0)

    hyperparameters = _load_hyperparameters()
    extra_trees = build_model_zoo(hyperparameters)["extra_trees"]
    extra_trees.fit(split.X_train, split.y_train)
    preds = extra_trees.predict(split.X_test)

    assert len(preds) == split.X_test.shape[0]
    assert set(preds).issubset(set(range(len(split.label_encoder.classes_))))
