from types import SimpleNamespace

import numpy as np
import pytest

from sws.inference import NoModelEngine, SklearnMlpEngine, load_sklearn_model_from_dir


class FakeScaler:
    def __init__(self):
        self.seen_shape = None

    def transform(self, sample):
        self.seen_shape = sample.shape
        return sample


class FakeClassifier:
    classes_ = np.array([0, 1])

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, sample):
        return np.array([self.probabilities])


def test_no_model_engine_never_generates_text():
    result = NoModelEngine().translate(np.zeros(126, dtype=np.float32))

    assert result.text == ""
    assert result.tokens == []
    assert result.confidence is None
    assert result.status == "model_not_loaded"


def test_sklearn_engine_uses_predict_proba_and_label():
    scaler = FakeScaler()
    engine = SklearnMlpEngine(scaler, FakeClassifier([0.2, 0.8]), {"1": "merci"})

    result = engine.translate(np.zeros(126, dtype=np.float32))

    assert scaler.seen_shape == (1, 126)
    assert result.text == "merci"
    assert result.confidence == pytest.approx(0.8)
    assert result.status == "ok"


def test_sklearn_engine_rejects_wrong_vector_size():
    engine = SklearnMlpEngine(FakeScaler(), FakeClassifier([1.0]), {"0": "bonjour"})

    result = engine.translate(np.zeros(125, dtype=np.float32))

    assert result.text == ""
    assert result.status == "inference_error"
    assert "125 valeurs" in result.message


def test_sklearn_engine_filters_low_confidence():
    engine = SklearnMlpEngine(
        FakeScaler(),
        FakeClassifier([0.55, 0.45]),
        {"0": "bonjour"},
        confidence_threshold=0.60,
    )

    result = engine.translate(np.zeros(126, dtype=np.float32))

    assert result.text == ""
    assert result.status == "low_confidence"


def test_sklearn_engine_reports_missing_labels_file_state():
    engine = SklearnMlpEngine(FakeScaler(), FakeClassifier([1.0]), None, load_status="labels_missing")

    result = engine.translate(np.zeros(126, dtype=np.float32))

    assert result.text == ""
    assert result.status == "labels_missing"


def test_sklearn_engine_reports_missing_predicted_label():
    engine = SklearnMlpEngine(FakeScaler(), FakeClassifier([0.1, 0.9]), {"0": "bonjour"})

    result = engine.translate(np.zeros(126, dtype=np.float32))

    assert result.text == ""
    assert result.status == "label_missing"


def test_load_model_rejects_missing_files(tmp_path):
    result = load_sklearn_model_from_dir(tmp_path)

    assert result.status == "model_invalid"
    assert result.engine.status == "model_not_loaded"


def test_load_model_rejects_bad_scaler_dimension(tmp_path):
    joblib = pytest.importorskip("joblib")
    sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")
    sklearn_neural = pytest.importorskip("sklearn.neural_network")

    scaler = sklearn_preprocessing.StandardScaler()
    scaler.n_features_in_ = 125
    classifier = sklearn_neural.MLPClassifier(hidden_layer_sizes=(256, 128))
    classifier.n_features_in_ = 126
    classifier.classes_ = np.array([0])

    joblib.dump(scaler, tmp_path / "scaler.pkl")
    joblib.dump(classifier, tmp_path / "mlp_classifier.pkl")

    result = load_sklearn_model_from_dir(tmp_path)

    assert result.status == "model_invalid"
    assert "125 valeurs" in result.message


def test_load_model_with_missing_labels_is_loaded_but_silent(tmp_path):
    joblib = pytest.importorskip("joblib")
    sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")
    sklearn_neural = pytest.importorskip("sklearn.neural_network")

    scaler = sklearn_preprocessing.StandardScaler()
    scaler.n_features_in_ = 126
    classifier = sklearn_neural.MLPClassifier(hidden_layer_sizes=(256, 128))
    classifier.n_features_in_ = 126
    classifier.classes_ = np.array([0])

    joblib.dump(scaler, tmp_path / "scaler.pkl")
    joblib.dump(classifier, tmp_path / "mlp_classifier.pkl")

    result = load_sklearn_model_from_dir(tmp_path)

    assert result.status == "labels_missing"
    assert result.engine.status == "labels_missing"
