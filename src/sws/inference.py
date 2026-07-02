from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np


EXPECTED_FEATURES = 126


@dataclass(frozen=True)
class TranslationResult:
    text: str
    tokens: list[str]
    confidence: float | None
    status: str
    latency_ms: float | None = None
    message: str | None = None


@dataclass(frozen=True)
class ModelLoadResult:
    engine: "TranslationEngine"
    status: str
    message: str


class TranslationEngine:
    status: str = "model_not_loaded"

    def translate(self, landmarks: np.ndarray) -> TranslationResult:
        raise NotImplementedError


class NoModelEngine(TranslationEngine):
    status = "model_not_loaded"

    def translate(self, landmarks: np.ndarray) -> TranslationResult:
        return TranslationResult(
            text="",
            tokens=[],
            confidence=None,
            status="model_not_loaded",
            message="Aucun modele charge",
        )


class SklearnMlpEngine(TranslationEngine):
    def __init__(
        self,
        scaler: Any,
        classifier: Any,
        labels: dict[str, str] | None,
        *,
        confidence_threshold: float = 0.60,
        load_status: str = "model_ready",
    ) -> None:
        self.scaler = scaler
        self.classifier = classifier
        self.labels = labels
        self.confidence_threshold = confidence_threshold
        self.status = load_status

    def translate(self, landmarks: np.ndarray) -> TranslationResult:
        start = time.perf_counter()
        try:
            vector = np.asarray(landmarks, dtype=np.float32)
            if vector.shape != (EXPECTED_FEATURES,):
                return TranslationResult(
                    text="",
                    tokens=[],
                    confidence=None,
                    status="inference_error",
                    message=f"Vecteur invalide : {vector.size} valeurs au lieu de {EXPECTED_FEATURES}",
                )

            if not self.labels:
                return TranslationResult(
                    text="",
                    tokens=[],
                    confidence=None,
                    status="labels_missing",
                    message="Modele charge, labels manquants",
                )

            sample = vector.reshape(1, EXPECTED_FEATURES)
            transformed = self.scaler.transform(sample)
            probabilities = np.asarray(self.classifier.predict_proba(transformed))[0]
            best_index = int(np.argmax(probabilities))
            confidence = float(probabilities[best_index])
            latency = (time.perf_counter() - start) * 1000

            if confidence < self.confidence_threshold:
                return TranslationResult(
                    text="",
                    tokens=[],
                    confidence=confidence,
                    status="low_confidence",
                    latency_ms=latency,
                )

            predicted_class = self.classifier.classes_[best_index]
            label = self.labels.get(str(predicted_class))
            if not label:
                return TranslationResult(
                    text="",
                    tokens=[],
                    confidence=confidence,
                    status="label_missing",
                    latency_ms=latency,
                    message=f"Label manquant pour la classe {predicted_class}",
                )

            return TranslationResult(
                text=label,
                tokens=[label],
                confidence=confidence,
                status="ok",
                latency_ms=latency,
            )
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            return TranslationResult(
                text="",
                tokens=[],
                confidence=None,
                status="inference_error",
                latency_ms=(time.perf_counter() - start) * 1000,
                message=str(exc),
            )


def load_sklearn_model_from_dir(model_dir: str | Path) -> ModelLoadResult:
    model_path = Path(model_dir).expanduser()
    if not model_path.exists() or not model_path.is_dir():
        return _invalid(f"Dossier modele introuvable : {model_path}")

    scaler_path = model_path / "scaler.pkl"
    classifier_path = model_path / "mlp_classifier.pkl"
    labels_path = model_path / "labels.json"

    if not scaler_path.exists():
        return _invalid("Modele invalide : scaler.pkl absent")
    if not classifier_path.exists():
        return _invalid("Modele invalide : mlp_classifier.pkl absent")

    try:
        import joblib
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return _invalid(f"Dependance scikit-learn/joblib indisponible : {exc}")

    try:
        scaler = joblib.load(scaler_path)
        classifier = joblib.load(classifier_path)
    except Exception as exc:
        return _invalid(f"Modele invalide : erreur de chargement ({exc})")

    if not isinstance(scaler, StandardScaler):
        return _invalid("Modele invalide : scaler.pkl n'est pas un StandardScaler")
    if not isinstance(classifier, MLPClassifier):
        return _invalid("Modele invalide : mlp_classifier.pkl n'est pas un MLPClassifier")

    scaler_features = getattr(scaler, "n_features_in_", None)
    if scaler_features != EXPECTED_FEATURES:
        return _invalid(
            f"Modele invalide : scaler.pkl attend {scaler_features} valeurs au lieu de {EXPECTED_FEATURES}"
        )

    classifier_features = getattr(classifier, "n_features_in_", None)
    if classifier_features != EXPECTED_FEATURES:
        return _invalid(
            "Modele invalide : mlp_classifier.pkl attend "
            f"{classifier_features} valeurs au lieu de {EXPECTED_FEATURES}"
        )

    if not hasattr(classifier, "classes_"):
        return _invalid("Modele invalide : classes_ absent du classifieur")
    if not hasattr(classifier, "predict_proba"):
        return _invalid("Modele invalide : predict_proba absent du classifieur")

    labels, label_message = _load_labels(labels_path)
    if labels is None:
        engine = SklearnMlpEngine(scaler, classifier, None, load_status="labels_missing")
        return ModelLoadResult(engine, "labels_missing", label_message)

    missing = [str(value) for value in classifier.classes_ if str(value) not in labels]
    if missing:
        engine = SklearnMlpEngine(scaler, classifier, labels, load_status="model_ready")
        return ModelLoadResult(
            engine,
            "model_ready",
            "Modele charge, mais certains labels manquent : " + ", ".join(missing),
        )

    engine = SklearnMlpEngine(scaler, classifier, labels, load_status="model_ready")
    return ModelLoadResult(engine, "model_ready", "Modele charge")


def _invalid(message: str) -> ModelLoadResult:
    return ModelLoadResult(NoModelEngine(), "model_invalid", message)


def _load_labels(path: Path) -> tuple[dict[str, str] | None, str]:
    if not path.exists():
        return None, "Modele charge, labels manquants"

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"Labels invalides : {exc}"

    if not isinstance(raw, dict):
        return None, "Labels invalides : le fichier doit contenir un objet JSON"

    labels: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value.strip():
            return None, "Labels invalides : chaque label doit etre une chaine non vide"
        labels[key] = value.strip()

    if not labels:
        return None, "Labels invalides : aucun label defini"
    return labels, "Labels charges"
