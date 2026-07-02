from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


EXPECTED_FEATURES = 126
LANDMARKS_PER_HAND = 21


@dataclass(frozen=True)
class HandVectorResult:
    vector: np.ndarray
    left_present: bool
    right_present: bool


def extract_hands_126(
    multi_hand_landmarks: Iterable[Any] | None,
    multi_handedness: Iterable[Any] | None = None,
) -> HandVectorResult:
    hands = list(multi_hand_landmarks or [])
    handedness = list(multi_handedness or [])
    result = {
        "Left": np.zeros(63, dtype=np.float32),
        "Right": np.zeros(63, dtype=np.float32),
    }
    present = {"Left": False, "Right": False}

    labels = [_handedness_label(item) for item in handedness]
    if len(labels) < len(hands):
        labels = _fallback_labels_from_wrist_x(hands)

    for hand, label in zip(hands, labels):
        if label not in result:
            continue
        result[label] = _flatten_landmarks(hand)
        present[label] = True

    vector = np.concatenate([result["Left"], result["Right"]]).astype(np.float32)
    return HandVectorResult(vector=vector, left_present=present["Left"], right_present=present["Right"])


def process_frame_bgr(frame_bgr: np.ndarray, hands_processor: Any) -> HandVectorResult:
    frame_rgb = _bgr_to_rgb(frame_bgr)
    processed = hands_processor.process(frame_rgb)
    return extract_hands_126(
        getattr(processed, "multi_hand_landmarks", None),
        getattr(processed, "multi_handedness", None),
    )


class MediaPipeHandsExtractor:
    def __init__(self, *, max_num_hands: int = 2) -> None:
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr: np.ndarray) -> HandVectorResult:
        return process_frame_bgr(frame_bgr, self._hands)

    def close(self) -> None:
        self._hands.close()


def _flatten_landmarks(hand: Any) -> np.ndarray:
    landmarks = getattr(hand, "landmark", hand)
    values: list[float] = []
    for landmark in list(landmarks)[:LANDMARKS_PER_HAND]:
        values.extend(
            [
                float(getattr(landmark, "x", 0.0)),
                float(getattr(landmark, "y", 0.0)),
                float(getattr(landmark, "z", 0.0)),
            ]
        )
    while len(values) < 63:
        values.append(0.0)
    return np.asarray(values[:63], dtype=np.float32)


def _handedness_label(item: Any) -> str | None:
    classifications = getattr(item, "classification", None)
    if classifications:
        label = getattr(classifications[0], "label", None)
        if label in {"Left", "Right"}:
            return label
    label = getattr(item, "label", None)
    if label in {"Left", "Right"}:
        return label
    return None


def _fallback_labels_from_wrist_x(hands: list[Any]) -> list[str]:
    if len(hands) == 1:
        return ["Left"]

    indexed = []
    for index, hand in enumerate(hands[:2]):
        landmarks = getattr(hand, "landmark", hand)
        wrist = list(landmarks)[0]
        indexed.append((float(getattr(wrist, "x", 0.0)), index))

    labels = [None] * len(hands)
    for label, (_, index) in zip(["Left", "Right"], sorted(indexed)):
        labels[index] = label
    return [(label or "Left") for label in labels]


def _bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    try:
        import cv2

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return frame_bgr[..., ::-1].copy()
