from types import SimpleNamespace

import numpy as np

from sws.preprocessing import extract_hands_126, process_frame_bgr


def landmark(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def hand(offset):
    return SimpleNamespace(
        landmark=[landmark(offset + i, offset + i + 0.1, offset + i + 0.2) for i in range(21)]
    )


def handed(label):
    return SimpleNamespace(classification=[SimpleNamespace(label=label)])


def test_extracts_126_values_left_then_right():
    left = hand(1.0)
    right = hand(100.0)

    result = extract_hands_126([right, left], [handed("Right"), handed("Left")])

    assert result.vector.shape == (126,)
    assert result.left_present is True
    assert result.right_present is True
    np.testing.assert_allclose(result.vector[:3], [1.0, 1.1, 1.2])
    np.testing.assert_allclose(result.vector[63:66], [100.0, 100.1, 100.2])


def test_missing_hand_is_filled_with_zeros():
    right = hand(10.0)

    result = extract_hands_126([right], [handed("Right")])

    assert result.left_present is False
    assert result.right_present is True
    assert np.all(result.vector[:63] == 0)
    np.testing.assert_allclose(result.vector[63:66], [10.0, 10.1, 10.2])


def test_process_frame_converts_bgr_to_rgb_before_mediapipe():
    captured = {}

    class FakeHands:
        def process(self, frame_rgb):
            captured["pixel"] = frame_rgb[0, 0].tolist()
            return SimpleNamespace(multi_hand_landmarks=[], multi_handedness=[])

    frame_bgr = np.array([[[1, 2, 3]]], dtype=np.uint8)

    process_frame_bgr(frame_bgr, FakeHands())

    assert captured["pixel"] == [3, 2, 1]


def test_fallback_uses_wrist_horizontal_order():
    left = hand(0.1)
    right = hand(0.8)

    result = extract_hands_126([right, left], None)

    np.testing.assert_allclose(result.vector[:3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(result.vector[63:66], [0.8, 0.9, 1.0])
