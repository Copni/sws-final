from pathlib import Path

import numpy as np

from sws.capture import WebcamCapture, is_probably_black_frame
from sws.inference import TranslationResult
from sws.output import compose_frame, export_mp4
from sws.subtitles import PredictionStabilizer


def translation(text, confidence=0.9):
    return TranslationResult(text=text, tokens=[text], confidence=confidence, status="ok")


def test_stabilizer_emits_partial_then_final_without_duplicate_final():
    stabilizer = PredictionStabilizer(partial_min_frames=2, final_min_frames=3)
    events = []

    events.extend(stabilizer.update(translation("bonjour"), 0.0))
    events.extend(stabilizer.update(translation("bonjour"), 0.1))
    events.extend(stabilizer.update(translation("bonjour"), 0.2))
    events.extend(stabilizer.update(translation("bonjour"), 0.3))

    assert [event.kind for event in events] == ["partial", "final"]
    assert [event.text for event in events] == ["bonjour", "bonjour"]


def test_compose_frame_adds_white_subtitle_area_under_video():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)

    composed = compose_frame(frame, "")

    assert composed.shape == (106, 20, 3)
    assert np.all(composed[:10] == 0)
    assert np.all(composed[10:] == 255)


def test_webcam_queue_keeps_only_latest_frame():
    capture = WebcamCapture(0)

    for index in range(3):
        capture._frames.append(index)  # queue behavior only

    assert list(capture._frames) == [2]


def test_black_frame_detection():
    assert is_probably_black_frame(np.zeros((10, 10, 3), dtype=np.uint8)) is True
    assert is_probably_black_frame(np.full((10, 10, 3), 80, dtype=np.uint8)) is False


def test_export_rejects_fake_mp4(tmp_path):
    fake = tmp_path / "fake.mp4"
    fake.write_text("not a video", encoding="utf-8")

    ok, message = export_mp4(fake, tmp_path / "out.mp4", lambda frame, media_time: frame)

    assert ok is False
    assert "Impossible" in message or "aucune frame" in message
