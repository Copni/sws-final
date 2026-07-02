from pathlib import Path

import numpy as np

from sws.capture import WebcamCapture, is_probably_black_frame
from sws.inference import TranslationResult
from sws.output import compose_frame, draw_debug_overlay, export_mp4
from sws.preprocessing import HandVectorResult
from sws.subtitles import PredictionStabilizer, SubtitleBuffer, normalize_subtitle_text


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


def test_subtitle_buffer_chains_final_words_and_current_partial():
    buffer = SubtitleBuffer(max_words=4)

    buffer.apply(stabilizer_events("final", "bonjour"))
    buffer.apply(stabilizer_events("final", "merci"))
    text = buffer.apply(stabilizer_events("partial", "cafe"))

    assert text == "bonjour merci cafe"


def test_subtitle_buffer_drops_oldest_words():
    buffer = SubtitleBuffer(max_words=3)

    for text in ["un deux", "trois", "quatre"]:
        buffer.apply(stabilizer_events("final", text))

    assert buffer.current_text == "deux trois quatre"


def test_subtitle_text_repairs_common_mojibake():
    assert normalize_subtitle_text("cafÃ©") == "café"
    assert normalize_subtitle_text("  déjà   vu  ") == "déjà vu"


def test_compose_frame_adds_white_subtitle_area_under_video():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)

    composed = compose_frame(frame, "")

    assert composed.shape == (106, 20, 3)
    assert np.all(composed[:10] == 0)
    assert np.all(composed[10:] == 255)


def test_compose_frame_keeps_subtitle_area_renderable_with_accents():
    frame = np.zeros((80, 240, 3), dtype=np.uint8)

    composed = compose_frame(frame, "déjà café ?", subtitle_height=80)

    subtitle_area = composed[80:]
    assert subtitle_area.min() < 255


def test_draw_debug_overlay_adds_skeleton_and_confidence_without_mutating_source():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    vector = np.zeros(126, dtype=np.float32)
    for index in range(21):
        vector[index * 3] = 0.2 + index * 0.02
        vector[index * 3 + 1] = 0.2 + index * 0.01
    hand_result = HandVectorResult(vector=vector, left_present=True, right_present=False)

    debug = draw_debug_overlay(frame, hand_result, 0.87)

    assert debug.shape == frame.shape
    assert np.any(debug != frame)
    assert np.all(frame == 0)


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


def stabilizer_events(kind, text):
    return [SubtitleEventForTest(kind=kind, text=text)]


class SubtitleEventForTest:
    def __init__(self, kind, text):
        self.kind = kind
        self.text = text
