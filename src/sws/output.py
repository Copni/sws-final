from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Callable

import numpy as np


def compose_frame(
    frame_bgr: np.ndarray,
    subtitle_text: str = "",
    *,
    subtitle_height: int = 96,
) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    composed = np.full((height + subtitle_height, width, 3), 255, dtype=np.uint8)
    composed[:height, :width] = frame_bgr

    if subtitle_text:
        _draw_subtitle(composed, subtitle_text, width, height, subtitle_height)
    return composed


def export_mp4(
    source_path: str | Path,
    output_path: str | Path,
    compose: Callable[[np.ndarray, float], np.ndarray],
    *,
    progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[bool, str]:
    try:
        import cv2
    except Exception as exc:
        return False, f"OpenCV indisponible : {exc}"

    source = Path(source_path)
    destination = Path(output_path)
    if source.suffix.lower() != ".mp4" or not source.exists():
        return False, "Fichier MP4 source invalide"
    if destination.exists():
        return False, "Le fichier de destination existe deja"

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return False, "Impossible de lire le MP4 source"

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        return False, "Le MP4 source ne contient aucune frame lisible"

    first_composed = compose(first_frame, 0.0)
    height, width = first_composed.shape[:2]
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4.tmp",
        dir=str(destination.parent),
    )
    temp_path = Path(temp_file.name)
    temp_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        temp_path.unlink(missing_ok=True)
        return False, "Impossible de creer le MP4 de sortie"

    try:
        index = 0
        writer.write(first_composed)
        if progress and total:
            progress(1 / total)

        while True:
            if should_cancel and should_cancel():
                return False, "Export annule"
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            media_time = index / fps
            writer.write(compose(frame, media_time))
            if progress and total:
                progress(min(1.0, (index + 1) / total))
    finally:
        writer.release()
        capture.release()

    if should_cancel and should_cancel():
        temp_path.unlink(missing_ok=True)
        return False, "Export annule"

    check = cv2.VideoCapture(str(temp_path))
    readable = check.isOpened() and check.read()[0]
    check.release()
    if not readable:
        temp_path.unlink(missing_ok=True)
        return False, "Le fichier exporte n'est pas relisible"

    temp_path.replace(destination)
    return True, "Export termine. Audio non conserve dans cette version."


class VirtualCameraSink:
    def __init__(self) -> None:
        self._camera = None
        self.message = ""

    @property
    def active(self) -> bool:
        return self._camera is not None

    def start(self, width: int, height: int, fps: float) -> tuple[bool, str]:
        try:
            import pyvirtualcam
        except Exception:
            return False, "Camera virtuelle indisponible : pyvirtualcam n'est pas installe"

        try:
            self._camera = pyvirtualcam.Camera(width=width, height=height, fps=max(1, int(fps)))
            self.message = f"Camera virtuelle active : {self._camera.device}"
            return True, self.message
        except Exception as exc:
            return False, (
                "Camera virtuelle indisponible. Verifiez OBS Virtual Camera sous Windows "
                f"ou v4l2loopback sous Linux. Detail : {exc}"
            )

    def send_bgr(self, frame_bgr: np.ndarray) -> None:
        if not self._camera:
            return
        frame_rgb = frame_bgr[..., ::-1]
        self._camera.send(frame_rgb)
        self._camera.sleep_until_next_frame()

    def close(self) -> None:
        if self._camera:
            self._camera.close()
            self._camera = None


def _draw_subtitle(
    composed: np.ndarray,
    text: str,
    width: int,
    video_height: int,
    subtitle_height: int,
) -> None:
    try:
        import cv2
    except Exception:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.72
    thickness = 2
    max_width = max(40, width - 32)
    lines = _wrap_text(text, max_width, font, scale, thickness)[:2]
    total_line_height = 30 * len(lines)
    start_y = video_height + max(30, (subtitle_height - total_line_height) // 2 + 22)

    for offset, line in enumerate(lines):
        size, _ = cv2.getTextSize(line, font, scale, thickness)
        x = max(12, (width - size[0]) // 2)
        y = start_y + offset * 32
        cv2.putText(composed, line, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _wrap_text(text: str, max_width: int, font: int, scale: float, thickness: int) -> list[str]:
    import cv2

    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
        if current and width > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]
