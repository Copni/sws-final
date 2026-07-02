from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from pathlib import Path
import sys

import numpy as np


@dataclass(frozen=True)
class VideoFrame:
    bgr: np.ndarray
    timestamp: float
    width: int
    height: int
    fps: float
    media_time: float | None = None


@dataclass(frozen=True)
class WebcamDevice:
    index: int
    backend: int
    backend_name: str
    width: int
    height: int

    @property
    def display_name(self) -> str:
        size = f"{self.width}x{self.height}" if self.width and self.height else "taille inconnue"
        return f"Webcam {self.index} - {self.backend_name} ({size})"


class WebcamCapture:
    def __init__(self, index: int, backend: int | None = None) -> None:
        self.index = index
        self.backend = backend
        self._capture = None
        self._frames: deque[VideoFrame] = deque(maxlen=1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.fps = 30.0

    def start(self) -> tuple[bool, str]:
        import cv2

        self._capture = _open_capture(self.index, self.backend)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            return False, f"Impossible d'ouvrir la webcam {self.index}"

        ok, frame = self._capture.read()
        if not ok or frame is None:
            backend_name = _backend_name(cv2, self.backend)
            self._capture.release()
            self._capture = None
            return False, f"Webcam {self.index} ouverte avec {backend_name}, mais aucune image lisible"

        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 30.0)
        height, width = frame.shape[:2]
        with self._lock:
            self._frames.append(VideoFrame(frame, time.monotonic(), width, height, self.fps))
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True, f"Webcam {self.index} active ({_backend_name(cv2, self.backend)})"

    def read_latest(self) -> VideoFrame | None:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._capture:
            self._capture.release()
        self._thread = None
        self._capture = None

    def _loop(self) -> None:
        while not self._stop.is_set() and self._capture:
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            height, width = frame.shape[:2]
            video_frame = VideoFrame(frame, time.monotonic(), width, height, self.fps)
            with self._lock:
                self._frames.append(video_frame)


class Mp4Reader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._capture = None
        self.fps = 25.0
        self.frame_count = 0
        self._index = 0

    def open(self) -> tuple[bool, str]:
        import cv2

        if self.path.suffix.lower() != ".mp4" or not self.path.exists():
            return False, "Fichier MP4 invalide"

        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            return False, "Impossible de lire le MP4"
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 25.0)
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._index = 0
        return True, f"MP4 charge : {self.path.name}"

    def read(self) -> VideoFrame | None:
        if not self._capture:
            return None
        ok, frame = self._capture.read()
        if not ok:
            return None
        height, width = frame.shape[:2]
        media_time = self._index / self.fps
        self._index += 1
        return VideoFrame(frame, time.monotonic(), width, height, self.fps, media_time)

    def close(self) -> None:
        if self._capture:
            self._capture.release()
        self._capture = None


def list_webcam_devices(max_index: int = 8) -> list[WebcamDevice]:
    try:
        import cv2
    except Exception:
        return []

    devices: list[WebcamDevice] = []
    old_log_level = cv2.getLogLevel() if hasattr(cv2, "getLogLevel") else None
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)
    try:
        for index in range(max_index + 1):
            for backend_name, backend in _candidate_backends(cv2):
                capture = _open_capture(index, backend)
                try:
                    if not capture.isOpened():
                        continue
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        continue
                    height, width = frame.shape[:2]
                    devices.append(WebcamDevice(index, backend, backend_name, width, height))
                    break
                finally:
                    capture.release()
    finally:
        if old_log_level is not None and hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(old_log_level)
    return devices


def list_webcam_indices(max_index: int = 5) -> list[int]:
    return [device.index for device in list_webcam_devices(max_index)]


def is_probably_black_frame(frame: np.ndarray, *, mean_threshold: float = 5.0) -> bool:
    return float(frame.mean()) < mean_threshold


def _open_capture(index: int, backend: int | None):
    import cv2

    if backend is None or backend == cv2.CAP_ANY:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, backend)


def _candidate_backends(cv2) -> list[tuple[str, int]]:
    if sys.platform.startswith("win"):
        return [
            ("DirectShow", cv2.CAP_DSHOW),
            ("Media Foundation", cv2.CAP_MSMF),
            ("Auto", cv2.CAP_ANY),
        ]
    if sys.platform.startswith("linux"):
        return [
            ("V4L2", cv2.CAP_V4L2),
            ("Auto", cv2.CAP_ANY),
        ]
    return [("Auto", cv2.CAP_ANY)]


def _backend_name(cv2, backend: int | None) -> str:
    if backend is None or backend == cv2.CAP_ANY:
        return "Auto"
    names = {
        getattr(cv2, "CAP_DSHOW", -1): "DirectShow",
        getattr(cv2, "CAP_MSMF", -2): "Media Foundation",
        getattr(cv2, "CAP_V4L2", -3): "V4L2",
    }
    return names.get(backend, str(backend))
