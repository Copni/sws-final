from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from itertools import count
import unicodedata

from sws.inference import TranslationResult


@dataclass(frozen=True)
class SubtitleEvent:
    event_id: int
    kind: str
    text: str
    timestamp: float
    confidence: float | None = None


class PredictionStabilizer:
    def __init__(self, partial_min_frames: int = 3, final_min_frames: int = 4) -> None:
        self.partial_min_frames = partial_min_frames
        self.final_min_frames = final_min_frames
        self._window: deque[str] = deque(maxlen=final_min_frames)
        self._ids = count(1)
        self._current_partial: str | None = None
        self._last_final: str | None = None

    def update(self, result: TranslationResult, timestamp: float) -> list[SubtitleEvent]:
        normalized_text = normalize_subtitle_text(result.text)
        if not normalized_text:
            return []

        self._window.append(normalized_text)
        text, occurrences = Counter(self._window).most_common(1)[0]
        events: list[SubtitleEvent] = []

        if (
            occurrences >= self.partial_min_frames
            and text != self._current_partial
            and text != self._last_final
        ):
            self._current_partial = text
            events.append(
                SubtitleEvent(next(self._ids), "partial", text, timestamp, result.confidence)
            )

        if occurrences >= self.final_min_frames and text != self._last_final:
            self._last_final = text
            self._current_partial = None
            events.append(SubtitleEvent(next(self._ids), "final", text, timestamp, result.confidence))

        return events

    def reset(self) -> None:
        self._window.clear()
        self._current_partial = None
        self._last_final = None


class SubtitleBuffer:
    def __init__(self, max_words: int = 10) -> None:
        self.max_words = max_words
        self._final_words: deque[str] = deque()
        self._partial_text: str | None = None
        self.current_text = ""

    def apply(self, events: list[SubtitleEvent]) -> str:
        for event in events:
            text = normalize_subtitle_text(event.text)
            if not text:
                continue

            if event.kind == "partial":
                self._partial_text = text
            elif event.kind == "final":
                words = text.split()
                if not self._ends_with(words):
                    self._final_words.extend(words)
                    self._trim()
                if self._partial_text == text:
                    self._partial_text = None
            else:
                self._partial_text = text

            self._refresh()
        return self.current_text

    def clear(self) -> None:
        self._final_words.clear()
        self._partial_text = None
        self.current_text = ""

    def _trim(self) -> None:
        while len(self._final_words) > self.max_words:
            self._final_words.popleft()

    def _refresh(self) -> None:
        words = list(self._final_words)
        if self._partial_text:
            partial_words = self._partial_text.split()
            if not self._ends_with(partial_words):
                words.extend(partial_words)
        self.current_text = " ".join(words[-self.max_words :])

    def _ends_with(self, words: list[str]) -> bool:
        return bool(words) and list(self._final_words)[-len(words) :] == words


def normalize_subtitle_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFC", text).strip()
    if not cleaned:
        return ""

    if "Ã" in cleaned or "Â" in cleaned:
        try:
            repaired = cleaned.encode("latin-1").decode("utf-8")
        except UnicodeError:
            repaired = cleaned
        else:
            cleaned = repaired

    return " ".join(cleaned.split())
