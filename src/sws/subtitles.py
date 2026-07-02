from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from itertools import count

from sws.inference import TranslationResult


@dataclass(frozen=True)
class SubtitleEvent:
    event_id: int
    kind: str
    text: str
    timestamp: float
    confidence: float | None = None


class PredictionStabilizer:
    def __init__(self, partial_min_frames: int = 3, final_min_frames: int = 8) -> None:
        self.partial_min_frames = partial_min_frames
        self.final_min_frames = final_min_frames
        self._window: deque[str] = deque(maxlen=final_min_frames)
        self._ids = count(1)
        self._current_partial: str | None = None
        self._last_final: str | None = None

    def update(self, result: TranslationResult, timestamp: float) -> list[SubtitleEvent]:
        if not result.text:
            return []

        self._window.append(result.text)
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
    def __init__(self) -> None:
        self.current_text = ""

    def apply(self, events: list[SubtitleEvent]) -> str:
        for event in events:
            self.current_text = event.text
        return self.current_text

    def clear(self) -> None:
        self.current_text = ""
