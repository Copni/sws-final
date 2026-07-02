from __future__ import annotations

from dataclasses import dataclass

from sws.inference import NoModelEngine, TranslationEngine, load_sklearn_model_from_dir


@dataclass
class AppState:
    model_status: str = "model_not_loaded"
    model_message: str = "Aucun modele charge"
    source_status: str = "Aucune source"
    output_status: str = "Sortie inactive"
    fps: float = 0.0
    latency_ms: float | None = None


class AppController:
    def __init__(self) -> None:
        self.state = AppState()
        self.engine: TranslationEngine = NoModelEngine()

    def load_model_dir(self, model_dir: str) -> AppState:
        result = load_sklearn_model_from_dir(model_dir)
        self.engine = result.engine
        self.state.model_status = result.status
        self.state.model_message = result.message
        return self.state

    def clear_model(self) -> AppState:
        self.engine = NoModelEngine()
        self.state.model_status = "model_not_loaded"
        self.state.model_message = "Aucun modele charge"
        return self.state
