from pathlib import Path
from typing import Any, Final, Protocol

from lesson_notes.model import Segment


class Transcriber(Protocol):
    def transcribe(self, audio: Path) -> list[Segment]: ...


INITIAL_PROMPT: Final[str] = "Bonjour, ça va ? Привет, как дела? Aujourd'hui мы учим le passé composé."


class WhisperTranscriber:
    def __init__(self, model: str, compute_type: str = "int8") -> None:
        self.model_name: str = model
        self.compute_type: str = compute_type
        self._model: Any = None

    def transcribe(self, audio: Path) -> list[Segment]:
        if self._model is None:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]  # faster-whisper ships no py.typed marker

            self._model = WhisperModel(self.model_name, device="cpu", compute_type=self.compute_type)
        segments: Any = self._model.transcribe(
            str(audio),
            language=None,
            multilingual=True,
            vad_filter=True,
            initial_prompt=INITIAL_PROMPT,
        )[0]
        result: list[Segment] = []
        raw: Any
        for raw in segments:
            text: str = str(raw.text).strip()
            if text:
                result.append(Segment(float(raw.start), float(raw.end), text))
        return result
