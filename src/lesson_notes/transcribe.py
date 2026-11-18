import wave
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from lesson_notes.model import Segment


class Transcriber(Protocol):
    def transcribe(self, audio: Path) -> list[Segment]: ...


INITIAL_PROMPT: Final[str] = "Bonjour, ça va ? Привет, как дела? Aujourd'hui мы учим le passé composé."


def read_wav(path: Path) -> np.ndarray:
    """Read a 16-bit mono 16 kHz WAV into float32 samples in [-1, 1].

    faster-whisper gets samples, not a path: its own decoder passes PyAV an argument
    that PyAV 19 rejects (`metadata_errors`), and the track is already a 16 kHz WAV.
    """
    wav: wave.Wave_read
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
            raise ValueError(f"{path}: expected 16-bit mono 16 kHz WAV")
        frames: bytes = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


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
            read_wav(audio),
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
