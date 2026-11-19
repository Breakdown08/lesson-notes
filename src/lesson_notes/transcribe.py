import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Final, Protocol

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


@dataclass(frozen=True)
class Pin:
    """A Hugging Face revision of a faster-whisper model and the SHA-256 of its `model.bin`."""

    revision: str
    sha256: str


# Checked on 2026-11-19 against the weights used for the first real lesson (2026-11-18).
# The SHA-256 is also the name of the LFS blob on Hugging Face.
PINS: Final[dict[str, Pin]] = {
    "large-v3-turbo": Pin(
        revision="0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
        sha256="e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da",
    ),
}


def verify_weights(path: Path, expected: str) -> None:
    """Raise if the file's SHA-256 is not `expected`."""
    f: BinaryIO
    with path.open("rb") as f:
        actual: str = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"{path}: SHA-256 {actual}, expected {expected}")


def load_model(name: str, compute_type: str) -> object:
    """A faster-whisper model; a pinned name is downloaded at its revision and checked first.

    Names outside `PINS` (another size, a local path) load as faster-whisper resolves them.
    """
    import faster_whisper  # type: ignore[import-untyped]  # faster-whisper ships no py.typed marker
    import faster_whisper.utils  # type: ignore[import-untyped]  # same

    source: str = name
    pin: Pin | None = PINS.get(name)
    if pin is not None:
        source = str(faster_whisper.utils.download_model(name, revision=pin.revision))
        verify_weights(Path(source) / "model.bin", pin.sha256)
    return faster_whisper.WhisperModel(source, device="cpu", compute_type=compute_type)


class WhisperTranscriber:
    def __init__(self, model: str, compute_type: str = "int8") -> None:
        self.model_name: str = model
        self.compute_type: str = compute_type
        self._model: Any = None

    def transcribe(self, audio: Path) -> list[Segment]:
        if self._model is None:
            loaded: Any = load_model(self.model_name, self.compute_type)  # untyped faster-whisper object
            self._model = loaded
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
