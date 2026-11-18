import wave
from pathlib import Path

import numpy as np
import pytest

from lesson_notes.model import Segment
from lesson_notes.transcribe import WhisperTranscriber, read_wav


def write_wav(path: Path, samples: list[int], rate: int = 16000, channels: int = 1) -> None:
    wav: wave.Wave_write
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.array(samples, dtype="<i2").tobytes())


def test_read_wav_scales_to_float32(tmp_path: Path) -> None:
    path: Path = tmp_path / "a.wav"
    write_wav(path, [0, 16384, -32768])
    samples: np.ndarray = read_wav(path)
    assert samples.dtype == np.float32
    assert samples.tolist() == [0.0, 0.5, -1.0]


@pytest.mark.parametrize(("rate", "channels"), [(48000, 1), (16000, 2)])
def test_read_wav_rejects_other_formats(tmp_path: Path, rate: int, channels: int) -> None:
    path: Path = tmp_path / "a.wav"
    write_wav(path, [0, 0], rate=rate, channels=channels)
    with pytest.raises(ValueError, match="16 kHz"):
        read_wav(path)


class FakeRaw:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start: float = start
        self.end: float = end
        self.text: str = text


class FakeModel:
    def __init__(self) -> None:
        self.audio: object = None

    def transcribe(self, audio: object, **kwargs: object) -> tuple[list[FakeRaw], None]:
        self.audio = audio
        return [FakeRaw(0.0, 1.0, " Bonjour "), FakeRaw(1.0, 2.0, "  ")], None


def test_model_gets_samples_not_a_path(tmp_path: Path) -> None:
    """faster-whisper's own decoder breaks on PyAV 19, so it must never see a path."""
    path: Path = tmp_path / "a.wav"
    write_wav(path, [0, 16384])
    transcriber: WhisperTranscriber = WhisperTranscriber("tiny")
    model: FakeModel = FakeModel()
    transcriber._model = model
    segments: list[Segment] = transcriber.transcribe(path)
    assert isinstance(model.audio, np.ndarray)
    assert segments == [Segment(0.0, 1.0, "Bonjour")]
