import hashlib
import wave
from pathlib import Path

import numpy as np
import pytest

from lesson_notes.model import Segment
from lesson_notes.transcribe import PINS, Pin, WhisperTranscriber, load_model, read_wav, verify_weights


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


def test_verify_weights_accepts_the_right_hash(tmp_path: Path) -> None:
    path: Path = tmp_path / "model.bin"
    path.write_bytes(b"weights")
    verify_weights(path, hashlib.sha256(b"weights").hexdigest())


def test_verify_weights_rejects_another_hash(tmp_path: Path) -> None:
    path: Path = tmp_path / "model.bin"
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_weights(path, hashlib.sha256(b"weights").hexdigest())


def test_default_model_is_pinned() -> None:
    assert "large-v3-turbo" in PINS
    pin: Pin = PINS["large-v3-turbo"]
    assert len(pin.revision) == 40
    assert len(pin.sha256) == 64


class Loaded:
    def __init__(self, source: str, **kwargs: object) -> None:
        self.source: str = source


def test_pinned_model_downloads_at_revision_and_checks_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import faster_whisper  # type: ignore[import-untyped]  # faster-whisper ships no py.typed marker
    import faster_whisper.utils  # type: ignore[import-untyped]  # same

    (tmp_path / "model.bin").write_bytes(b"weights")
    calls: list[tuple[str, str | None]] = []

    def download(name: str, revision: str | None = None) -> str:
        calls.append((name, revision))
        return str(tmp_path)

    monkeypatch.setattr(faster_whisper.utils, "download_model", download)
    monkeypatch.setattr(faster_whisper, "WhisperModel", Loaded)
    monkeypatch.setitem(PINS, "pinned", Pin("r" * 40, hashlib.sha256(b"weights").hexdigest()))
    model: object = load_model("pinned", "int8")
    assert calls == [("pinned", "r" * 40)]
    assert isinstance(model, Loaded)
    assert model.source == str(tmp_path)

    monkeypatch.setitem(PINS, "pinned", Pin("r" * 40, "0" * 64))
    with pytest.raises(ValueError, match="SHA-256"):
        load_model("pinned", "int8")


def test_unpinned_model_loads_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    import faster_whisper
    import faster_whisper.utils

    def download(name: str, revision: str | None = None) -> str:
        raise AssertionError("an unpinned name must not be downloaded here")

    monkeypatch.setattr(faster_whisper.utils, "download_model", download)
    monkeypatch.setattr(faster_whisper, "WhisperModel", Loaded)
    model: object = load_model("tiny", "int8")
    assert isinstance(model, Loaded)
    assert model.source == "tiny"
