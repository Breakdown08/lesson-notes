"""Shared fakes and generators for the offline tests: no network, no recognition models."""

import math
import wave
from array import array
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import av
import numpy as np
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from lesson_notes.config import Settings
from lesson_notes.model import Segment

RATE: int = 16000
LOUD: int = 12000  # amplitude of a "loud" track
QUIET: int = 2000  # amplitude of a "quiet" track
RMS_THRESHOLD: float = 6000.0  # between the RMS of QUIET and of LOUD tones
RECORDING_NAME: str = "2026-11-16 18-00-03.mkv"
LESSON_DATE: date = date(2026, 11, 16)


def make_recording(
    path: Path,
    amplitudes: Sequence[int] = (LOUD, QUIET),
    seconds: float = 1.5,
    with_video: bool = False,
) -> Path:
    """MKV with one pcm_s16le audio stream per amplitude (a 440 Hz tone), optionally a video stream first."""
    container: Any = av.open(str(path), "w", format="matroska")
    video: Any = None
    i: int
    amp: int
    _amp: int
    vpacket: Any
    packet: Any
    if with_video:
        video = container.add_stream("mpeg4", rate=10)
        video.width = 64
        video.height = 48
        video.pix_fmt = "yuv420p"
    streams: list[Any] = []
    for _amp in amplitudes:
        stream: Any = container.add_stream("pcm_s16le", rate=RATE)
        stream.layout = "mono"
        streams.append(stream)
    if video is not None:
        for i in range(int(seconds * 10)):
            vframe: Any = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24")
            vframe.pts = i
            for vpacket in video.encode(vframe):
                container.mux(vpacket)
        for vpacket in video.encode(None):
            container.mux(vpacket)
    total: int = int(seconds * RATE)
    chunk: int = 1024
    start: int
    for start in range(0, total, chunk):
        n: int = min(chunk, total - start)
        t: Any = (np.arange(start, start + n) / RATE).astype(np.float64)
        for stream, amp in zip(streams, amplitudes, strict=True):
            samples: Any = (amp * math.sqrt(2) * np.sin(2 * math.pi * 440 * t)).astype(np.int16)
            frame: Any = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = RATE
            frame.pts = start
            for packet in stream.encode(frame):
                container.mux(packet)
    for stream in streams:
        for packet in stream.encode(None):
            container.mux(packet)
    container.close()
    return path


def wav_info(path: Path) -> tuple[int, int, int, float]:
    """(channels, sample width in bytes, frame rate, duration in seconds)."""
    w: wave.Wave_read
    with wave.open(str(path), "rb") as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes() / w.getframerate()


def wav_rms(path: Path) -> float:
    w: wave.Wave_read
    with wave.open(str(path), "rb") as w:
        raw: bytes = w.readframes(w.getnframes())
    samples: array[int] = array("h")
    samples.frombytes(raw)
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class FakeTranscriber:
    """Records the WAV passed to each call (path, existence and RMS at call time)."""

    def __init__(self, per_call: Sequence[list[Segment]] = ()) -> None:
        self.per_call: list[list[Segment]] = list(per_call)
        self.paths: list[Path] = []
        self.rms: list[float] = []

    def transcribe(self, audio: Path) -> list[Segment]:
        index: int = len(self.paths)
        self.paths.append(audio)
        self.rms.append(wav_rms(audio))
        return self.per_call[index] if index < len(self.per_call) else []


class ExplodingTranscriber:
    def transcribe(self, audio: Path) -> list[Segment]:
        raise AssertionError("transcriber must not be called")


class RecordingLLM(BaseChatModel):
    """Remembers every list of messages it receives; answers `answer` or raises `error`."""

    answer: str = "  Summary body.\n"
    error: Exception | None = None
    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "recording-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.answer))])


def make_llm_fake(answer: str = "  Summary body.\n", error: Exception | None = None) -> RecordingLLM:
    llm: RecordingLLM = RecordingLLM(answer=answer, error=error, calls=[])
    return llm


def make_settings(
    tmp_path: Path,
    output_dir: Path | None = None,
    summary_copy_dir: Path | None = None,
    student_track: int = 1,
    teacher_track: int = 2,
) -> Settings:
    out: Path = tmp_path / "out"
    copy: Path = tmp_path / "notes"
    rec: Path = tmp_path / "recordings"
    d: Path
    for d in (out, copy, rec):
        d.mkdir(exist_ok=True)
    return Settings(
        recordings_dir=rec,
        output_dir=output_dir or out,
        summary_copy_dir=summary_copy_dir or copy,
        xai_api_key="test-key",
        student_track=student_track,
        teacher_track=teacher_track,
    )


STUDENT_SEGMENTS: list[Segment] = [Segment(9.0, 11.0, "Ça va bien, merci.")]
TEACHER_SEGMENTS: list[Segment] = [Segment(5.0, 7.0, "Bonjour, ça va ?"), Segment(20.0, 22.0, "Très bien.")]
