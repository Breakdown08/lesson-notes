import wave
from pathlib import Path
from typing import Any, Final

import av

SAMPLE_RATE: Final = 16000


class TrackNotFoundError(Exception): ...


def count_audio_tracks(recording: Path) -> int:
    container: Any
    with av.open(str(recording)) as container:
        return len(container.streams.audio)


def extract_track(recording: Path, track: int, dest: Path) -> Path:
    """Decode audio stream number `track` (1-based, audio streams only) with PyAV,
    resample to 16 kHz mono 16-bit PCM and write a WAV file to `dest`. Returns `dest`."""
    container: Any
    wav: wave.Wave_write
    with av.open(str(recording)) as container:
        streams: list[Any] = list(container.streams.audio)
        if track < 1 or track > len(streams):
            raise TrackNotFoundError(
                f"Audio track {track} requested, but {recording.name} has {len(streams)} audio track(s)"
            )
        stream: Any = streams[track - 1]
        resampler: Any = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        with wave.open(str(dest), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            packet: Any
            frame: Any
            out: Any
            for packet in container.demux(stream):
                for frame in packet.decode():
                    for out in resampler.resample(frame):
                        wav.writeframes(bytes(out.planes[0])[: out.samples * 2])
            for out in resampler.resample(None):
                wav.writeframes(bytes(out.planes[0])[: out.samples * 2])
    return dest
