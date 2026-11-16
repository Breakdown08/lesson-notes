from pathlib import Path

import pytest
from helpers import LOUD, QUIET, RMS_THRESHOLD, make_recording, wav_info, wav_rms

from lesson_notes.audio import SAMPLE_RATE, TrackNotFoundError, count_audio_tracks, extract_track


def test_10_counts_two_audio_tracks(tmp_path: Path) -> None:
    rec: Path = make_recording(tmp_path / "a.mkv")
    assert count_audio_tracks(rec) == 2


def test_10_video_stream_is_not_counted(tmp_path: Path) -> None:
    rec: Path = make_recording(tmp_path / "v.mkv", with_video=True)
    assert count_audio_tracks(rec) == 2


def test_11_extract_track_one(tmp_path: Path) -> None:
    rec: Path = make_recording(tmp_path / "a.mkv", seconds=1.5)
    dest: Path = tmp_path / "t1.wav"
    assert extract_track(rec, 1, dest) == dest
    channels: int
    width: int
    rate: int
    duration: float
    channels, width, rate, duration = wav_info(dest)
    assert (channels, width, rate) == (1, 2, 16000)
    assert SAMPLE_RATE == 16000
    assert abs(duration - 1.5) <= 0.1
    assert wav_rms(dest) > RMS_THRESHOLD


def test_11_extract_track_two(tmp_path: Path) -> None:
    rec: Path = make_recording(tmp_path / "a.mkv", seconds=1.5)
    dest: Path = tmp_path / "t2.wav"
    assert extract_track(rec, 2, dest) == dest
    channels: int
    width: int
    rate: int
    duration: float
    channels, width, rate, duration = wav_info(dest)
    assert (channels, width, rate) == (1, 2, 16000)
    assert abs(duration - 1.5) <= 0.1
    rms: float = wav_rms(dest)
    assert rms < RMS_THRESHOLD
    assert rms > 0.5 * QUIET  # the second tone is really there, not silence
    assert LOUD > RMS_THRESHOLD > QUIET


def test_11_track_numbers_ignore_video_stream(tmp_path: Path) -> None:
    rec: Path = make_recording(tmp_path / "v.mkv", with_video=True)
    first: Path = extract_track(rec, 1, tmp_path / "1.wav")
    second: Path = extract_track(rec, 2, tmp_path / "2.wav")
    assert wav_rms(first) > RMS_THRESHOLD > wav_rms(second)


@pytest.mark.parametrize("track", [3, 0, -1])
def test_12_track_out_of_range(tmp_path: Path, track: int) -> None:
    rec: Path = make_recording(tmp_path / "a.mkv")
    dest: Path = tmp_path / "x.wav"
    info: pytest.ExceptionInfo[TrackNotFoundError]
    with pytest.raises(TrackNotFoundError) as info:
        extract_track(rec, track, dest)
    text: str = str(info.value)
    assert str(track) in text
    assert "2" in text
    assert not dest.exists()
