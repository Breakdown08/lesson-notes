import os
from datetime import date
from pathlib import Path

import pytest
from helpers import RECORDING_NAME, make_settings

from lesson_notes.config import ConfigError, Settings
from lesson_notes.plan import (
    LessonExistsError,
    LessonPlan,
    RecordingNotFoundError,
    lesson_date,
    pick_latest_recording,
    plan_lesson,
)

TRANSCRIPT_TEMPLATE: str = (
    "# Lesson 2026-11-16 — transcript\n\nRecording: {name}\n\n_No speech recognized._\n"
)


def touch(path: Path, mtime: float | None = None) -> Path:
    path.write_text("x")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_23_picks_newest_by_mtime(tmp_path: Path) -> None:
    touch(tmp_path / "old.mkv", 1_000_000)
    touch(tmp_path / "new.MKV", 3_000_000)
    touch(tmp_path / "mid.mp4", 2_000_000)
    touch(tmp_path / "newest.txt", 9_000_000)
    sub: Path = tmp_path / "sub.mkv"
    sub.mkdir()
    os.utime(sub, (9_500_000, 9_500_000))
    assert pick_latest_recording(tmp_path) == tmp_path / "new.MKV"


def test_23_mov_is_a_recording(tmp_path: Path) -> None:
    touch(tmp_path / "a.mov", 5_000_000)
    touch(tmp_path / "b.mkv", 1_000_000)
    assert pick_latest_recording(tmp_path) == tmp_path / "a.mov"


def test_23_empty_directory(tmp_path: Path) -> None:
    touch(tmp_path / "notes.txt")
    (tmp_path / "dir.mkv").mkdir()
    with pytest.raises(RecordingNotFoundError):
        pick_latest_recording(tmp_path)


def test_24_date_from_obs_name_with_space(tmp_path: Path) -> None:
    assert lesson_date(touch(tmp_path / "2026-11-16 18-00-03.mkv", 1_000_000)) == date(2026, 11, 16)


def test_24_date_from_name_with_underscore(tmp_path: Path) -> None:
    assert lesson_date(touch(tmp_path / "2026-11-16_18-00-03.mkv", 1_000_000)) == date(2026, 11, 16)


def test_24_date_from_name_without_time(tmp_path: Path) -> None:
    assert lesson_date(touch(tmp_path / "2026-11-16.mkv", 1_000_000)) == date(2026, 11, 16)


def local_noon(y: int, m: int, d: int) -> float:
    import datetime

    return datetime.datetime(y, m, d, 12, 0, 0).timestamp()


def test_24_date_falls_back_to_mtime(tmp_path: Path) -> None:
    rec: Path = touch(tmp_path / "lesson.mkv", local_noon(2026, 3, 5))
    assert lesson_date(rec) == date(2026, 3, 5)


def test_24_invalid_date_in_name_falls_back_to_mtime(tmp_path: Path) -> None:
    rec: Path = touch(tmp_path / "2026-13-40 x.mkv", local_noon(2026, 3, 5))
    assert lesson_date(rec) == date(2026, 3, 5)


def test_25_paths(tmp_path: Path) -> None:
    s: Settings = make_settings(tmp_path)
    rec: Path = touch(tmp_path / RECORDING_NAME)
    plan: LessonPlan = plan_lesson(rec, s, False)
    assert plan.recording == rec
    assert plan.lesson_date == date(2026, 11, 16)
    assert plan.lesson_dir == s.output_dir / "2026-11-16"
    assert plan.summary_copy_path == s.summary_copy_dir / "2026-11-16.md"
    assert plan.resume_from_transcript is False


def test_26_missing_output_dir(tmp_path: Path) -> None:
    s: Settings = make_settings(tmp_path)
    missing: Path = tmp_path / "typo-out"
    broken: Settings = make_settings(tmp_path, output_dir=missing)
    rec: Path = touch(tmp_path / RECORDING_NAME)
    info: pytest.ExceptionInfo[ConfigError]
    with pytest.raises(ConfigError) as info:
        plan_lesson(rec, broken, False)
    assert str(missing) in str(info.value)
    assert not missing.exists()
    assert s.output_dir.exists()


def test_26_missing_summary_copy_dir(tmp_path: Path) -> None:
    missing: Path = tmp_path / "typo-notes"
    broken: Settings = make_settings(tmp_path, summary_copy_dir=missing)
    rec: Path = touch(tmp_path / RECORDING_NAME)
    info: pytest.ExceptionInfo[ConfigError]
    with pytest.raises(ConfigError) as info:
        plan_lesson(rec, broken, True)
    assert str(missing) in str(info.value)
    assert not missing.exists()


def test_26_output_dir_is_a_file(tmp_path: Path) -> None:
    afile: Path = touch(tmp_path / "afile")
    broken: Settings = make_settings(tmp_path, output_dir=afile)
    with pytest.raises(ConfigError):
        plan_lesson(touch(tmp_path / RECORDING_NAME), broken, False)


def lesson_dir_with(s: Settings, transcript_of: str | None, summary: bool) -> Path:
    d: Path = s.output_dir / "2026-11-16"
    d.mkdir()
    if transcript_of is not None:
        (d / "transcript.md").write_text(TRANSCRIPT_TEMPLATE.format(name=transcript_of), encoding="utf-8")
    if summary:
        (d / "summary.md").write_text("old summary\n", encoding="utf-8")
    return d


def test_27_no_lesson_dir(tmp_path: Path) -> None:
    s: Settings = make_settings(tmp_path)
    plan: LessonPlan = plan_lesson(touch(tmp_path / RECORDING_NAME), s, False)
    assert plan.resume_from_transcript is False
    assert not plan.lesson_dir.exists()  # planning writes nothing


def test_27_dir_without_transcript(tmp_path: Path) -> None:
    s: Settings = make_settings(tmp_path)
    lesson_dir_with(s, None, False)
    plan: LessonPlan = plan_lesson(touch(tmp_path / RECORDING_NAME), s, False)
    assert plan.resume_from_transcript is False


def test_27_transcript_of_another_recording(tmp_path: Path) -> None:
    s: Settings = make_settings(tmp_path)
    lesson_dir_with(s, "other.mkv", False)
    info: pytest.ExceptionInfo[LessonExistsError]
    with pytest.raises(LessonExistsError) as info:
        plan_lesson(touch(tmp_path / RECORDING_NAME), s, False)
    text: str = str(info.value)
    assert "2026-11-16" in text
    assert "other.mkv" in text
    assert RECORDING_NAME in text
    assert "--force" in text


def test_27_transcript_and_summary_exist(tmp_path: Path) -> None:
    s: Settings = make_settings(tmp_path)
    lesson_dir_with(s, RECORDING_NAME, True)
    info: pytest.ExceptionInfo[LessonExistsError]
    with pytest.raises(LessonExistsError) as info:
        plan_lesson(touch(tmp_path / RECORDING_NAME), s, False)
    assert "--force" in str(info.value)


def test_27_only_transcript_of_same_recording_resumes(tmp_path: Path) -> None:
    s: Settings = make_settings(tmp_path)
    lesson_dir_with(s, RECORDING_NAME, False)
    plan: LessonPlan = plan_lesson(touch(tmp_path / RECORDING_NAME), s, False)
    assert plan.resume_from_transcript is True


def test_27_resume_does_not_need_the_recording_file(tmp_path: Path) -> None:
    s: Settings = make_settings(tmp_path)
    lesson_dir_with(s, RECORDING_NAME, False)
    plan: LessonPlan = plan_lesson(tmp_path / RECORDING_NAME, s, False)
    assert plan.resume_from_transcript is True


@pytest.mark.parametrize(
    ("transcript_of", "summary"),
    [(None, False), ("other.mkv", False), (RECORDING_NAME, True), (RECORDING_NAME, False)],
)
def test_27_force_never_raises_and_never_resumes(
    tmp_path: Path, transcript_of: str | None, summary: bool
) -> None:
    s: Settings = make_settings(tmp_path)
    lesson_dir_with(s, transcript_of, summary)
    plan: LessonPlan = plan_lesson(touch(tmp_path / RECORDING_NAME), s, True)
    assert plan.resume_from_transcript is False
