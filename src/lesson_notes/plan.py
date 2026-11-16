import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Final

from lesson_notes.config import ConfigError, Settings
from lesson_notes.render import read_recording_name

RECORDING_SUFFIXES: Final = frozenset({".mkv", ".mp4", ".mov"})
_DATE_RE: Final[re.Pattern[str]] = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[ _]|$)")


class RecordingNotFoundError(Exception): ...


class LessonExistsError(Exception): ...


def pick_latest_recording(directory: Path) -> Path:
    """Newest by st_mtime among regular files in `directory` (not recursive) with a suffix
    in RECORDING_SUFFIXES (case-insensitive). None → RecordingNotFoundError."""
    if not directory.is_dir():
        raise RecordingNotFoundError(f"Recordings directory does not exist: {directory}")
    candidates: list[Path] = [
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in RECORDING_SUFFIXES
    ]
    if not candidates:
        raise RecordingNotFoundError(f"No recordings (.mkv, .mp4, .mov) in {directory}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def lesson_date(recording: Path) -> date:
    """Date from an OBS default name, otherwise the local date of the file's st_mtime."""
    match: re.Match[str] | None = _DATE_RE.match(recording.stem)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass
    return datetime.fromtimestamp(recording.stat().st_mtime).date()


@dataclass(frozen=True)
class LessonPlan:
    recording: Path
    lesson_date: date
    lesson_dir: Path
    summary_copy_path: Path
    resume_from_transcript: bool


def plan_lesson(recording: Path, settings: Settings, force: bool) -> LessonPlan:
    directory: Path
    for directory in (settings.output_dir, settings.summary_copy_dir):
        if not directory.is_dir():
            raise ConfigError(f"Directory does not exist or is not a directory: {directory}")
    day: date = lesson_date(recording)
    stamp: str = day.isoformat()
    lesson_dir: Path = settings.output_dir / stamp
    copy_path: Path = settings.summary_copy_dir / f"{stamp}.md"
    resume: bool = False
    transcript: Path = lesson_dir / "transcript.md"
    if not force and transcript.is_file():
        existing: str | None = read_recording_name(transcript.read_text(encoding="utf-8"))
        if existing != recording.name:
            raise LessonExistsError(
                f"{lesson_dir} already holds a transcript of another recording ({existing}), "
                f"not {recording.name}; use --force to overwrite"
            )
        if (lesson_dir / "summary.md").exists():
            raise LessonExistsError(
                f"{lesson_dir} already has a transcript and a summary for {recording.name}; "
                "use --force to overwrite"
            )
        resume = True
    return LessonPlan(recording, day, lesson_dir, copy_path, resume)
