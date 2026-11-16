import math
import re
from datetime import date

from lesson_notes.model import Segment, Speaker, Utterance

_RECORDING_RE: re.Pattern[str] = re.compile(r"^Recording: (.+)$", re.MULTILINE)


def merge_tracks(student: list[Segment], teacher: list[Segment]) -> list[Utterance]:
    """Label segments, drop empty text, sort by (start, speaker) — "student" < "teacher"."""
    items: list[Utterance] = []
    speaker: Speaker
    segments: list[Segment]
    segment: Segment
    for speaker, segments in (("student", student), ("teacher", teacher)):
        for segment in segments:
            text: str = segment.text.strip()
            if text:
                items.append(Utterance(speaker, segment.start, segment.end, text))
    return sorted(items, key=lambda u: (u.start, u.speaker))


def format_timestamp(seconds: float) -> str:
    total: int = math.floor(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def render_transcript(lesson_date: date, recording_name: str, utterances: list[Utterance]) -> str:
    head: str = f"# Lesson {lesson_date.isoformat()} — transcript\n\nRecording: {recording_name}\n\n"
    if not utterances:
        return head + "_No speech recognized._\n"
    lines: list[str] = [
        f"- [{format_timestamp(u.start)}] **{u.speaker.capitalize()}:** {u.text}" for u in utterances
    ]
    return head + "\n".join(lines) + "\n"


def read_recording_name(transcript_md: str) -> str | None:
    """Value of the first line matching ^Recording: (.+)$, else None."""
    match: re.Match[str] | None = _RECORDING_RE.search(transcript_md)
    return match.group(1) if match else None


def render_summary(lesson_date: date, body: str) -> str:
    return f"# Lesson {lesson_date.isoformat()} — summary\n\n{body.strip()}\n"
