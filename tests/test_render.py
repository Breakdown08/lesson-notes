from datetime import date

import pytest

from lesson_notes.model import Segment, Utterance
from lesson_notes.render import (
    format_timestamp,
    merge_tracks,
    read_recording_name,
    render_summary,
    render_transcript,
)

D: date = date(2026, 11, 16)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, "00:00:00"), (65.9, "00:01:05"), (3725, "01:02:05"), (360000, "100:00:00")],
)
def test_13_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


def test_14_merge_labels_and_sorts_by_start() -> None:
    student: list[Segment] = [Segment(9.0, 11.0, "b"), Segment(30.0, 31.0, "d")]
    teacher: list[Segment] = [Segment(5.0, 7.0, "a"), Segment(20.0, 22.0, "c")]
    merged: list[Utterance] = merge_tracks(student, teacher)
    assert merged == [
        Utterance("teacher", 5.0, 7.0, "a"),
        Utterance("student", 9.0, 11.0, "b"),
        Utterance("teacher", 20.0, 22.0, "c"),
        Utterance("student", 30.0, 31.0, "d"),
    ]


def test_14_student_goes_first_on_equal_start() -> None:
    merged: list[Utterance] = merge_tracks([Segment(5.0, 6.0, "s")], [Segment(5.0, 6.0, "t")])
    assert [u.speaker for u in merged] == ["student", "teacher"]


def test_14_empty_and_blank_text_dropped() -> None:
    student: list[Segment] = [Segment(1.0, 2.0, ""), Segment(3.0, 4.0, "ok")]
    teacher: list[Segment] = [Segment(1.0, 2.0, "   "), Segment(2.0, 3.0, "\n\t")]
    merged: list[Utterance] = merge_tracks(student, teacher)
    assert merged == [Utterance("student", 3.0, 4.0, "ok")]


def test_15_render_transcript_exact() -> None:
    utterances: list[Utterance] = [
        Utterance("teacher", 5.0, 7.0, "Bonjour, ça va ?"),
        Utterance("student", 9.0, 11.0, "Ça va bien, merci."),
    ]
    text: str = render_transcript(D, "2026-11-16 18-00-03.mkv", utterances)
    assert text == (
        "# Lesson 2026-11-16 — transcript\n"
        "\n"
        "Recording: 2026-11-16 18-00-03.mkv\n"
        "\n"
        "- [00:00:05] **Teacher:** Bonjour, ça va ?\n"
        "- [00:00:09] **Student:** Ça va bien, merci.\n"
    )


def test_15_render_transcript_without_speech() -> None:
    text: str = render_transcript(D, "r.mkv", [])
    assert text == ("# Lesson 2026-11-16 — transcript\n\nRecording: r.mkv\n\n_No speech recognized._\n")


def test_16_read_recording_name_roundtrip() -> None:
    utterances: list[Utterance] = [Utterance("teacher", 5.0, 7.0, "Bonjour")]
    text: str = render_transcript(D, "2026-11-16 18-00-03.mkv", utterances)
    assert read_recording_name(text) == "2026-11-16 18-00-03.mkv"
    assert read_recording_name(render_transcript(D, "x y.mp4", [])) == "x y.mp4"


def test_16_read_recording_name_absent() -> None:
    assert read_recording_name("# Lesson\n\nno such line\n") is None


def test_16_read_recording_name_takes_first_match() -> None:
    assert read_recording_name("Recording: a.mkv\nRecording: b.mkv\n") == "a.mkv"


def test_17_render_summary() -> None:
    assert render_summary(D, "  body\n\n") == "# Lesson 2026-11-16 — summary\n\nbody\n"
