from datetime import date
from pathlib import Path

import pytest
from helpers import (
    LESSON_DATE,
    RECORDING_NAME,
    RMS_THRESHOLD,
    STUDENT_SEGMENTS,
    TEACHER_SEGMENTS,
    ExplodingTranscriber,
    FakeTranscriber,
    RecordingLLM,
    make_llm_fake,
    make_recording,
    make_settings,
)

from lesson_notes.config import Settings
from lesson_notes.graph import PipelineResult, build_graph, run_pipeline
from lesson_notes.plan import LessonPlan
from lesson_notes.summarize import SummaryError

NODES: set[str] = {
    "transcribe",
    "write_transcript",
    "load_transcript",
    "summarize",
    "write_summary",
    "copy_summary",
}
EXPECTED_TRANSCRIPT: str = (
    "# Lesson 2026-11-16 — transcript\n"
    "\n"
    "Recording: 2026-11-16 18-00-03.mkv\n"
    "\n"
    "- [00:00:05] **Teacher:** Bonjour, ça va ?\n"
    "- [00:00:09] **Student:** Ça va bien, merci.\n"
    "- [00:00:20] **Teacher:** Très bien.\n"
)
EXPECTED_SUMMARY: str = "# Lesson 2026-11-16 — summary\n\nSummary body.\n"


def make_plan(settings: Settings, recording: Path, resume: bool = False) -> LessonPlan:
    return LessonPlan(
        recording=recording,
        lesson_date=LESSON_DATE,
        lesson_dir=settings.output_dir / "2026-11-16",
        summary_copy_path=settings.summary_copy_dir / "2026-11-16.md",
        resume_from_transcript=resume,
    )


def fake_transcriber() -> FakeTranscriber:
    return FakeTranscriber([STUDENT_SEGMENTS, TEACHER_SEGMENTS])


def test_28_graph_nodes(tmp_path: Path) -> None:
    s: Settings = make_settings(tmp_path)
    plan: LessonPlan = make_plan(s, tmp_path / RECORDING_NAME)
    graph: object = build_graph(plan, s, fake_transcriber(), make_llm_fake())
    names: set[str] = set(graph.get_graph().nodes)  # type: ignore[attr-defined]
    assert names - {"__start__", "__end__"} == NODES


def test_29_end_to_end(tmp_path: Path) -> None:
    s: Settings = make_settings(tmp_path)
    rec: Path = make_recording(tmp_path / RECORDING_NAME)
    plan: LessonPlan = make_plan(s, rec)
    tr: FakeTranscriber = fake_transcriber()
    llm: RecordingLLM = make_llm_fake(answer="  Summary body.\n")
    result: PipelineResult = run_pipeline(plan, s, tr, llm)

    assert len(tr.paths) == 2
    assert tr.rms[0] > RMS_THRESHOLD  # student track, loud
    assert tr.rms[1] < RMS_THRESHOLD  # teacher track
    assert result.transcript_path == plan.lesson_dir / "transcript.md"
    assert result.summary_path == plan.lesson_dir / "summary.md"
    assert result.summary_copy_path == plan.summary_copy_path
    assert result.resumed is False
    assert result.transcript_path.read_bytes() == EXPECTED_TRANSCRIPT.encode("utf-8")
    assert result.summary_path.read_bytes() == EXPECTED_SUMMARY.encode("utf-8")
    assert result.summary_copy_path.read_bytes() == result.summary_path.read_bytes()
    assert len(llm.calls) == 1
    assert llm.calls[0][-1].content == EXPECTED_TRANSCRIPT


def test_29_existing_copy_is_overwritten(tmp_path: Path) -> None:
    s: Settings = make_settings(tmp_path)
    rec: Path = make_recording(tmp_path / RECORDING_NAME)
    plan: LessonPlan = make_plan(s, rec)
    plan.summary_copy_path.write_text("old copy\n")
    run_pipeline(plan, s, fake_transcriber(), make_llm_fake())
    assert plan.summary_copy_path.read_text() == EXPECTED_SUMMARY


def test_30_swapped_tracks(tmp_path: Path) -> None:
    s: Settings = make_settings(tmp_path, student_track=2, teacher_track=1)
    rec: Path = make_recording(tmp_path / RECORDING_NAME)
    tr: FakeTranscriber = fake_transcriber()
    run_pipeline(make_plan(s, rec), s, tr, make_llm_fake())
    assert len(tr.paths) == 2
    assert tr.rms[0] < RMS_THRESHOLD  # track 2 (quiet) went first, as the student
    assert tr.rms[1] > RMS_THRESHOLD


def test_31_temporary_wavs_are_removed(tmp_path: Path) -> None:
    s: Settings = make_settings(tmp_path)
    rec: Path = make_recording(tmp_path / RECORDING_NAME)
    tr: FakeTranscriber = fake_transcriber()
    run_pipeline(make_plan(s, rec), s, tr, make_llm_fake())
    assert len(tr.paths) == 2
    path: Path
    for path in tr.paths:
        assert not path.exists()


def test_32_resume_from_transcript(tmp_path: Path) -> None:
    s: Settings = make_settings(tmp_path)
    plan: LessonPlan = make_plan(s, tmp_path / "gone.mkv", resume=True)
    plan.lesson_dir.mkdir()
    text: str = (
        "# Lesson 2026-11-16 — transcript\n\nRecording: gone.mkv\n\n- [00:00:05] **Teacher:** Salut — ça\n"
    )
    (plan.lesson_dir / "transcript.md").write_text(text, encoding="utf-8")
    llm: RecordingLLM = make_llm_fake()
    result: PipelineResult = run_pipeline(plan, s, ExplodingTranscriber(), llm)
    assert result.resumed is True
    assert len(llm.calls) == 1
    assert llm.calls[0][-1].content == text
    assert (plan.lesson_dir / "summary.md").read_text() == EXPECTED_SUMMARY
    assert plan.summary_copy_path.read_text() == EXPECTED_SUMMARY
    assert (plan.lesson_dir / "transcript.md").read_text(encoding="utf-8") == text


def test_33_llm_failure_keeps_transcript(tmp_path: Path) -> None:
    s: Settings = make_settings(tmp_path)
    rec: Path = make_recording(tmp_path / RECORDING_NAME)
    plan: LessonPlan = make_plan(s, rec)
    llm: RecordingLLM = make_llm_fake(error=RuntimeError("quota"))
    with pytest.raises(SummaryError):
        run_pipeline(plan, s, fake_transcriber(), llm)
    assert (plan.lesson_dir / "transcript.md").read_text(encoding="utf-8") == EXPECTED_TRANSCRIPT
    assert not (plan.lesson_dir / "summary.md").exists()
    assert not plan.summary_copy_path.exists()


def test_date_is_taken_from_plan(tmp_path: Path) -> None:
    s: Settings = make_settings(tmp_path)
    rec: Path = make_recording(tmp_path / RECORDING_NAME)
    plan: LessonPlan = LessonPlan(
        recording=rec,
        lesson_date=date(2030, 1, 2),
        lesson_dir=s.output_dir / "2030-01-02",
        summary_copy_path=s.summary_copy_dir / "2030-01-02.md",
        resume_from_transcript=False,
    )
    run_pipeline(plan, s, fake_transcriber(), make_llm_fake())
    assert (plan.lesson_dir / "summary.md").read_text().startswith("# Lesson 2030-01-02 — summary\n")
    assert (
        (plan.lesson_dir / "transcript.md")
        .read_text(encoding="utf-8")
        .startswith("# Lesson 2030-01-02 — transcript\n")
    )
