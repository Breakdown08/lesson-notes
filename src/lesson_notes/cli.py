import argparse
import logging
import os
import sys
from pathlib import Path

from lesson_notes.audio import TrackNotFoundError
from lesson_notes.config import ConfigError, Settings, load_settings
from lesson_notes.graph import PipelineResult, run_pipeline
from lesson_notes.model import Segment
from lesson_notes.plan import (
    LessonExistsError,
    LessonPlan,
    RecordingNotFoundError,
    pick_latest_recording,
    plan_lesson,
)
from lesson_notes.summarize import SummaryError, make_llm
from lesson_notes.transcribe import Transcriber, WhisperTranscriber

__all__: list[str] = ["WhisperTranscriber", "main", "make_llm"]


class _ResumeTranscriber:
    """Stands in when resuming from a transcript: no recognition model is created or loaded."""

    def transcribe(self, audio: Path) -> list[Segment]:
        raise RuntimeError("recognition is not expected when resuming from a transcript")


def _parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="lesson-notes")
    sub: argparse._SubParsersAction[argparse.ArgumentParser] = parser.add_subparsers(
        dest="command", required=True
    )
    run: argparse.ArgumentParser = sub.add_parser("run", help="transcribe and summarize a lesson recording")
    run.add_argument("recording", nargs="?", type=Path, help="recording file (default: the newest one)")
    run.add_argument("--force", action="store_true", help="overwrite an existing lesson folder")
    run.add_argument("--env-file", type=Path, default=Path(".env"), help="env file (default: ./.env)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args: argparse.Namespace = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        settings: Settings = load_settings(args.env_file, os.environ)
        recording: Path
        if args.recording is not None:
            recording = args.recording
            if not recording.is_file():
                raise RecordingNotFoundError(f"Recording does not exist: {recording}")
        else:
            recording = pick_latest_recording(settings.recordings_dir)
        plan: LessonPlan = plan_lesson(recording, settings, args.force)
        if plan.resume_from_transcript:
            print("Transcript exists, resuming from summary", file=sys.stderr)
        transcriber: Transcriber = (
            _ResumeTranscriber()
            if plan.resume_from_transcript
            else WhisperTranscriber(settings.whisper_model, settings.whisper_compute_type)
        )
        result: PipelineResult = run_pipeline(plan, settings, transcriber, make_llm(settings))
    except (
        ConfigError,
        RecordingNotFoundError,
        LessonExistsError,
        TrackNotFoundError,
        SummaryError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result.transcript_path)
    print(result.summary_path)
    print(result.summary_copy_path)
    return 0
