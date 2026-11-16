import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import pytest
from helpers import (
    RECORDING_NAME,
    STUDENT_SEGMENTS,
    TEACHER_SEGMENTS,
    FakeTranscriber,
    make_llm_fake,
    make_recording,
)

from lesson_notes.cli import main

ENV_NAMES: list[str] = [
    "LESSON_NOTES_RECORDINGS_DIR",
    "LESSON_NOTES_OUTPUT_DIR",
    "LESSON_NOTES_SUMMARY_COPY_DIR",
    "LESSON_NOTES_LLM_PROVIDER",
    "XAI_API_KEY",
    "XAI_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "LESSON_NOTES_STUDENT_TRACK",
    "LESSON_NOTES_TEACHER_TRACK",
    "LESSON_NOTES_WHISPER_MODEL",
    "LESSON_NOTES_WHISPER_COMPUTE_TYPE",
    "LESSON_NOTES_SUMMARY_LANGUAGE",
]
OTHER_TRANSCRIPT: str = (
    "# Lesson 2026-11-16 — transcript\n\nRecording: other.mkv\n\n_No speech recognized._\n"
)
SAME_TRANSCRIPT: str = (
    f"# Lesson 2026-11-16 — transcript\n\nRecording: {RECORDING_NAME}\n\n_No speech recognized._\n"
)


def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    name: str
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    rec: Path = tmp_path / "recordings"
    out: Path = tmp_path / "out"
    notes: Path = tmp_path / "notes"
    d: Path
    for d in (rec, out, notes):
        d.mkdir()
    monkeypatch.setenv("LESSON_NOTES_RECORDINGS_DIR", str(rec))
    monkeypatch.setenv("LESSON_NOTES_OUTPUT_DIR", str(out))
    monkeypatch.setenv("LESSON_NOTES_SUMMARY_COPY_DIR", str(notes))
    monkeypatch.setenv("XAI_API_KEY", "xk")
    monkeypatch.chdir(tmp_path)
    return rec, out, notes


def must_not_be_called(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("model must not be created before planning succeeds")


def env_file_arg(tmp_path: Path) -> list[str]:
    return ["--env-file", str(tmp_path / "absent.env")]


def test_34_missing_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    name: str
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    code: int = main(["run", *env_file_arg(tmp_path)])
    err: str = capsys.readouterr().err
    assert code == 1
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_35_empty_recordings_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_env(monkeypatch, tmp_path)
    code: int = main(["run", *env_file_arg(tmp_path)])
    err: str = capsys.readouterr().err
    assert code == 1
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_35_missing_recordings_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_env(monkeypatch, tmp_path)
    monkeypatch.setenv("LESSON_NOTES_RECORDINGS_DIR", str(tmp_path / "nowhere"))
    code: int = main(["run", *env_file_arg(tmp_path)])
    assert code == 1
    assert capsys.readouterr().err.startswith("error:")


def test_36_folder_taken_by_another_recording(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec: Path
    out: Path
    _unused: Path
    rec, out, _unused = clean_env(monkeypatch, tmp_path)
    recording: Path = rec / RECORDING_NAME
    recording.write_bytes(b"")
    (out / "2026-11-16").mkdir()
    (out / "2026-11-16" / "transcript.md").write_text(OTHER_TRANSCRIPT, encoding="utf-8")
    monkeypatch.setattr("lesson_notes.cli.WhisperTranscriber", must_not_be_called)
    monkeypatch.setattr("lesson_notes.cli.make_llm", must_not_be_called)
    code: int = main(["run", str(recording), *env_file_arg(tmp_path)])
    err: str = capsys.readouterr().err
    assert code == 1
    assert err.startswith("error:")
    assert "--force" in err
    assert "Traceback" not in err


def test_36_config_error_before_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec: Path
    _unused1: Path
    _unused2: Path
    rec, _unused1, _unused2 = clean_env(monkeypatch, tmp_path)
    monkeypatch.setenv("LESSON_NOTES_OUTPUT_DIR", str(tmp_path / "typo"))
    monkeypatch.setattr("lesson_notes.cli.WhisperTranscriber", must_not_be_called)
    monkeypatch.setattr("lesson_notes.cli.make_llm", must_not_be_called)
    code: int = main(["run", str(rec / RECORDING_NAME), *env_file_arg(tmp_path)])
    assert code == 1
    assert capsys.readouterr().err.startswith("error:")


def test_explicit_recording_does_not_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_env(monkeypatch, tmp_path)
    monkeypatch.setattr("lesson_notes.cli.WhisperTranscriber", must_not_be_called)
    monkeypatch.setattr("lesson_notes.cli.make_llm", must_not_be_called)
    code: int = main(["run", str(tmp_path / "ghost.mkv"), *env_file_arg(tmp_path)])
    err: str = capsys.readouterr().err
    assert code == 1
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_37_unknown_option() -> None:
    info: pytest.ExceptionInfo[SystemExit]
    with pytest.raises(SystemExit) as info:
        main(["run", "--bogus"])
    assert info.value.code == 2


def fake_factories(monkeypatch: pytest.MonkeyPatch) -> FakeTranscriber:
    tr: FakeTranscriber = FakeTranscriber([STUDENT_SEGMENTS, TEACHER_SEGMENTS])
    factory: Callable[..., FakeTranscriber] = lambda *a, **k: tr  # noqa: E731
    monkeypatch.setattr("lesson_notes.cli.WhisperTranscriber", factory)
    monkeypatch.setattr("lesson_notes.cli.make_llm", lambda settings: make_llm_fake())
    return tr


def test_success_prints_three_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec: Path
    out: Path
    notes: Path
    rec, out, notes = clean_env(monkeypatch, tmp_path)
    recording: Path = make_recording(rec / RECORDING_NAME)
    fake_factories(monkeypatch)
    code: int = main(["run", *env_file_arg(tmp_path)])  # latest recording picked from recordings dir
    captured: Any = capsys.readouterr()
    assert code == 0
    assert captured.out.splitlines() == [
        str(out / "2026-11-16" / "transcript.md"),
        str(out / "2026-11-16" / "summary.md"),
        str(notes / "2026-11-16.md"),
    ]
    assert "Transcript exists, resuming from summary" not in captured.err
    assert (notes / "2026-11-16.md").exists()
    assert recording.exists()


def test_resume_prints_notice_to_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec: Path
    out: Path
    notes: Path
    rec, out, notes = clean_env(monkeypatch, tmp_path)
    recording: Path = rec / RECORDING_NAME
    recording.write_bytes(b"")
    (out / "2026-11-16").mkdir()
    (out / "2026-11-16" / "transcript.md").write_text(SAME_TRANSCRIPT, encoding="utf-8")
    monkeypatch.setattr("lesson_notes.cli.WhisperTranscriber", must_not_be_called)
    monkeypatch.setattr("lesson_notes.cli.make_llm", lambda settings: make_llm_fake())
    code: int = main(["run", str(recording), *env_file_arg(tmp_path)])
    captured: Any = capsys.readouterr()
    assert code == 0
    assert "Transcript exists, resuming from summary" in captured.err
    assert len(captured.out.splitlines()) == 3
    assert (notes / "2026-11-16.md").exists()


def test_38_importing_transcribe_does_not_load_faster_whisper() -> None:
    code: str = "import sys, lesson_notes.transcribe; sys.exit(1 if 'faster_whisper' in sys.modules else 0)"
    done: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
