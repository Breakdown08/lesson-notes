from pathlib import Path

import pytest

from lesson_notes.config import ConfigError, Settings, load_settings

ROOT: Path = Path(__file__).resolve().parents[1]
ALL_VARS: list[str] = [
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


def base_env(**extra: str) -> dict[str, str]:
    env: dict[str, str] = {
        "LESSON_NOTES_RECORDINGS_DIR": "/data/rec",
        "LESSON_NOTES_OUTPUT_DIR": "/data/out",
        "LESSON_NOTES_SUMMARY_COPY_DIR": "/data/notes",
        "XAI_API_KEY": "xk",
    }
    env.update(extra)
    return env


def config_error(env: dict[str, str]) -> str:
    info: pytest.ExceptionInfo[ConfigError]
    with pytest.raises(ConfigError) as info:
        Settings.from_env(env)
    text: str = str(info.value)
    return text


def test_01_defaults() -> None:
    s: Settings = Settings.from_env(base_env())
    assert s.llm_provider == "xai"
    assert s.xai_api_key == "xk"
    assert s.anthropic_api_key is None
    assert s.anthropic_model == "claude-haiku-5-5"
    assert s.xai_model == "grok-4.7"
    assert (s.student_track, s.teacher_track) == (1, 2)
    assert s.whisper_model == "large-v3-turbo"
    assert s.whisper_compute_type == "int8"
    assert s.summary_language == "Russian"
    assert s.recordings_dir == Path("/data/rec")


def test_02_missing_variables_all_named_in_one_error() -> None:
    text: str = config_error({"LESSON_NOTES_OUTPUT_DIR": "~/out"})
    assert "LESSON_NOTES_RECORDINGS_DIR" in text
    assert "LESSON_NOTES_SUMMARY_COPY_DIR" in text
    assert "XAI_API_KEY" in text
    assert "ANTHROPIC_API_KEY" not in text


def test_02_empty_and_blank_values_count_as_missing() -> None:
    env: dict[str, str] = base_env(LESSON_NOTES_OUTPUT_DIR="", XAI_API_KEY="   ")
    text: str = config_error(env)
    assert "LESSON_NOTES_OUTPUT_DIR" in text
    assert "XAI_API_KEY" in text
    assert "LESSON_NOTES_RECORDINGS_DIR" not in text


def test_03_anthropic_provider_needs_only_its_key() -> None:
    env: dict[str, str] = base_env(LESSON_NOTES_LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY="k")
    del env["XAI_API_KEY"]
    s: Settings = Settings.from_env(env)
    assert s.llm_provider == "anthropic"
    assert s.anthropic_api_key == "k"
    assert s.xai_api_key is None


def test_03_anthropic_provider_without_its_key_fails() -> None:
    env: dict[str, str] = base_env(LESSON_NOTES_LLM_PROVIDER="anthropic")
    del env["XAI_API_KEY"]
    text: str = config_error(env)
    assert "ANTHROPIC_API_KEY" in text
    assert "XAI_API_KEY" not in text


def test_04_provider_is_stripped_and_lowercased() -> None:
    s: Settings = Settings.from_env(base_env(LESSON_NOTES_LLM_PROVIDER=" Anthropic ", ANTHROPIC_API_KEY="k"))
    assert s.llm_provider == "anthropic"


def test_04_unknown_provider_names_variable_value_and_allowed() -> None:
    text: str = config_error(base_env(LESSON_NOTES_LLM_PROVIDER="openai"))
    assert "LESSON_NOTES_LLM_PROVIDER" in text
    assert "openai" in text
    assert "xai" in text
    assert "anthropic" in text


def test_04_unknown_provider_is_checked_before_missing_keys() -> None:
    text: str = config_error({"LESSON_NOTES_LLM_PROVIDER": "openai"})
    assert "openai" in text
    assert "LESSON_NOTES_RECORDINGS_DIR" not in text


def test_05_unselected_provider_key_is_still_read() -> None:
    s: Settings = Settings.from_env(base_env(ANTHROPIC_API_KEY="k"))
    assert s.llm_provider == "xai"
    assert s.anthropic_api_key == "k"


def test_06_tilde_is_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/someone")
    s: Settings = Settings.from_env(
        base_env(LESSON_NOTES_RECORDINGS_DIR="~/rec", LESSON_NOTES_OUTPUT_DIR="~/out")
    )
    assert s.recordings_dir == Path("/home/someone/rec")
    assert s.output_dir == Path("/home/someone/out")


@pytest.mark.parametrize("value", ["0", "abc", "-1", "1.5"])
def test_07_bad_student_track(value: str) -> None:
    assert "LESSON_NOTES_STUDENT_TRACK" in config_error(base_env(LESSON_NOTES_STUDENT_TRACK=value))


def test_07_bad_teacher_track() -> None:
    assert "LESSON_NOTES_TEACHER_TRACK" in config_error(base_env(LESSON_NOTES_TEACHER_TRACK="0"))


def test_07_equal_tracks_rejected() -> None:
    config_error(base_env(LESSON_NOTES_STUDENT_TRACK="2", LESSON_NOTES_TEACHER_TRACK="2"))


def test_07_custom_tracks_accepted() -> None:
    s: Settings = Settings.from_env(base_env(LESSON_NOTES_STUDENT_TRACK="2", LESSON_NOTES_TEACHER_TRACK="1"))
    assert (s.student_track, s.teacher_track) == (2, 1)


def test_08_env_file_is_read_and_environ_overrides(tmp_path: Path) -> None:
    env_file: Path = tmp_path / ".env"
    env_file.write_text(
        "LESSON_NOTES_RECORDINGS_DIR=/f/rec\n"
        "LESSON_NOTES_OUTPUT_DIR=/f/out\n"
        "LESSON_NOTES_SUMMARY_COPY_DIR=/f/notes\n"
        "XAI_API_KEY=from-file\n"
        "XAI_MODEL=file-model\n"
    )
    s: Settings = load_settings(env_file, {"XAI_MODEL": "env-model"})
    assert s.recordings_dir == Path("/f/rec")
    assert s.xai_api_key == "from-file"
    assert s.xai_model == "env-model"


def test_08_missing_env_file_is_not_an_error(tmp_path: Path) -> None:
    s: Settings = load_settings(tmp_path / "absent.env", base_env())
    assert s.xai_api_key == "xk"


def test_08_no_env_file_reads_environ_only() -> None:
    s: Settings = load_settings(None, base_env())
    assert s.output_dir == Path("/data/out")


def test_09_env_example_lists_all_variables() -> None:
    lines: list[str] = (ROOT / ".env.example").read_text().splitlines()
    name: str
    for name in ALL_VARS:
        assert any(line.startswith(f"{name}=") for line in lines), name
    assert "LESSON_NOTES_LLM_PROVIDER=xai" in lines
    assert "XAI_API_KEY=" in [line.strip() for line in lines]
    assert "ANTHROPIC_API_KEY=" in [line.strip() for line in lines]


def test_09_gitignore_has_env() -> None:
    lines: list[str] = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in [line.strip() for line in lines]
