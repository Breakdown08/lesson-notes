from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

from dotenv import dotenv_values


class ConfigError(Exception): ...


type LlmProvider = Literal["xai", "anthropic"]

PROVIDERS: Final[tuple[str, ...]] = ("xai", "anthropic")
KEY_VARS: Final[dict[str, str]] = {"xai": "XAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def _get(environ: Mapping[str, str], name: str) -> str | None:
    value: str | None = environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _track(environ: Mapping[str, str], name: str, default: int) -> int:
    raw: str | None = _get(environ, name)
    if raw is None:
        return default
    try:
        number: int = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer >= 1, got {raw!r}") from None
    if number < 1:
        raise ConfigError(f"{name} must be an integer >= 1, got {raw!r}")
    return number


@dataclass(frozen=True)
class Settings:
    recordings_dir: Path
    output_dir: Path
    summary_copy_dir: Path
    llm_provider: LlmProvider = "xai"
    xai_api_key: str | None = None
    xai_model: str = "grok-4.7"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-5-5"
    student_track: int = 1
    teacher_track: int = 2
    whisper_model: str = "large-v3-turbo"
    whisper_compute_type: str = "int8"
    summary_language: str = "Russian"

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> Settings:
        provider: str = (_get(environ, "LESSON_NOTES_LLM_PROVIDER") or "xai").lower()
        if provider not in PROVIDERS:
            raise ConfigError(
                f"LESSON_NOTES_LLM_PROVIDER: got {provider!r}, expected one of: "
                + ", ".join(repr(p) for p in PROVIDERS)
            )
        required: list[str] = [
            "LESSON_NOTES_RECORDINGS_DIR",
            "LESSON_NOTES_OUTPUT_DIR",
            "LESSON_NOTES_SUMMARY_COPY_DIR",
            KEY_VARS[provider],
        ]
        missing: list[str] = [name for name in required if _get(environ, name) is None]
        if missing:
            raise ConfigError("Missing required environment variables: " + ", ".join(missing))
        student: int = _track(environ, "LESSON_NOTES_STUDENT_TRACK", 1)
        teacher: int = _track(environ, "LESSON_NOTES_TEACHER_TRACK", 2)
        if student == teacher:
            raise ConfigError(
                f"LESSON_NOTES_STUDENT_TRACK and LESSON_NOTES_TEACHER_TRACK must differ, both are {student}"
            )

        def path(name: str) -> Path:
            return Path(cast(str, _get(environ, name))).expanduser()

        return cls(
            recordings_dir=path("LESSON_NOTES_RECORDINGS_DIR"),
            output_dir=path("LESSON_NOTES_OUTPUT_DIR"),
            summary_copy_dir=path("LESSON_NOTES_SUMMARY_COPY_DIR"),
            llm_provider=cast(LlmProvider, provider),
            xai_api_key=_get(environ, "XAI_API_KEY"),
            xai_model=_get(environ, "XAI_MODEL") or "grok-4.7",
            anthropic_api_key=_get(environ, "ANTHROPIC_API_KEY"),
            anthropic_model=_get(environ, "ANTHROPIC_MODEL") or "claude-haiku-5-5",
            student_track=student,
            teacher_track=teacher,
            whisper_model=_get(environ, "LESSON_NOTES_WHISPER_MODEL") or "large-v3-turbo",
            whisper_compute_type=_get(environ, "LESSON_NOTES_WHISPER_COMPUTE_TYPE") or "int8",
            summary_language=_get(environ, "LESSON_NOTES_SUMMARY_LANGUAGE") or "Russian",
        )


def load_settings(env_file: Path | None, environ: Mapping[str, str]) -> Settings:
    """Read env_file with dotenv_values (if given and it exists) and let environ override it."""
    merged: dict[str, str] = {}
    if env_file is not None and env_file.is_file():
        name: str
        value: str | None
        for name, value in dotenv_values(env_file).items():
            if value is not None:
                merged[name] = value
    merged.update(environ)
    return Settings.from_env(merged)
