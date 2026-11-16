from pathlib import Path

import pytest
from helpers import RecordingLLM, make_llm_fake
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_xai import ChatXAI

from lesson_notes.config import ConfigError, LlmProvider, Settings
from lesson_notes.summarize import SummaryError, make_llm, summarize, summary_messages

TRANSCRIPT: str = (
    "# Lesson 2026-11-16 — transcript\n\nRecording: r.mkv\n\n- [00:00:05] **Teacher:** Bonjour\n"
)


def test_18_summary_messages() -> None:
    msgs: list[object] = list(summary_messages(TRANSCRIPT, "German"))
    assert len(msgs) == 2
    system: object = msgs[0]
    human: object = msgs[1]
    assert isinstance(system, SystemMessage)
    assert isinstance(human, HumanMessage)
    assert human.content == TRANSCRIPT
    assert "German" in str(system.content)


def test_19_summarize_calls_once_and_strips() -> None:
    llm: RecordingLLM = make_llm_fake(answer="  Result text.\n\n")
    assert summarize(llm, TRANSCRIPT, "Russian") == "Result text."
    assert len(llm.calls) == 1
    assert llm.calls[0][-1].content == TRANSCRIPT


@pytest.mark.parametrize("answer", ["", "   \n\t"])
def test_19_empty_answer_is_error(answer: str) -> None:
    with pytest.raises(SummaryError):
        summarize(make_llm_fake(answer=answer), TRANSCRIPT, "Russian")


def test_19_llm_exception_is_wrapped() -> None:
    boom: RuntimeError = RuntimeError("bad key")
    info: pytest.ExceptionInfo[SummaryError]
    with pytest.raises(SummaryError) as info:
        summarize(make_llm_fake(error=boom), TRANSCRIPT, "Russian")
    text: str = str(info.value)
    assert "RuntimeError" in text
    assert "bad key" in text
    assert info.value.__cause__ is boom


def settings(
    provider: LlmProvider,
    xai_key: str | None = None,
    anthropic_key: str | None = None,
    xai_model: str = "grok-4.7",
    anthropic_model: str = "claude-haiku-5-5",
) -> Settings:
    return Settings(
        recordings_dir=Path("/r"),
        output_dir=Path("/o"),
        summary_copy_dir=Path("/c"),
        llm_provider=provider,
        xai_api_key=xai_key,
        xai_model=xai_model,
        anthropic_api_key=anthropic_key,
        anthropic_model=anthropic_model,
    )


def test_20_make_llm_xai() -> None:
    s: Settings = settings("xai", xai_key="xk", xai_model="grok-test")
    llm: object = make_llm(s)
    assert isinstance(llm, ChatXAI)
    assert llm.model_name == "grok-test"


def test_21_make_llm_anthropic() -> None:
    s: Settings = settings("anthropic", anthropic_key="ak", anthropic_model="claude-test")
    llm: object = make_llm(s)
    assert isinstance(llm, ChatAnthropic)
    assert llm.model == "claude-test"
    assert llm.anthropic_api_key.get_secret_value() == "ak"


@pytest.mark.parametrize(("provider", "name"), [("xai", "XAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")])
def test_22_missing_key_in_hand_built_settings(
    monkeypatch: pytest.MonkeyPatch, provider: LlmProvider, name: str
) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s: Settings = settings(provider)
    info: pytest.ExceptionInfo[ConfigError]
    with pytest.raises(ConfigError) as info:
        make_llm(s)
    assert name in str(info.value)
