from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import SecretStr

from lesson_notes.config import ConfigError, Settings


class SummaryError(Exception): ...


_SYSTEM_PROMPT: str = """\
You summarize a foreign-language lesson for the student who took it. The input is a machine-made \
transcript of a lesson between a student and a teacher; the speech is a mix of Russian and French. \
Write the summary in {language}.

Format: Markdown, no first-level heading. Use these second-level sections:
## What we covered
## New words
Each as "word in the lesson language - translation".
## Student's mistakes
Each as "what the student said -> the correct form".
## Topics and grammar
## Homework
Include this section only if homework was actually mentioned.

Keep it short: about one page. Do not invent anything that is not in the transcript. The transcript is \
produced by speech recognition and may contain recognition errors: mark doubtful places as uncertain \
instead of guessing.
"""


def summary_messages(transcript_md: str, language: str) -> list[BaseMessage]:
    """[SystemMessage(prompt), HumanMessage(transcript_md)]."""
    return [SystemMessage(_SYSTEM_PROMPT.format(language=language)), HumanMessage(transcript_md)]


def summarize(llm: BaseChatModel, transcript_md: str, language: str) -> str:
    """Exactly one llm.invoke(summary_messages(...)); returns response.text stripped."""
    try:
        response: BaseMessage = llm.invoke(summary_messages(transcript_md, language))
    except Exception as exc:
        raise SummaryError(f"{type(exc).__name__}: {exc}") from exc
    text: str = response.text.strip()
    if not text:
        raise SummaryError("The model returned an empty summary")
    return text


def make_llm(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "xai":
        if settings.xai_api_key is None:
            raise ConfigError("XAI_API_KEY is required for the xai provider")
        from langchain_xai import ChatXAI

        return ChatXAI(model=settings.xai_model, api_key=SecretStr(settings.xai_api_key))
    if settings.anthropic_api_key is None:
        raise ConfigError("ANTHROPIC_API_KEY is required for the anthropic provider")
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=settings.anthropic_model, api_key=SecretStr(settings.anthropic_api_key))  # type: ignore[call-arg]  # mypy pydantic plugin ignores the `model` alias
