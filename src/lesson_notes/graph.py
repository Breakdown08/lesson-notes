import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from lesson_notes.audio import extract_track
from lesson_notes.config import Settings
from lesson_notes.model import Segment, Utterance
from lesson_notes.plan import LessonPlan
from lesson_notes.render import merge_tracks, render_summary, render_transcript
from lesson_notes.summarize import summarize
from lesson_notes.transcribe import Transcriber

log: logging.Logger = logging.getLogger(__name__)


class LessonState(TypedDict, total=False):
    transcript_md: str
    summary_md: str


# ty bug: any class-based TypedDict is rejected as StateT (it lacks __optional_keys__ as a protocol member)
type CompiledLessonGraph = CompiledStateGraph[LessonState, None, LessonState, LessonState]  # ty: ignore[invalid-type-arguments]


@dataclass(frozen=True)
class PipelineResult:
    transcript_path: Path
    summary_path: Path
    summary_copy_path: Path
    resumed: bool


def build_graph(
    plan: LessonPlan, settings: Settings, transcriber: Transcriber, llm: BaseChatModel
) -> CompiledLessonGraph:
    transcript_path: Path = plan.lesson_dir / "transcript.md"

    def transcribe(state: LessonState) -> LessonState:
        segments: list[list[Segment]] = []
        tmp: str
        track: int
        label: str
        with tempfile.TemporaryDirectory() as tmp:
            for track, label in ((settings.student_track, "student"), (settings.teacher_track, "teacher")):
                log.info("Extracting track %d…", track)
                wav: Path = extract_track(plan.recording, track, Path(tmp) / f"{label}.wav")
                log.info("Transcribing %s…", label)
                segments.append(transcriber.transcribe(wav))
        utterances: list[Utterance] = merge_tracks(segments[0], segments[1])
        return {"transcript_md": render_transcript(plan.lesson_date, plan.recording.name, utterances)}

    def write_transcript(state: LessonState) -> LessonState:
        plan.lesson_dir.mkdir(exist_ok=True)
        transcript_path.write_text(state["transcript_md"], encoding="utf-8")
        return {}

    def load_transcript(state: LessonState) -> LessonState:
        return {"transcript_md": transcript_path.read_text(encoding="utf-8")}

    def summarize_node(state: LessonState) -> LessonState:
        log.info("Summarizing…")
        body: str = summarize(llm, state["transcript_md"], settings.summary_language)
        return {"summary_md": render_summary(plan.lesson_date, body)}

    def write_summary(state: LessonState) -> LessonState:
        (plan.lesson_dir / "summary.md").write_text(state["summary_md"], encoding="utf-8")
        return {}

    def copy_summary(state: LessonState) -> LessonState:
        plan.summary_copy_path.write_text(state["summary_md"], encoding="utf-8")
        return {}

    def route(state: LessonState) -> str:
        return "load_transcript" if plan.resume_from_transcript else "transcribe"

    # ty bug, same as above: langgraph's own documented usage with a TypedDict state
    builder: StateGraph[LessonState, None, LessonState, LessonState] = StateGraph(  # ty: ignore[invalid-type-arguments]
        LessonState  # ty: ignore[invalid-argument-type]
    )
    builder.add_node("transcribe", transcribe)
    builder.add_node("write_transcript", write_transcript)
    builder.add_node("load_transcript", load_transcript)
    builder.add_node("summarize", summarize_node)
    builder.add_node("write_summary", write_summary)
    builder.add_node("copy_summary", copy_summary)
    builder.add_conditional_edges(START, route, ["transcribe", "load_transcript"])
    builder.add_edge("transcribe", "write_transcript")
    builder.add_edge("write_transcript", "summarize")
    builder.add_edge("load_transcript", "summarize")
    builder.add_edge("summarize", "write_summary")
    builder.add_edge("write_summary", "copy_summary")
    builder.add_edge("copy_summary", END)
    return builder.compile()


def run_pipeline(
    plan: LessonPlan, settings: Settings, transcriber: Transcriber, llm: BaseChatModel
) -> PipelineResult:
    build_graph(plan, settings, transcriber, llm).invoke({})
    return PipelineResult(
        transcript_path=plan.lesson_dir / "transcript.md",
        summary_path=plan.lesson_dir / "summary.md",
        summary_copy_path=plan.summary_copy_path,
        resumed=plan.resume_from_transcript,
    )
