from dataclasses import dataclass
from typing import Literal

type Speaker = Literal["student", "teacher"]


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Utterance:
    speaker: Speaker
    start: float
    end: float
    text: str
