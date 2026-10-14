"""Wire types for the System One API: questions in, typed answers out."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

State = Union[str, dict[str, Any], list[Any]]

# Instructions, option descriptions, and level descriptions may be plain text or
# JSON structure; structure is rendered compactly into the prompt.
Text = Union[str, dict[str, Any], list[Any]]

# The hosted services' request limits, so a client written against them never
# sees a different answer here. Above the tokenizer's single-token codes a
# Choice is answered in two stages (see Engine).
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
MAX_QUESTIONS = 64

# The hosted SDK sends its default model name on every request. Treating these as
# "whichever model this server loaded" is what lets that client talk to ruling
# unchanged; the response still reports the model that actually answered.
DEFAULT_MODEL_ALIASES = ("jev-latest", "default")


def render_text(value: Text) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def has_text(value: Text | None) -> bool:
    return value is not None and bool(render_text(value).strip())


class Strict(BaseModel):
    """A question refuses fields it does not know, as the hosted services do, so a typo
    such as `instruction` fails loudly instead of asking a different question."""

    model_config = ConfigDict(extra="forbid")


class Choice(Strict):
    """Pick one option from a closed set. Criteria maps option key to an optional description.

    The schema admits one option, as the hosted services' does; `check_askable` is what
    refuses it, naming the question.
    """

    type: Literal["choice"] = "choice"
    instructions: Text | None = None
    criteria: dict[str, Text | None] = Field(min_length=1, max_length=MAX_CHOICE_OPTIONS)


class Score(Strict):
    """Place the state on an ordered scale. Criteria lists the levels from lowest to highest."""

    type: Literal["score"] = "score"
    instructions: Text | None = None
    criteria: list[Text] = Field(min_length=2, max_length=MAX_SCORE_LEVELS)

    @field_validator("criteria")
    @classmethod
    def levels_are_distinct(cls, levels: list[Text]) -> list[Text]:
        rendered = [render_text(level) for level in levels]
        if len(set(rendered)) != len(rendered):
            raise ValueError("score levels must be distinct")
        return levels


class NoulCriteria(Strict):
    """Optional descriptions of what a true and a false answer mean."""

    true: Text | None = None
    false: Text | None = None


class Noul(Strict):
    """Judge whether a statement about the state is true. The statement is the instructions,
    or, without them, what the criteria say a true or a false answer means."""

    type: Literal["noul"] = "noul"
    instructions: Text | None = None
    criteria: NoulCriteria | None = None

    def described(self) -> bool:
        return self.criteria is not None and (has_text(self.criteria.true) or has_text(self.criteria.false))


Question = Annotated[Union[Choice, Score, Noul], Field(discriminator="type")]


# What a noul asks when it has no instructions and only its criteria describe the answers.
DESCRIBED_NOUL = "Which description fits the state?"


def question_statement(question: Question) -> str:
    """The question's instructions as prompt text, or the standing question a described noul asks."""
    return render_text(question.instructions) if has_text(question.instructions) else DESCRIBED_NOUL


class InvalidOptions(ValueError):
    """A well-formed question the model cannot answer, named so the error can point at it."""

    def __init__(self, question_id: str, message: str):
        super().__init__(f"questions.{question_id}: {message}")
        self.question_id = question_id


def check_askable(questions: dict[str, Question], max_options: int) -> None:
    """Refuse what the hosted services refuse after schema validation: a choice outside
    2..`max_options`, or a question with nothing to ask."""
    for qid, question in questions.items():
        if isinstance(question, Choice) and not 2 <= len(question.criteria) <= max_options:
            raise InvalidOptions(qid, f"a choice needs 2 to {max_options} options, got {len(question.criteria)}")
        if not has_text(question.instructions) and not (isinstance(question, Noul) and question.described()):
            raise InvalidOptions(qid, "needs instructions (a yes/no question may describe true or false instead)")


class SystemOneRequest(BaseModel):
    model: str | None = None
    state: State
    questions: dict[str, Question] = Field(min_length=1, max_length=MAX_QUESTIONS)

    @model_validator(mode="after")
    def ids_are_non_empty(self) -> "SystemOneRequest":
        if any(not qid for qid in self.questions):
            raise ValueError("question ids must be non-empty")
        return self


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


Answer = Annotated[Union[ChoiceAnswer, ScoreAnswer, NoulAnswer], Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    id: str | None = None
    """The decision's id in the server's ledger, for reporting its outcome; absent when nothing is logged."""
    model: str
    answers: dict[str, Answer]
    usage: Usage


class ModelInfo(BaseModel):
    """One served model. `name`, `description` and `release_date` are what the hosted
    API reports and what its SDK decodes; the rest is detail only a local engine has.
    `name` is what a request sends as `model`; `base` is the weights behind it."""

    name: str
    description: str
    release_date: str
    id: str
    base: str
    revision: str | None
    adapter: str | None
    calibration: dict[str, float]
    max_input_tokens: int
    max_branch_tokens: int
    max_options: int
    rotations: int
    prior_debias: bool


class ModelList(BaseModel):
    models: list[ModelInfo]


def option_keys(question: Choice | Score | Noul) -> list[str]:
    """The ordered answer set a question is scored over."""
    if isinstance(question, Choice):
        return list(question.criteria)
    if isinstance(question, Score):
        return [str(i) for i in range(len(question.criteria))]
    return ["yes", "no"]
