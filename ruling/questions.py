"""Wire types for the System One API: questions in, typed answers out."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

State = Union[str, dict[str, Any], list[Any]]

# Instructions, option descriptions, and level descriptions may be plain text or
# JSON structure; structure is rendered compactly into the prompt.
Text = Union[str, dict[str, Any], list[Any]]

# Matches the hosted service. Above the tokenizer's single-token codes a Choice
# is answered in two stages (see Engine).
MAX_CHOICE_OPTIONS = 255

# The hosted SDK sends its default model name on every request. Treating these as
# "whichever model this server loaded" is what lets that client talk to ruling
# unchanged; the response still reports the model that actually answered.
DEFAULT_MODEL_ALIASES = ("jev-latest", "default")


def render_text(value: Text) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _non_empty(value: Text) -> Text:
    if not render_text(value).strip():
        raise ValueError("text must not be empty")
    return value


class Choice(BaseModel):
    """Pick one option from a closed set. Criteria maps option key to an optional description."""

    type: Literal["choice"] = "choice"
    instructions: Text
    criteria: dict[str, Text | None] = Field(min_length=2, max_length=MAX_CHOICE_OPTIONS)

    _instructions = field_validator("instructions")(_non_empty)


class Score(BaseModel):
    """Place the state on an ordered scale. Criteria lists the levels from lowest to highest."""

    type: Literal["score"] = "score"
    instructions: Text
    criteria: list[Text] = Field(min_length=2)

    _instructions = field_validator("instructions")(_non_empty)

    @field_validator("criteria")
    @classmethod
    def levels_are_distinct(cls, levels: list[Text]) -> list[Text]:
        rendered = [render_text(level) for level in levels]
        if len(set(rendered)) != len(rendered):
            raise ValueError("score levels must be distinct")
        return levels


class NoulCriteria(BaseModel):
    """Optional descriptions of what a true and a false answer mean."""

    true: Text | None = None
    false: Text | None = None


class Noul(BaseModel):
    """Judge whether a statement about the state is true."""

    type: Literal["noul"] = "noul"
    instructions: Text
    criteria: NoulCriteria | None = None

    _instructions = field_validator("instructions")(_non_empty)


Question = Annotated[Union[Choice, Score, Noul], Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    model: str | None = None
    state: State
    questions: dict[str, Question] = Field(min_length=1)

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
    model: str
    answers: dict[str, Answer]
    usage: Usage


class ModelInfo(BaseModel):
    """One served model. `name`, `description` and `release_date` are what the hosted
    API reports and what its SDK decodes; the rest is detail only a local engine has."""

    name: str
    description: str
    release_date: str
    id: str
    revision: str | None
    adapter: str | None
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
