"""Wire types for the System One API: questions in, typed answers out."""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Iterator, Mapping
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

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

# An image is {"type": "image", "media_type", "data": <base64>} anywhere in the state,
# PNG, JPEG or WebP, at most 8 per decision and 5 MB each.
MAX_IMAGES = 8
MAX_IMAGE_BYTES = 5 * 2**20
IMAGE_REFERENCE = "<image {}>"
IMAGE_MENTION = re.compile(r"<image (\d+)>")
ImageMediaType = Literal["image/png", "image/jpeg", "image/webp"]
IMAGE_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
"""The media type an image file's suffix implies, for clients reading images off disk."""

MAX_PREDICT_ROWS = 50_000
"""Rows one `/v1/predict` or `/v1/evaluations` request may carry. A batch holds one admission slot
for its whole duration, so the limit bounds how long one client can keep a slot."""

# The hosted SDK sends its default model name on every request. Treating these as
# "whichever model this server loaded" is what lets that client talk to ruling
# unchanged; the response still reports the model that actually answered.
DEFAULT_MODEL_ALIASES = ("jev-latest", "jev-preview", "default")


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


class Box(Strict):
    """Where something is in an image. The instructions say what to find and, when the state holds
    several images, which one as `<image N>`. A box is never "not found": the answer is always four
    edges, with a confidence to say how sure the model is, as the hosted API's box is."""

    type: Literal["box"] = "box"
    instructions: Text


Scorable = Union[Choice, Score, Noul]
"""The question types answered from a distribution over options; a box is generated instead."""
Question = Annotated[Union[Choice, Score, Noul, Box], Field(discriminator="type")]


# What a noul asks when it has no instructions and only its criteria describe the answers.
DESCRIBED_NOUL = "Which description fits the state?"


def question_statement(question: Scorable) -> str:
    """The question's instructions as prompt text, or the standing question a described noul asks."""
    return render_text(question.instructions) if has_text(question.instructions) else DESCRIBED_NOUL


class Image(Strict):
    type: Literal["image"]
    media_type: ImageMediaType
    data: str

    @field_validator("data")
    @classmethod
    def decodes_within_the_limit(cls, data: str) -> str:
        try:
            size = len(base64.b64decode(data, validate=True))
        except binascii.Error as exc:
            raise ValueError(f"image data must be base64: {exc}") from exc
        if size > MAX_IMAGE_BYTES:
            raise ValueError(f"an image may be at most {MAX_IMAGE_BYTES // 2**20} MB, got {size / 2**20:.1f} MB")
        return data

    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{self.data}"


def _is_image(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") == "image"


def state_images(state: State) -> list[Image]:
    """Every image in the state, depth first in document order, each validated."""
    if _is_image(state):
        return [Image.model_validate(state)]
    children = state.values() if isinstance(state, dict) else state if isinstance(state, list) else ()
    return [image for child in children for image in state_images(child)]


def without_images(state: State) -> State:
    """The state with each image replaced by its reference, `<image 1>` and on, in `state_images` order."""
    counter = iter(range(1, MAX_IMAGES + 1))

    def replace(value: Any) -> Any:
        if _is_image(value):
            return IMAGE_REFERENCE.format(next(counter))
        if isinstance(value, dict):
            return {key: replace(child) for key, child in value.items()}
        if isinstance(value, list):
            return [replace(child) for child in value]
        return value

    return replace(state)


def with_images(state: State, images: list[Image]) -> State:
    """The state with `images` after everything in it, so they take the next `<image N>` numbers:
    appended to a list, added as `images` to an object, and a text state becomes `{"text", "images"}`."""
    blocks = [image.model_dump() for image in images]
    if not blocks:
        return state
    if isinstance(state, list):
        return [*state, *blocks]
    if isinstance(state, dict):
        existing = state.get("images")
        if isinstance(existing, list):
            return {**state, "images": [*existing, *blocks]}
        if existing is None:
            return {**state, "images": blocks}
        return {"state": state, "images": blocks}
    return {"text": state, "images": blocks}


IMAGE_SIGNATURES: tuple[tuple[bytes, ImageMediaType], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"), (b"RIFF", "image/webp"))


def sniff_media_type(data: bytes) -> ImageMediaType:
    """What the bytes are, from their signature; a client's `format` field is not trusted over them."""
    for signature, media_type in IMAGE_SIGNATURES:
        if data.startswith(signature) and (media_type != "image/webp" or data[8:12] == b"WEBP"):
            return media_type
    raise ValueError("not a PNG, JPEG or WebP image")


class Asset(Strict):
    """An image sent beside the text and named in it, as the hosted API's `assets` carries one:
    the bytes as `data_base64`, as the state's `media_type` + `data`, or in a multipart part."""

    type: Literal["image"]
    format: str | None = None
    data_base64: str | None = None
    data: str | None = None
    media_type: ImageMediaType | None = None
    part: str | None = None

    @model_validator(mode="after")
    def one_source(self) -> "Asset":
        sources = [name for name in ("data_base64", "data", "part") if getattr(self, name) is not None]
        if len(sources) != 1:
            raise ValueError("an asset needs exactly one of data_base64, data, or part")
        return self

    def image(self, parts: Mapping[str, bytes]) -> Image:
        if self.part is not None:
            if self.part not in parts:
                raise ValueError(f"no multipart part named {self.part!r}")
            raw = parts[self.part]
        else:
            try:
                raw = base64.b64decode(self.data_base64 or self.data, validate=True)
            except binascii.Error as exc:
                raise ValueError(f"image data must be base64: {exc}") from exc
        return Image(type="image", media_type=self.media_type or sniff_media_type(raw),
                     data=base64.b64encode(raw).decode())


def _strings(value: Any) -> Iterator[str]:
    """Every string in a JSON value, depth first in document order; object keys are names, not text."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _replaced(value: Any, names: dict[str, str]) -> Any:
    if isinstance(value, str):
        for name in sorted(names, key=len, reverse=True):
            value = value.replace(name, names[name])
        return value
    if isinstance(value, dict):
        return {key: _replaced(child, names) for key, child in value.items()}
    if isinstance(value, list):
        return [_replaced(child, names) for child in value]
    return value


class UnsupportedModality(ValueError):
    """A state holds images, or a question needs one located, and the model that would answer cannot."""


def check_readable(state: State, reads_images: bool) -> None:
    if not reads_images and state_images(state):
        raise UnsupportedModality("this model reads only text; the state holds images")


class InvalidOptions(ValueError):
    """A well-formed question the model cannot answer, named so the error can point at it."""

    def __init__(self, question_id: str, message: str):
        super().__init__(f"questions.{question_id}: {message}")
        self.question_id = question_id


def check_locatable(questions: dict[str, Question], state: State, locates: bool) -> None:
    """Refuse a box question the selected model cannot answer, or one with no image to search."""
    boxes = [qid for qid, question in questions.items() if isinstance(question, Box)]
    if not boxes:
        return
    if not locates:
        raise UnsupportedModality("this model does not locate; a box question needs a vision: model")
    if not state_images(state):
        raise InvalidOptions(boxes[0], "a box needs an image in the state to search")


def target_image(question_id: str, box: Box, count: int) -> int:
    """Which of the state's `count` images a box searches: the one its instructions name as `<image N>`,
    or the only one; naming none of several, or several, is an error on the question."""
    named = sorted({int(n) for n in IMAGE_MENTION.findall(render_text(box.instructions))})
    if not named:
        if count == 1:
            return 1
        raise InvalidOptions(question_id, f"the state holds {count} images; name the one to search as <image N>")
    if len(named) > 1:
        raise InvalidOptions(question_id, "a box searches one image; the instructions name several")
    if named[0] > count:
        raise InvalidOptions(question_id, f"the instructions name <image {named[0]}> but the state holds {count}")
    return named[0]


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
    assets: dict[str, Asset] | None = None
    """Images named in the state, instructions or criteria by the key they sit under here; `resolve_assets`
    turns them into `<image N>` references and images in the state before anything reads the request."""
    # Marks traffic that must never become training data, such as benchmarks and evaluations.
    exclude_from_training: bool = False

    @field_validator("state")
    @classmethod
    def images_are_valid(cls, state: State) -> State:
        return check_images(state)

    @model_validator(mode="after")
    def ids_are_non_empty(self) -> "SystemOneRequest":
        if any(not qid for qid in self.questions):
            raise ValueError("question ids must be non-empty")
        return self


def resolve_assets(request: SystemOneRequest, parts: Mapping[str, bytes] | None = None) -> SystemOneRequest:
    """The request with each asset's image in the state and each mention of its name, wherever the text
    says it, replaced by that image's `<image N>` reference. Assets are numbered after the state's own
    images, in the order the request first mentions them; an asset nothing mentions is an error, as a
    multipart part no asset names is."""
    parts = parts or {}
    assets = request.assets or {}
    unnamed = sorted(set(parts) - {asset.part for asset in assets.values()})
    if unnamed:
        raise ValueError(f"multipart parts {unnamed} are named by no asset")
    if not assets:
        return request
    images = {}
    for key, asset in assets.items():
        try:
            images[key] = asset.image(parts)
        except ValueError as exc:
            raise ValueError(f"assets.{key}: {exc}") from exc
    questions = {qid: question.model_dump(exclude_none=True) for qid, question in request.questions.items()}
    texts = list(_strings(request.state)) + list(_strings(questions))
    first: dict[str, tuple[int, int]] = {}
    for position, text in enumerate(texts):
        for key in assets:
            at = text.find(key)
            if at >= 0 and (key not in first or (position, at) < first[key]):
                first[key] = (position, at)
    unmentioned = sorted(set(assets) - set(first))
    if unmentioned:
        raise ValueError(f"assets {unmentioned} are mentioned nowhere in the state or questions")
    ordered = sorted(assets, key=first.__getitem__)
    state = with_images(request.state, [images[key] for key in ordered])
    inline = len(state_images(request.state))
    names = {key: IMAGE_REFERENCE.format(inline + number) for number, key in enumerate(ordered, 1)}
    return request.model_copy(update={"state": check_images(_replaced(state, names)),
                                      "questions": {qid: QUESTION.validate_python(_replaced(q, names))
                                                    for qid, q in questions.items()},
                                      "assets": None})


def check_images(state: State) -> State:
    """Validate every image in the state and their count; returns the state unchanged."""
    images = state_images(state)
    if len(images) > MAX_IMAGES:
        raise ValueError(f"a state may hold at most {MAX_IMAGES} images, got {len(images)}")
    return state


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


class BoxCoordinates(BaseModel):
    """Edges as fractions of the image, from its top-left corner: `top` and `bottom` of the height,
    `left` and `right` of the width, each from zero to one."""

    top: float = Field(ge=0, le=1)
    bottom: float = Field(ge=0, le=1)
    left: float = Field(ge=0, le=1)
    right: float = Field(ge=0, le=1)


class BoxAnswer(BaseModel):
    type: Literal["box"] = "box"
    box: BoxCoordinates
    confidence: float


Answer = Annotated[Union[ChoiceAnswer, ScoreAnswer, NoulAnswer, BoxAnswer], Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int = 0


QUESTION = TypeAdapter(Question)


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
    long_context_tokens: int | None
    """Both limits for a request sent with `TypeSafe-Danger-Zone-Long-Context: true`; None where the
    engine has no limit of its own."""
    max_options: int
    rotations: int
    prior_debias: bool
    modalities: list[Literal["text", "image"]]
    question_types: list[Literal["choice", "score", "noul", "box"]]


class ModelList(BaseModel):
    models: list[ModelInfo]


def option_keys(question: Scorable) -> list[str]:
    """The ordered answer set a question is scored over."""
    if isinstance(question, Choice):
        return list(question.criteria)
    if isinstance(question, Score):
        return [str(i) for i in range(len(question.criteria))]
    return ["yes", "no"]
