"""Render state and questions into the text the model scores.

The state is rendered once as a prefix; each question is rendered as a suffix.
The engine encodes the prefix a single time and evaluates every suffix against
it, which is what keeps per-question cost flat as questions are added.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ruling.questions import (
    IMAGE_REFERENCE,
    Choice,
    Noul,
    Score,
    State,
    question_statement,
    render_text,
    state_images,
    without_images,
)

SYSTEM_PROMPT = (
    "You are a decision engine embedded in software. You are shown a state and one "
    "question about it. Answer with a single letter naming the best option. "
    "Never write anything except that letter."
)

# Splits the rendered user message into the shared prefix and the per-question
# suffix after the chat template has been applied. It must never occur in user data.
SPLIT_MARKER = "\x00RULING_SPLIT\x00"

# A content-free question used to measure how much the model prefers each
# answer letter regardless of content (Zhao et al. 2021, contextual calibration).
PRIOR_STATE = "No information is available."
PRIOR_INSTRUCTIONS = "Pick an option."
PRIOR_OPTION = "No description."


@dataclass(frozen=True)
class RenderedQuestion:
    text: str
    keys: list[str]
    codes: list[str]
    # The parts of `text`, for a layout that orders them differently.
    question: str = ""
    options: str = ""
    closing: str = ""


def render_state(state: State) -> str:
    """The state as prompt text, each image as a reference such as `<image 1>`; an engine that reads
    images supplies them beside the text."""
    state = without_images(state)
    body = state if isinstance(state, str) else json.dumps(state, indent=2, ensure_ascii=False)
    return f"State:\n{body}\n\n"


def render_question(question: Choice | Score | Noul, codes: list[str], rotation: int = 0) -> RenderedQuestion:
    """Render one question under a given `rotation` of its answer order.

    Choice and Noul rotate cyclically. Score has two orderings, lowest to
    highest and highest to lowest, because level order is part of the question.
    Averaging over orderings cancels the positional bias small models show
    toward particular letters.
    """
    instructions = question_statement(question)
    if isinstance(question, Choice):
        keys = _rotate(list(question.criteria), rotation)
        lines = [_option_line(code, key, question.criteria[key]) for code, key in zip(codes, keys)]
        heading = "Options:"
        closing = "Respond with only the letter of the best option."
    elif isinstance(question, Score):
        levels = list(enumerate(question.criteria))
        if rotation % 2:
            levels.reverse()
            heading = "Levels, from highest to lowest:"
        else:
            heading = "Levels, from lowest to highest:"
        keys = [str(i) for i, _ in levels]
        lines = [_option_line(code, render_text(level), None) for code, (_, level) in zip(codes, levels)]
        closing = "Respond with only the letter of the level that fits best."
    else:
        keys = _rotate(["yes", "no"], rotation)
        descriptions = {"yes": None, "no": None}
        if question.criteria is not None:
            descriptions = {"yes": question.criteria.true, "no": question.criteria.false}
        lines = [_option_line(code, key, descriptions[key]) for code, key in zip(codes, keys)]
        heading = "Is the statement true?"
        closing = "Respond with only the letter."
    used = codes[: len(keys)]
    question = f"Question: {instructions}"
    options = "\n".join([heading, *lines])
    text = "\n".join([question, options, closing])
    return RenderedQuestion(text=text, keys=keys, codes=used, question=question, options=options, closing=closing)


def hosted_user_message(state: State, rendered: RenderedQuestion) -> str | list[dict]:
    """The user turn for an OpenAI-compatible host: question and options first,
    the state after them, the answer cue last.

    Such a host caches the prompt prefix it has already processed and reuses
    it for the next request that starts the same way. Across calls the
    question and its options repeat and the state changes, so with the state
    first nothing after it is reused and every call pays for the whole prompt:
    on a CPU host at 36 tokens a second, 18 s against 1 s. The local engine
    keeps the state first, since it encodes the state once and scores every
    question against it.

    A state with images becomes content parts, each image after the state text.
    """
    text = f"{rendered.question}\n{rendered.options}\n\n{render_state(state)}"
    images = state_images(state)
    if not images:
        return text + rendered.closing
    # Each image follows the state under the reference the state text uses for it.
    parts = [{"type": "text", "text": text}]
    for number, image in enumerate(images, 1):
        parts += [{"type": "text", "text": f"{IMAGE_REFERENCE.format(number)}:"},
                  {"type": "image_url", "image_url": {"url": image.data_url()}}]
    return parts + [{"type": "text", "text": "\n\n" + rendered.closing}]


def prior_question(cardinality: int) -> Choice:
    return Choice(instructions=PRIOR_INSTRUCTIONS, criteria={f"option{i + 1}": PRIOR_OPTION for i in range(cardinality)})


def _rotate(keys: list[str], rotation: int) -> list[str]:
    shift = rotation % len(keys)
    return keys[shift:] + keys[:shift]


def _option_line(code: str, label: str, description: object) -> str:
    return f"{code}. {label}: {render_text(description)}" if description is not None else f"{code}. {label}"
