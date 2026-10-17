"""Ask a question in plain words, and get a typed answer with its work shown.

A decision model cannot write the question. So before anything is written, the
model is asked nine typed questions *about* the input, in one request: can it be
judged at all, what kind of question it is, whether it is several questions,
whether it is finished and fit to show, and the answer itself read three ways,
as a yes-or-no, as a degree, and as a chance. For most questions one of those
three already is the answer and nothing is generated. Only a pick ("what is the
best text editor") or a scale with its own units ("how hot is the sun") needs
options written, and then the same model writes them as a short JSON tool call,
on a leash: it may only write the kinds of question the rules asked for, and
only the model's judgment of what it wrote is ever shown as the answer. Last,
each shown answer is sent back as "is this right?".

The nine questions and the rules follow lmjtfy (MIT OR Apache-2.0); see
THIRD_PARTY.md. Their wording is behavior: it decides which rule fires, and it is
measured by the routing cases in datasets/routing/ask25.jsonl.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from ruling.questions import (
    Answer,
    Choice,
    ChoiceAnswer,
    Image,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Question,
    ScoreAnswer,
    Score,
    SystemOneRequest,
    Usage,
)

# The runner-up reading of what kind of question it is counts too when it has at least this much.
ALSO = 0.3
# A yes-or-no is said as "Yes." or "No." from this probability; between, as "Yes?" or "No?".
SURE = 0.7
# The most questions the writer may draft for one input.
MAX_DRAFTS = 4
# Options and levels the writer may give one question: enough for a real field, few enough to read.
WRITER_OPTIONS = (2, 8)
WRITER_LEVELS = (3, 7)
# Tokens the writer may spend on its reply: four drafts with descriptions fit well inside it.
WRITER_TOKENS = 600

DEGREES = ["Not at all", "Slightly", "Moderately", "Very", "Extremely"]
KINDS = {
    "yes_or_no": "It asks whether something is so. A yes or a no answers it.",
    "pick_one": "It asks which, who or what: the best or right one among several possibilities. Naming one answers it.",
    "how_much": "It asks for a degree, an amount or a rating. A position on a scale answers it.",
    "how_likely": "It asks for the chance, the odds or the probability of something. A percentage answers it.",
}
YES_MEANS = "The answer to the question in the input is yes."
NO_MEANS = "The answer to the question in the input is no."

FACTS: dict[str, Question] = {
    "answerable": Noul(
        instructions="Is the input a question you can answer with your types?",
        criteria=NoulCriteria(
            true="The input asks something that can be judged: a yes-or-no question, a question with a best answer "
                 "among options, or a question of degree.",
            false="The input is anything else: a greeting, a statement, a command, or a request to write, explain, "
                  "summarize or do something.")),
    "kind": Choice(instructions="What kind of question is the input?", criteria=dict(KINDS)),
    "several": Noul(
        instructions="Does the input ask more than one separate question?",
        criteria=NoulCriteria(true="It asks two or more separate questions, each needing its own answer.",
                              false="It asks one question, even if that question mentions several things.")),
    "yes": Noul(instructions="Answer the question in the input.", criteria=NoulCriteria(true=YES_MEANS, false=NO_MEANS)),
    "degree": Score(instructions="Answer the question in the input as a degree: how much, how good, how strong, how likely.",
                    criteria=list(DEGREES)),
    "scale": Noul(
        instructions="Would the input be answered badly on a general scale from 'Not at all' to 'Extremely'?",
        criteria=NoulCriteria(
            true="Yes. What it asks about has its own units, ranges, grades or named levels, and a good answer would use them.",
            false="No. A general scale answers it well enough, or it is not a how-much question at all.")),
    "chance": Noul(instructions="Is the thing the input asks the chances of so?",
                   criteria=NoulCriteria(true="It is so, or it will happen.", false="It is not so, or it will not happen.")),
    "fit": Noul(
        instructions="Is the input fit to show to strangers on a public page?",
        criteria=NoulCriteria(true="It is harmless to show anyone.",
                              false="It has a slur, harassment, a threat, sexual content, or a private person's name or "
                                    "personal details.")),
    "whole": Noul(
        instructions="Is the input finished, or was it cut off part way through being typed?",
        criteria=NoulCriteria(
            true="Finished. Its last word is a whole word, and nothing it needs is missing after it. A greeting, a "
                 "statement or a few words can be finished; it need not be a question.",
            false="Cut off. It ends in part of a word, or on a word that needs more after it, such as a, the, is, "
                  "of, how or which.")),
}

# What each kind of question is answered by: a fact the first request already holds, or a draft the writer makes.
DIRECT = {"yes_or_no": "yes", "how_likely": "chance", "how_much": "degree"}
TOOL_OF_KIND = {"pick_one": "choice", "how_much": "score"}
KIND_OF_TOOL = {"noul": "yes_or_no", "choice": "pick_one", "score": "how_much"}

WRITER_SYSTEM = """\
You sit between a person and a decision model. The decision model cannot write text. It only judges, in three ways:
- noul: a yes-or-no question, answered with the probability of yes.
- choice: one of several options that you list, answered with a probability for each.
- score: a position on a scale whose levels you write, lowest first.

Turn the person's input into the question or questions that answer it, as JSON.
- Never answer the question yourself. Write only the JSON object, nothing else.
- A yes-or-no question becomes a noul. So does "how likely is X": ask whether X; the probability is the answer.
- "Which", "who", "what is the best" and other questions with a best answer become a choice. Give 2 to 8 real, \
specific options, each with a one-line description.
- "How good", "how much", "how many", "how spicy", "rate this" become a score with 3 to 7 levels, lowest first. \
The levels cover every possible answer: for a count or an amount the first is none or zero when that is possible \
and the last is open-ended ("more than 5"). Name each level by a range or a word.
- Use one question. Use more only when the input plainly asks several separate things, and never more than 4.
- Write "instructions" as one clear question. The decision model sees the person's input beside it, and nothing else.

Reply with exactly this shape, one object per question:
{"calls": [
  {"tool": "noul", "instructions": "...", "yes_means": "...", "no_means": "..."},
  {"tool": "choice", "instructions": "...", "options": ["label: one-line description", "label: description"]},
  {"tool": "score", "instructions": "...", "levels": ["lowest", "...", "highest"]}
]}"""


class AskRequest(BaseModel):
    model: str | None = None
    input: str = Field(min_length=1)
    images: list[Image] | None = None


class Shown(BaseModel):
    """One answer on the page: which reading it answers, who wrote the question, and what was said."""

    kind: str
    source: Literal["fact", "writer"]
    question: Question
    answer: Answer
    said: str
    checked: float | None = None
    """The model's probability that `said` is a right answer, asked afterward; absent when nothing was checked."""


class AskResponse(BaseModel):
    model: str
    input: str
    facts: dict[str, Answer]
    readings: list[str]
    refused: bool
    unfinished: bool
    fit: float
    answers: list[Shown]
    wrote: str | None = None
    """The writer's reply, byte for byte, when it was called."""
    dropped: list[str] = []
    """What the writer wrote that was not used, each with the reason."""
    usage: Usage


class Writer(Protocol):
    def write(self, text: str, allowed: tuple[str, ...]) -> str: ...


class LocalWriter:
    """The served model writing a tool call, with its thinking off and sampling at zero."""

    def __init__(self, engine):
        self.engine = engine

    def write(self, text: str, allowed: tuple[str, ...]) -> str:
        from mlx_lm import batch_generate
        from mlx_lm.sample_utils import make_sampler

        from ruling.engine import TEMPLATE_KWARGS

        leash = "" if set(allowed) == set(KIND_OF_TOOL) else f"\nWrite only these tools: {', '.join(allowed)}."
        messages = [{"role": "system", "content": WRITER_SYSTEM + leash},
                    {"role": "user", "content": f"Input: {text}"}]
        tokens = self.engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True, **TEMPLATE_KWARGS)
        with self.engine._lock:
            self.engine._activate()
            return batch_generate(self.engine.model, self.engine.tokenizer, [tokens], max_tokens=WRITER_TOKENS,
                                  sampler=make_sampler(temp=0.0)).texts[0]


def writer_for(engine) -> Writer | None:
    """The writer an engine can supply: a local text model writes with its own weights; others cannot write."""
    from ruling.engine import Engine  # imported here so the client, which shares this module's models, stays light

    return LocalWriter(engine) if type(engine) is Engine else None


def state_for(text: str, images: list[Image] | None) -> dict:
    state: dict = {"input": text}
    if images:
        state["images"] = [image.model_dump() for image in images]
    return state


@dataclass(frozen=True)
class Plan:
    """What the first request settled and what is still to do."""

    refused: bool
    unfinished: bool
    readings: list[str]
    direct: list[tuple[str, str]]
    """(kind, fact id) pairs the first request already answered."""
    wanted: tuple[str, ...]
    """Tools the writer is asked for; empty when nothing needs writing."""


def noul(answers: dict[str, Answer], fact: str) -> float:
    return answers[fact].noul


def readings_of(answers: dict[str, Answer]) -> list[str]:
    """The kinds the model read the input as: the likeliest, and any runner-up with at least ALSO."""
    kind: ChoiceAnswer = answers["kind"]
    ranked = sorted(kind.probabilities.items(), key=lambda item: -item[1])
    return [k for i, (k, p) in enumerate(ranked) if i == 0 or p >= ALSO]


def plan(answers: dict[str, Answer]) -> Plan:
    unfinished = noul(answers, "whole") < 0.5
    if noul(answers, "answerable") < 0.5:
        return Plan(refused=True, unfinished=unfinished, readings=[], direct=[], wanted=())
    readings = readings_of(answers)
    if noul(answers, "several") >= 0.5:
        return Plan(refused=False, unfinished=unfinished, readings=readings, direct=[], wanted=tuple(KIND_OF_TOOL))
    direct, wanted = [], []
    for kind in readings:
        if kind == "how_much" and noul(answers, "scale") >= 0.5:
            wanted.append("score")
        elif kind in DIRECT:
            direct.append((kind, DIRECT[kind]))
        else:
            wanted.append(TOOL_OF_KIND[kind])
    return Plan(refused=False, unfinished=unfinished, readings=readings, direct=direct, wanted=tuple(wanted))


@dataclass(frozen=True)
class Drafts:
    questions: dict[str, Question]
    dropped: list[str]


def parse_drafts(reply: str, allowed: tuple[str, ...]) -> Drafts:
    """The questions in a writer's reply, each checked, and what was left out and why."""
    calls, dropped = calls_in(reply)
    questions: dict[str, Question] = {}
    dropped += [f"call {i + 1} is over the cap of {MAX_DRAFTS}" for i in range(MAX_DRAFTS, len(calls))]
    for i, call in enumerate(calls[:MAX_DRAFTS]):
        try:
            questions[f"q{i + 1}"] = draft(call, allowed)
        except ValueError as error:
            dropped.append(f"call {i + 1}: {error}")
    return Drafts(questions, dropped)


def calls_in(reply: str) -> tuple[list, list[str]]:
    """The writer's calls: the whole object when it parses, else each call that parses on its own.

    A small model closes a brace with a bracket now and then, or runs on past the token cap; the
    calls before the slip are still good questions."""
    reply = closers_matched(reply)
    match = re.search(r"\{.*\}", reply, re.S)
    if match is None:
        return [], ["the writer wrote no JSON object"]
    try:
        calls = json.loads(match.group(0)).get("calls", [])
        return (calls, []) if isinstance(calls, list) else ([], ["the writer's `calls` is not a list"])
    except (json.JSONDecodeError, AttributeError) as error:
        whole = f"the writer's JSON does not parse: {error}"
    decoder = json.JSONDecoder()
    calls, dropped = [], []
    for i, start in enumerate(re.finditer(r"\{\s*\"tool\"", reply)):
        try:
            call, _ = decoder.raw_decode(reply, start.start())
            calls.append(call)
        except json.JSONDecodeError:
            dropped.append(f"call {i + 1} does not parse")
    return calls, ([whole] if not calls else []) + dropped


def closers_matched(text: str) -> str:
    """The text with every `}` or `]` that closes the other kind of bracket replaced by the right one.

    The one slip small models make in this shape is closing a list with a brace or an object with a
    bracket; the structure they meant is unambiguous from what was opened, so it is restored."""
    out, stack, in_string, escaped = [], [], False, False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if stack:
                char = "}" if stack.pop() == "{" else "]"
        out.append(char)
    return "".join(out)


def draft(call: object, allowed: tuple[str, ...]) -> Question:
    if not isinstance(call, dict):
        raise ValueError("not an object")
    tool = call.get("tool")
    if tool not in KIND_OF_TOOL:
        raise ValueError(f"tool {tool!r} is not noul, choice or score")
    if tool not in allowed:
        raise ValueError(f"a {tool} was not asked for")
    instructions = text_field(call, "instructions")
    if tool == "noul":
        return Noul(instructions=instructions,
                    criteria=NoulCriteria(true=text_field(call, "yes_means"), false=text_field(call, "no_means")))
    if tool == "choice":
        options = call.get("options")
        if isinstance(options, list):
            # "label: description" strings, or bare labels; a map of label to description is taken too.
            options = dict(labeled(str(o)) for o in options)
        if not isinstance(options, dict) or not all(isinstance(k, str) and k.strip() for k in options):
            raise ValueError("options must be a list of 'label: description' strings")
        if not WRITER_OPTIONS[0] <= len(options) <= WRITER_OPTIONS[1]:
            raise ValueError(f"a choice takes {WRITER_OPTIONS[0]} to {WRITER_OPTIONS[1]} options, got {len(options)}")
        return Choice(instructions=instructions,
                      criteria={k.strip(): (str(v).strip() or None) if v is not None else None for k, v in options.items()})
    levels = call.get("levels")
    if not isinstance(levels, list) or not all(isinstance(level, str) and level.strip() for level in levels):
        raise ValueError("levels must be a list of names, lowest first")
    if not WRITER_LEVELS[0] <= len(levels) <= WRITER_LEVELS[1]:
        raise ValueError(f"a score takes {WRITER_LEVELS[0]} to {WRITER_LEVELS[1]} levels, got {len(levels)}")
    return Score(instructions=instructions, criteria=[level.strip() for level in levels])


def labeled(option: str) -> tuple[str, str | None]:
    label, sep, description = option.partition(": ")
    return label.strip(), (description.strip() or None) if sep else None


def text_field(call: dict, name: str) -> str:
    value = call.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def kind_of(question: Question) -> str:
    return KIND_OF_TOOL[question.type]


def said(kind: str, answer: Answer) -> str:
    """The answer in words: Yes. / No? for a yes-or-no, a percentage for a chance, else the winning option or level."""
    if isinstance(answer, NoulAnswer):
        if kind == "how_likely":
            return f"{round(answer.noul * 100)}%."
        word = "Yes" if answer.noul >= 0.5 else "No"
        return f"{word}." if max(answer.noul, 1 - answer.noul) >= SURE else f"{word}?"
    if isinstance(answer, ScoreAnswer):
        top = max(answer.probabilities, key=answer.probabilities.__getitem__)
        return f"{answer.legend[top]}."
    return f"{answer.choice}."


def check_question(shown: Shown) -> Noul:
    return Noul(instructions=f"Is this a right answer to the question in `input`? The answer: {shown.said}")


def ask(engine, writer: Writer | None, text: str, images: list[Image] | None = None) -> AskResponse:
    state = state_for(text, images)
    tokens = 0

    def judge(questions: dict[str, Question]) -> dict[str, Answer]:
        nonlocal tokens
        response = engine.evaluate(SystemOneRequest(state=state, questions=questions))
        tokens += response.usage.input_tokens
        return response.answers

    facts = judge(FACTS)
    settled = plan(facts)
    shown: list[Shown] = []
    wrote, dropped = None, []
    if not settled.refused:
        for kind, fact in settled.direct:
            shown.append(Shown(kind=kind, source="fact", question=FACTS[fact], answer=facts[fact],
                               said=said(kind, facts[fact])))
        if settled.wanted:
            if writer is None:
                dropped.append("this model cannot write options; serve a local text model to draft them")
            else:
                wrote = writer.write(text, settled.wanted)
                drafts = parse_drafts(wrote, settled.wanted)
                dropped.extend(drafts.dropped)
                if drafts.questions:
                    judged = judge(drafts.questions)
                    for qid, question in drafts.questions.items():
                        kind = kind_of(question)
                        shown.append(Shown(kind=kind, source="writer", question=question, answer=judged[qid],
                                           said=said(kind, judged[qid])))
        if shown:
            checks = judge({f"check_{i}": check_question(s) for i, s in enumerate(shown)})
            shown = [s.model_copy(update={"checked": checks[f"check_{i}"].noul}) for i, s in enumerate(shown)]
    return AskResponse(model=engine.model_id, input=text, facts=facts, readings=settled.readings, refused=settled.refused,
                       unfinished=settled.unfinished, fit=noul(facts, "fit"), answers=shown, wrote=wrote,
                       dropped=dropped, usage=Usage(input_tokens=tokens))
