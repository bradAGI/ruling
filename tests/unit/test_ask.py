"""A plain question becomes typed questions: the rules, the writer's leash, and what is said."""

import base64
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ruling import ask as asking
from ruling.ask import FACTS, Drafts, Plan, ask, parse_drafts, plan, said
from ruling.calibration import Calibration
from ruling.engine import RawScore, answer
from ruling.questions import Choice, Noul, NoulAnswer, Score, ScoreAnswer, SystemOneResponse, Usage
from ruling.registry import Registry
from ruling.server import create_app
from ruling import registry as registry_module


def facts(**overrides):
    """The nine answers for a plain yes-or-no question, with any of them overridden by id.

    A noul is given as its probability of yes, `kind` as a map over the four kinds, `degree` as a list."""
    base = {"answerable": 0.95, "kind": {"yes_or_no": 0.9, "pick_one": 0.05, "how_much": 0.03, "how_likely": 0.02},
            "several": 0.1, "yes": 0.8, "degree": [0.1, 0.2, 0.4, 0.2, 0.1], "scale": 0.1, "chance": 0.6,
            "fit": 0.9, "whole": 0.95}
    base.update(overrides)
    answers = {}
    for fact, value in base.items():
        question = FACTS[fact]
        if isinstance(question, Noul):
            answers[fact] = NoulAnswer(noul=value)
        elif isinstance(question, Choice):
            answers[fact] = answer(question, RawScore(keys=list(question.criteria), logits=np.log(list(value.values()))),
                                   Calibration())
        else:
            answers[fact] = answer(question, RawScore(keys=[str(i) for i in range(len(value))], logits=np.log(value)),
                                   Calibration())
    return answers


def test_a_yes_or_no_is_answered_from_the_first_request_alone():
    settled = plan(facts())
    assert settled == Plan(refused=False, unfinished=False, readings=["yes_or_no"], direct=[("yes_or_no", "yes")], wanted=())


def test_what_the_model_cannot_judge_is_refused_and_nothing_else_is_asked():
    settled = plan(facts(answerable=0.2, kind={"yes_or_no": 0.9, "pick_one": 0.08, "how_much": 0.01, "how_likely": 0.01}))
    assert settled.refused and settled.direct == [] and settled.wanted == ()


def test_a_cut_off_input_is_still_answered_as_it_stands():
    settled = plan(facts(whole=0.2))
    assert settled.unfinished and settled.direct == [("yes_or_no", "yes")]


@pytest.mark.parametrize("kind, scale, direct, wanted", [
    ("how_likely", 0.1, [("how_likely", "chance")], ()),
    ("how_much", 0.1, [("how_much", "degree")], ()),
    ("how_much", 0.9, [], ("score",)),       # its own units: the writer drafts the scale
    ("pick_one", 0.1, [], ("choice",)),      # options have to be written
])
def test_each_kind_is_answered_directly_or_handed_to_the_writer(kind, scale, direct, wanted):
    kinds = {k: 0.02 for k in FACTS["kind"].criteria}
    kinds[kind] = 0.94
    settled = plan(facts(kind=kinds, scale=scale))
    assert settled.readings == [kind] and settled.direct == direct and settled.wanted == wanted


def test_a_runner_up_reading_with_enough_probability_is_answered_too():
    kinds = {"yes_or_no": 0.55, "pick_one": 0.4, "how_much": 0.03, "how_likely": 0.02}
    settled = plan(facts(kind=kinds))
    assert settled.readings == ["yes_or_no", "pick_one"]
    assert settled.direct == [("yes_or_no", "yes")] and settled.wanted == ("choice",)


def test_several_questions_go_whole_to_the_writer_with_every_tool():
    settled = plan(facts(several=0.8))
    assert settled.direct == [] and set(settled.wanted) == {"noul", "choice", "score"}


def test_the_writers_reply_is_read_leniently_but_checked_strictly():
    reply = '''Sure! ```json
{"calls": [
  {"tool": "choice", "instructions": "Which editor is best?", "options": ["Vim: modal", "Emacs: lisp", "VS Code"]},
  {"tool": "noul", "instructions": "Is \\"C++\\" better than C#?", "yes_means": "C++ is better", "no_means": "C# is better"},
  {"tool": "score", "instructions": "How many?", "levels": ["none", "1 to 2", "more than 2"]},
  {"tool": "choice", "instructions": "One option", "options": {"only": "one"}},
  {"tool": "essay", "instructions": "Write it"},
  {"tool": "score", "instructions": "Fifth", "levels": ["a", "b", "c"]}
]}
```'''
    drafts = parse_drafts(reply, ("choice", "noul"))
    assert list(drafts.questions) == ["q1", "q2"]
    assert drafts.questions["q1"] == Choice(instructions="Which editor is best?",
                                            criteria={"Vim": "modal", "Emacs": "lisp", "VS Code": None})
    assert drafts.questions["q2"].instructions == 'Is "C++" better than C#?'
    assert drafts.dropped == ["call 5 is over the cap of 4", "call 6 is over the cap of 4",
                              "call 3: a score was not asked for",
                              "call 4: a choice takes 2 to 8 options, got 1"]


def test_a_list_closed_with_a_brace_or_an_object_with_a_bracket_is_read_as_meant():
    # The two slips a 4B makes: `"]}` for a list of options, and `}]` for the object around one.
    reply = ('{"calls": [{"tool": "choice", "instructions": "Which?", "options": ["Vim: modal", "Emacs: lisp"}]}, '
             '{"tool": "noul", "instructions": "Is it? [a]", "yes_means": "y", "no_means": "n"]]}')
    drafts = parse_drafts(reply, ("choice", "noul"))
    assert [q.type for q in drafts.questions.values()] == ["choice", "noul"] and drafts.dropped == []
    assert drafts.questions["q1"].criteria == {"Vim": "modal", "Emacs": "lisp"}
    assert drafts.questions["q2"].instructions == "Is it? [a]"      # brackets inside strings are left alone


def test_calls_before_the_token_cap_are_kept_and_the_cut_off_one_is_named():
    reply = ('{"calls": [{"tool": "choice", "instructions": "Which?", "options": ["Vim: modal", "Emacs: lisp"]}, '
             '{"tool": "noul", "instructions": "Is it?", "yes_means": "y", "no_means": "n"}, {"tool": "sc')
    drafts = parse_drafts(reply, ("choice", "noul"))
    assert [q.type for q in drafts.questions.values()] == ["choice", "noul"]
    assert drafts.dropped == ["call 3 does not parse"]


@pytest.mark.parametrize("reply, reason", [
    ("I cannot help with that.", "no JSON object"),
    ('{"calls": [1, 2,]}', "does not parse"),
    ('{"calls": "none"}', "not a list"),
])
def test_a_reply_without_usable_calls_says_why(reply, reason):
    drafts = parse_drafts(reply, ("choice",))
    assert drafts.questions == {} and reason in drafts.dropped[0]


def test_what_is_said_follows_the_probabilities():
    assert said("yes_or_no", NoulAnswer(noul=0.8)) == "Yes."
    assert said("yes_or_no", NoulAnswer(noul=0.6)) == "Yes?"
    assert said("yes_or_no", NoulAnswer(noul=0.31)) == "No?"
    assert said("yes_or_no", NoulAnswer(noul=0.1)) == "No."
    assert said("how_likely", NoulAnswer(noul=0.284)) == "28%."
    assert said("how_much", ScoreAnswer(score=2.1, legend={"0": "Not at all", "1": "Very"},
                                        probabilities={"0": 0.3, "1": 0.7}, confidence=0.4)) == "Very."


class ScriptedEngine:
    """Answers every noul from a script keyed by instructions, choices and scores from a script keyed by id."""

    model_id = "scripted"
    prior_debias = reads_images = False
    rotations = max_options = 1
    max_input_tokens = max_branch_tokens = 10_000
    revision = adapter = None

    def __init__(self, nouls: dict[str, float], others: dict[str, list[float]]):
        self.nouls, self.others, self.calibration, self.requests = nouls, others, Calibration(), []

    def evaluate(self, request):
        self.requests.append(request)
        answers = {}
        for qid, question in request.questions.items():
            if isinstance(question, Noul):
                answers[qid] = NoulAnswer(noul=self.nouls.get(question.instructions, 0.5))
            else:
                keys = list(question.criteria) if isinstance(question, Choice) else [str(i) for i in range(len(question.criteria))]
                probs = self.others.get(qid) or [1 / len(keys)] * len(keys)
                answers[qid] = answer(question, RawScore(keys=keys, logits=np.log(probs)), self.calibration)
        return SystemOneResponse(model=self.model_id, answers=answers, usage=Usage(input_tokens=len(request.questions)))


class ScriptedWriter:
    def __init__(self, reply: str):
        self.reply, self.calls = reply, []

    def write(self, text, allowed, images=None):
        self.calls.append((text, allowed, images))
        return self.reply


PICK = {"kind": [0.05, 0.9, 0.03, 0.02]}
# A 1x1 PNG, the smallest readable picture.
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
A_QUESTION = {FACTS["answerable"].instructions: 0.95, FACTS["whole"].instructions: 0.9, FACTS["fit"].instructions: 0.9,
              FACTS["several"].instructions: 0.1, FACTS["scale"].instructions: 0.1}


def test_a_pick_is_written_judged_and_checked_and_the_page_can_see_all_of_it():
    engine = ScriptedEngine({**A_QUESTION, "Is this a right answer to the question in `input`? The answer: Vim.": 0.85},
                            {**PICK, "q1": [0.7, 0.3]})
    writer = ScriptedWriter('{"calls": [{"tool": "choice", "instructions": "Which editor?", "options": {"Vim": "a", "Emacs": "b"}}]}')
    response = ask(engine, writer, "what is the best text editor")
    assert writer.calls == [("what is the best text editor", ("choice",), None)]
    [shown] = response.answers
    assert shown.kind == "pick_one" and shown.source == "writer" and shown.said == "Vim." and shown.checked == 0.85
    assert shown.question.criteria == {"Vim": "a", "Emacs": "b"}
    assert response.readings == ["pick_one"] and not response.refused and response.wrote == writer.reply
    # Three requests: the facts, the draft, the check; usage counts every question of each.
    assert [list(r.questions) for r in engine.requests] == [list(FACTS), ["q1"], ["check_0"]]
    assert response.usage.input_tokens == len(FACTS) + 2


def test_pictures_reach_every_request_and_the_writer():
    from ruling.ask import writer_messages
    from ruling.questions import Image as ImageBlock
    picture = ImageBlock(type="image", media_type="image/png", data=base64.b64encode(PNG).decode())
    engine = ScriptedEngine(A_QUESTION, {**PICK, "q1": [0.2, 0.8]})
    writer = ScriptedWriter('{"calls": [{"tool": "choice", "instructions": "What is in <image 1>?", "options": ["cat", "dog"]}]}')
    response = ask(engine, writer, "what is this?", [picture])
    assert writer.calls[0][2] == [picture]
    assert all(r.state == {"input": "what is this?", "images": [picture.model_dump()]} for r in engine.requests)
    assert response.answers[0].said == "dog."
    seen, unseen = writer_messages("x", ("choice",), 2, sees=True)[0]["content"], writer_messages("x", ("choice",), 1, sees=False)[0]["content"]
    assert "<image 1>, <image 2>" in seen and "shown to you" in seen and "cannot see" in unseen
    assert "picture" not in writer_messages("x", ("choice",), 0, sees=True)[0]["content"]


def test_without_a_writer_a_pick_says_so_instead_of_answering_wrong():
    engine = ScriptedEngine(A_QUESTION, PICK)
    response = ask(engine, None, "best pizza topping")
    assert response.answers == [] and response.wrote is None
    assert response.dropped == ["this model cannot write options; serve a local text model to draft them"]


def test_a_refused_input_asks_nothing_more():
    engine = ScriptedEngine({FACTS["answerable"].instructions: 0.1}, {})
    writer = ScriptedWriter("{}")
    response = ask(engine, writer, "write me a poem")
    assert response.refused and response.answers == [] and writer.calls == [] and len(engine.requests) == 1


@pytest.fixture(autouse=True)
def no_hub_lookups(monkeypatch):
    monkeypatch.setattr(registry_module, "model_released", lambda model_id: "2026-01-01")


def test_the_route_answers_a_question_and_counts_it_as_a_decision(monkeypatch):
    engine = ScriptedEngine({**A_QUESTION, FACTS["yes"].instructions: 0.9,
                             "Is this a right answer to the question in `input`? The answer: Yes.": 0.7}, {})
    monkeypatch.setattr(asking, "writer_for", lambda engine: None)
    app = create_app(Registry.single(engine), "k", 4)
    with TestClient(app) as client:
        refused = client.post("/v1/ask", json={"input": "is water wet?"})
        assert refused.status_code == 401
        headers = {"authorization": "Bearer k"}
        assert client.post("/v1/ask", json={"input": ""}, headers=headers).status_code == 422
        assert client.post("/v1/ask", json={"input": "x", "model": "nope"}, headers=headers).json()["error"]["code"] == "unknown_model"
        body = client.post("/v1/ask", json={"input": "is water wet?"}, headers=headers).json()
        assert body["answers"][0]["said"] == "Yes." and body["answers"][0]["checked"] == 0.7
        assert body["facts"]["kind"]["type"] == "choice" and body["model"] == "scripted"
        [bucket] = client.get("/v1/usage", headers=headers).json()["usage"]
        assert bucket["decisions"] == 1 and bucket["input_tokens"] == body["usage"]["input_tokens"]
