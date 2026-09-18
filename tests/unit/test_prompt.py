import json

from ruling.prompt import render_question, render_state
from ruling.questions import Choice, Noul, Score

CODES = list("ABCDEFG")


def test_state_object_is_rendered_as_json():
    text = render_state({"k": "v", "n": [1, 2]})
    assert json.loads(text.removeprefix("State:\n")) == {"k": "v", "n": [1, 2]}


def test_choice_rotation_shifts_codes_but_keeps_keys_set():
    question = Choice(instructions="q", criteria={"x": "one", "y": "two", "z": None})
    plain = render_question(question, CODES, rotation=0)
    shifted = render_question(question, CODES, rotation=1)
    assert plain.keys == ["x", "y", "z"]
    assert shifted.keys == ["y", "z", "x"]
    assert plain.codes == shifted.codes == ["A", "B", "C"]
    assert "A. x: one" in plain.text and "A. y: two" in shifted.text
    assert "C. z" in plain.text and "C. z:" not in plain.text


def test_score_has_exactly_two_orderings():
    question = Score(instructions="q", criteria=["lo", "mid", "hi"])
    forward = render_question(question, CODES, rotation=0)
    reverse = render_question(question, CODES, rotation=1)
    assert forward.keys == ["0", "1", "2"] and "lowest to highest" in forward.text
    assert reverse.keys == ["2", "1", "0"] and "highest to lowest" in reverse.text
    assert "A. hi" in reverse.text
    assert render_question(question, CODES, rotation=2).text == forward.text


def test_structured_instructions_and_descriptions_render_as_json():
    question = Choice(instructions={"task": "route"}, criteria={"a": {"desc": "x"}, "b": None})
    text = render_question(question, CODES).text
    assert 'Question: {"task": "route"}' in text and 'A. a: {"desc": "x"}' in text


def test_noul_rotation_swaps_yes_no():
    question = Noul(instructions="q")
    assert render_question(question, CODES, rotation=1).keys == ["no", "yes"]
