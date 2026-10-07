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


def test_hosted_message_puts_the_stable_question_before_the_state():
    """A host that caches prompt prefixes must see the parts that repeat first:
    the question and its options, then the state, then the answer cue."""
    import os
    from ruling.prompt import hosted_user_message, render_question
    from ruling.questions import Choice

    question = Choice(instructions="Which type is the name?", criteria={"ORG": "a company", "PERSON": "a person"})
    rendered = render_question(question, ["A", "B"], 0)
    a = hosted_user_message({"name": "OpenAI"}, rendered)
    b = hosted_user_message({"name": "Meta"}, rendered)
    shared = os.path.commonprefix([a, b])
    assert rendered.question in shared and rendered.options in shared
    assert a.index('"OpenAI"') > a.index(rendered.options)
    assert a.endswith(rendered.closing)
    # The local engine's layout, state then question, is untouched.
    assert rendered.text == f"{rendered.question}\n{rendered.options}\n{rendered.closing}"
