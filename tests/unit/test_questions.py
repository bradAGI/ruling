import pytest
from pydantic import ValidationError

from ruling.questions import (MAX_QUESTIONS, MAX_SCORE_LEVELS, Choice, InvalidOptions, Noul, NoulCriteria, Score,
                              SystemOneRequest, check_askable, option_keys)


def test_choice_requires_an_option():
    with pytest.raises(ValidationError):
        Choice(instructions="pick", criteria={})


def test_score_levels_must_be_distinct():
    with pytest.raises(ValidationError):
        Score(instructions="rate", criteria=["low", "low"])


def test_request_discriminates_on_type():
    request = SystemOneRequest.model_validate(
        {
            "state": "x",
            "questions": {
                "a": {"type": "choice", "instructions": "q", "criteria": {"x": None, "y": "desc"}},
                "b": {"type": "score", "instructions": "q", "criteria": ["lo", "hi"]},
                "c": {"type": "noul", "instructions": "q"},
            },
        }
    )
    assert isinstance(request.questions["a"], Choice)
    assert isinstance(request.questions["b"], Score)
    assert isinstance(request.questions["c"], Noul)


def test_request_rejects_unknown_type():
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "x", "questions": {"a": {"type": "rank", "instructions": "q"}}})


def test_choice_is_capped_at_the_hosted_limit():
    Choice(instructions="q", criteria={f"o{i}": None for i in range(255)})
    with pytest.raises(ValidationError):
        Choice(instructions="q", criteria={f"o{i}": None for i in range(256)})


def test_structured_instructions_are_kept_as_structure():
    assert Noul(instructions={"claim": "x"}).instructions == {"claim": "x"}


def test_option_keys_per_type():
    assert option_keys(Choice(instructions="q", criteria={"b": None, "a": None})) == ["b", "a"]
    assert option_keys(Score(instructions="q", criteria=["lo", "mid", "hi"])) == ["0", "1", "2"]
    assert option_keys(Noul(instructions="q")) == ["yes", "no"]


def test_a_request_carries_at_most_the_hosted_question_limit():
    noul = {"type": "noul", "instructions": "q"}
    SystemOneRequest.model_validate({"state": "x", "questions": {f"q{i}": noul for i in range(MAX_QUESTIONS)}})
    with pytest.raises(ValidationError, match="questions"):
        SystemOneRequest.model_validate({"state": "x", "questions": {f"q{i}": noul for i in range(MAX_QUESTIONS + 1)}})


def test_a_score_has_at_most_the_hosted_level_limit():
    Score(instructions="rate", criteria=[f"l{i}" for i in range(MAX_SCORE_LEVELS)])
    with pytest.raises(ValidationError, match="criteria"):
        Score(instructions="rate", criteria=[f"l{i}" for i in range(MAX_SCORE_LEVELS + 1)])


def test_a_one_option_choice_is_well_formed_but_not_askable():
    """The hosted schema admits one option; the model-level check is what refuses it, by question."""
    one = Choice(instructions="pick", criteria={"only": None})
    with pytest.raises(InvalidOptions, match="2 to 255") as caught:
        check_askable({"q": one}, max_options=255)
    assert caught.value.question_id == "q"


def test_more_options_than_the_model_reads_is_not_askable():
    wide = Choice(instructions="pick", criteria={f"o{i}": None for i in range(27)})
    check_askable({"q": wide}, max_options=27)
    with pytest.raises(InvalidOptions, match="2 to 26"):
        check_askable({"q": wide}, max_options=26)


@pytest.mark.parametrize("question", [
    Choice(instructions=None, criteria={"a": None, "b": None}),
    Choice(instructions="   ", criteria={"a": None, "b": None}),
    Score(instructions=None, criteria=["lo", "hi"]),
    Noul(instructions=None),
    Noul(instructions="", criteria=NoulCriteria(true=None, false="  ")),
])
def test_a_question_with_nothing_to_ask_is_not_askable(question):
    with pytest.raises(InvalidOptions, match="needs instructions"):
        check_askable({"q": question}, max_options=255)


def test_a_noul_may_describe_true_or_false_instead_of_instructions():
    check_askable({"q": Noul(criteria=NoulCriteria(true="the customer is angry"))}, max_options=255)
    check_askable({"q": Noul(criteria=NoulCriteria(false={"tone": "calm"}))}, max_options=255)
    check_askable({"q": Noul(instructions={"claim": "x"})}, max_options=255)


def test_a_question_refuses_fields_it_does_not_know():
    with pytest.raises(ValidationError, match="instruction"):
        Noul.model_validate({"type": "noul", "instruction": "typo"})
    with pytest.raises(ValidationError, match="bogus"):
        NoulCriteria.model_validate({"true": "x", "bogus": 1})
