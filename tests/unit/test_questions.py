import pytest
from pydantic import ValidationError

from ruling.questions import Choice, Noul, Score, SystemOneRequest, option_keys


def test_choice_requires_two_options():
    with pytest.raises(ValidationError):
        Choice(instructions="pick", criteria={"only": None})


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


def test_structured_text_must_not_be_empty():
    with pytest.raises(ValidationError):
        Noul(instructions="   ")
    assert Noul(instructions={"claim": "x"}).instructions == {"claim": "x"}


def test_option_keys_per_type():
    assert option_keys(Choice(instructions="q", criteria={"b": None, "a": None})) == ["b", "a"]
    assert option_keys(Score(instructions="q", criteria=["lo", "mid", "hi"])) == ["0", "1", "2"]
    assert option_keys(Noul(instructions="q")) == ["yes", "no"]
