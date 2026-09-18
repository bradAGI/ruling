import math

import pytest

from ruling.hosted import Budget, BudgetExceeded, letter_log_probs


def top(pairs):
    return [{"token": token, "logprob": logprob} for token, logprob in pairs]


def test_letter_log_probs_merge_spacing_variants_and_normalize():
    result = letter_log_probs(top([("A", -0.1), (" A", -3.0), ("B", -2.5), ("<eos>", -9.0)]), ["A", "B", "C"])
    probs = [math.exp(v) for v in result]
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[1] > probs[2] > 0
    # " A" and "A" are the same answer; their probability mass adds
    assert probs[0] / probs[1] == pytest.approx((math.exp(-0.1) + math.exp(-3.0)) / math.exp(-2.5))


def test_missing_letters_get_less_than_the_least_likely_returned_token():
    result = letter_log_probs(top([("B", -0.01), ("A", -6.0), ("x", -12.0)]), ["A", "B", "C"])
    assert result[2] < result[0] and math.exp(result[2]) < math.exp(-12.0)


def test_no_letter_in_top_logprobs_is_an_error():
    with pytest.raises(ValueError, match="none of the answer letters"):
        letter_log_probs(top([("Hello", -0.1)]), ["A", "B"])


def test_budget_refuses_spending_past_the_cap():
    budget = Budget(limit_usd=0.01)
    budget.charge(0.006)
    budget.check()
    budget.charge(0.005)
    with pytest.raises(BudgetExceeded):
        budget.check()
    assert budget.spent_usd == pytest.approx(0.011)


def test_top_logprobs_are_extracted_or_reported_missing():
    from ruling.hosted import first_token_top_logprobs

    good = {"choices": [{"logprobs": {"content": [{"token": "A", "logprob": 0.0, "top_logprobs": [{"token": "A", "logprob": 0.0}]}]}}]}
    assert first_token_top_logprobs(good) == [{"token": "A", "logprob": 0.0}]
    assert first_token_top_logprobs({"choices": [{"logprobs": None}]}) is None
    assert first_token_top_logprobs({"choices": [{"logprobs": {"content": []}}]}) is None


def test_extremely_unlikely_tokens_do_not_underflow():
    result = letter_log_probs(top([("A", 0.0), ("B", -9999.0), ("<pad>", -float("inf"))]), ["A", "B", "C"])
    assert all(math.isfinite(v) for v in result)
    assert result[0] == pytest.approx(0.0, abs=1e-9) and result[1] < -9000 and result[2] < -9000
