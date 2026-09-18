"""The live TypeSafe path, exercised against a ruling server, which speaks the same protocol."""

import numpy as np
import pytest

from ruling.against import collect_live
from ruling.evals import Record, collect, report

pytestmark = pytest.mark.integration

RECORD = Record.model_validate({
    "state": {"message": "My card was charged twice for one order. Refund the duplicate."},
    "questions": {
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Charges and refunds", "technical": "Bugs", "sales": "Pricing"}},
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Can wait", "This week", "Today"]},
        "refund": {"type": "noul", "instructions": "The customer asks for a refund.",
                   "criteria": {"true": "Money back is requested", "false": "No refund is mentioned"}},
    },
    "labels": {"team": "billing", "refund": True},
})


def test_live_answers_become_references_and_agree_with_the_same_engine(live_server, engine):
    live = collect_live([RECORD], base_url=live_server, api_key=None)
    assert set(live) == {(0, "team"), (0, "urgency"), (0, "refund")}
    assert set(live[(0, "team")]) == {"billing", "technical", "sales"}
    assert set(live[(0, "urgency")]) == {"0", "1", "2"}
    assert live[(0, "refund")]["yes"] + live[(0, "refund")]["no"] == pytest.approx(1.0)
    observations = collect(engine, [RECORD], live)
    summary = report(observations, engine.calibration)["question_types"]
    for question_type in ("choice", "score", "noul"):
        ref = summary[question_type]["references"]["live"]
        assert ref["argmax_agreement"] == 1.0
        assert ref["mean_total_variation"] < 0.05
    assert summary["choice"]["references"]["live"]["reference_accuracy"] == 1.0


def test_live_rejects_bad_credentials(live_server_with_key):
    import httpx
    with pytest.raises(httpx.HTTPStatusError):
        collect_live([RECORD], base_url=live_server_with_key, api_key="wrong")
    assert collect_live([RECORD], base_url=live_server_with_key, api_key="s3cret")
