"""Real calls to OpenRouter; each run costs a fraction of a cent. Skipped without OPENROUTER_API_KEY."""

import os

import pytest

from ruling.calibration import Calibration
from ruling.hosted import Budget, BudgetExceeded, OpenRouterEngine
from ruling.questions import Choice, Noul, SystemOneRequest

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY unset")]

MODEL = "openrouter:deepseek/deepseek-v4-flash-0731"
PROVIDERS = ["parasail", "coreweave", "streamlake"]


def test_hosted_teacher_answers_an_obvious_question_within_budget():
    budget = Budget(limit_usd=0.01)
    engine = OpenRouterEngine(MODEL, Calibration(), rotations=3, providers=PROVIDERS, budget=budget)
    response = engine.evaluate(SystemOneRequest(
        state="My card was charged twice for order 4471. Please refund the duplicate charge.",
        questions={"team": Choice(instructions="Which team should handle this?",
                                  criteria={"billing": "Charges and refunds", "technical": "Bugs", "sales": "Pricing"}),
                   "refund": Noul(instructions="The customer asks for a refund.")}))
    team = response.answers["team"]
    assert team.choice == "billing" and team.probabilities["billing"] > 0.9
    assert response.answers["refund"].noul > 0.9
    assert 0 < budget.spent_usd < 0.01


def test_budget_stops_calls_once_spent():
    budget = Budget(limit_usd=0.0)
    budget.charge(0.0001)
    engine = OpenRouterEngine(MODEL, Calibration(), rotations=1, providers=PROVIDERS, budget=budget)
    with pytest.raises(BudgetExceeded):
        engine.score("x", {"q": Noul(instructions="q")})


def test_api_errors_carry_the_providers_explanation():
    engine = OpenRouterEngine(MODEL, Calibration(), rotations=1, providers=["no-such-provider"], budget=Budget(limit_usd=0.01))
    with pytest.raises(RuntimeError, match="OpenRouter 404 .*no-such-provider"):
        engine.score("x", {"q": Noul(instructions="q")})
