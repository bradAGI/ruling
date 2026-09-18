import datetime as dt

import numpy as np
import pytest

from ruling.evals import Record
from ruling.states import GENERATORS, access_decision, generate, inventory_answers, orders_answers, services_answers


def test_every_generator_yields_valid_labeled_records_deterministically():
    a = generate(200, seed=7)
    b = generate(200, seed=7)
    assert [r.model_dump() for r in a] == [r.model_dump() for r in b]
    families = {r.family for r in a}
    assert families == {f"state_{name}" for name in GENERATORS}
    for r in a:
        assert isinstance(r, Record) and set(r.labels) == set(r.questions)


def test_orders_answers_follow_the_stated_policy():
    state = {
        "today": "2026-09-20",
        "refund_policy_days": 14,
        "orders": [
            {"id": "A1", "amount_usd": 120.0, "status": "delivered", "delivered_on": "2026-09-10"},
            {"id": "A2", "amount_usd": 900.5, "status": "delivered", "delivered_on": "2026-08-01"},
            {"id": "A3", "amount_usd": 40.0, "status": "pending", "delivered_on": None},
        ],
    }
    answers = orders_answers(state, subject="A1", threshold=1000.0)
    assert answers["subject_refundable"] is True
    assert orders_answers(state, subject="A2", threshold=1000.0)["subject_refundable"] is False
    assert orders_answers(state, subject="A3", threshold=1000.0)["subject_refundable"] is False
    assert answers["largest"] == "A2"
    assert answers["pending"] == 1
    assert answers["delivered_total_over"] is True


def test_services_answers_flag_breaches_and_worst_error_rate():
    state = {"thresholds": {"error_rate": 0.02, "p95_ms": 800},
             "services": [{"name": "api", "error_rate": 0.031, "p95_ms": 420},
                          {"name": "web", "error_rate": 0.004, "p95_ms": 950},
                          {"name": "jobs", "error_rate": 0.0, "p95_ms": 100}]}
    answers = services_answers(state, subject="web")
    assert answers["subject_breach"] is True and answers["worst_error"] == "api" and answers["breaches"] == 2
    assert services_answers(state, subject="jobs")["subject_breach"] is False


@pytest.mark.parametrize("role,classification,approvals,expected", [
    ("engineer", "internal", [], "permit"),
    ("contractor", "internal", [], "needs_approval"),
    ("contractor", "internal", ["manager"], "permit"),
    ("engineer", "restricted", ["manager"], "needs_approval"),
    ("engineer", "restricted", ["manager", "security"], "permit"),
    ("contractor", "restricted", ["manager", "security"], "deny"),
])
def test_access_decision_matrix(role, classification, approvals, expected):
    assert access_decision(role, classification, approvals) == expected


def test_inventory_answers_use_days_of_cover():
    state = {"items": [{"sku": "S1", "on_hand": 30, "daily_sales": 10, "reorder_point": 40},
                       {"sku": "S2", "on_hand": 500, "daily_sales": 5, "reorder_point": 100}]}
    answers = inventory_answers(state, subject="S1")
    assert answers["subject_reorder"] is True and answers["lowest_cover"] == "S1" and answers["subject_cover_level"] == 0
    assert inventory_answers(state, subject="S2")["subject_cover_level"] == 3


def test_generated_records_survive_the_jsonl_round_trip(tmp_path):
    from ruling.build import write
    from ruling.evals import load_dataset

    records = generate(40, seed=1)
    path = tmp_path / "states.jsonl"
    write([r.model_dump(mode="json") for r in records], path)
    assert [r.model_dump() for r in load_dataset(path)] == [r.model_dump() for r in records]
