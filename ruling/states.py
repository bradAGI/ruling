"""Synthetic program state with answers computed in code.

Each generator builds a JSON state, including any policy the question depends
on, and asks Choice, Score, and Noul questions whose answers a pure function
derives from that state. Labels are therefore exact. This is the part of the
training mix that teaches reading structured state rather than prose.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from ruling.evals import Record

NAMES = ["Avery", "Blake", "Casey", "Devon", "Emery", "Finley", "Harper", "Jordan", "Kai", "Logan", "Morgan", "Quinn",
         "Reese", "Rowan", "Sage", "Taylor"]
SERVICES = ["api", "web", "auth", "billing", "search", "jobs", "gateway", "notifications", "reports", "uploads"]
RESOURCES = ["payroll export", "customer database", "build server", "design drive", "analytics warehouse", "wiki",
             "production logs", "finance ledger"]


# ---- orders -------------------------------------------------------------------

def orders_answers(state: dict, subject: str, threshold: float) -> dict:
    today = dt.date.fromisoformat(state["today"])
    orders = {o["id"]: o for o in state["orders"]}
    order = orders[subject]
    refundable = (order["status"] == "delivered"
                  and (today - dt.date.fromisoformat(order["delivered_on"])).days <= state["refund_policy_days"])
    delivered_total = sum(o["amount_usd"] for o in state["orders"] if o["status"] == "delivered")
    return {
        "subject_refundable": refundable,
        "largest": max(state["orders"], key=lambda o: o["amount_usd"])["id"],
        "pending": sum(1 for o in state["orders"] if o["status"] == "pending"),
        "delivered_total_over": delivered_total > threshold,
    }


def _orders(rng: np.random.Generator) -> Record:
    today = dt.date(2026, 1, 1) + dt.timedelta(days=int(rng.integers(0, 300)))
    count = int(rng.integers(3, 8))
    ids = [f"{chr(65 + int(rng.integers(26)))}{int(rng.integers(100, 999))}" for _ in range(count)]
    ids = list(dict.fromkeys(ids))
    orders = []
    for oid in ids:
        status = str(rng.choice(["pending", "shipped", "delivered", "delivered", "refunded", "cancelled"]))
        delivered = today - dt.timedelta(days=int(rng.integers(0, 45))) if status in ("delivered", "refunded") else None
        orders.append({"id": oid, "customer": str(rng.choice(NAMES)), "amount_usd": round(float(rng.uniform(5, 1500)), 2),
                       "status": status, "delivered_on": delivered.isoformat() if delivered else None})
    amounts = [o["amount_usd"] for o in orders]
    if len(set(amounts)) != len(amounts):
        return _orders(rng)
    state = {"today": today.isoformat(), "refund_policy_days": int(rng.choice([7, 14, 30])), "orders": orders}
    subject = str(rng.choice(ids))
    threshold = float(rng.choice([500, 1000, 2000, 3000]))
    answers = orders_answers(state, subject, threshold)
    return Record(
        family="state_orders",
        state=state,
        questions={
            "subject_refundable": {"type": "noul", "instructions": f"Under the refund policy, order {subject} can still be refunded today."},
            "largest": {"type": "choice", "instructions": "Which order has the largest amount?", "criteria": {i: None for i in ids}},
            "pending": {"type": "score", "instructions": "How many orders are still pending?",
                        "criteria": ["none", "one", "two", "three or more"]},
            "delivered_total_over": {"type": "noul",
                                     "instructions": f"The delivered orders add up to more than ${threshold:,.0f}."},
        },
        labels={**{k: answers[k] for k in ("subject_refundable", "largest", "delivered_total_over")},
                "pending": min(answers["pending"], 3)},
    )


# ---- services -----------------------------------------------------------------

def services_answers(state: dict, subject: str) -> dict:
    limits = state["thresholds"]

    def breach(s):
        return s["error_rate"] > limits["error_rate"] or s["p95_ms"] > limits["p95_ms"]

    services = {s["name"]: s for s in state["services"]}
    return {
        "subject_breach": breach(services[subject]),
        "worst_error": max(state["services"], key=lambda s: s["error_rate"])["name"],
        "breaches": sum(1 for s in state["services"] if breach(s)),
    }


def _services(rng: np.random.Generator) -> Record:
    names = list(rng.choice(SERVICES, size=int(rng.integers(3, 7)), replace=False))
    services = [{"name": str(n), "error_rate": round(float(rng.choice([0.0, rng.uniform(0, 0.01), rng.uniform(0.01, 0.08)])), 4),
                 "p95_ms": int(rng.integers(40, 1600))} for n in names]
    rates = [s["error_rate"] for s in services]
    if len(set(rates)) != len(rates):
        return _services(rng)
    state = {"thresholds": {"error_rate": float(rng.choice([0.01, 0.02, 0.05])), "p95_ms": int(rng.choice([500, 800, 1200]))},
             "services": services}
    subject = str(rng.choice(names))
    answers = services_answers(state, subject)
    return Record(
        family="state_services",
        state=state,
        questions={
            "subject_breach": {"type": "noul", "instructions": f"The {subject} service is breaching at least one threshold."},
            "worst_error": {"type": "choice", "instructions": "Which service has the highest error rate?",
                            "criteria": {str(n): None for n in names}},
            "breaches": {"type": "score", "instructions": "How many services breach a threshold?",
                         "criteria": ["none", "one", "two", "three or more"]},
        },
        labels={"subject_breach": answers["subject_breach"], "worst_error": answers["worst_error"],
                "breaches": min(answers["breaches"], 3)},
    )


# ---- access requests ----------------------------------------------------------

ACCESS_POLICY = ("Internal resources: employees are permitted; contractors need manager approval. "
                 "Restricted resources: employees need both manager and security approval; contractors are always denied.")


def access_decision(role: str, classification: str, approvals: list[str]) -> str:
    if classification == "internal":
        return "permit" if role != "contractor" or "manager" in approvals else "needs_approval"
    if role == "contractor":
        return "deny"
    return "permit" if {"manager", "security"} <= set(approvals) else "needs_approval"


def _access(rng: np.random.Generator) -> Record:
    role = str(rng.choice(["engineer", "analyst", "contractor"]))
    classification = str(rng.choice(["internal", "restricted"]))
    approvals = [a for a in ("manager", "security") if rng.random() < 0.5]
    state = {"policy": ACCESS_POLICY,
             "request": {"requester": str(rng.choice(NAMES)), "role": role, "employment": "contractor" if role == "contractor" else "employee",
                         "resource": str(rng.choice(RESOURCES)), "classification": classification, "approvals": approvals}}
    return Record(
        family="state_access",
        state=state,
        questions={
            "decision": {"type": "choice", "instructions": "What does the policy say to do with this access request?",
                         "criteria": {"permit": "Grant access now", "needs_approval": "Hold until the missing approval arrives",
                                      "deny": "Refuse the request"}},
            "has_security": {"type": "noul", "instructions": "Security has approved this request."},
        },
        labels={"decision": access_decision(role, classification, approvals), "has_security": "security" in approvals},
    )


# ---- inventory ----------------------------------------------------------------

COVER_LEVELS = ["under a week", "one to two weeks", "two to four weeks", "more than four weeks"]


def _cover_level(days: float) -> int:
    return 0 if days < 7 else 1 if days < 14 else 2 if days < 28 else 3


def inventory_answers(state: dict, subject: str) -> dict:
    items = {i["sku"]: i for i in state["items"]}
    item = items[subject]
    return {
        "subject_reorder": item["on_hand"] <= item["reorder_point"],
        "lowest_cover": min(state["items"], key=lambda i: i["on_hand"] / i["daily_sales"])["sku"],
        "subject_cover_level": _cover_level(item["on_hand"] / item["daily_sales"]),
    }


def _inventory(rng: np.random.Generator) -> Record:
    skus = list(dict.fromkeys(f"SKU-{int(rng.integers(1000, 9999))}" for _ in range(int(rng.integers(3, 7)))))
    items = [{"sku": s, "on_hand": int(rng.integers(0, 600)), "daily_sales": int(rng.integers(1, 40)),
              "reorder_point": int(rng.integers(20, 200))} for s in skus]
    covers = [i["on_hand"] / i["daily_sales"] for i in items]
    if len(set(covers)) != len(covers):
        return _inventory(rng)
    state = {"items": items}
    subject = str(rng.choice(skus))
    answers = inventory_answers(state, subject)
    return Record(
        family="state_inventory",
        state=state,
        questions={
            "subject_reorder": {"type": "noul", "instructions": f"{subject} is at or below its reorder point."},
            "lowest_cover": {"type": "choice", "instructions": "Which item will run out soonest at its current daily sales?",
                             "criteria": {s: None for s in skus}},
            "subject_cover_level": {"type": "score", "instructions": f"How long will current stock of {subject} last?",
                                    "criteria": COVER_LEVELS},
        },
        labels=answers,
    )


GENERATORS = {"orders": _orders, "services": _services, "access": _access, "inventory": _inventory}


def generate(count: int, seed: int) -> list[Record]:
    rng = np.random.default_rng(seed)
    makers = list(GENERATORS.values())
    return [makers[i % len(makers)](rng) for i in range(count)]
