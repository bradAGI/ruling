"""Print one model's held-out and JevBench numbers from the files assets/bench/score.sh wrote.

    uv run python assets/bench/summary.py <name> [<name> ...]
"""
import json
import sys
from pathlib import Path

from ruling.calibration import coverage_at_error
from ruling.evals import ERROR_BUDGET

# The scripts are tracked; what they write is not, so the results live under runs/ like every other run.
BENCH = Path(__file__).resolve().parents[2] / "runs" / "bench"
SETS = ["typesafe102", "every", "authored144", "perturbations108"]
TIERS = ["easy", "original", "hard"]


def heldout(name: str, dataset: str) -> dict:
    report = json.loads((BENCH / f"{dataset}-{name}-r1.json").read_text())
    types = {t: s for t, s in report["question_types"].items() if s.get("labeled_count")}
    n = sum(s["labeled_count"] for s in types.values())
    correct = sum(round(s["accuracy"] * s["labeled_count"]) for s in types.values())
    return {"n": n, "correct": correct, "accuracy": correct / n, "coverage": pooled_coverage(BENCH / f"{dataset}-{name}-r1.predictions.jsonl"),
            "ece": {t: s["expected_calibration_error"] for t, s in types.items()},
            "flips": report["question_types"].get("choice", {}).get("option_reversal_flips"),
            "p50_ms": report["latency_ms"]["p50"], "p95_ms": report["latency_ms"]["p95"]}


def pooled_coverage(predictions: Path) -> float:
    """Coverage at the error budget over every labeled question of the set, all types pooled, as the README reports it."""
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    rows = [r for r in rows if r.get("label") is not None]
    return coverage_at_error([max(r["probabilities"]) for r in rows],
                             [max(range(len(r["probabilities"])), key=r["probabilities"].__getitem__) == r["label"] for r in rows],
                             ERROR_BUDGET)


def jevbench(name: str, tier: str) -> dict:
    summary = json.loads((BENCH / f"jevbench-{name}" / f"{tier}.summary.json").read_text())
    return {k: summary[k] for k in ("n_planned", "n_attempted", "n_scorable", "n_correct", "accuracy", "ece", "operational_success")}


for name in sys.argv[1:]:
    print(f"## {name}")
    for dataset in SETS:
        if (BENCH / f"{dataset}-{name}-r1.json").exists():
            h = heldout(name, dataset)
            print(f"{dataset:18s} acc {h['accuracy']:.3f} ({h['correct']}/{h['n']})  coverage@5% {h['coverage']:.3f}  "
                  f"ece {', '.join(f'{t} {v:.3f}' for t, v in h['ece'].items())}  reversal flips {h['flips']}  "
                  f"p50 {h['p50_ms']:.0f} ms p95 {h['p95_ms']:.0f} ms")
    tiers = {t: jevbench(name, t) for t in TIERS if (BENCH / f"jevbench-{name}" / f"{t}.summary.json").exists()}
    if tiers:
        total = sum(t["n_correct"] for t in tiers.values()); planned = sum(t["n_planned"] for t in tiers.values())
        print(f"jevbench all {total}/{planned} = {total / planned:.3f}  " + "  ".join(
            f"{t} {v['n_correct']}/{v['n_planned']} = {v['accuracy']:.3f} (ok {v['operational_success']})"
            for t, v in tiers.items()))
