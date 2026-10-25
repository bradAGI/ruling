"""Learn from your own traffic: train on the decisions whose outcomes came back, and serve the result
only when it is better.

An adapters directory holds one version per run, `v1`, `v2`, …, and `current`,
a relative link to the version being served (`RULING_ADAPTER=<dir>/current`).
A run exports the ledger's resolved decisions, drops any whose state is in a
held-out set, trains the next version (from `current`'s weights when there is
one), scores both versions on the held-out sets, and repoints `current` only
when the new one has lower log loss without losing accuracy overall or on any
one question. Each version is
scored at its own fitted temperature: training sharpens an adapter faster than
it corrects it, and a version that is right more often must not lose for being
overconfident. The candidate's temperatures are saved beside its weights, so
`RULING_CALIBRATION=<dir>/current/calibration.json` serves the promoted
version calibrated. Every run leaves its records, its training log and the
comparison in its version directory, promoted or not.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from ruling.calibration import Calibration, Provenance
from ruling.config import Settings
from ruling.engine import ADAPTER_CONFIG, ADAPTER_WEIGHTS
from ruling.evals import Observation, Record, collect, decision_blocks, fit, load_dataset, target_of
from ruling.ledger import DecisionLedger
from ruling.metrics import credit
from ruling.registry import CALIBRATION_FILE, CURRENT, PREVIEW, build_engine
from ruling.train import TRAINING, Trainer, TrainConfig, _state_key

REPORT = "retrain.json"
CALIBRATION = CALIBRATION_FILE
DECISIONS = "decisions.jsonl"
VERSION = re.compile(r"v(\d+)")
SIGNIFICANCE = 0.05
"""How unlikely by chance a question's losses must be before the candidate counts as worse on it."""


def should_promote(current: dict, candidate: dict, worse: list[str]) -> tuple[bool, str]:
    """Held-out log loss must fall and accuracy must not: a sharper but less often right adapter is not better.
    Nor is one that is worse on any question in `worse`, however much it gains elsewhere."""
    if worse:
        return False, f"held-out answers got worse on {', '.join(worse)}"
    if candidate["nll"] >= current["nll"]:
        return False, f"held-out log loss {candidate['nll']:.4f} is not below {current['nll']:.4f}"
    if candidate["accuracy"] < current["accuracy"]:
        return False, f"held-out accuracy {candidate['accuracy']:.4f} fell below {current['accuracy']:.4f}"
    return True, (f"held-out log loss {current['nll']:.4f} -> {candidate['nll']:.4f}, "
                  f"accuracy {current['accuracy']:.4f} -> {candidate['accuracy']:.4f}")


def worse_questions(current: list[Observation], candidate: list[Observation]) -> list[str]:
    """The questions the candidate gets right less often than the served version beyond chance: on the held-out
    rows where the two are not equally right, an exact two-sided sign test (McNemar's) of losses against wins."""
    served = {(o.record, o.question_id): o for o in current}
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for new in candidate:
        now = served[(new.record, new.question_id)]
        if new.label is None or new.logits is None or now.logits is None:
            continue
        before, after = credit(now.logits, target_of(now)), credit(new.logits, target_of(new))
        if before != after:
            tally[new.question_id][after > before] += 1
    return sorted(q for q, (lost, won) in tally.items() if lost > won and _sign_test(lost, lost + won) < SIGNIFICANCE)


def _sign_test(larger: int, total: int) -> float:
    """The two-sided probability of a split at least this uneven when each side is equally likely."""
    return min(1.0, 2 * sum(math.comb(total, k) for k in range(larger, total + 1)) / 2 ** total)


def score(observations: list[Observation], provenance: Provenance) -> tuple[dict, Calibration]:
    """Held-out metrics at the temperatures that fit these observations best, and that calibration.

    A temperature never changes which answer wins, so accuracy is the raw model's; log loss
    becomes a measure of how well the model ranks answers, not of how sharply it states them."""
    calibration = fit(observations, provenance)
    return decision_blocks(observations, calibration)["overall"], calibration


def next_version(adapters: Path) -> Path:
    numbers = [int(m.group(1)) for p in adapters.iterdir() if (m := VERSION.fullmatch(p.name))] if adapters.exists() else []
    return adapters / f"v{max(numbers, default=0) + 1}"


def point(adapters: Path, link: str, version: Path) -> None:
    """Point `link` at `version` in one rename, so a server starting meanwhile sees the old or the new, never neither."""
    staged = adapters / f".{link}.next"
    staged.unlink(missing_ok=True)
    staged.symlink_to(version.name)
    os.replace(staged, adapters / link)


def promote(adapters: Path, version: Path) -> None:
    point(adapters, CURRENT, version)


def promote_version(adapters: Path, name: str) -> Path:
    """Serve the trained version `name` (a rollback when it is older), refusing anything that is not one."""
    version = adapters / name
    if not VERSION.fullmatch(name) or not (version / ADAPTER_WEIGHTS).exists():
        trained = sorted((p.name for p in adapters.iterdir() if VERSION.fullmatch(p.name) and (p / ADAPTER_WEIGHTS).exists()),
                         key=lambda n: int(n[1:])) if adapters.exists() else []
        raise ValueError(f"{name!r} is not a trained version in {adapters}; trained: {', '.join(trained) or 'none'}")
    promote(adapters, version)
    return version


def trainable(records: list[Record], held_out: list[Record]) -> tuple[list[Record], int]:
    """The records whose states appear in no held-out set, and how many were dropped for appearing in one."""
    held = {_state_key(r.state) for r in held_out}
    kept = [r for r in records if _state_key(r.state) not in held]
    return kept, len(records) - len(kept)


def retrain(ledger: DecisionLedger, adapters: Path, model: str, holdout: list[Path], settings: Settings,
            min_records: int, train_options: dict, log=print, auto_promote: bool = True) -> dict | None:
    """One retraining run; returns its report, or None when too few decisions have outcomes yet. The new version
    becomes the preview either way; it becomes current when it is better and `auto_promote` is on."""
    held_out = [r for path in holdout for r in load_dataset(path)]
    records, dropped = trainable(list(ledger.records()), held_out)
    if dropped:
        log(f"dropped {dropped} decisions whose states are in the held-out sets")
    if len(records) < min_records:
        log(f"{len(records)} decisions have outcomes; waiting for {min_records}")
        return None

    current = adapters / CURRENT
    start = current.resolve() if current.exists() else None
    version = next_version(adapters)
    version.mkdir(parents=True)
    data = version / DECISIONS
    data.write_text("".join(r.model_dump_json(exclude_none=True) + "\n" for r in records))
    Trainer(TrainConfig(model=model, data=[data], output=version, holdout=holdout, from_adapter=start,
                        max_image_tokens=settings.max_image_tokens, **_lora_layout(start), **train_options)).run(log)
    # The trainer's copy of the model is unreachable now; return its memory before loading the scorer's.
    mx.clear_cache()

    served = build_engine(replace(settings, model=model, adapter_path=start, calibration_path=None, cascade_to=None,
                                  failover_to=None))
    candidate = served.variant(version, Calibration())
    observed = {name: collect(engine, held_out) for name, engine in (("current", served), ("candidate", candidate))}
    scored = {name: score(observations, engine.provenance)
              for (name, observations), engine in zip(observed.items(), (served, candidate))}
    scored["candidate"][1].save(version / CALIBRATION)
    scores = {name: {**metrics, "temperatures": calibration.temperatures}
              for name, (metrics, calibration) in scored.items()}
    worse = worse_questions(observed["current"], observed["candidate"])
    better, reason = should_promote(scores["current"], scores["candidate"], worse)
    promoted = better and auto_promote
    if better and not auto_promote:
        reason += f"; held as the preview: `ruling promote {version.name}` serves it"
    report = {"version": version.name, "started_from": start.name if start else None, "records": len(records),
              "dropped_held_out": dropped, "holdout": [str(p) for p in holdout], **scores, "worse": worse,
              "better": better, "promoted": promoted, "reason": reason}
    (version / REPORT).write_text(json.dumps(report, indent=2) + "\n")
    point(adapters, PREVIEW, version)
    if promoted:
        promote(adapters, version)
    log(f"{version.name}: {'promoted' if promoted else 'kept ' + (start.name if start else 'the base model')}; {reason}")
    return report


def history(adapters: Path) -> dict:
    """Every finished run's report in version order, each with its training curve, and the versions `current`
    and `preview` point at.

    A version still training, or one whose run failed, has no report and is left out."""
    links = {name: adapters / name for name in (CURRENT, PREVIEW)}
    reports = [_report(version)
               for version in sorted((p for p in adapters.iterdir() if VERSION.fullmatch(p.name)),
                                     key=lambda p: int(p.name[1:]))
               if (version / REPORT).exists()] if adapters.exists() else []
    return {**{name: link.resolve().name if link.exists() else None for name, link in links.items()},
            "versions": reports}


def _report(version: Path) -> dict:
    """The version's report, with the history its trainer logged under `training` when the log is there."""
    report = json.loads((version / REPORT).read_text())
    if (version / TRAINING).exists():
        report["training"] = json.loads((version / TRAINING).read_text())["history"]
    return report


def _lora_layout(adapter: Path | None) -> dict:
    """The LoRA settings to train with: the starting adapter's, which its weights only fit, else the trainer's defaults."""
    if adapter is None:
        return {}
    saved = json.loads((adapter / ADAPTER_CONFIG).read_text())
    lora = saved["lora_parameters"]
    return {"lora_layers": saved["num_layers"], "lora_rank": lora["rank"], "lora_scale": lora["scale"],
            "lora_dropout": lora["dropout"]}
