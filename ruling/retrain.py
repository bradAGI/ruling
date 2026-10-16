"""Learn from your own traffic: train on the decisions whose outcomes came back, and serve the result
only when it is better.

An adapters directory holds one version per run, `v1`, `v2`, …, and `current`,
a relative link to the version being served (`RULING_ADAPTER=<dir>/current`).
A run exports the ledger's resolved decisions, drops any whose state is in a
held-out set, trains the next version (from `current`'s weights when there is
one), scores both versions on the held-out sets, and repoints `current` only
when the new one has lower log loss without losing accuracy. Every run leaves
its records, its training log and the comparison in its version directory,
promoted or not.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import mlx.core as mx

from ruling.calibration import Calibration
from ruling.config import Settings
from ruling.engine import ADAPTER_CONFIG, Engine
from ruling.evals import Record, collect, decision_blocks, load_dataset
from ruling.ledger import DecisionLedger
from ruling.train import Trainer, TrainConfig, _state_key

CURRENT = "current"
REPORT = "retrain.json"
DECISIONS = "decisions.jsonl"
VERSION = re.compile(r"v(\d+)")


def should_promote(current: dict, candidate: dict) -> tuple[bool, str]:
    """Held-out log loss must fall and accuracy must not: a sharper but less often right adapter is not better."""
    if candidate["nll"] >= current["nll"]:
        return False, f"held-out log loss {candidate['nll']:.4f} is not below {current['nll']:.4f}"
    if candidate["accuracy"] < current["accuracy"]:
        return False, f"held-out accuracy {candidate['accuracy']:.4f} fell below {current['accuracy']:.4f}"
    return True, (f"held-out log loss {current['nll']:.4f} -> {candidate['nll']:.4f}, "
                  f"accuracy {current['accuracy']:.4f} -> {candidate['accuracy']:.4f}")


def next_version(adapters: Path) -> Path:
    numbers = [int(m.group(1)) for p in adapters.iterdir() if (m := VERSION.fullmatch(p.name))] if adapters.exists() else []
    return adapters / f"v{max(numbers, default=0) + 1}"


def promote(adapters: Path, version: Path) -> None:
    """Point `current` at `version` in one rename, so a server starting meanwhile sees the old or the new, never neither."""
    staged = adapters / f".{CURRENT}.next"
    staged.unlink(missing_ok=True)
    staged.symlink_to(version.name)
    os.replace(staged, adapters / CURRENT)


def trainable(records: list[Record], held_out: list[Record]) -> tuple[list[Record], int]:
    """The records whose states appear in no held-out set, and how many were dropped for appearing in one."""
    held = {_state_key(r.state) for r in held_out}
    kept = [r for r in records if _state_key(r.state) not in held]
    return kept, len(records) - len(kept)


def retrain(ledger: DecisionLedger, adapters: Path, model: str, holdout: list[Path], settings: Settings,
            min_records: int, train_options: dict, log=print) -> dict | None:
    """One retraining run; returns its report, or None when too few decisions have outcomes yet."""
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
                        **_lora_layout(start), **train_options)).run(log)
    # The trainer's copy of the model is unreachable now; return its memory before loading the scorer's.
    mx.clear_cache()

    served = Engine(model, Calibration(), settings.max_input_tokens, settings.rotations, settings.prior_debias,
                    start, settings.max_branch_tokens)
    candidate = served.variant(version, Calibration())
    scores = {name: decision_blocks(collect(engine, held_out), Calibration())["overall"]
              for name, engine in (("current", served), ("candidate", candidate))}
    promoted, reason = should_promote(scores["current"], scores["candidate"])
    report = {"version": version.name, "started_from": start.name if start else None, "records": len(records),
              "dropped_held_out": dropped, "holdout": [str(p) for p in holdout], **scores,
              "promoted": promoted, "reason": reason}
    (version / REPORT).write_text(json.dumps(report, indent=2) + "\n")
    if promoted:
        promote(adapters, version)
    log(f"{version.name}: {'promoted' if promoted else 'kept ' + (start.name if start else 'the base model')}; {reason}")
    return report


def _lora_layout(adapter: Path | None) -> dict:
    """The LoRA settings to train with: the starting adapter's, which its weights only fit, else the trainer's defaults."""
    if adapter is None:
        return {}
    saved = json.loads((adapter / ADAPTER_CONFIG).read_text())
    lora = saved["lora_parameters"]
    return {"lora_layers": saved["num_layers"], "lora_rank": lora["rank"], "lora_scale": lora["scale"],
            "lora_dropout": lora["dropout"]}
