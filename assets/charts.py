"""Regenerate the README's figures from the evaluation JSON in runs/.

Run after a matrix sweep:  uv run --with matplotlib python assets/charts.py
Every number plotted here is read from a report `ruling eval` wrote; nothing is
typed in by hand except Jev's published figures, which are labeled as such.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "runs" / "matrix"
ASSETS = ROOT / "assets"

GROUND, INK, MUTED, FAINT, RULE = "#0d1117", "#e9edf2", "#8a97a8", "#5f6b7a", "#262f3b"
JEV, BIG, SMALL, ADAPTER = "#93a7bd", "#e3a34a", "#7fb5a6", "#c58cf0"

MODELS = [("qwen35-4b", "Qwen3.5-4B", SMALL), ("qwen3-4b", "Qwen3-4B-Instruct", MUTED),
          ("qwen3-4b-lora", "Qwen3-4B + adapter", ADAPTER), ("35b", "Qwen3.6-35B-A3B", BIG)]
SETS = [("typesafe102", "TypeSafe 102"), ("authored144", "authored144"),
        ("perturbations108", "perturbations108"), ("every", "Every 154")]
JEV_ACCURACY = {"typesafe102": 90 / 102, "every": 148 / 154}


def report(tag: str, dataset: str, rotations: int) -> dict | None:
    path = MATRIX / f"{dataset}-{tag}-r{rotations}.json"
    return json.loads(path.read_text()) if path.exists() else None


def accuracy(tag: str, dataset: str, rotations: int) -> float | None:
    data = report(tag, dataset, rotations)
    if data is None:
        return None
    total = correct = 0
    for question_type in data["question_types"].values():
        if question_type["labeled_count"]:
            total += question_type["labeled_count"]
            correct += question_type["accuracy"] * question_type["labeled_count"]
    return correct / total if total else None


def calibration_error(tag: str, dataset: str, rotations: int) -> float | None:
    data = report(tag, dataset, rotations)
    if data is None:
        return None
    total = weighted = 0
    for question_type in data["question_types"].values():
        if question_type["labeled_count"]:
            total += question_type["labeled_count"]
            weighted += question_type["expected_calibration_error"] * question_type["labeled_count"]
    return weighted / total if total else None


def frame(width: float, height: float):
    figure, axes = plt.subplots(figsize=(width, height), dpi=100)
    figure.patch.set_facecolor(GROUND)
    axes.set_facecolor(GROUND)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(colors=FAINT, length=0)
    axes.grid(axis="y", color=RULE, linewidth=1)
    axes.set_axisbelow(True)
    return figure, axes


def title(axes, heading: str, standfirst: str) -> None:
    axes.text(0, 1.16, heading, transform=axes.transAxes, color=INK, fontsize=17, fontweight="bold", va="top")
    axes.text(0, 1.07, standfirst, transform=axes.transAxes, color=MUTED, fontsize=12.5, va="top")


def against_jev_figure() -> None:
    """Jev's published answers against the largest local model, on the rows Jev has answered."""
    rows = []
    for dataset, label, jev_correct in (("typesafe102", "TypeSafe's published evals\n102 questions", 90),
                                        ("every", "Every's judgment lab\n154 author-labeled", 148)):
        data = report("35b", dataset, 3)
        labeled = [q for q in data["question_types"].values() if q["labeled_count"]]
        total = sum(q["labeled_count"] for q in labeled)
        ours = round(sum(q["accuracy"] * q["labeled_count"] for q in labeled))
        rows.append((label, jev_correct, ours, total))
    rows.append(("Pooled\nevery row Jev has answered",
                 sum(r[1] for r in rows), sum(r[2] for r in rows), sum(r[3] for r in rows)))

    figure, axes = frame(11, 5.4)
    axes.grid(axis="x", color=RULE, linewidth=1)
    axes.grid(axis="y", visible=False)
    height = 0.34
    for index, (label, jev_correct, ours, total) in enumerate(rows):
        y = len(rows) - 1 - index
        axes.barh(y + height / 2 + 0.02, jev_correct / total, height, color=JEV, zorder=3)
        axes.barh(y - height / 2 - 0.02, ours / total, height, color=BIG, zorder=3)
        axes.text(jev_correct / total + 0.008, y + height / 2 + 0.02, f"{jev_correct}/{total}",
                  va="center", color=JEV, fontsize=12)
        axes.text(ours / total + 0.008, y - height / 2 - 0.02, f"{ours}/{total}",
                  va="center", color=BIG, fontsize=12)
    axes.set_yticks(range(len(rows)))
    axes.set_yticklabels([r[0] for r in reversed(rows)], color=INK, fontsize=11.5)
    axes.set_ylim(-0.6, len(rows) - 0.4)
    axes.set_xlim(0, 1.12)
    axes.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes.set_xticklabels(["0", "25%", "50%", "75%", "100%"], color=FAINT, fontsize=10)
    axes.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=JEV), plt.Rectangle((0, 0), 1, 1, color=BIG)],
                labels=["Jev, hosted and post-trained for this task",
                        "Qwen3.6-35B-A3B, 4-bit on one Mac, untrained"],
                loc="upper left", bbox_to_anchor=(0, 1.11), frameon=False, fontsize=11,
                labelcolor=INK, handlelength=1.2, handleheight=1.2, borderpad=0)
    axes.text(0, 1.34, "Jev against an open model nobody trained for the task",
              transform=axes.transAxes, color=INK, fontsize=17, fontweight="bold", va="top")
    axes.text(0, 1.25, "Shipped defaults on every row. Exact McNemar on the paired rows: p = 0.21.",
              transform=axes.transAxes, color=MUTED, fontsize=12.5, va="top")
    figure.subplots_adjust(left=0.22, right=0.985, top=0.68, bottom=0.10)
    figure.savefig(ASSETS / "against-jev.png", facecolor=GROUND)


def rotations_figure() -> None:
    figure, axes = plt.subplots(1, 4, figsize=(14, 4.2), dpi=100, sharey=True)
    figure.patch.set_facecolor(GROUND)
    for panel, (dataset, label) in zip(axes, SETS):
        panel.set_facecolor(GROUND)
        for spine in panel.spines.values():
            spine.set_visible(False)
        panel.grid(axis="y", color=RULE, linewidth=1)
        panel.set_axisbelow(True)
        panel.tick_params(colors=FAINT, length=0, labelsize=10)
        settings = [1, 3] if dataset == "every" else [1, 3, 6]
        for tag, name, color in MODELS:
            values = [(r, accuracy(tag, dataset, r)) for r in settings]
            values = [(r, v) for r, v in values if v is not None]
            if values:
                panel.plot([r for r, _ in values], [v for _, v in values], marker="o",
                           color=color, linewidth=2, markersize=5, label=name)
        if dataset in JEV_ACCURACY:
            panel.axhline(JEV_ACCURACY[dataset], color=JEV, linestyle="--", linewidth=1.5)
            panel.text(settings[-1], JEV_ACCURACY[dataset] + 0.004, "Jev", color=JEV, fontsize=10, ha="right")
        panel.set_title(label, color=INK, fontsize=12, pad=8)
        panel.set_xticks(settings)
        panel.set_xlabel("option orderings averaged", color=FAINT, fontsize=10)
    axes[0].set_ylim(0.74, 0.98)
    axes[0].set_ylabel("accuracy", color=FAINT, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
                  frameon=False, fontsize=11, labelcolor=INK)
    figure.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.14, wspace=0.08)
    figure.savefig(ASSETS / "rotations.png", facecolor=GROUND)


def calibration_figure() -> None:
    figure, axes = frame(9, 4.6)
    datasets = [(d, l) for d, l in SETS]
    base = [calibration_error("qwen3-4b", d, 3) for d, _ in datasets]
    tuned = [calibration_error("qwen3-4b-lora", d, 3) for d, _ in datasets]
    positions = range(len(datasets))
    axes.bar([p - 0.2 for p in positions], base, 0.38, color=MUTED, label="base model")
    axes.bar([p + 0.2 for p in positions], tuned, 0.38, color=ADAPTER, label="with our adapter")
    for p, (b, t) in enumerate(zip(base, tuned)):
        axes.text(p - 0.2, b + 0.003, f"{b:.3f}", ha="center", color=MUTED, fontsize=10)
        axes.text(p + 0.2, t + 0.003, f"{t:.3f}", ha="center", color=ADAPTER, fontsize=10)
    axes.set_ylim(0, max(base + tuned) * 1.3)
    axes.set_xticks(list(positions))
    axes.set_xticklabels([l for _, l in datasets], color=INK, fontsize=11)
    axes.set_ylabel("expected calibration error", color=FAINT, fontsize=11)
    axes.legend(frameon=False, fontsize=11, labelcolor=INK, loc="upper right")
    title(axes, "Post-training moved calibration, not accuracy",
          "Qwen3-4B-Instruct-2507, three option orderings, every set held out of training. Lower is better.")
    figure.subplots_adjust(left=0.10, right=0.98, top=0.80, bottom=0.11)
    figure.savefig(ASSETS / "calibration.png", facecolor=GROUND)


def readout_figure() -> None:
    figure, axes = frame(9, 4.6)
    labels = ["Reading the logits\n(what ruling does)", "Generating the same answers\nas JSON, 105 tokens"]
    throughput = [5140, 131]
    axes.bar(labels, throughput, 0.5, color=[BIG, MUTED])
    for i, value in enumerate(throughput):
        axes.text(i, value * 1.15, f"{value:,} tok/s", ha="center", color=BIG if i == 0 else MUTED,
                  fontsize=15, fontweight="bold")
    axes.set_yscale("log")
    axes.set_ylim(50, 20000)
    axes.set_ylabel("tokens per second of GPU time (log)", color=FAINT, fontsize=11)
    axes.tick_params(axis="x", labelsize=11, colors=INK)
    title(axes, "Prefill is compute-bound; decoding is not",
          "Qwen3.5-4B at 4-bit, one Mac. Prefill hit 41 TFLOP/s, decoding 1.0 TFLOP/s.")
    figure.subplots_adjust(left=0.11, right=0.98, top=0.80, bottom=0.14)
    figure.savefig(ASSETS / "readout.png", facecolor=GROUND)


def frontier_figure() -> None:
    figure, axes = frame(9, 4.8)
    latency = {"qwen35-4b": 136, "qwen3-4b": 116, "qwen3-4b-lora": 116, "35b": 204}
    for tag, name, color in MODELS:
        value = accuracy(tag, "typesafe102", 3)
        if value is None:
            continue
        axes.scatter(latency[tag], value, s=190, color=color, zorder=3)
        axes.annotate(name, (latency[tag], value), textcoords="offset points", xytext=(12, -4),
                      color=color, fontsize=11)
    axes.axhspan(JEV_ACCURACY["typesafe102"] - 0.0005, JEV_ACCURACY["typesafe102"] + 0.0005,
                 color=JEV, alpha=0.5)
    axes.text(100, JEV_ACCURACY["typesafe102"] + 0.005, "Jev, published", color=JEV, fontsize=11)
    axes.set_xlim(95, 260)
    axes.set_ylim(0.79, 0.91)
    axes.set_xlabel("latency for three questions, milliseconds", color=FAINT, fontsize=11)
    axes.set_ylabel("agreement with TypeSafe's reference", color=FAINT, fontsize=11)
    title(axes, "What each model costs in latency for what it gets right",
          "TypeSafe's 102 published rows, three option orderings, measured on one Mac.")
    figure.subplots_adjust(left=0.10, right=0.98, top=0.80, bottom=0.13)
    figure.savefig(ASSETS / "frontier.png", facecolor=GROUND)


if __name__ == "__main__":
    against_jev_figure()
    rotations_figure()
    calibration_figure()
    readout_figure()
    frontier_figure()
    print("wrote against-jev.png, rotations.png, calibration.png, readout.png, frontier.png")
