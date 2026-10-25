"""Decision metrics in the form hosted decision platforms report, so a ruling report reads like theirs.

Inputs are parallel lists of probability vectors and target vectors over the
same keys; a target may be one-hot or a soft distribution, whose most likely
option counts as the correct answer for accuracy.
"""

from __future__ import annotations

import numpy as np

THRESHOLDS = (0.5, 0.7, 0.9, 0.95)
"""Top-1 confidence thresholds for selective accuracy. Reported under their string form; the names never change."""

PROBABILITY_FLOOR = 1e-12
"""Log loss is taken on probabilities floored here, so an impossible answer costs a finite amount."""


def credit(p: np.ndarray, y: np.ndarray) -> float:
    """How right one answer is: the share of the options tied for p's top that are tied for y's top."""
    return float(np.isin(np.flatnonzero(p == p.max()), np.flatnonzero(y == y.max())).mean())


def decision_metrics(probs: list[np.ndarray], targets: list[np.ndarray], bins: int = 10) -> dict:
    """Pooled metrics over paired distributions and targets; `{"n": 0}` for empty input.

    - accuracy: argmax(p) equals argmax(y); when k options tie for p's top, each counts 1/k, so a decision
      scores |top(p) ∩ top(y)| / k and no metric depends on the order the options are listed in
    - nll: −Σ y log p, averaged
    - brier: Σ (p − y)², averaged
    - ece: expected calibration error over `bins` equal-width bins on top-1 confidence
    - mean_confidence: average top-1 probability
    - selective: coverage and accuracy among decisions whose top-1 confidence is at least each threshold
    """
    if len(probs) != len(targets):
        raise ValueError("probs and targets must have the same length")
    n = len(probs)
    if n == 0:
        return {"n": 0}
    confidence, correct, nll, brier = (np.empty(n) for _ in range(4))
    for i, (p, y) in enumerate(zip(probs, targets)):
        p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if p.shape != y.shape:
            raise ValueError(f"decision {i}: probabilities have {p.size} entries, target has {y.size}")
        confidence[i] = p.max()
        correct[i] = credit(p, y)
        nll[i] = -(y * np.log(np.maximum(p, PROBABILITY_FLOOR))).sum()
        brier[i] = ((p - y) ** 2).sum()
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        # A confidence of exactly zero belongs to the first bin rather than to none.
        member = ((lo < confidence) & (confidence <= hi)) | ((b == 0) & (confidence == 0))
        if member.any():
            ece += member.mean() * abs(confidence[member].mean() - correct[member].mean())
    selective = {}
    for threshold in THRESHOLDS:
        covered = confidence >= threshold
        selective[str(threshold)] = {
            "coverage": float(covered.mean()),
            "accuracy": float(correct[covered].mean()) if covered.any() else None,
        }
    return {
        "n": n,
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece": float(ece),
        "mean_confidence": float(confidence.mean()),
        "selective": selective,
    }
