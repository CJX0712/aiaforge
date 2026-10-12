"""Metrics for attribute-inference evaluation.

Author: 晨星 (CJX0712)
"""

from collections.abc import Sequence

import numpy as np


def auc_score(y_true, scores) -> float:
    """ROC-AUC for binary S. Monotonic in ``scores``; NaN if only one class."""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, scores))
    except Exception:
        return _manual_auc(y_true, scores)


def _manual_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = float(ranks[y_true == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def accuracy_score(y_true, scores) -> float:
    """Accuracy at the 0.5 decision threshold (classes are balanced)."""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    pred = (scores >= 0.5).astype(int)
    return float((pred == y_true).mean())


def mean_std(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def significant(mean_a: float, std_a: float, mean_b: float, std_b: float) -> bool:
    """SOP significance gate: mean difference exceeds half the sum of std devs."""
    return (mean_a - mean_b) > 0.5 * (std_a + std_b)


def aggregate(per_seed: list[dict], methods: list[str], metric: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for m in methods:
        key = f"{m}_{metric}"
        vals = [row[key] for row in per_seed if key in row and not _isnan(row[key])]
        if vals:
            out[m] = mean_std(vals)
    return out


def _isnan(x) -> bool:
    return isinstance(x, float) and x != x
