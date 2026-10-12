"""AiaForge benchmark pipeline.

Single deterministic entry point: generate -> train target -> fit attacks ->
evaluate on the held-out test split -> aggregate across seeds.

Invariants (anti-leakage, checked by tests):
* the target model sees only the *train* partition;
* attack models are trained on the *aux* partition;
* all metrics are computed on the *test* partition;
* the three partitions are disjoint.

Author: 晨星 (CJX0712)
"""

import numpy as np

from aia.methods import all_attacks
from aia.target import build_target
from core.config import Config
from core.seed import set_all
from core.types import AiaDataset, BenchmarkReport
from data.dgp import generate_split
from eval.metrics import accuracy_score, aggregate, auc_score, significant

METHOD_NAMES = ["majority", "label_only", "conf_inversion", "attack_lr", "attack_nn", "aiafuse"]


def run_one_seed(cfg: Config, seed: int):
    """Execute one fully-independent seed. Returns (row, split)."""
    set_all(seed)
    split = generate_split(cfg, seed)
    target = build_target(cfg.target_model, seed)
    Xtr = np.column_stack([split.train.X_pub, split.train.S.astype(float)])
    target.fit(Xtr, split.train.Y)

    methods = all_attacks()
    scores: dict[str, np.ndarray] = {}
    row: dict = {"seed": seed}
    skipped: list[str] = []
    for m in methods:
        try:
            m.fit(split.aux, target)
            sc = m.attack(split.test.X_pub, split.test.Y, target)
        except Exception as exc:
            skipped.append(f"{m.name}:{type(exc).__name__}")
            continue
        scores[m.name] = np.asarray(sc, dtype=float)
        row[f"{m.name}_auc"] = auc_score(split.test.S, scores[m.name])
        row[f"{m.name}_acc"] = accuracy_score(split.test.S, scores[m.name])
    return row, split, skipped


def _failure_analysis(cfg: Config, split: AiaDataset, scores: dict[str, np.ndarray]) -> list[dict]:
    """Derive >=3 typical failure cases from the last seed's results."""
    S = split.test.S
    fails: list[dict] = []
    fuse = scores.get("aiafuse")
    if fuse is None:
        return fails
    hard_pos = np.where((S == 1) & (fuse < 0.5))[0]
    hard_neg = np.where((S == 0) & (fuse >= 0.5))[0]
    for idx in list(hard_pos[:3]):
        fails.append(
            {
                "case": "false_negative",
                "index": int(idx),
                "true_S": int(S[idx]),
                "fuse_score": float(fuse[idx]),
                "note": "sensitive attribute S=1 but fusion predicted 0 (low model-leakage signal)",
            }
        )
    for idx in list(hard_neg[:3]):
        fails.append(
            {
                "case": "false_positive",
                "index": int(idx),
                "true_S": int(S[idx]),
                "fuse_score": float(fuse[idx]),
                "note": (
                    "sensitive attribute S=0 but fusion predicted 1 (confidence inversion misled)"
                ),
            }
        )
    return fails


def run_benchmark(cfg: Config, seeds: list[int] | None = None) -> BenchmarkReport:
    if seeds is None:
        seeds = [cfg.seed + k * 101 for k in range(cfg.n_seeds)]

    per_seed: list[dict] = []
    skipped_all: list[str] = []
    last_split: AiaDataset | None = None
    last_scores: dict[str, np.ndarray] = {}

    for s in seeds:
        row, split, skipped = run_one_seed(cfg, s)
        per_seed.append(row)
        if skipped:
            skipped_all.extend(skipped)
        last_split = split

    # recompute last_scores for failure analysis from a fresh run is avoided;
    # instead keep scores from the final seed by re-running cheaply.
    if last_split is not None:
        set_all(seeds[-1])
        target = build_target(cfg.target_model, seeds[-1])
        Xtr = np.column_stack([last_split.train.X_pub, last_split.train.S.astype(float)])
        target.fit(Xtr, last_split.train.Y)
        for m in all_attacks():
            try:
                m.fit(last_split.aux, target)
                last_scores[m.name] = np.asarray(
                    m.attack(last_split.test.X_pub, last_split.test.Y, target), dtype=float
                )
            except Exception:
                continue

    agg_auc = aggregate(per_seed, METHOD_NAMES, "auc")
    agg_acc = aggregate(per_seed, METHOD_NAMES, "acc")

    verdict: dict = {}
    if "aiafuse" in agg_auc and "majority" in agg_auc:
        a = agg_auc["aiafuse"]
        b = agg_auc["majority"]
        diff = a["mean"] - b["mean"]
        sig = significant(a["mean"], a["std"], b["mean"], b["std"])
        verdict["aiafuse_vs_majority"] = {
            "diff": diff,
            "significant": sig,
            "gate_threshold": cfg.threshold_auc_lift,
            "pass": bool(diff >= cfg.threshold_auc_lift and sig),
        }
    # non-inferiority vs strongest single attack
    singles = {k: v for k, v in agg_auc.items() if k not in ("majority", "aiafuse")}
    if singles and "aiafuse" in agg_auc:
        best_single = max(singles, key=lambda k: singles[k]["mean"])
        a = agg_auc["aiafuse"]
        bs = singles[best_single]
        verdict["aiafuse_vs_best_single"] = {
            "best_single": best_single,
            "diff": a["mean"] - bs["mean"],
            "non_inferior": bool(a["mean"] >= bs["mean"] - 0.5 * (a["std"] + bs["std"])),
        }

    failures = _failure_analysis(cfg, last_split, last_scores) if last_split else []

    report = BenchmarkReport(
        config=cfg.as_dict(),
        per_seed=per_seed,
        aggregate={"auc": agg_auc, "accuracy": agg_acc, "verdict": verdict},
        skipped=sorted(set(skipped_all)),
        failures=failures,
    )
    return report


class Pipeline:
    """Thin OO wrapper matching the architecture contract ``cli -> pipeline``."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config.from_env()

    def run(self, seeds: list[int] | None = None) -> BenchmarkReport:
        return run_benchmark(self.cfg, seeds)
