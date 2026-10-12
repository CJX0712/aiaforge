"""AiaForge command-line entry point.

Author: 晨星 (CJX0712)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config
from pipeline.pipeline import METHOD_NAMES, run_benchmark


def _print_report(rep) -> None:
    agg = rep.aggregate
    auc = agg.get("auc", {})
    print("\n=== AiaForge benchmark (AUC, higher = stronger attribute inference) ===")
    header = f"{'method':<16}{'mean':>10}{'std':>10}"
    print(header)
    print("-" * len(header))
    for m in METHOD_NAMES:
        if m in auc:
            print(f"{m:<16}{auc[m]['mean']:>10.4f}{auc[m]['std']:>10.4f}")
    v = agg.get("verdict", {})
    if "aiafuse_vs_majority" in v:
        x = v["aiafuse_vs_majority"]
        print(
            f"\naiafuse vs majority: ΔAUC={x['diff']:+.4f} "
            f"(gate>={x['gate_threshold']:.2f}) significant={x['significant']} "
            f"=> {'PASS' if x['pass'] else 'FAIL'}"
        )
    if "aiafuse_vs_best_single" in v:
        x = v["aiafuse_vs_best_single"]
        print(
            f"aiafuse vs best single ({x['best_single']}): "
            f"ΔAUC={x['diff']:+.4f} non_inferior={x['non_inferior']}"
        )
    if rep.skipped:
        print(f"\nskipped: {', '.join(rep.skipped)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="aiaforge",
        description="World-class attribute inference attack benchmark (AiaFuse flagship).",
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--target-model", choices=["mlp", "logreg"], default=None)
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--out", type=str, default=None, help="path to write benchmark.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config.from_env()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.target_model is not None:
        cfg.target_model = args.target_model
    if args.n_seeds is not None:
        cfg.n_seeds = args.n_seeds
    cfg.validate()

    rep = run_benchmark(cfg)
    if not args.quiet:
        _print_report(rep)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep.to_dict(), f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
