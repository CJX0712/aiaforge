"""AiaForge end-to-end demo.

Generates synthetic data, trains the target model, runs every attack, prints a
benchmark table, writes ``benchmark.json`` to the project root, and performs the
bit-identical determinism check (same seeds -> identical core metrics).

Author: 晨星 (CJX0712)
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cli import _print_report  # noqa: E402
from core.config import Config  # noqa: E402
from core.seed import set_all  # noqa: E402
from pipeline.pipeline import run_benchmark  # noqa: E402


def grade(rep) -> str:
    v = rep.aggregate.get("verdict", {})
    vm = v.get("aiafuse_vs_majority", {})
    vb = v.get("aiafuse_vs_best_single", {})
    if vm.get("pass") and vb.get("non_inferior"):
        return "S"
    if vm.get("pass"):
        return "A"
    if "aiafuse" in rep.aggregate.get("auc", {}):
        return "B"
    return "C"


def main() -> int:
    set_all(7)
    cfg = Config.from_env()
    cfg.n_seeds = 3
    cfg.validate()

    t0 = time.time()
    rep = run_benchmark(cfg)
    elapsed = time.time() - t0

    out = os.path.join(ROOT, "benchmark.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep.to_dict(), f, indent=2, ensure_ascii=False)

    # determinism: re-run with identical seeds, compare core metrics (bit-identical)
    rep2 = run_benchmark(cfg)
    det = json.dumps(rep.to_dict()["per_seed"], sort_keys=True) == json.dumps(
        rep2.to_dict()["per_seed"], sort_keys=True
    )

    print(f"\n[run_demo] elapsed={elapsed:.1f}s  determinism_bit_identical={det}")
    _print_report(rep)

    g = grade(rep)
    print(f"\n=== Quality grade: {g} ===")
    print(f"benchmark.json -> {out}")

    if rep.failures:
        print("\nFailure cases (derived from results):")
        for fc in rep.failures[:6]:
            print(
                f"  - {fc['case']} idx={fc['index']} true_S={fc['true_S']} "
                f"fuse={fc['fuse_score']:.3f}: {fc['note']}"
            )
    if rep.skipped:
        print(f"\nskipped: {', '.join(rep.skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
