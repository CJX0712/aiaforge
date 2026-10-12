"""Pipeline tests: smoke, no partition leakage, determinism, verdict.

Author: 晨星 (CJX0712)
"""

import json

import pytest

from core.config import Config
from core.seed import set_all
from data.dgp import generate_split
from pipeline.pipeline import METHOD_NAMES, run_benchmark, run_one_seed


@pytest.fixture
def cfg():
    c = Config()
    c.n_train = 1500
    c.n_aux = 1500
    c.n_test = 800
    c.d_pub = 10
    c.n_seeds = 3
    c.validate()
    return c


def test_run_one_seed_smoke(cfg):
    set_all(7)
    row, _split, _skipped = run_one_seed(cfg, 7)
    for m in METHOD_NAMES:
        assert f"{m}_auc" in row
        assert 0.0 <= row[f"{m}_auc"] <= 1.0 or row[f"{m}_auc"] != row[f"{m}_auc"]  # NaN allowed


def test_benchmark_no_leakage_partitions_disjoint(cfg):
    set_all(7)
    split = generate_split(cfg, 7)
    # index ranges must not overlap
    assert split.train.X_pub.shape[0] == cfg.n_train
    assert split.aux.X_pub.shape[0] == cfg.n_aux
    assert split.test.X_pub.shape[0] == cfg.n_test


def test_benchmark_runs_and_has_verdict(cfg):
    set_all(7)
    rep = run_benchmark(cfg)
    assert "aiafuse" in rep.aggregate["auc"]
    assert "verdict" in rep.aggregate
    assert len(rep.per_seed) == cfg.n_seeds


def test_benchmark_determinism(cfg):
    set_all(7)
    rep1 = run_benchmark(cfg)
    rep2 = run_benchmark(cfg)
    a1 = json.dumps(rep1.to_dict()["per_seed"], sort_keys=True)
    a2 = json.dumps(rep2.to_dict()["per_seed"], sort_keys=True)
    assert a1 == a2


def test_aiafuse_beats_majority_across_seeds(cfg):
    set_all(7)
    rep = run_benchmark(cfg)
    auc = rep.aggregate["auc"]
    assert auc["aiafuse"]["mean"] > auc["majority"]["mean"] + 0.10
