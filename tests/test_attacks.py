"""Attack-method tests: baselines, flagship beats baselines, offline fallback.

Author: 晨星 (CJX0712)
"""

import numpy as np
import pytest

from aia.methods import (
    AiaFuse,
    AttackLR,
    AttackNN,
    ConfInversionAttack,
    MajorityAttack,
)
from aia.target import TargetNumpy, build_target
from core.config import Config
from core.seed import set_all
from data.dgp import generate_split
from eval.metrics import auc_score


@pytest.fixture
def env():
    c = Config()
    c.n_train = 2000
    c.n_aux = 2000
    c.n_test = 1000
    c.d_pub = 12
    c.validate()
    set_all(7)
    split = generate_split(c, 7)
    target = build_target(c.target_model, 7)
    Xtr = np.column_stack([split.train.X_pub, split.train.S.astype(float)])
    target.fit(Xtr, split.train.Y)
    return c, split, target


def test_majority_near_random(env):
    _, split, target = env
    m = MajorityAttack().fit(split.aux, target)
    s = m.attack(split.test.X_pub, split.test.Y, target)
    assert np.allclose(s, s[0])
    auc = auc_score(split.test.S, s)
    assert 0.40 <= auc <= 0.60


def test_real_attacks_beat_majority(env):
    _, split, target = env
    base = MajorityAttack().fit(split.aux, target)
    base_auc = auc_score(split.test.S, base.attack(split.test.X_pub, split.test.Y, target))
    for cls in (ConfInversionAttack, AttackLR, AttackNN, AiaFuse):
        m = cls().fit(split.aux, target)
        sc = m.attack(split.test.X_pub, split.test.Y, target)
        a = auc_score(split.test.S, sc)
        assert a > base_auc + 0.02, f"{cls.__name__} AUC={a:.3f} <= majority {base_auc:.3f}"


def test_aiafuse_non_inferior_to_best(env):
    _, split, target = env
    comps = {
        c.name: c.fit(split.aux, target).attack(split.test.X_pub, split.test.Y, target)
        for c in (ConfInversionAttack(), AttackLR(), AttackNN())
    }
    fuse = AiaFuse().fit(split.aux, target)
    fs = fuse.attack(split.test.X_pub, split.test.Y, target)
    best = max(comps.values(), key=lambda x: auc_score(split.test.S, x))
    assert auc_score(split.test.S, fs) >= auc_score(split.test.S, best) - 0.03


def test_offline_target_numpy_end_to_end():
    c = Config()
    c.n_train = 1500
    c.n_aux = 1500
    c.n_test = 800
    c.d_pub = 10
    c.validate()
    set_all(7)
    split = generate_split(c, 7)
    t = TargetNumpy(seed=7)
    Xtr = np.column_stack([split.train.X_pub, split.train.S.astype(float)])
    t.fit(Xtr, split.train.Y)
    m = AiaFuse().fit(split.aux, t)
    sc = m.attack(split.test.X_pub, split.test.Y, t)
    assert auc_score(split.test.S, sc) > 0.55


def test_attack_nn_offline_fallback(monkeypatch, env):
    monkeypatch.setattr("aia.target.available_sklearn", lambda: False)
    _, split, target = env  # target may be sklearn; only the meta falls back
    m = AttackNN()
    m.fit(split.aux, target)
    assert m._kind == "numpy"
    sc = m.attack(split.test.X_pub, split.test.Y, target)
    assert auc_score(split.test.S, sc) > 0.55
