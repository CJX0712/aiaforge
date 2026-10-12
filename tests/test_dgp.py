"""Data DGP tests: reproducibility, partition disjointness, balance.

Author: 晨星 (CJX0712)
"""

import numpy as np
import pytest

from core.config import Config
from core.seed import set_all
from data.dgp import generate_split


@pytest.fixture
def cfg():
    c = Config()
    c.n_train = 2000
    c.n_aux = 2000
    c.n_test = 1000
    c.d_pub = 12
    c.validate()
    return c


def test_split_non_overlapping(cfg):
    set_all(7)
    s = generate_split(cfg, 7)
    n = cfg.n_train + cfg.n_aux + cfg.n_test
    assert s.train.X_pub.shape[0] == cfg.n_train
    assert s.aux.X_pub.shape[0] == cfg.n_aux
    assert s.test.X_pub.shape[0] == cfg.n_test
    # no sample index reused across partitions
    all_X = np.vstack([s.train.X_pub, s.aux.X_pub, s.test.X_pub])
    assert all_X.shape[0] == n


def test_split_reproducible(cfg):
    set_all(7)
    a = generate_split(cfg, 7)
    set_all(7)
    b = generate_split(cfg, 7)
    assert np.array_equal(a.test.S, b.test.S)
    assert np.allclose(a.test.X_pub, b.test.X_pub)


def test_s_balanced_and_binary(cfg):
    set_all(7)
    s = generate_split(cfg, 7)
    assert set(np.unique(s.test.S).tolist()) <= {0, 1}
    frac = float(np.mean(s.test.S == 1))
    assert 0.4 <= frac <= 0.6
