"""Core-layer tests: config, seed determinism, types.

Author: 晨星 (CJX0712)
"""

import numpy as np
import pytest

from core.config import Config
from core.errors import ConfigError
from core.seed import set_all
from core.types import AiaDataset


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("AIAFORGE_SEED", "42")
    monkeypatch.setenv("AIAFORGE_D_PUB", "8")
    cfg = Config.from_env()
    assert cfg.seed == 42
    assert cfg.d_pub == 8


def test_config_validate_bounds():
    cfg = Config(seed=-1)
    with pytest.raises(ConfigError):
        cfg.validate()
    cfg = Config(label_noise=1.5)
    with pytest.raises(ConfigError):
        cfg.validate()
    cfg = Config(target_model="xgb")
    with pytest.raises(ConfigError):
        cfg.validate()


def test_seed_determinism():
    set_all(7)
    a = np.random.standard_normal(5)
    set_all(7)
    b = np.random.standard_normal(5)
    assert np.array_equal(a, b)


def test_dataset_post_init_raises_on_shape():
    with pytest.raises(ValueError):
        AiaDataset(X_pub=np.zeros((5, 2)), S=np.zeros(3), Y=np.zeros(5))
    with pytest.raises(ValueError):
        AiaDataset(X_pub=np.zeros((5, 2)), S=np.zeros(5), Y=np.zeros((5, 1)))
