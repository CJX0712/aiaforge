"""AiaForge configuration with ENV_XXX_* overrides and schema validation.

Author: 晨星 (CJX0712)
"""

import os
from dataclasses import dataclass, fields
from typing import Any

from core.errors import ConfigError


@dataclass
class Config:
    """Global, validated run configuration.

    Any field may be overridden by an environment variable of the form
    ``AIAFORGE_<FIELD>`` (uppercased, e.g. ``AIAFORGE_SEED=42``).
    """

    # determinism
    seed: int = 7
    n_seeds: int = 3

    # dataset sizing (tuned for the 60s CPU demo budget)
    n_train: int = 2000
    n_aux: int = 2000
    n_test: int = 1000
    d_pub: int = 12

    # DGP difficulty knobs (sweet-spot tuned, see docs/architecture.md)
    s_mag: float = 0.85  # separation of X_pub mean by S
    s_eff: float = 0.65  # direct S -> Y logit effect
    y_noise: float = 1.0  # label logistic noise scale
    label_noise: float = 0.18  # Y flip probability (overfitting fuel)

    # target model
    target_model: str = "mlp"  # "mlp" | "logreg"

    # evaluation
    threshold_auc_lift: float = 0.10  # S-grade gate vs majority baseline
    significance_seeds: int = 3

    _ENV_PREFIX = "AIAFORGE_"

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        for f in fields(cls):
            if f.name.startswith("_"):
                continue
            env_key = cls._ENV_PREFIX + f.name.upper()
            if env_key in os.environ:
                raw = os.environ[env_key]
                cfg = cfg._set(f.name, _coerce(raw, f.type))
        cfg.validate()
        return cfg

    def _set(self, name: str, value: Any) -> "Config":
        setattr(self, name, value)
        return self

    def validate(self) -> None:
        if self.seed < 0:
            raise ConfigError("seed must be >= 0")
        for name in ("n_train", "n_aux", "n_test", "d_pub", "n_seeds"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive, got {getattr(self, name)}")
        if not (0.0 <= self.label_noise <= 1.0):
            raise ConfigError("label_noise must be in [0, 1]")
        if self.y_noise <= 0:
            raise ConfigError("y_noise must be > 0")
        if self.target_model not in ("mlp", "logreg"):
            raise ConfigError(f"unknown target_model {self.target_model!r}")
        if self.threshold_auc_lift <= 0:
            raise ConfigError("threshold_auc_lift must be > 0")

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}


def _coerce(raw: str, typ: type | str) -> Any:
    t = str(typ)
    if "int" in t:
        return int(raw)
    if "float" in t:
        return float(raw)
    if "bool" in t:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return raw
