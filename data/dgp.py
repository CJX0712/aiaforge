"""Synthetic data generator for the attribute-inference threat model.

Threat model
------------
A target classifier is trained to predict a label ``Y`` from public features
``X_pub`` concatenated with the sensitive attribute ``S`` (which the *model*
saw during training). At attack time the adversary knows ``X_pub`` and ``Y``
but not ``S``; it probes the model by feeding both candidate values
``S_guess ∈ {0, 1}`` and tries to recover the true ``S``.

The generator is fully deterministic given ``seed``; the split (train / aux /
test) is contiguous and non-overlapping so no sample the target model or the
attack models ever see leaks across partitions.

Author: 晨星 (CJX0712)
"""

import numpy as np

from core.errors import DataError
from core.types import AiaDataset, Split


def _sample(cfg, n: int, rng: np.random.Generator) -> AiaDataset:
    d = cfg.d_pub
    dx = rng.standard_normal(d)
    dx /= np.linalg.norm(dx)
    w = rng.standard_normal(d)
    w /= np.linalg.norm(w)

    S = rng.integers(0, 2, size=n)
    sgn = (2 * S - 1).reshape(-1, 1)

    # X_pub differs by S along direction dx -> S is recoverable from X_pub,
    # but only partially; the model's overfit confidence adds extra signal.
    X_pub = rng.standard_normal((n, d)) + sgn * cfg.s_mag * dx

    logit = X_pub @ w + cfg.s_eff * (2 * S - 1)
    py = 1.0 / (1.0 + np.exp(-(logit + rng.normal(0.0, cfg.y_noise, n))))
    Y = (rng.random(n) < py).astype(int)
    flip = rng.random(n) < cfg.label_noise
    Y[flip] = 1 - Y[flip]
    return AiaDataset(X_pub=X_pub, S=S, Y=Y)


def generate_split(cfg, seed: int) -> Split:
    """Generate a non-overlapping train/aux/test split from one RNG stream."""
    rng = np.random.default_rng(seed)
    N = cfg.n_train + cfg.n_aux + cfg.n_test
    full = _sample(cfg, N, rng)
    i1 = cfg.n_train
    i2 = i1 + cfg.n_aux

    def sl(lo: int, hi: int) -> AiaDataset:
        return AiaDataset(X_pub=full.X_pub[lo:hi], S=full.S[lo:hi], Y=full.Y[lo:hi])

    return Split(train=sl(0, i1), aux=sl(i1, i2), test=sl(i2, N))


def generate(cfg, seed: int) -> AiaDataset:
    """Generate a single undifferentiated dataset (used by tests)."""
    rng = np.random.default_rng(seed)
    N = getattr(cfg, "n_train", 1000)
    return _sample(cfg, N, rng)


def from_arrays(
    X_pub: np.ndarray, S: np.ndarray, Y: np.ndarray, *, validate: bool = True
) -> AiaDataset:
    """Build an :class:`AiaDataset` from external arrays (CSV/DB loader hook)."""
    try:
        X_pub = np.asarray(X_pub, dtype=float)
        S = np.asarray(S).reshape(-1)
        Y = np.asarray(Y).reshape(-1)
    except Exception as exc:
        raise DataError(f"failed to coerce input arrays: {exc}") from exc
    if validate and (X_pub.ndim != 2 or S.ndim != 1 or Y.ndim != 1):
        raise DataError("X_pub must be 2-D; S, Y must be 1-D")
    return AiaDataset(X_pub=X_pub, S=S, Y=Y)
