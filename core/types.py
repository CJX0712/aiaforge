"""AiaForge core data types.

Author: 晨星 (CJX0712)
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AiaDataset:
    """Synthetic record bundle for the attribute-inference threat model.

    X_pub : public, attacker-observable features (the model's real input at
            attack time is ``X_pub`` stacked with a *guessed* ``S``).
    S     : sensitive binary attribute the attacker wants to recover.
    Y     : label the model was trained to predict.
    """

    X_pub: np.ndarray
    S: np.ndarray
    Y: np.ndarray

    def __post_init__(self) -> None:
        if self.X_pub.ndim != 2:
            raise ValueError("X_pub must be 2-D (n, d)")
        n = self.X_pub.shape[0]
        if self.S.shape != (n,):
            raise ValueError(f"S must be 1-D length {n}, got {self.S.shape}")
        if self.Y.shape != (n,):
            raise ValueError(f"Y must be 1-D length {n}, got {self.Y.shape}")
        if not np.issubdtype(self.S.dtype, np.integer) and not np.issubdtype(
            self.S.dtype, np.bool_
        ):
            self.S = self.S.astype(int)
        self.S = np.ascontiguousarray(self.S)
        self.Y = np.ascontiguousarray(self.Y)


@dataclass
class Split:
    """Train / auxiliary / test partition of an :class:`AiaDataset`."""

    train: AiaDataset
    aux: AiaDataset
    test: AiaDataset


@dataclass
class AttackResult:
    """Per-sample attack output: ``scores[i] = P(S=1 | evidence)``."""

    scores: np.ndarray
    name: str


@dataclass
class BenchmarkReport:
    """Aggregated, JSON-serialisable benchmark result."""

    system: str = "AiaForge"
    config: dict = field(default_factory=dict)
    per_seed: list[dict] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "system": self.system,
            "config": self.config,
            "per_seed": self.per_seed,
            "aggregate": self.aggregate,
            "skipped": self.skipped,
            "failures": self.failures,
        }
