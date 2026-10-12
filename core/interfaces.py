"""AiaForge runtime interfaces (Protocol contracts).

Author: 晨星 (CJX0712)
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TargetModel(Protocol):
    """A classifier trained to predict Y from ``[X_pub, S_guess]``.

    At attack time the attacker only knows ``X_pub`` and ``Y``; it probes the
    model by feeding both candidate sensitive values ``S_guess ∈ {0, 1}``.
    """

    classes_: np.ndarray

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "TargetModel": ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n, n_classes) probability matrix."""


@runtime_checkable
class Attack(Protocol):
    """An attribute-inference attack producing ``P(S=1 | evidence)`` scores."""

    name: str

    def fit(self, aux: "object", target: TargetModel | None) -> "Attack": ...

    def attack(self, X_pub: np.ndarray, Y: np.ndarray, target: TargetModel) -> np.ndarray:
        """Return (n,) scores; larger == more confident S=1."""


def available_target_model(name: str) -> bool:  # pragma: no cover - thin guard
    return name in ("mlp", "logreg")
