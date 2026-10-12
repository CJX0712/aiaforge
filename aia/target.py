"""Target models: the classifier an adversary attacks.

Two production backends (scikit-learn) plus a zero-dependency NumPy fallback.
The model is *always* trained on ``[X_pub, S]`` so that the sensitive attribute
is a real input feature during training — this is what the attacker probes.

Author: 晨星 (CJX0712)
"""

import numpy as np

from aia._numpy_models import NumpyLogReg
from core.interfaces import TargetModel


def available_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


class TargetMLP:
    """scikit-learn MLPClassifier backend (production Tier-0)."""

    def __init__(self, seed: int = 7, hidden=(32, 16), max_iter: int = 400):
        self.seed = seed
        self.hidden = hidden
        self.max_iter = max_iter
        self.clf = None
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "TargetMLP":
        from sklearn.neural_network import MLPClassifier

        self.clf = MLPClassifier(
            hidden_layer_sizes=self.hidden,
            max_iter=self.max_iter,
            random_state=self.seed,
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            early_stopping=False,
        )
        self.clf.fit(X, Y)
        self.classes_ = np.asarray([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)


class TargetLogReg:
    """scikit-learn LogisticRegression backend (production Tier-0)."""

    def __init__(self, seed: int = 7, C: float = 1.0, max_iter: int = 2000):
        self.seed = seed
        self.C = C
        self.max_iter = max_iter
        self.clf = None
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "TargetLogReg":
        from sklearn.linear_model import LogisticRegression

        self.clf = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.seed)
        self.clf.fit(X, Y)
        self.classes_ = np.asarray([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)


class TargetNumpy:
    """Offline fallback: pure-NumPy logistic regression."""

    def __init__(self, seed: int = 7):
        self.seed = seed
        self._m: NumpyLogReg | None = None
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "TargetNumpy":
        self._m = NumpyLogReg(lr=0.05, n_iter=3000, seed=self.seed, l2=1e-3).fit(X, Y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._m.predict_proba(X)


def build_target(kind: str, seed: int) -> TargetModel:
    if kind == "mlp":
        return TargetMLP(seed=seed)
    if kind == "logreg":
        return TargetLogReg(seed=seed)
    raise ValueError(f"unknown target model kind {kind!r}")
