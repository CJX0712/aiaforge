"""Zero-dependency NumPy logistic-regression fallback.

Used when scikit-learn is unavailable so the whole system still runs offline.
Deterministic given ``seed`` and single-threaded BLAS (pinned in ``core.seed``).

Author: 晨星 (CJX0712)
"""

import numpy as np


class NumpyLogReg:
    """Minimal L2-regularised logistic regression via full-batch gradient descent."""

    def __init__(self, lr: float = 0.05, n_iter: int = 3000, seed: int = 0, l2: float = 1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.seed = seed
        self.l2 = l2
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "NumpyLogReg":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y).astype(int).reshape(-1)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        w = rng.standard_normal(d) * 1e-2
        b = np.zeros(1)
        yb = Y.astype(float)
        for _ in range(self.n_iter):
            z = X @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = (p - yb) @ X / n + self.l2 * w
            grad_b = (p - yb).mean()
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        self.coef_ = w
        self.intercept_ = b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        z = X @ self.coef_ + self.intercept_
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p, p])
