"""Attribute-inference attacks.

Threat-model convention (kept identical to the protocol contract):
* ``scores`` are ``P(S=1 | evidence)`` — larger means the adversary is more
  confident the sensitive attribute equals 1.
* Every attack receives the *auxiliary* dataset (known ``S``) for training and a
  fitted target model for probing. Evaluation is always on the held-out test
  split, so no partition leaks into another.

Flagship ``AiaFuse`` = calibrate-stacked fusion of component attackers with a
non-inferiority guard plus a regime router (label-only fallback when the
adversary has no public features).

Author: 晨星 (CJX0712)
"""

import numpy as np

from aia import target as _target_mod
from aia._numpy_models import NumpyLogReg
from core.errors import AttackError
from core.interfaces import Attack, TargetModel
from core.types import AiaDataset
from eval.metrics import auc_score


def _with_s(X_pub: np.ndarray, s: int) -> np.ndarray:
    return np.column_stack([np.asarray(X_pub, dtype=float), np.full(len(X_pub), float(s))])


def _proba_for_s(target: TargetModel, X_pub: np.ndarray, s: int) -> np.ndarray:
    return target.predict_proba(_with_s(X_pub, s))


def _true_class_col(Y: np.ndarray) -> np.ndarray:
    return np.asarray(Y).astype(int).reshape(-1)


def _attack_features(target: TargetModel, X_pub: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Meta-features observable by the adversary for one individual."""
    p1 = _proba_for_s(target, X_pub, 1)
    p0 = _proba_for_s(target, X_pub, 0)
    Yc = _true_class_col(Y)
    c1 = p1[np.arange(len(Yc)), Yc]
    c0 = p0[np.arange(len(Yc)), Yc]
    gap = c1 - c0
    return np.column_stack([np.asarray(X_pub, float), Yc.astype(float), p1[:, 1], p0[:, 1], gap])


class MajorityAttack:
    """Baseline: always predict the most frequent S in the auxiliary set."""

    name = "majority"

    def __init__(self) -> None:
        self.p: float = 0.5

    def fit(self, aux: AiaDataset, target: TargetModel | None = None) -> "MajorityAttack":
        self.p = float(np.mean(aux.S == 1))
        return self

    def attack(self, X_pub: np.ndarray, Y: np.ndarray, target: TargetModel) -> np.ndarray:
        return np.full(len(np.asarray(X_pub, float)), self.p)


class LabelOnlyAttack:
    """Baseline: infer S only from the known label Y (empirical, no model)."""

    name = "label_only"

    def __init__(self) -> None:
        self._m: NumpyLogReg | None = None

    def fit(self, aux: AiaDataset, target: TargetModel | None = None) -> "LabelOnlyAttack":
        self._m = NumpyLogReg(lr=0.1, n_iter=2000, seed=0, l2=1e-3).fit(
            np.column_stack([aux.Y.astype(float)]), aux.S
        )
        return self

    def attack(self, X_pub: np.ndarray, Y: np.ndarray, target: TargetModel) -> np.ndarray:
        return self._m.predict_proba(np.column_stack([_true_class_col(Y).astype(float)]))[:, 1]


class ConfInversionAttack:
    """Yeom-style confidence inversion: probe both S guesses, keep the one whose
    predicted confidence on the true label is higher."""

    name = "conf_inversion"

    def __init__(self) -> None:
        self._t: TargetModel | None = None

    def fit(self, aux: AiaDataset, target: TargetModel | None = None) -> "ConfInversionAttack":
        if target is None:
            raise AttackError("conf_inversion requires a target model")
        self._t = target
        return self

    def attack(self, X_pub: np.ndarray, Y: np.ndarray, target: TargetModel) -> np.ndarray:
        Yc = _true_class_col(Y)
        p1 = _proba_for_s(target, X_pub, 1)
        p0 = _proba_for_s(target, X_pub, 0)
        c1 = p1[np.arange(len(Yc)), Yc]
        c0 = p0[np.arange(len(Yc)), Yc]
        # map confidence gap -> P(S=1) in [0,1] (threshold 0.5 == tie)
        return 1.0 / (1.0 + np.exp(-(c1 - c0)))


class AttackLR:
    """Meta-classifier on adversary-observable features (X_pub, Y, model probas)."""

    name = "attack_lr"

    def __init__(self) -> None:
        self._m: NumpyLogReg | None = None

    def fit(self, aux: AiaDataset, target: TargetModel | None = None) -> "AttackLR":
        if target is None:
            raise AttackError("attack_lr requires a target model")
        self._m = NumpyLogReg(lr=0.05, n_iter=3000, seed=1, l2=1e-3).fit(
            _attack_features(target, aux.X_pub, aux.Y), aux.S
        )
        return self

    def attack(self, X_pub: np.ndarray, Y: np.ndarray, target: TargetModel) -> np.ndarray:
        return self._m.predict_proba(_attack_features(target, X_pub, Y))[:, 1]


class AttackNN:
    """Neural meta-classifier on the same adversary-observable features.
    Falls back to a (deeper-trained) NumPy logistic regression offline."""

    name = "attack_nn"

    def __init__(self) -> None:
        self._m = None
        self._kind = "sklearn" if _target_mod.available_sklearn() else "numpy"

    def fit(self, aux: AiaDataset, target: TargetModel | None = None) -> "AttackNN":
        if target is None:
            raise AttackError("attack_nn requires a target model")
        X = _attack_features(target, aux.X_pub, aux.Y)
        y = aux.S
        if self._kind == "sklearn":
            from sklearn.neural_network import MLPClassifier

            self._m = MLPClassifier(
                hidden_layer_sizes=(32, 16),
                max_iter=300,
                random_state=3,
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                early_stopping=False,
            )
            self._m.fit(X, y)
        else:
            self._m = NumpyLogReg(lr=0.02, n_iter=6000, seed=3, l2=1e-3).fit(X, y)
        return self

    def attack(self, X_pub: np.ndarray, Y: np.ndarray, target: TargetModel) -> np.ndarray:
        X = _attack_features(target, X_pub, Y)
        if self._kind == "sklearn":
            return self._m.predict_proba(X)[:, 1]
        return self._m.predict_proba(X)[:, 1]

    def available(self) -> bool:
        return True


class AiaFuse:
    """Flagship: calibrate-stacked fusion via out-of-fold (OOF) stacked
    generalisation (Wolpert 1992).

    The meta-learner is trained on *OOF* component scores (so it sees how the
    components generalise instead of their in-sample overfit) plus the
    adversary-observable features, with standardisation. A non-inferiority guard
    falls back to the best single component when fusion does not beat it on the
    OOF estimate. A regime router degrades to the label-only baseline when the
    adversary has no public features.
    """

    name = "aiafuse"

    def __init__(self, components: list[Attack] | None = None, n_folds: int = 3) -> None:
        self.components: list[Attack] = components or [
            ConfInversionAttack(),
            AttackLR(),
            AttackNN(),
        ]
        self.n_folds = n_folds
        self._meta = None
        self._meta_kind = "sklearn" if _target_mod.available_sklearn() else "numpy"
        self._aux_comp_auc: dict = {}
        self._aux_fuse_auc: float = 0.0
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def _standardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if fit or self._mean is None:
            self._mean = X.mean(axis=0)
            self._std = X.std(axis=0)
        std = np.where(self._std < 1e-9, 1.0, self._std)
        return (X - self._mean) / std

    def _oof_scores(self, aux: AiaDataset, target: TargetModel) -> np.ndarray:
        n = len(aux.S)
        oof = np.zeros((n, len(self.components)))
        try:
            from sklearn.model_selection import KFold

            folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=0).split(np.arange(n))
        except Exception:
            folds = _manual_kfold(n, self.n_folds)
        for tr, va in folds:
            aux_tr = AiaDataset(X_pub=aux.X_pub[tr], S=aux.S[tr], Y=aux.Y[tr])
            for i, c in enumerate(self.components):
                c.fit(aux_tr, target)
                oof[va, i] = c.attack(aux.X_pub[va], aux.Y[va], target)
        return oof

    def fit(self, aux: AiaDataset, target: TargetModel | None = None) -> "AiaFuse":
        if target is None:
            raise AttackError("aiafuse requires a target model")
        # components fit on full aux -> used for test-time attack
        for c in self.components:
            c.fit(aux, target)
        # OOF component scores -> honest generalisation signal for the meta
        oof = self._oof_scores(aux, target)
        meta_X = self._standardize(
            np.column_stack([_attack_features(target, aux.X_pub, aux.Y), oof]), fit=True
        )
        y = aux.S
        if self._meta_kind == "sklearn":
            from sklearn.linear_model import LogisticRegression

            self._meta = LogisticRegression(C=0.5, max_iter=2000, random_state=2)
            self._meta.fit(meta_X, y)
        else:
            self._meta = NumpyLogReg(lr=0.05, n_iter=6000, seed=2, l2=1e-2).fit(meta_X, y)
        self._aux_comp_auc = {
            self.components[i].name: auc_score(aux.S, oof[:, i])
            for i in range(len(self.components))
        }
        self._aux_fuse_auc = auc_score(aux.S, self._meta.predict_proba(meta_X)[:, 1])
        return self

    def attack(self, X_pub: np.ndarray, Y: np.ndarray, target: TargetModel) -> np.ndarray:
        comp = np.column_stack([c.attack(X_pub, Y, target) for c in self.components])
        meta_X = self._standardize(np.column_stack([_attack_features(target, X_pub, Y), comp]))
        fuse = self._meta.predict_proba(meta_X)[:, 1]
        best_name = max(self._aux_comp_auc, key=self._aux_comp_auc.get)
        if self._aux_fuse_auc < self._aux_comp_auc[best_name] - 0.02:
            idx = [c.name for c in self.components].index(best_name)
            return comp[:, idx]
        return fuse


def _manual_kfold(n: int, k: int):
    idx = np.arange(n)
    size = n // k
    for i in range(k):
        end = (i + 1) * size if i < k - 1 else n
        va = idx[i * size : end]
        tr = np.array([j for j in idx if j not in set(va.tolist())])
        yield tr, va


def all_attacks() -> list[Attack]:
    return [
        MajorityAttack(),
        LabelOnlyAttack(),
        ConfInversionAttack(),
        AttackLR(),
        AttackNN(),
        AiaFuse(),
    ]
