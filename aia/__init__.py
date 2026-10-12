"""AiaForge attack module.

Author: 晨星 (CJX0712)
"""

from aia.methods import (
    AiaFuse,
    AttackLR,
    AttackNN,
    ConfInversionAttack,
    LabelOnlyAttack,
    MajorityAttack,
    all_attacks,
)
from aia.target import (
    TargetLogReg,
    TargetMLP,
    TargetNumpy,
    available_sklearn,
    build_target,
)

__all__ = [
    "AiaFuse",
    "AttackLR",
    "AttackNN",
    "ConfInversionAttack",
    "LabelOnlyAttack",
    "MajorityAttack",
    "TargetLogReg",
    "TargetMLP",
    "TargetNumpy",
    "all_attacks",
    "available_sklearn",
    "build_target",
]
