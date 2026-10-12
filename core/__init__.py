"""AiaForge core: shared types, errors, config, interfaces, deterministic seeding.

Author: 晨星 (CJX0712)
"""

from core.config import Config
from core.errors import (
    AiaError,
    AttackError,
    ConfigError,
    DataError,
    ModelError,
    PipelineError,
)
from core.interfaces import Attack, TargetModel
from core.seed import is_seeded, set_all
from core.types import AiaDataset, AttackResult, BenchmarkReport, Split

__all__ = [
    "AiaDataset",
    "AiaError",
    "Attack",
    "AttackError",
    "AttackResult",
    "BenchmarkReport",
    "Config",
    "ConfigError",
    "DataError",
    "ModelError",
    "PipelineError",
    "Split",
    "TargetModel",
    "is_seeded",
    "set_all",
]
