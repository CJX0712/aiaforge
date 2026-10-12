"""Global deterministic seeding for AiaForge.

Pinning BLAS/OpenMP threads to 1 is essential: NumPy/SciPy linear algebra on
multithreaded runtimes is non-deterministic across runs, which would break the
bit-identical reproducibility gate.

Author: 晨星 (CJX0712)
"""

import os
import random

_STATE: dict[str, object] = {"seeded": False, "seed": None}


def set_all(seed: int = 7) -> None:
    """Seed numpy / random / thread env once, deterministically."""
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    import numpy as np

    np.random.seed(seed)
    random.seed(seed)
    _STATE["seeded"] = True
    _STATE["seed"] = seed


def is_seeded() -> bool:
    return bool(_STATE["seeded"])


def get_seed() -> int | None:
    return _STATE["seed"]  # type: ignore[return-value]
