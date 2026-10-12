"""AiaForge error taxonomy (E100~E500).

Author: 晨星 (CJX0712)
"""


class AiaError(Exception):
    """Base error for AiaForge. Every subclass carries a stable code."""

    code = "E000"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        self.message = message
        self.code = code or self.__class__.code
        super().__init__(f"[{self.code}] {message}")


class ConfigError(AiaError):
    """Invalid configuration / environment override."""

    code = "E100"


class DataError(AiaError):
    """Synthetic data generation or loading failure."""

    code = "E200"


class ModelError(AiaError):
    """Target model training / inference failure."""

    code = "E300"


class AttackError(AiaError):
    """Attack method failure."""

    code = "E400"


class PipelineError(AiaError):
    """Pipeline orchestration failure."""

    code = "E500"
