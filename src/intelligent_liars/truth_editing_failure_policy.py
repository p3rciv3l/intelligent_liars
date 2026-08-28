"""Study-wide failures that must cross scoring and scheduling seams unchanged."""

from __future__ import annotations


class PaidJudgeCircuitOpen(RuntimeError):
    """The paid semantic judge failed, so optimization must stop immediately."""


__all__ = ["PaidJudgeCircuitOpen"]
