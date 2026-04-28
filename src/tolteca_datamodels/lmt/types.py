"""LMT types and constants."""

from __future__ import annotations

from enum import StrEnum, auto

__all__ = ["LmtObsGoalType"]


class LmtObsGoalType(StrEnum):
    """LMT observation goal types."""

    engineering = auto()
    science = auto()
    calibration = auto()
    pointing = auto()
    focus = auto()
    astigmatism = auto()
    beammap = auto()
    oof = auto()
    unspecified = auto()
