"""LMT types and constants."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import ClassVar

import astropy.units as u

from ..lmt_base.metadata import LmtInstruMixin

__all__ = ["LmtInfo", "LmtObsGoalType"]


class LmtInfo:
    """LMT telescope information, site constants, and readout interfaces."""

    instru: ClassVar = "lmt"
    name: ClassVar = "LMT"
    name_long: ClassVar = "Large Millimeter Telescope"

    # Geodetic site coordinates, accepted by EarthLocation.from_geodetic.
    location: ClassVar = {
        "lon": "-97d18m52.6s",
        "lat": "+18d59m10s",
        "height": 4640 << u.m,
    }
    timezone_local: ClassVar = "America/Mexico_City"

    # LMT-owned telescope control.
    interface_instru: ClassVar[dict[str, LmtInstruMixin]] = {
        "tel": LmtInstruMixin(interface="tel", instru="lmt", instru_component="tcs"),
    }
    interfaces: ClassVar = list(interface_instru)


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
