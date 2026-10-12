"""LMT types and constants."""

from __future__ import annotations

from enum import IntEnum, StrEnum, auto
from typing import ClassVar, Literal, Self

import astropy.units as u
from pydantic import Field
from tollan.utils.typing import get_literal_value

from ..lmt_base.metadata import InterfaceInfoMixin
from ..lmt_base.types import LabelT

__all__ = [
    "LmtInfo",
    "LmtInterfaceInfo",
    "LmtMaster",
    "LmtObsGoal",
    "LmtObsPgm",
    "LmtRawObsMasterCode",
]


class LmtInterfaceInfo(InterfaceInfoMixin):
    """LMT interface information with optional instrument affiliation."""

    instru: LabelT | None = Field(
        default=None, description="Instrument whose data conventions apply"
    )


class LmtMaster(StrEnum):
    """LMT acquisition masters."""

    tcs = auto()
    """The telescope control system."""


class LmtRawObsMasterCode(IntEnum):
    """Acquisition master codes in LMT telescope data."""

    TCS = 0

    @classmethod
    def get_master(cls, master: Self) -> LmtMaster:
        """Return the acquisition master.

        Parameters
        ----------
        master : LmtRawObsMasterCode
            Acquisition master code.

        Returns
        -------
        LmtMaster
            Corresponding master label.
        """
        return LmtMaster(cls(master).name.lower())


class LmtInfo:
    """LMT telescope information, site constants, and readout interfaces."""

    type InstruT = Literal["lmt"]

    instru: ClassVar[InstruT] = get_literal_value(InstruT)
    name: ClassVar = "LMT"
    name_long: ClassVar = "Large Millimeter Telescope"

    # Geodetic site coordinates, accepted by EarthLocation.from_geodetic.
    location: ClassVar = {
        "lon": "-97d18m52.6s",
        "lat": "+18d59m10s",
        "height": 4640 << u.m,
    }
    timezone_local: ClassVar = "America/Mexico_City"

    master_code_master: ClassVar[dict[int, LmtMaster]] = {
        master_code.value: LmtRawObsMasterCode.get_master(master_code)
        for master_code in LmtRawObsMasterCode
    }
    """Telescope acquisition-master codes and their labels."""
    master_master_code: ClassVar[dict[str, int]] = {
        master: master_code for master_code, master in master_code_master.items()
    }

    # LMT-owned telescope control.
    interface_interface_info: ClassVar[dict[str, LmtInterfaceInfo]] = {
        "tel": LmtInterfaceInfo(
            interface="tel",
            instru=None,
            producer=instru,
            interface_kind="tel",
        ),
    }
    interfaces: ClassVar = list(interface_interface_info)


class LmtObsPgm(StrEnum):
    """LMT observation programs."""

    vnasweep = auto()
    targsweep = auto()
    tune = auto()
    lissajous = auto()


class LmtObsGoal(StrEnum):
    """LMT observation goals."""

    engineering = auto()
    science = auto()
    calibration = auto()
    pointing = auto()
    focus = auto()
    astigmatism = auto()
    beammap = auto()
    oof = auto()
    unspecified = auto()
