"""LMT types and constants."""

from __future__ import annotations

from enum import IntEnum, StrEnum, auto
from typing import ClassVar, Literal, Self, cast

import astropy.units as u
from tollan.utils.typing import get_literal_value

from ..core.types import DataModelInfo
from ..lmt_base.metadata import InterfaceInfoMixin

__all__ = [
    "LmtDataModel",
    "LmtInfo",
    "LmtInterfaceInfo",
    "LmtObsGoal",
    "LmtObsPgm",
    "LmtRawObsMasterCode",
]


class LmtDataModel(DataModelInfo):
    """LMT telescope data-model information."""

    name = "lmt"


class LmtInterfaceInfo(InterfaceInfoMixin):
    """Identity and data-model memberships of an LMT interface."""


class LmtRawObsMasterCode(IntEnum):
    """Acquisition master codes in LMT telescope data."""

    TCS = 0

    @classmethod
    def get_master(cls, master: Self) -> LmtInfo.MasterT:
        """Return the acquisition master.

        Parameters
        ----------
        master : LmtRawObsMasterCode
            Acquisition master code.

        Returns
        -------
        LmtInfo.MasterT
            Corresponding master label.
        """
        return cast("LmtInfo.MasterT", cls(master).name.lower())


class LmtInfo:
    """LMT telescope information, site constants, and readout interfaces."""

    type InstruT = Literal["lmt"]
    type MasterT = Literal["tcs"]

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

    master_description: ClassVar[dict[MasterT, str]] = {
        "tcs": "Telescope control system",
    }
    """Descriptions of acquisition master labels."""
    master_code_master: ClassVar[dict[int, MasterT]] = {
        master_code.value: LmtRawObsMasterCode.get_master(master_code)
        for master_code in LmtRawObsMasterCode
    }
    """Telescope acquisition-master codes and their labels."""
    master_master_code: ClassVar[dict[MasterT, int]] = {
        master: master_code for master_code, master in master_code_master.items()
    }

    # LMT-owned telescope control.
    interface_interface_info: ClassVar[dict[str, LmtInterfaceInfo]] = {
        "tel": LmtInterfaceInfo(
            interface="tel",
            instru=instru,
            data_models=[LmtDataModel.name],
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
    map = auto()


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
