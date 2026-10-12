"""TolTEC data acquisition database specifics."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Self, cast

if TYPE_CHECKING:
    from .types import ToltecInfo


class ToltecDBRawObsMasterCode(IntEnum):
    """Acquisition master codes in the TolTEC database."""

    TCS = 0
    ICS = 1
    CLIP = 2

    @classmethod
    def get_master(cls, master: Self) -> ToltecInfo.MasterT:
        """Return the acquisition master.

        Parameters
        ----------
        master : ToltecDBRawObsMasterCode
            Database acquisition master code.

        Returns
        -------
        ToltecInfo.MasterT
            Corresponding master label.
        """
        return cast("ToltecInfo.MasterT", cls(master).name.lower())


class ToltecDBRawObsTypeCode(IntEnum):
    """Observation type codes in the TolTEC database."""

    Nominal = 0
    """Nominal observation."""

    Timestream = 1
    """Time stream from timestream_save operation."""

    VNA = 2
    """VnaSweep."""

    TARG = 3
    """TargetSweep."""

    TUNE = 4
    """TUNE."""

    @classmethod
    def get_obs_type(cls, raw_obs_type: Self) -> ToltecInfo.ObsTypeT:
        """Return the acquisition observation type.

        Parameters
        ----------
        raw_obs_type : ToltecDBRawObsTypeCode
            Database raw-observation type code.

        Returns
        -------
        ToltecInfo.ObsTypeT
            Corresponding observation-type label.
        """
        return cast("ToltecInfo.ObsTypeT", cls(raw_obs_type).name.lower())
