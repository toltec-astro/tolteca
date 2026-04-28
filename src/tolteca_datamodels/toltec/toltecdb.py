"""TolTEC data acquisition database specifics."""

from __future__ import annotations

from enum import IntEnum
from typing import Self

from .types import ToltecDataKind, ToltecMasterType


class ToltecDBRawObsMaster(IntEnum):
    """Master constants in-sync with ``toltec/master`` table in the toltec db."""

    TCS = 0
    ICS = 1
    CLIP = 2

    @classmethod
    def get_master_type(cls, master: Self) -> ToltecMasterType:
        """Return the name of the master."""
        result = {
            0: ToltecMasterType.tcs,
            1: ToltecMasterType.ics,
            2: ToltecMasterType.clip,
        }.get(master)
        assert result is not None
        return result


class ToltecDBRawObsType(IntEnum):
    """The is in line with the ``toltec/obstype`` table in the toltec db."""

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
    def get_data_kind(cls, raw_obs_type: Self) -> ToltecDataKind:
        """Return the data kind for ``raw_obs_type``."""
        return {
            0: ToltecDataKind.RawTimeStream,
            1: ToltecDataKind.RawTimeStream,
            2: ToltecDataKind.VnaSweep,
            3: ToltecDataKind.TargetSweep,
            4: ToltecDataKind.Tune,
        }.get(raw_obs_type, ToltecDataKind.Unknown)
