"""TolTEC data types."""

from __future__ import annotations

from enum import Flag, StrEnum, auto
from typing import ClassVar, Literal, get_args

import astropy.units as u

__all__ = [
    "ToltecArrayNameT",
    "ToltecArrayType",
    "ToltecDataKind",
    "ToltecInfo",
    "ToltecMasterNameT",
    "ToltecMasterType",
]


class ToltecDataKind(Flag):
    """TolTEC data kinds."""

    # Raw KIDs data
    VnaSweep = auto()
    """A full range sweep on a regular grid."""

    TargetSweep = auto()
    """A target sweep on a list of frequencies."""

    Tune = auto()
    """A pack of two back-to-back target sweeps to improve the probe tones."""

    RawSweep = VnaSweep | TargetSweep | Tune
    """The sum of all sweep kinds."""

    RawTimeStream = auto()
    """A continuous capturing of the data at given probe tones."""

    RawKidsData = RawSweep | RawTimeStream
    """The sum of all raw kids data kinds."""

    # Reduced KIDs data kinds
    D21 = auto()
    ReducedVnaSweep = auto()
    ReducedTargetSweep = auto()
    ReducedSweep = ReducedVnaSweep | ReducedTargetSweep
    SolvedTimeStream = auto()
    ReducedKidsData = D21 | ReducedSweep | SolvedTimeStream

    # Raw and reduced sum types
    Sweep = RawSweep | ReducedSweep
    TimeStream = RawTimeStream | SolvedTimeStream
    KidsData = RawKidsData | ReducedKidsData

    # Observing-time reduction data types.

    TargFreqsDat = auto()
    """The legacy targ_freqs.dat consumed by tlaloc."""

    TargAmpsDat = auto()
    """The legacy default_amps.dat consumed by tlaloc."""

    KidsModelParamsTable = auto()
    """The legacy kids model fitting result from kidscpp."""

    KidsPropTable = auto()
    """The KIDs finding/fitting results."""

    TonePropTable = auto()
    """The table listing tone properties, derived from KPT."""

    ChanPropTable = auto()
    """The table listing channel properties, derived from KPT."""

    ArrayPropTable = auto()
    """The array property table derived from beammapping."""

    PointingTable = auto()
    """The pointing property table derived from pointing observation."""

    KidsTableData = KidsModelParamsTable | KidsPropTable | TonePropTable | ChanPropTable
    TableData = KidsTableData | ArrayPropTable | PointingTable

    # Raw infrastructural kinds
    Hwpr = auto()
    """The half wave place rotation data."""

    Wyatt = auto()
    """The wyatt robot arm."""

    LmtTel = auto()
    """LMT telescope file"""

    LmtTel2 = auto()
    """Supplementary LMT telescope file with data at their original sample rate."""

    HouseKeeping = auto()
    """The house keeping data."""

    # settings and configurations.
    LmtOtScript = auto()
    """LMT OT script."""

    ToltecaConfig = auto()
    """TolTECA Yaml config."""

    Unknown = auto()
    """Unknown data."""


class ToltecMasterType(StrEnum):
    """Toltec master types."""

    tcs = auto()
    """The telescope control system."""

    ics = auto()
    """The instrument control system."""

    clip = auto()
    """The ROACH manager."""


type ToltecMasterNameT = Literal[
    "tcs",
    "ics",
    "clip",
]
"""Toltec master names."""


class ToltecArrayType(StrEnum):
    """Toltec array types."""

    a1100 = auto()
    """The 1.1 mm array."""

    a1400 = auto()
    """The 1.4 mm array."""

    a2000 = auto()
    """The 2.0 mm array."""


type ToltecArrayNameT = Literal[
    "a1100",
    "a1400",
    "a2000",
]
"""Toltec array names."""


class ToltecInfo:
    """Toltec instrument information and constants."""

    masters: ClassVar[list[ToltecMasterNameT]] = list(get_args(ToltecMasterNameT))

    # roach interfaces
    roaches: ClassVar = list(range(13))
    roach_interface: ClassVar = {roach: f"toltec{roach}" for roach in roaches}
    interface_roach: ClassVar = {v: k for k, v in roach_interface.items()}
    roach_interfaces: ClassVar = list(roach_interface.values())

    # interfaces
    interfaces: ClassVar = [*roach_interfaces, "hwpr"]

    # arrays
    arrays: ClassVar = list(range(3))
    array_names: ClassVar[list[ToltecArrayNameT]] = list(get_args(ToltecArrayNameT))

    interface_array_name: ClassVar[dict[str, ToltecArrayNameT]] = {
        "toltec0": "a1100",
        "toltec1": "a1100",
        "toltec2": "a1100",
        "toltec3": "a1100",
        "toltec4": "a1100",
        "toltec5": "a1100",
        "toltec6": "a1100",
        "toltec7": "a1400",
        "toltec8": "a1400",
        "toltec9": "a1400",
        "toltec10": "a1400",
        "toltec11": "a2000",
        "toltec12": "a2000",
    }
    fov_diameter: ClassVar = 4 << u.arcmin
