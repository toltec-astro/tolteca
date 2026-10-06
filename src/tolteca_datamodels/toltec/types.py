"""TolTEC data types."""

from __future__ import annotations

from enum import Flag, StrEnum, auto
from typing import Annotated, ClassVar, Literal

import astropy.units as u
from pydantic import Field
from tollan.utils.typing import get_literal_values

from ..lmt_base.metadata import LmtInstruMixin
from .toltecdb import ToltecDBRawObsMasterCode, ToltecDBRawObsTypeCode

__all__ = [
    "RoachT",
    "ToltecArray",
    "ToltecArrayNameT",
    "ToltecDataKind",
    "ToltecInfo",
    "ToltecMaster",
    "ToltecMasterNameT",
]


class ToltecDataKind(Flag):
    """TolTEC data kinds."""

    # Raw KIDs data
    vnasweep = auto()
    """A full range sweep on a regular grid."""

    targsweep = auto()
    """A target sweep on a list of frequencies."""

    tune = auto()
    """A pack of two back-to-back target sweeps to improve the probe tones."""

    raw_sweep = vnasweep | targsweep | tune
    """The sum of all sweep kinds."""

    raw_timestream = auto()
    """A continuous capturing of the data at given probe tones."""

    raw_kids_data = raw_sweep | raw_timestream
    """The sum of all raw kids data kinds."""

    # Reduced KIDs data kinds
    d21 = auto()
    reduced_vnasweep = auto()
    reduced_targsweep = auto()
    reduced_sweep = reduced_vnasweep | reduced_targsweep
    solved_timestream = auto()
    reduced_kids_data = d21 | reduced_sweep | solved_timestream

    # Raw and reduced sum types
    sweep = raw_sweep | reduced_sweep
    timestream = raw_timestream | solved_timestream
    kids_data = raw_kids_data | reduced_kids_data

    # Observing-time reduction data types.

    targ_freqs_dat = auto()
    """The legacy targ_freqs.dat consumed by tlaloc."""

    targ_amps_dat = auto()
    """The legacy default_amps.dat consumed by tlaloc."""

    kids_model_params_table = auto()
    """The legacy kids model fitting result from kidscpp."""

    kids_prop_table = auto()
    """The KIDs finding/fitting results."""

    tone_prop_table = auto()
    """The table listing tone properties, derived from KPT."""

    chan_prop_table = auto()
    """The table listing channel properties, derived from KPT."""

    array_prop_table = auto()
    """The array property table derived from beammapping."""

    pointing_table = auto()
    """The pointing property table derived from pointing observation."""

    kids_table_data = (
        kids_model_params_table | kids_prop_table | tone_prop_table | chan_prop_table
    )
    table_data = kids_table_data | array_prop_table | pointing_table

    # Raw infrastructural kinds
    hwpr = auto()
    """The half wave place rotation data."""

    wyatt = auto()
    """The wyatt robot arm."""

    lmt_tel = auto()
    """LMT telescope file"""

    lmt_tel2 = auto()
    """Supplementary LMT telescope file with data at their original sample rate."""

    housekeeping = auto()
    """The house keeping data."""

    # settings and configurations.
    lmt_ot_script = auto()
    """LMT OT script."""

    tolteca_config = auto()
    """TolTECA Yaml config."""

    unknown = auto()
    """Unknown data."""


class ToltecMaster(StrEnum):
    """TolTEC acquisition masters."""

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


class ToltecArray(StrEnum):
    """TolTEC detector arrays."""

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

    type InstruT = Literal["toltec"]
    type MasterT = ToltecMasterNameT
    type MasterCodeT = Annotated[
        int, Field(ge=min(ToltecDBRawObsMasterCode), le=max(ToltecDBRawObsMasterCode))
    ]
    type ObsTypeT = Literal["nominal", "timestream", "vna", "targ", "tune"]
    type ObsTypeCodeT = Annotated[
        int, Field(ge=min(ToltecDBRawObsTypeCode), le=max(ToltecDBRawObsTypeCode))
    ]
    type ArrayT = Annotated[int, Field(ge=0, le=2)]
    type ArrayNameT = ToltecArrayNameT

    obs_type_code_obs_type: ClassVar[dict[int, str]] = {
        code.value: code.name.lower() for code in ToltecDBRawObsTypeCode
    }
    """Recorded acquisition types, independent of individual block data kinds."""

    masters: ClassVar[list[ToltecMasterNameT]] = get_literal_values(ToltecMasterNameT)

    master_code_master: ClassVar[dict[int, ToltecMaster]] = {
        master_code.value: ToltecDBRawObsMasterCode.get_master(master_code)
        for master_code in ToltecDBRawObsMasterCode
    }
    """Supported acquisition-master codes and their labels."""
    master_master_code: ClassVar[dict[str, int]] = {
        master: master_code for master_code, master in master_code_master.items()
    }

    # roach interfaces
    roaches: ClassVar = list(range(13))
    roach_interface: ClassVar = {roach: f"toltec{roach}" for roach in roaches}
    interface_roach: ClassVar = {v: k for k, v in roach_interface.items()}
    roach_interfaces: ClassVar = list(roach_interface.values())

    # TolTEC-owned readout, auxiliary hardware, and housekeeping.
    interface_instru: ClassVar[dict[str, LmtInstruMixin]] = {
        **{
            interface: LmtInstruMixin(
                interface=interface, instru="toltec", instru_component="roach"
            )
            for interface in roach_interfaces
        },
        "hwpr": LmtInstruMixin(
            interface="hwpr", instru="toltec", instru_component="hwpr"
        ),
        "wyatt": LmtInstruMixin(
            interface="wyatt", instru="toltec", instru_component="wyatt"
        ),
        "toltec_hk": LmtInstruMixin(
            interface="toltec_hk", instru="toltec", instru_component="hk"
        ),
    }
    interfaces: ClassVar = list(interface_instru)

    # arrays
    arrays: ClassVar = list(range(3))
    array_names: ClassVar[list[ToltecArrayNameT]] = get_literal_values(ToltecArrayNameT)
    array_array_name: ClassVar = dict(zip(arrays, array_names, strict=True))
    array_name_array: ClassVar = {v: k for k, v in array_array_name.items()}

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


type RoachT = Annotated[
    int, Field(ge=min(ToltecInfo.roaches), le=max(ToltecInfo.roaches))
]
