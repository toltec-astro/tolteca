"""TolTEC data types."""

from __future__ import annotations

from enum import Flag, auto
from typing import Annotated, ClassVar, Literal

import astropy.units as u
from pydantic import Field
from tollan.utils.typing import get_literal_value, get_literal_values

from ..core.types import DataModelInfo
from ..lmt.types import LmtDataModel, LmtInfo
from ..lmt_base.metadata import InterfaceInfoMixin
from .toltec_db import ToltecDBRawObsMasterCode, ToltecDBRawObsTypeCode

__all__ = [
    "ToltecDataKind",
    "ToltecDataModel",
    "ToltecInfo",
]


class ToltecDataModel(DataModelInfo):
    """TolTEC data-model information, including telescope companions."""

    name = "toltec"


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


def _make_roach_interface_info(
    interfaces: list[str], *, instru: str, data_model: str
) -> dict[str, InterfaceInfoMixin]:
    """Describe the ROACH interfaces with a shared instrument and data model."""
    return {
        interface: InterfaceInfoMixin(
            interface=interface,
            instru=instru,
            data_models=frozenset({data_model}),
            interface_kind="roach",
        )
        for interface in interfaces
    }


class ToltecInfo:
    """Toltec instrument information and constants."""

    type InstruT = Literal["toltec"]
    instru: ClassVar[InstruT] = get_literal_value(InstruT)
    type MasterT = Literal["tcs", "ics", "clip"]
    type ObsTypeT = Literal["nominal", "timestream", "vna", "targ", "tune"]
    type ObsTypeCodeT = Annotated[
        int, Field(ge=min(ToltecDBRawObsTypeCode), le=max(ToltecDBRawObsTypeCode))
    ]
    type ArrayT = Annotated[int, Field(ge=0, le=2)]
    type ArrayNameT = Literal["a1100", "a1400", "a2000"]

    obs_type_code_obs_type: ClassVar[dict[int, ObsTypeT]] = {
        code.value: ToltecDBRawObsTypeCode.get_obs_type(code)
        for code in ToltecDBRawObsTypeCode
    }
    """Recorded acquisition types, independent of individual block data kinds."""

    obs_type_data_kind: ClassVar[dict[ObsTypeT, ToltecDataKind]] = {
        "nominal": ToltecDataKind.raw_timestream,
        "timestream": ToltecDataKind.raw_timestream,
        "vna": ToltecDataKind.vnasweep,
        "targ": ToltecDataKind.targsweep,
        "tune": ToltecDataKind.tune,
    }
    """Raw KIDs data kinds for acquisition observation-type labels."""

    masters: ClassVar[list[MasterT]] = get_literal_values(MasterT)
    master_description: ClassVar[dict[MasterT, str]] = {
        "tcs": "Telescope control system",
        "ics": "Instrument control system",
        "clip": "ROACH manager",
    }
    """Descriptions of acquisition master labels."""

    master_code_master: ClassVar[dict[int, MasterT]] = {
        master_code.value: ToltecDBRawObsMasterCode.get_master(master_code)
        for master_code in ToltecDBRawObsMasterCode
    }
    """Supported acquisition-master codes and their labels."""
    master_master_code: ClassVar[dict[MasterT, int]] = {
        master: master_code for master_code, master in master_code_master.items()
    }

    # roach interfaces
    roaches: ClassVar = list(range(13))
    type RoachT = Annotated[int, Field(ge=min(roaches), le=max(roaches))]

    roach_interface: ClassVar = {roach: f"toltec{roach}" for roach in roaches}
    interface_roach: ClassVar = {v: k for k, v in roach_interface.items()}
    roach_interfaces: ClassVar = list(roach_interface.values())

    # TolTEC readouts, auxiliary streams, and companion telescope data.
    interface_interface_info: ClassVar[dict[str, InterfaceInfoMixin]] = {
        **_make_roach_interface_info(
            roach_interfaces, instru=instru, data_model=ToltecDataModel.name
        ),
        "hwpr": InterfaceInfoMixin(
            interface="hwpr",
            instru=instru,
            data_models=frozenset({ToltecDataModel.name}),
            interface_kind="hwpr",
        ),
        "wyatt": InterfaceInfoMixin(
            interface="wyatt",
            instru=instru,
            data_models=frozenset({ToltecDataModel.name}),
            interface_kind="wyatt",
        ),
        "toltec_hk": InterfaceInfoMixin(
            interface="toltec_hk",
            instru=instru,
            data_models=frozenset({ToltecDataModel.name}),
            interface_kind="toltec_hk",
        ),
        "tel": LmtInfo.interface_interface_info["tel"].model_copy(
            update={"data_models": frozenset({LmtDataModel.name, ToltecDataModel.name})}
        ),
    }
    interfaces: ClassVar = list(interface_interface_info)

    # arrays
    arrays: ClassVar = list(range(3))
    array_names: ClassVar[list[ArrayNameT]] = get_literal_values(ArrayNameT)
    array_name_description: ClassVar[dict[ArrayNameT, str]] = {
        "a1100": "1.1 mm detector array",
        "a1400": "1.4 mm detector array",
        "a2000": "2.0 mm detector array",
    }
    """Descriptions of detector array names."""
    array_array_name: ClassVar = dict(zip(arrays, array_names, strict=True))
    array_name_array: ClassVar = {v: k for k, v in array_array_name.items()}

    interface_array_name: ClassVar[dict[str, ArrayNameT]] = {
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
