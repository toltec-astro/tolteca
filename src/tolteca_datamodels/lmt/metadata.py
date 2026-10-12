"""LMT telescope metadata models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from tollan.config.types import SkyCoordField

from ..lmt_base._filepath import FilepathOrigMetadataMixin
from ..lmt_base.metadata import LmtRawObsMetadata
from ._filepath import LmtFilepathInfo
from .types import LmtInfo, LmtMaster, LmtObsGoal, LmtObsPgm, LmtRawObsMasterCode

__all__ = [
    "LmtTelInterfaceKindT",
    "LmtTelInterfaceT",
    "LmtTelMetadata",
]

type LmtTelInterfaceT = Literal["tel"]
type LmtTelInterfaceKindT = Literal["tel"]


class LmtTelMetadata(LmtRawObsMetadata, FilepathOrigMetadataMixin[LmtFilepathInfo]):
    """LMT telescope control system (TCS) metadata."""

    producer: LmtInfo.InstruT = Field(
        description="Large Millimeter Telescope",
    )
    interface: LmtTelInterfaceT = Field(description="Data interface identifier")
    interface_kind: LmtTelInterfaceKindT = Field(
        description="Telescope data interface",
    )
    master: LmtMaster = Field(description="Acquisition master label")
    master_code: LmtRawObsMasterCode = Field(
        description="Recorded acquisition master code",
    )
    obs_pgm: LmtObsPgm = Field(description="Observation program")
    obs_goal: LmtObsGoal = Field(
        description="Observation goal",
    )
    target: SkyCoordField = Field(
        description="Main target sky coordinate",
    )
    target_ref: SkyCoordField = Field(
        description="Reference target sky coordinate",
    )
