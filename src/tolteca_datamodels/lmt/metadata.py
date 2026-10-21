"""LMT telescope metadata models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from tollan.config.types import SkyCoordField

from ..lmt_base.filepath import FilepathOrigMetadataMixin
from ..lmt_base.metadata import LmtRawObsMetadata
from .types import LmtInfo, LmtObsGoal, LmtObsPgm, LmtRawObsMasterCode

__all__ = [
    "LmtTelInterfaceKindT",
    "LmtTelInterfaceT",
    "LmtTelMetadata",
]

type LmtTelInterfaceT = Literal["tel"]
type LmtTelInterfaceKindT = Literal["tel"]


class LmtTelMetadata(LmtRawObsMetadata, FilepathOrigMetadataMixin):
    """LMT telescope control system (TCS) metadata."""

    instru: LmtInfo.InstruT = Field(
        description="Large Millimeter Telescope",
    )
    interface: LmtTelInterfaceT = Field(description="Data interface identifier")
    interface_kind: LmtTelInterfaceKindT = Field(
        description="Telescope data interface",
    )
    master: LmtInfo.MasterT = Field(description="Acquisition master label")
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
