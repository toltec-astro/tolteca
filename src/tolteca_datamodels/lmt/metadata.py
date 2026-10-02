"""LMT telescope metadata models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from tollan.config.types import SkyCoordField

from ..lmt_base import LmtRawObsMetadata
from ..lmt_base._filepath import FilepathOrigMetadataMixin
from ..lmt_base.types import LabelT
from ._filepath import LmtFilepathInfo
from .types import LmtInfo, LmtObsGoal, LmtObsPgm

__all__ = ["LmtTelMetadata"]


class LmtTelMetadata(LmtRawObsMetadata, FilepathOrigMetadataMixin[LmtFilepathInfo]):
    """LMT telescope control system (TCS) metadata."""

    master: LabelT = Field(description="Acquisition master label")
    master_code: int = Field(
        description="Recorded acquisition master code",
    )
    instru: LmtInfo.InstruT = Field(
        default=LmtInfo.instru,
        description="Large Millimeter Telescope.",
    )
    instru_component: Literal["tcs"] = Field(
        default="tcs",
        description="Telescope Control System.",
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
