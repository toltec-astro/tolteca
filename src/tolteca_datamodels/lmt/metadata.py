"""LMT telescope metadata models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic.dataclasses import dataclass
from tollan.config.types import SkyCoordField

from ..core import metadata_dataclass_config
from ..lmt_base import LmtRawObsMetadata
from .types import LmtObsGoalType

__all__ = ["LmtTelMetadata"]


@dataclass(config=metadata_dataclass_config, frozen=True, kw_only=True)
class LmtTelMetadata(LmtRawObsMetadata):
    """LMT telescope control system (TCS) metadata."""

    # Override base fields with LMT telescope-specific types and defaults
    master: Literal["tcs"] | str = Field(
        default="tcs",
        description="Master controller",
    )
    instru: Literal["lmt"] | str = Field(
        default="lmt",
        description="LMT instrument",
    )
    instru_component: Literal["tel"] | str = Field(
        default="tel",
        description="Telescope component",
    )
    obs_goal: LmtObsGoalType = Field(
        default=LmtObsGoalType.unspecified,
        description="Observation goal",
    )
    target: SkyCoordField | None = Field(
        default=None,
        description="Target sky coordinate",
    )
    target_off: SkyCoordField | None = Field(
        default=None,
        description="Target offset coordinate",
    )
