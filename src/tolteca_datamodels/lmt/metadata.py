"""LMT telescope metadata models."""

from __future__ import annotations

from functools import cached_property
from typing import Literal

import astropy.units as u
from pydantic import Field, computed_field
from pydantic.dataclasses import dataclass
from tollan.config.types import SkyCoordField, TimeQuantityField

from ..core import metadata_dataclass_config
from ..lmt_base import LmtRawObsMetadata
from ..toltec.file import FilepathInfo
from ..toltec.toltecdb import ToltecDBRawObsMaster

__all__ = ["LmtTelMetadata"]


@dataclass(config=metadata_dataclass_config, frozen=True, kw_only=True)
class LmtTelMetadata(LmtRawObsMetadata):
    """LMT telescope control system (TCS) metadata."""

    master: ToltecDBRawObsMaster | None = Field(
        description="Recorded acquisition master code, or None when unavailable",
    )
    instru: Literal["lmt"] = Field(
        default="lmt",
        description="Large Millimeter Telescope.",
    )
    instru_component: Literal["tcs"] = Field(
        default="tcs",
        description="Telescope Control System.",
    )
    obs_pgm: str = Field(description="Observation program")
    obs_goal: str = Field(
        description="Observation goal",
    )
    target: SkyCoordField = Field(
        description="Main target sky coordinate",
    )
    target_ref: SkyCoordField = Field(
        description="Reference target sky coordinate",
    )
    filepath_orig: str = Field(description="Original acquisition file path")

    @computed_field
    @cached_property
    def t_elapsed(self) -> TimeQuantityField:
        """Elapsed time between the first and last telescope samples, in seconds.

        This is ``t1 - t0`` from the telescope timestamps, not the header's
        integration time ``t_exp``. The two durations need not agree.
        """
        return (self.t1 - self.t0).to(u.s)

    @computed_field
    @cached_property
    def master_str(self) -> str | None:
        """Return the acquisition master's name, or None when unavailable."""
        if self.master is None:
            return None
        return ToltecDBRawObsMaster.get_master_type(self.master)

    @computed_field(repr=False)
    @cached_property
    def filepath_info_orig(self) -> FilepathInfo:
        """Parsed identifiers from the recorded acquisition path.

        An unrecognized acquisition filename raises ValueError.
        """
        parsed = FilepathInfo.from_path(self.filepath_orig)
        assert parsed is not None
        return parsed
