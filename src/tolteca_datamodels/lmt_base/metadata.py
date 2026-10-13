"""LMT observation metadata."""

from __future__ import annotations

from functools import cached_property

import astropy.units as u
from pydantic import Field, computed_field
from tollan.config.types import TimeQuantityField, UnixTimeField

from ..core import DataModelsT, FrozenDataModel
from .funcs import make_quartet, make_uid_obs, make_uid_raw_obs, make_uid_raw_obs_file
from .types import LabelT, SeqNumT

__all__ = [
    "InterfaceInfoMixin",
    "LmtObsQuartetMixin",
    "LmtRawObsMetadata",
]


class LmtObsQuartetMixin(FrozenDataModel):
    """Mixin for the LMT observation quartet."""

    master: LabelT | None = Field(
        description="Acquisition master label",
    )
    obsnum: SeqNumT = Field(description="Observation number")
    subobsnum: SeqNumT = Field(description="Sub-observation number")
    scannum: SeqNumT = Field(description="Scan number")

    @computed_field
    @cached_property
    def quartet(self) -> str | None:
        """Identifier of the observation."""
        return make_quartet(self.master, self.obsnum, self.subobsnum, self.scannum)


class InterfaceInfoMixin(FrozenDataModel):
    """Complete identity and classification of a data interface."""

    instru: LabelT = Field(description="Instrument producing the data")
    data_models: DataModelsT = Field(description="Applicable data models")
    interface_kind: LabelT = Field(description="Class of acquisition interface")
    interface: LabelT = Field(description="Data interface identifier")


class LmtRawObsMetadata(LmtObsQuartetMixin, InterfaceInfoMixin):
    """LMT raw observation metadata."""

    t0: UnixTimeField = Field(description="Observation start time")
    t1: UnixTimeField = Field(description="Observation end time")
    t_int: TimeQuantityField = Field(description="Integration time")

    @computed_field
    @cached_property
    def t_elapsed(self) -> TimeQuantityField:
        """Elapsed observation time."""
        return (self.t1 - self.t0).to(u.s)

    @computed_field
    @cached_property
    def uid_obs(self) -> LabelT | None:
        """Identifier of the observation."""
        return make_uid_obs(self.master, self.obsnum)

    @computed_field
    @cached_property
    def uid_raw_obs(self) -> LabelT | None:
        """Identifier of the raw observation."""
        return make_uid_raw_obs(self.master, self.obsnum, self.subobsnum, self.scannum)

    @computed_field
    @cached_property
    def uid_raw_obs_file(self) -> LabelT | None:
        """Identifier of the raw observation file."""
        return make_uid_raw_obs_file(
            self.master, self.obsnum, self.subobsnum, self.scannum, self.interface
        )
