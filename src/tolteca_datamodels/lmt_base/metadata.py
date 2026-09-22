"""LMT base metadata models.

Pydantic dataclasses with Field descriptions for self-documenting metadata.
All metadata classes are frozen (immutable) DTOs.
"""

from __future__ import annotations

from functools import cached_property

import astropy.units as u
from pydantic import Field, SerializeAsAny, computed_field
from pydantic.dataclasses import dataclass
from tollan.config.types import TimeQuantityField, UnixTimeField

from ..core import metadata_dataclass_config
from .file import FilepathInfo

__all__ = [
    "LmtInstruMixin",
    "LmtObsQuartetMixin",
    "LmtRawObsMetadata",
]


@dataclass(config=metadata_dataclass_config, frozen=True, kw_only=True)
class LmtObsQuartetMixin:
    """Mixin for the LMT observation quartet."""

    master: str | int | None = Field(
        description="Recorded master controller name or code, or None when unavailable",
    )
    obsnum: int = Field(description="Observation number")
    subobsnum: int = Field(description="Sub-observation number")
    scannum: int = Field(description="Scan number")

    @computed_field
    @cached_property
    def master_str(self) -> str | None:
        """Master controller name or code as text, or None when unavailable."""
        return None if self.master is None else str(self.master)

    @computed_field
    @cached_property
    def quartet(self) -> str | None:
        """Return ``master-obsnum-subobsnum-scannum``, without zero padding.

        The master component is ``master_str``. Return None when that name is
        unavailable.
        """
        if self.master_str is None:
            return None
        return f"{self.master_str}-{self.obsnum}-{self.subobsnum}-{self.scannum}"


@dataclass(config=metadata_dataclass_config, frozen=True, kw_only=True)
class LmtInstruMixin:
    """Mixin for LMT instrument metadata."""

    interface: str = Field(description="Data interface identifier")
    instru: str = Field(description="Instrument name")
    instru_component: str | None = Field(
        default=None,
        description="Instrument component or subsystem",
    )


@dataclass(config=metadata_dataclass_config, frozen=True, kw_only=True)
class LmtRawObsMetadata(LmtObsQuartetMixin, LmtInstruMixin):
    """LMT raw observation metadata.

    ``t0`` and ``t1`` are scalar Astropy times. Serialized records store them as
    Unix seconds, matching the acquisition timestamps. Construct them from
    ``Time`` objects or numeric Unix timestamps.
    """

    t0: UnixTimeField = Field(description="Observation start time")
    t1: UnixTimeField = Field(description="Observation end time")
    t_int: TimeQuantityField = Field(description="Integration time")
    filepath_orig: str | None = Field(
        default=None,
        description="Original acquisition file path",
    )

    @computed_field
    @cached_property
    def t_elapsed(self) -> TimeQuantityField:
        """Elapsed time between t0 and t1, in seconds."""
        return (self.t1 - self.t0).to(u.s)

    @computed_field(repr=False)
    @cached_property
    def filepath_info_orig(self) -> SerializeAsAny[FilepathInfo] | None:
        """Parsed acquisition path, or None when absent or unrecognized."""
        if self.filepath_orig is None:
            return None
        return FilepathInfo.from_path(self.filepath_orig, raise_on_error=False)

    @computed_field
    @cached_property
    def uid_raw_obs(self) -> str | None:
        """Identifier of the raw observation, or None when unavailable."""
        return self.quartet

    @computed_field
    @cached_property
    def uid_raw_obs_file(self) -> str | None:
        """Identifier of the observation's interface data, or None if unavailable."""
        if self.quartet is None:
            return None
        return f"{self.quartet}-{self.interface}"
