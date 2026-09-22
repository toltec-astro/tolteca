"""LMT base metadata models.

Pydantic dataclasses with Field descriptions for self-documenting metadata.
All metadata classes are frozen (immutable) DTOs.
"""

from __future__ import annotations

from functools import cached_property

from pydantic import Field, computed_field
from pydantic.dataclasses import dataclass
from tollan.config.types import TimeQuantityField, UnixTimeField

from ..core import metadata_dataclass_config

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
        """Master controller name, or None when it is unavailable.

        Names are retained as supplied. Metadata classes with a controller-code
        mapping override this property to translate their numeric codes.
        """
        return self.master if isinstance(self.master, str) else None

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
    t_exp: TimeQuantityField = Field(description="Integration time")

    @computed_field
    @cached_property
    def uid_raw_obs(self) -> str | None:
        """Raw observation identifier, equal to ``quartet``."""
        return self.quartet

    @computed_field
    @cached_property
    def uid_raw_obs_file(self) -> str | None:
        """Return ``quartet-interface``, or None when the quartet is unavailable.

        This identifies an acquisition's interface, not a file's storage
        location or a particular copy of its contents.
        """
        if self.quartet is None:
            return None
        return f"{self.quartet}-{self.interface}"
