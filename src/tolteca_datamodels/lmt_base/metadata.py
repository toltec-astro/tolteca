"""LMT base metadata models.

Pydantic dataclasses with Field descriptions for self-documenting metadata.
All metadata classes are frozen (immutable) DTOs.
"""

from __future__ import annotations

from pydantic import Field
from pydantic.dataclasses import dataclass
from tollan.config.types import TimeField, TimeQuantityField

from ..core import metadata_dataclass_config

__all__ = [
    "LmtInstruMixin",
    "LmtObsQuartetMixin",
    "LmtRawObsMetadata",
]


@dataclass(config=metadata_dataclass_config, frozen=True, kw_only=True)
class LmtObsQuartetMixin:
    """Mixin for the LMT observation quartet."""

    master: str = Field(description="Master controller")
    obsnum: int = Field(description="Observation number")
    subobsnum: int = Field(description="Sub-observation number")
    scannum: int = Field(description="Scan number")


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
    """LMT raw observation metadata."""

    t0: TimeField = Field(description="Observation start time")
    t1: TimeField = Field(description="Observation end time")
    t_exp: TimeQuantityField = Field(description="Integration time")
