"""LMT telescope-specific data models.

This package contains metadata and types specific to LMT telescope systems
(TCS, environmental monitoring, etc.).
"""

from __future__ import annotations

from .metadata import LmtTelMetadata
from .tel import LmtTelAccessor, LmtTelIOMapper, LmtTelIOSchema, open_lmt_tel
from .types import LmtObsGoalType

__all__ = [
    "LmtObsGoalType",
    "LmtTelAccessor",
    "LmtTelIOMapper",
    "LmtTelIOSchema",
    "LmtTelMetadata",
    "open_lmt_tel",
]
