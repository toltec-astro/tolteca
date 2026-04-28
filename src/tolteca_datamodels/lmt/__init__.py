"""LMT telescope-specific data models.

This package contains metadata and types specific to LMT telescope systems
(TCS, environmental monitoring, etc.).
"""

from __future__ import annotations

from .metadata import LmtTelMetadata
from .types import LmtObsGoalType

__all__ = ["LmtObsGoalType", "LmtTelMetadata"]
