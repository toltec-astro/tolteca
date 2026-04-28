"""LMT base module - shared constructs for LMT telescope data.

This module provides base classes and mixins that are used by both
the LMT telescope package and instrument-specific packages (like TolTEC)
to avoid circular dependencies.
"""

from __future__ import annotations

from .metadata import (
    LmtInstruMixin,
    LmtObsQuartetMixin,
    LmtRawObsMetadata,
)

__all__ = [
    "LmtInstruMixin",
    "LmtObsQuartetMixin",
    "LmtRawObsMetadata",
]
