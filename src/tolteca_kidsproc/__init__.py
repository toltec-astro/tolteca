"""Generic KIDs data processing utilities.

Provides reference implementations of schema and mapper classes for
KIDs sweep and timestream data using tollan's accessor framework.

Also provides xarray accessor registration for convenient dataset access.
"""

from __future__ import annotations

# Export reference schema and mapper implementations
from .accessors.kids import (
    KidsAccessor,
    KidsMapper,
    KidsSchema,
)
from .accessors.views import (
    MultiSweepView,
    MultiTimestreamView,
    SweepView,
    TimestreamView,
)

__all__ = [
    "KidsAccessor",
    "KidsMapper",
    "KidsSchema",
    "MultiSweepView",
    "MultiTimestreamView",
    "SweepView",
    "TimestreamView",
]
