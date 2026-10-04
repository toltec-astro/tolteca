"""I/Q views, sweep derivatives, and timestream spectra for KIDs data.

Import this package to register the ``kids`` accessor on xarray Datasets and
DataTrees. Use ``accessors.make_kids_dataset`` to construct inputs and the
``analysis`` package to compute D21 or PSD results.
"""

from __future__ import annotations

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
