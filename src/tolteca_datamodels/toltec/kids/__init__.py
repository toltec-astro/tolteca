"""TolTEC KIDs data models and analysis.

This submodule provides schemas, mappers, and analysis classes for TolTEC
KIDs detector data, including raw data I/O and reduced data processing.

The module is organized to provide clear namespacing:
- core.py: I/O schema, mapper, and accessor
  (ToltecKidsIOSchema, ToltecKidsIOMapper, ToltecKidsAccessor)
- sweep.py: Sweep data schema, reducer, and views (namespaced)
  (ToltecSweepSchema, ToltecSweepMapper, SweepReducer, ReducedSweepView)
- timestream.py: Timestream reducer with PSD analysis (namespaced)
  (ToltecTimestreamSchema, ToltecTimestreamMapper, TimestreamReducer,
   ReducedTimestreamView)

Examples
--------
Access raw TolTEC data:
    >>> import xarray as xr
    >>> ds = xr.open_dataset("toltec_raw.nc")
    >>> ds.toltec_kids.meta.roach
    >>> ds.toltec_kids.f_tone

Reduce sweep data:
    >>> from tolteca_datamodels.toltec.kids import SweepReducer
    >>> reducer = SweepReducer()
    >>> dt = reducer(ds_raw)  # returns xr.DataTree
    >>> ds_reduced = dt.children["tolteca_datamodels.toltec.kids.sweep"].dataset

Access reduced sweep data via view or accessor:
    >>> from tolteca_datamodels.toltec.kids import ReducedSweepView
    >>> view = ReducedSweepView(dt)       # direct: auto-resolves child node
    >>> view = dt.toltec_kids.sweep       # accessor entry point
    >>> view.I
    >>> view.unc_I

Reduce timestream and compute PSDs:
    >>> from tolteca_datamodels.toltec.kids import TimestreamReducer
    >>> reducer = TimestreamReducer(psd_nperseg=1024, psd_stat_freq_range=(10.0, 100.0))
    >>> dt = reducer(ds_timestream)  # returns xr.DataTree
    >>> view = dt.toltec_kids.timestream          # accessor entry point
    >>> view.I_psd
    >>> view.I_psd_median
"""

from __future__ import annotations

from .core import (
    ToltecKidsAccessor,
    ToltecKidsIOMapper,
    ToltecKidsIOSchema,
)
from .zarr import open_zarr_dataset
from .sweep import (
    ReducedSweepView,
    SweepReducer,
    ToltecSweepMapper,
    ToltecSweepSchema,
)
from .timestream import (
    ReducedTimestreamView,
    TimestreamReducer,
    ToltecTimestreamMapper,
    ToltecTimestreamSchema,
)

__all__ = [
    "open_zarr_dataset",
    "ReducedSweepView",
    "ReducedTimestreamView",
    "SweepReducer",
    "TimestreamReducer",
    "ToltecKidsAccessor",
    "ToltecKidsIOMapper",
    "ToltecKidsIOSchema",
    "ToltecSweepMapper",
    "ToltecSweepSchema",
    "ToltecTimestreamMapper",
    "ToltecTimestreamSchema",
]
