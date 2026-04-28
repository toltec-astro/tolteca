"""TolTEC KIDs pipeline steps.

This package provides the pipeline step implementations for TolTEC KIDs
data processing, building on the ``tollan.pipeline`` step system and the
``tolteca_datamodels.toltec.kids`` reducers/views.

Pipeline steps store their typed context objects in
``data.attrs["__pipeline_context__"]`` on the ``xr.DataTree`` that flows
through the pipeline.  Access any step's context with::

    ctx = SweepCheck.get_context(dt)   # → SweepCheckContext (typed)
    ctx.data.bitmask_chan              # channel-level bitmask array

Typical pipeline use:

    from tolteca_datamodels.toltec.kids import SweepReducer
    from tolteca_kids import SweepCheck, KidsFind, ToltecKidsPipeline

    reducer = SweepReducer()
    dt = reducer(ds_raw)              # xr.DataTree with reduced sweep

    pipeline = ToltecKidsPipeline()   # assembles steps with defaults
    dt = pipeline(dt)                 # runs SweepCheck → KidsFind → ...

    ctx_sc = SweepCheck.get_context(dt)
    ctx_kf = KidsFind.get_context(dt)
"""

from __future__ import annotations

from .kids_find import (
    KidsFind,
    KidsFindConfig,
    KidsFindContext,
    KidsFindData,
    SegmentBitMask,
)
from .kids_plot import make_kids_find_figs
from .sweep_check import (
    SweepBitMask,
    SweepCheck,
    SweepCheckConfig,
    SweepCheckContext,
    SweepCheckData,
)
from .pipeline import (
    KIDS_FIND_GROUP,
    SWEEP_CHECK_GROUP,
    KidsPipeline,
    KidsPipelineConfig,
    KidsPipelineResult,
    has_kids_reduction,
    read_kids_find,
    read_sweep_check,
)

__all__ = [
    "KidsFind",
    "KidsFindConfig",
    "KidsFindContext",
    "KidsFindData",
    "SegmentBitMask",
    "SweepBitMask",
    "SweepCheck",
    "SweepCheckConfig",
    "SweepCheckContext",
    "SweepCheckData",
    "KIDS_FIND_GROUP",
    "SWEEP_CHECK_GROUP",
    "KidsPipeline",
    "KidsPipelineConfig",
    "KidsPipelineResult",
    "has_kids_reduction",
    "read_kids_find",
    "read_sweep_check",
    "make_kids_find_figs",
]
