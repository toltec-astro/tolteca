"""TolTEC KIDs pipeline steps and xarray accessor."""

# ruff: noqa: F405 -- __all__ includes the public steps imported for type checkers.
from __future__ import annotations

from typing import TYPE_CHECKING

from .accessor import ToltecKidsTreeAccessor
from .block_create import BlockCreate, BlockCreateContext
from .block_identify import BlockIdentify, BlockIdentifyContext
from .blocks import ToltecBlocksView, ToltecBlockView
from .tlaloc_output import TlalocOutput, TlalocOutputConfig

if TYPE_CHECKING:
    from ._public_steps import *  # noqa: F403 -- runtime exports resolved below


def __getattr__(name: str) -> object:
    """Import scientific steps only when requested by a processing client."""
    if name not in __all__:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    from . import _public_steps  # noqa: PLC0415

    value = getattr(_public_steps, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "KIDS_FIND_GROUP",
    "SWEEP_CHECK_GROUP",
    "BlockCreate",
    "BlockCreateContext",
    "BlockIdentify",
    "BlockIdentifyContext",
    "KidsContextConfig",
    "KidsFind",
    "KidsFindConfig",
    "KidsFindContext",
    "KidsFindData",
    "KidsFindPlot",
    "KidsFindPlotConfig",
    "KidsFindView",
    "KidsModelSelection",
    "KidsPipeline",
    "KidsPipelineConfig",
    "KidsPipelineResult",
    "ModelFit",
    "ModelFitBitMask",
    "ModelFitConfig",
    "ModelFitPlot",
    "ModelFitPlotArtifacts",
    "ModelFitPlotConfig",
    "ModelFitPlotContext",
    "ModelFitQualityConfig",
    "ModelFitSummaryViz",
    "PowerSetupConfig",
    "ReducedSweepView",
    "SegmentBitMask",
    "SweepBitMask",
    "SweepCheck",
    "SweepCheckConfig",
    "SweepCheckContext",
    "SweepCheckPlot",
    "SweepCheckPlotConfig",
    "SweepCheckView",
    "SweepReduce",
    "TimestreamCheck",
    "TimestreamCheckConfig",
    "TimestreamCheckContext",
    "TimestreamCheckPlot",
    "TimestreamCheckPlotConfig",
    "TimestreamCheckView",
    "TimestreamInspect",
    "TimestreamInspectConfig",
    "TimestreamInspectContext",
    "TimestreamReduce",
    "TimestreamSolve",
    "TimestreamSolveConfig",
    "TimestreamSolveContext",
    "TimestreamView",
    "TlalocOutput",
    "TlalocOutputConfig",
    "ToltecBlockView",
    "ToltecBlocksView",
    "ToltecKidsTreeAccessor",
    "TonePrep",
    "TonePrepConfig",
    "has_kids_reduction",
    "make_kids_find_figs",
    "prepare_model_binding",
    "read_kids_find",
    "read_sweep_check",
]
