"""Configured steps and pipeline assembly for TolTEC KIDs data."""

from functools import cached_property
from typing import Literal

from tollan.config import ConfigHandler, SubConfigKeyTransformer
from tollan.pipeline import SequentialPipeline

from tolteca_datamodels.toltec.types import ToltecDataKind

from .config import KidsConfig
from .dataprod_output import DataProdOutput
from .noop import NoOp
from .sweep_reduce import SweepReduce

__all__ = ["Kids", "KidsConfig"]


class Kids(SubConfigKeyTransformer[Literal["kids"]], ConfigHandler[KidsConfig]):
    """Construct configured KIDs steps from a runtime configuration snapshot.

    Examples
    --------
    >>> from tollan.config import RuntimeContext
    >>> kids = Kids(RuntimeContext())
    >>> kids.sweep_reduce.config is kids.config.sweep_reduce
    True
    >>> kids.sweep_pipeline.steps == [kids.sweep_reduce, kids.output]
    True
    """

    @cached_property
    def sweep_reduce(self) -> SweepReduce:
        """The configured raw sweep reduction step."""
        return SweepReduce(self.config.sweep_reduce)

    @cached_property
    def output(self) -> DataProdOutput:
        """The configured product-output step."""
        return DataProdOutput(self.config.output)

    @property
    def sweep_pipeline(self) -> SequentialPipeline:
        """Reduce raw sweeps, then optionally publish the result."""
        return SequentialPipeline(steps=[self.sweep_reduce, self.output])

    def pipeline_for(self, kind: ToltecDataKind) -> SequentialPipeline:
        """Select processing before executing any steps.

        Timestream and processed-input workflows are not yet implemented.
        They are rejected explicitly rather than passed to a raw reducer.
        """
        if kind != ToltecDataKind.Unknown and kind in ToltecDataKind.RawSweep:
            return self.sweep_pipeline
        msg = f"KIDs processing is not implemented for {kind.name}"
        raise NotImplementedError(msg)

    @property
    def noop_pipeline(self) -> SequentialPipeline:
        """Exercise step execution without reduction or product output."""
        return SequentialPipeline(steps=[NoOp()])
