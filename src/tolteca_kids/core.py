"""Configured steps and pipeline assembly for TolTEC KIDs data."""

from functools import cached_property
from typing import Literal

from tollan.config import ConfigHandler, SubConfigKeyTransformer
from tollan.pipeline import SequentialPipeline

from tolteca_datamodels.toltec.types import ToltecDataKind

from .block_create import BlockCreate
from .block_identify import BlockIdentify
from .config import KidsConfig
from .dataprod_output import DataProdOutput
from .input import Input
from .kids_find import KidsFind
from .kids_find_plot import KidsFindPlot
from .noop import NoOp
from .sweep_check import SweepCheck
from .sweep_check_plot import SweepCheckPlot
from .sweep_reduce import SweepReduce
from .timestream_reduce import TimestreamReduce
from .tlaloc_output import TlalocOutput
from .tone_prepare import TonePrepare

__all__ = ["Kids", "KidsConfig"]


class Kids(SubConfigKeyTransformer[Literal["kids"]], ConfigHandler[KidsConfig]):
    """Construct configured KIDs steps from a runtime configuration snapshot.

    Parameters
    ----------
    runtime_context : tollan.config.RuntimeContext
        Runtime containing the ``kids`` configuration section.

    Examples
    --------
    >>> from tollan.config import RuntimeContext
    >>> kids = Kids(RuntimeContext())
    >>> kids.sweep_reduce.config is kids.config.sweep_reduce
    True
    >>> kids.sweep_pipeline.steps == [
    ...     kids.sweep_reduce, kids.sweep_check, kids.sweep_check_plot,
    ...     kids.kids_find, kids.kids_find_plot, kids.tone_prepare]
    True
    """

    @cached_property
    def input(self) -> Input:
        """The configured input-selection step."""
        return Input(self.config.input)

    @cached_property
    def block_identify(self) -> BlockIdentify:
        """The recorded-block identification step."""
        return BlockIdentify()

    @cached_property
    def block_create(self) -> BlockCreate:
        """The block-scope creation step."""
        return BlockCreate()

    @cached_property
    def sweep_reduce(self) -> SweepReduce:
        """The configured raw sweep reduction step."""
        return SweepReduce(self.config.sweep_reduce)

    @cached_property
    def sweep_check(self) -> SweepCheck:
        """Configured sweep-quality check."""
        return SweepCheck(self.config.sweep_check)

    @cached_property
    def sweep_check_plot(self) -> SweepCheckPlot:
        """Configured standalone sweep diagnostics."""
        return SweepCheckPlot(self.config.sweep_check_plot)

    @cached_property
    def kids_find(self) -> KidsFind:
        """Configured resonance finding and input-probe matching."""
        return KidsFind(self.config.kids_find)

    @cached_property
    def kids_find_plot(self) -> KidsFindPlot:
        """Configured detection and matching diagnostics."""
        return KidsFindPlot(self.config.kids_find_plot)

    @cached_property
    def tone_prepare(self) -> TonePrepare:
        """Configured preparation of the next tone list."""
        return TonePrepare(self.config.tone_prepare)

    @cached_property
    def tlaloc_output(self) -> TlalocOutput:
        """Configured installation of prepared tone files."""
        return TlalocOutput(self.config.tlaloc_output)

    @cached_property
    def timestream_reduce(self) -> TimestreamReduce:
        """Configured Python timestream reduction step."""
        return TimestreamReduce(self.config.timestream_reduce)

    @property
    def timestream_pipeline(self) -> SequentialPipeline:
        """Processing operations for one timestream block."""
        return SequentialPipeline(
            steps=[
                self.timestream_reduce,
            ]
        )

    @cached_property
    def output(self) -> DataProdOutput:
        """The configured product-output step."""
        return DataProdOutput(
            self.config.output,
            workflow_config=self.config,
            registration=self.registration,
        )

    @cached_property
    def registration(self):
        """Configured index registration action, or None when disabled."""
        if not (self.config.output.enabled and self.config.output.registration):
            return None
        # DPDB is an optional output backend. File-only processing does not
        # import or initialize the database package.
        from tolteca_db.core import DataProdHandler  # noqa: PLC0415

        return DataProdHandler(self._rc).index.ingest

    @property
    def sweep_pipeline(self) -> SequentialPipeline:
        """Reduction and optional diagnostics for one sweep block."""
        return SequentialPipeline(
            steps=[
                self.sweep_reduce,
                self.sweep_check,
                self.sweep_check_plot,
                self.kids_find,
                self.kids_find_plot,
                self.tone_prepare,
            ]
        )

    def pipeline_for(self, kind: ToltecDataKind) -> SequentialPipeline:
        """Select the pipeline for a data kind.

        Parameters
        ----------
        kind : ToltecDataKind
            Input data classification.

        Returns
        -------
        tollan.pipeline.SequentialPipeline
            Pipeline for the input kind.

        Raises
        ------
        NotImplementedError
            If the input kind is unsupported.
        """
        if kind != ToltecDataKind.unknown and kind in ToltecDataKind.raw_sweep:
            return self.sweep_pipeline
        if kind == ToltecDataKind.raw_timestream:
            return self.timestream_pipeline
        msg = f"KIDs processing is not implemented for {kind.name}"
        raise NotImplementedError(msg)

    @property
    def noop_pipeline(self) -> SequentialPipeline:
        """No-op pipeline."""
        return SequentialPipeline(steps=[NoOp()])
