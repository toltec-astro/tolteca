"""Configured steps and pipeline assembly for TolTEC KIDs data."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Literal

from tollan.config import ConfigHandler, SubConfigKeyTransformer
from tollan.pipeline import SequentialPipeline

from tolteca_datamodels.toltec.types import ToltecDataKind

from .block_create import BlockCreate
from .block_identify import BlockIdentify
from .config import KidsConfig
from .dataprod_output import DataProdOutput
from .kids_find import KidsFind
from .kids_find_plot import KidsFindPlot
from .model_fit import ModelFit
from .model_fit_plot import ModelFitPlot
from .noop import NoOp
from .sweep_check import SweepCheck
from .sweep_check_plot import SweepCheckPlot
from .sweep_reduce import SweepReduce
from .timestream_check import TimestreamCheck
from .timestream_check_plot import TimestreamCheckPlot
from .timestream_inspect import TimestreamInspect
from .timestream_reduce import TimestreamReduce
from .tlaloc_output import TlalocOutput
from .tone_prep import TonePrep

if TYPE_CHECKING:
    from .input import Input

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
    ...     kids.sweep_reduce, kids.timestream_check, kids.timestream_check_plot,
    ...     kids.sweep_check, kids.sweep_check_plot,
    ...     kids.kids_find, kids.kids_find_plot, kids.tone_prep]
    True
    """

    @cached_property
    def input(self) -> Input:
        """The configured input-selection step."""
        # Already-open block processing does not use file-table discovery.
        from .input import Input  # noqa: PLC0415 -- optional workflow backend

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
    def tone_prep(self) -> TonePrep:
        """Configured preparation of the next tone list."""
        return TonePrep(self.config.tone_prep)

    @cached_property
    def model_fit(self) -> ModelFit | None:
        """Configured fitting step, or None when no fitting was requested."""
        return (
            None if self.config.model_fit is None else ModelFit(self.config.model_fit)
        )

    @cached_property
    def model_fit_plot(self) -> ModelFitPlot | None:
        """Optional diagnostics for the selected Python fit invocation."""
        return (
            None
            if self.config.model_fit_plot is None
            else ModelFitPlot(self.config.model_fit_plot)
        )

    @cached_property
    def tlaloc_output(self) -> TlalocOutput:
        """Configured installation of prepared tone files."""
        return TlalocOutput(self.config.tlaloc_output)

    @cached_property
    def timestream_reduce(self) -> TimestreamReduce:
        """Configured Python timestream reduction step."""
        return TimestreamReduce(self.config.timestream_reduce)

    @cached_property
    def timestream_check(self) -> TimestreamCheck:
        """Configured raw noise check shared by sweep and timestream routes."""
        return TimestreamCheck(self.config.timestream_check)

    @cached_property
    def timestream_check_plot(self) -> TimestreamCheckPlot:
        """Configured diagnostics from a saved raw noise check."""
        return TimestreamCheckPlot(self.config.timestream_check_plot)

    @property
    def timestream_pipeline(self) -> SequentialPipeline:
        """Processing operations for one timestream block."""
        return SequentialPipeline(
            steps=[
                self.timestream_reduce,
                self.timestream_check,
                self.timestream_check_plot,
                *(
                    [TimestreamInspect(self.config.timestream_inspect)]
                    if self.config.timestream_inspect is not None
                    else []
                ),
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

        ingest = DataProdHandler(self._rc).index.ingest
        if not self.config.quicklook.enabled:
            return ingest

        def register(paths, *, run_uids):
            result = ingest(paths, run_uids=run_uids)
            from tolteca_datamodels.data_prod import DataProdIndex  # noqa: PLC0415
            from tolteca_quicklook.observation import (  # noqa: PLC0415 -- optional workflow backend
                refresh_registered,
            )

            # The acquisition may also have been published independently as a
            # KIDs quicklook. Selection is resolved across all tagged products.
            scopes = set()
            for path in paths:
                if path.name != "index.yaml":
                    continue
                product = DataProdIndex.read(path).product
                if any(
                    item.meta.get("format") == "kids_quicklook"
                    for item in product.data_items
                ):
                    scopes.update(
                        a.uid
                        for a in product.assocs
                        if a.data_prod_assoc_type == "dpa_raw_obs_proc_raw_obs"
                    )
            refresh_registered(
                self._rc,
                scopes,
                self.config.output.path,
                self.config.output.namespace,
                self.config.output.tag,
            )
            return result

        return register

    @property
    def sweep_pipeline(self) -> SequentialPipeline:
        """Reduction and optional diagnostics for one sweep block."""
        return self.make_sweep_pipeline()

    def make_sweep_pipeline(
        self, *, model_fit: ModelFit | None = None
    ) -> SequentialPipeline:
        """Assemble the steps for one sweep, optionally fitting its acquired channels.

        Parameters
        ----------
        model_fit : ModelFit or None, optional
            Override the configured fitting step, for example to supply an
            explicit Python segment plan or bind native acquisition inputs.
            None uses kids.model_fit. TonePrep owns selection of the next comb.

        Returns
        -------
        SequentialPipeline
            Configured steps in execution order.
        """
        model_fit = self.model_fit if model_fit is None else model_fit
        return SequentialPipeline(
            steps=[
                self.sweep_reduce,
                self.timestream_check,
                self.timestream_check_plot,
                self.sweep_check,
                self.sweep_check_plot,
                self.kids_find,
                self.kids_find_plot,
                *([model_fit] if model_fit is not None else []),
                *([self.model_fit_plot] if self.model_fit_plot is not None else []),
                self.tone_prep,
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
