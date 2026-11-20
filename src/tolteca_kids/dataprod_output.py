"""Configured publication of completed KIDs data products."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path  # noqa: TC003 -- Pydantic resolves context fields at runtime
from typing import TYPE_CHECKING

from pydantic import Field, PrivateAttr
from tollan.pipeline import Step, StepContext, StepFailure, StepFailureKind

from tolteca_datamodels.data_prod import (
    DataProdIndex,
    DataProdStore,
)

from .config import DataProdOutputConfig, KidsConfig
from .product import make_kids_product_index, write_kids_product
from .sweep_reduce import SweepReduce
from .timestream_reduce import TimestreamReduce

if TYPE_CHECKING:
    from collections.abc import Callable

    import xarray as xr

    from tolteca_datamodels.data_prod import WorkflowRun

__all__ = [
    "DataProdOutput",
    "DataProdOutputConfig",
    "DataProdOutputContext",
]


class DataProdOutputContext(StepContext["DataProdOutput", DataProdOutputConfig]):
    """Product-output status and pinned run directory."""

    data: Path | None = Field(
        default=None, description="Pinned run directory after publication."
    )
    registered: bool = Field(
        default=False, description="Published indexes registered in DPDB."
    )
    _run: WorkflowRun | None = PrivateAttr(default=None)
    _registration: Callable | None = PrivateAttr(default=None)
    _output_index: DataProdIndex | None = PrivateAttr(default=None)
    _workflow_config: KidsConfig | None = PrivateAttr(default=None)


class DataProdOutput(Step[DataProdOutputConfig, DataProdOutputContext]):
    """Publish completed reductions of one acquisition in a data-product store.

    Parameters
    ----------
    *args, **kwargs
        A DataProdOutputConfig instance or its field values, as accepted by Step.
    run : WorkflowRun or None, optional
        Caller-owned execution for direct output calls. Several output
        steps may materialize the same computation with this identity.
    workflow_config : KidsConfig or None, optional
        Resolved workflow used for the inputs. Runner-recorded per-input
        snapshots take precedence. None retains only available step records.
    registration : callable or None, optional
        Index registration action accepting paths and ``run_uids``. Required
        when config.registration is True. ``Kids.output`` supplies this action.
    artifacts : ArtifactContext or None, optional
        Prepared acquisition artifacts to publish alongside block results.
    """

    def __init__(
        self,
        *args,
        run=None,
        workflow_config=None,
        artifacts=None,
        registration=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._run = run
        self._registration = registration
        self._artifacts = artifacts
        self._output_index = None
        self._last_context = None
        self._workflow_config = workflow_config

    @contextmanager
    def prepare(self, metadata, *, run):
        """Reserve output before computation and complete it after this step runs.

        Parameters
        ----------
        metadata : object
            Acquisition metadata exposing master, obsnum, subobsnum and scannum.
        run : WorkflowRun
            Caller-owned execution shared across the workflow's outputs.

        Yields
        ------
        DataProdOutput
            This output step. Invoke it on the computed acquisition in the scope.
            Incomplete output discards temporary files and marks the execution
            failed. Successful exit records the final path and registers saved
            indexes when configured.
        """
        store = DataProdStore(self.config.path, namespace=self.config.namespace)
        index = make_kids_product_index(
            metadata.master,
            int(metadata.obsnum),
            int(metadata.subobsnum),
            int(metadata.scannum),
            run=run,
        )
        self._last_context = None
        try:
            with store.materialize(
                index, run_name=self.config.run_name, tag=self.config.tag
            ) as bound:
                self._output_index = bound
                yield self
                _require_completed(self._last_context)
            self._last_context.data = bound.base_path
            _register(self._last_context, self._registration)
        except _IncompleteOutputError:
            run.fail()
        finally:
            self._output_index = None

    def create_context(self, data):
        """Create an output execution record on the supplied tree.

        Parameters
        ----------
        data : xarray.DataTree
            Working acquisition or interface collection.

        Returns
        -------
        DataProdOutputContext
            New record, replacing the prior invocation of this step alias.
        """
        context = super().create_context(data)
        if self._artifacts is not None:
            context.artifacts = self._artifacts
        context._run = self._run
        context._registration = self._registration
        context._output_index = self._output_index
        self._last_context = context
        context._workflow_config = self._workflow_config
        return context

    @classmethod
    def run(cls, data: xr.DataTree, context: DataProdOutputContext) -> None:
        """Publish reduced arrays and optionally register their indexes.

        Parameters
        ----------
        data : xarray.DataTree
            One working interface tree, or a collection with working trees at
            ``interfaces/<interface>``. Keep their input resources open until
            publication finishes. All interfaces must belong to one acquisition.
        context : DataProdOutputContext
            Output settings and execution record. Its data is the pinned run
            directory after publication; registered is True after the configured
            DB action succeeds. Missing reductions record failures and publish no
            artifacts. Registration errors retain the published files and raise.

        Returns
        -------
        None
            The pinned run directory is recorded in context.data.

        Raises
        ------
        FileExistsError
            The requested run UID or counter value is already reserved.
        ValueError
            Interfaces belong to different acquisitions or validation fails.
        """
        registration = context._registration
        context._registration = None
        if context.config.registration and registration is None:
            msg = "Supply a registration action when output.registration is enabled"
            raise ValueError(msg)
        output_index = context._output_index
        context._output_index = None
        trees = (
            dict(data["interfaces"].children)
            if "interfaces" in data.children
            else {"input": data}
        )
        for interface, tree in trees.items():
            scopes = [tree]
            if "blocks" in tree.children and not (
                SweepReduce.has_context(tree) or TimestreamReduce.has_context(tree)
            ):
                scopes = [block.tree for block in tree.toltec_kids.blocks]
            for scope in scopes:
                reducer = (
                    TimestreamReduce
                    if TimestreamReduce.has_context(scope)
                    else SweepReduce
                )
                reduced = (
                    reducer.get_context(scope) if reducer.has_context(scope) else None
                )
                if reduced is None or not reduced.completed or reduced.data is None:
                    context.failures.append(
                        StepFailure(
                            kind=StepFailureKind.missing_input,
                            message=(
                                f"{interface} {scope.path}: no completed reduction "
                                "is available for publication."
                            ),
                        )
                    )
        if context.failures:
            return
        store = DataProdStore(context.config.path, namespace=context.config.namespace)
        context.data = write_kids_product(
            trees if "interfaces" in data.children else data,
            store,
            output_config=context.config,
            output_index=output_index,
            run=context._run,
            workflow_config=context._workflow_config,
            artifacts=context.artifacts,
        )

        if output_index is None:
            _register(context, registration)


class _IncompleteOutputError(Exception):
    """Internal scope abort for a step that reported an output failure."""


def _require_completed(context):
    if context is None or not context.completed:
        msg = "Required KIDs output did not complete"
        raise _IncompleteOutputError(msg)


def _register(context, registration):
    if not context.config.registration:
        return
    assert registration is not None
    assert context.data is not None
    index = DataProdIndex.read(context.data / "index.yaml")
    assert index.run is not None
    store = DataProdStore(context.config.path, namespace=context.config.namespace)
    registration(
        [
            context.data / "index.yaml",
            context.data.parent / "index.yaml",
            store.namespace_path / "namespace.yaml",
        ],
        run_uids=[index.run.uid],
    )
    context.registered = True
