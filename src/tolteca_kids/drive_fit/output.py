"""Product publication step for the drive-fit workflow."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path  # noqa: TC003 -- Pydantic field.
from typing import TYPE_CHECKING

from pydantic import PrivateAttr
from tollan.pipeline import Step, StepContext

from tolteca_datamodels.data_prod import DataProdStore
from tolteca_kids.config import DataProdOutputConfig

from .product import _write_artifacts, _write_product, make_drive_fit_product_index

if TYPE_CHECKING:
    import xarray as xr


class DriveFitOutputStepContext(
    StepContext["DriveFitOutputStep", DataProdOutputConfig]
):
    """Publication path and registration state for an attenuation sequence."""

    data: Path | None = None
    registered: bool = False
    _index = PrivateAttr(default=None)


class DriveFitOutputStep(Step[DataProdOutputConfig, DriveFitOutputStepContext]):
    """Publish the prepared artifacts of a drive-fit collection.

    Parameters
    ----------
    *args, **kwargs
        Output configuration, as accepted by Step.
    index : DataProdIndex or None, optional
        Product identity and associations. None describes a drive-fit sequence.
    """

    def __init__(self, *args, index=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._index = index
        self._last_context = None

    @contextmanager
    def prepare(self, index, *, run):
        """Hold a destination while the caller computes and invokes this output.

        Parameters
        ----------
        index : DataProdIndex
            Product identity and primary associations.
        run : WorkflowRun
            Caller-owned execution to record in the output.

        Yields
        ------
        DriveFitOutputStep
            This step, with a destination-bound index. Successful exit updates
            its context.data to the installed directory.
        """
        store = DataProdStore(self.config.path, namespace=self.config.namespace)
        description = index.model_copy(update={"run": run})
        self._last_context = None
        with store.materialize(
            description, run_name=self.config.run_name, tag=self.config.tag
        ) as bound:
            self._index = bound
            try:
                yield self
                if self._last_context is None or not self._last_context.completed:
                    msg = "Drive-fit output did not complete"
                    raise ValueError(msg)
            finally:
                self._index = None
        self._last_context.data = bound.base_path

    def create_context(self, data):
        """Create the publication execution record.

        Parameters
        ----------
        data : xarray.DataTree
            Working collection.

        Returns
        -------
        DriveFitOutputStepContext
            New output context with caller-owned publication state.
        """
        context = super().create_context(data)
        context._index = self._index
        self._last_context = context
        return context

    @classmethod
    def run(cls, data: xr.DataTree, context: DriveFitOutputStepContext) -> None:
        """Write retained artifacts and record their receipts in the output index.

        Parameters
        ----------
        data : xarray.DataTree
            Collection with completed scientific step artifacts.
        context : DriveFitOutputStepContext
            Output settings and destination for the pinned path. The workflow
            registers this product after the publication run has finalized.

        Returns
        -------
        None
            Published path is retained in context.data.
        """
        index, context._index = context._index, None
        if index is not None and index.base_path is not None:
            _write_artifacts(data, index, context.config)
            context.data = index.base_path
        else:
            context.data = _write_product(
                data,
                context.config.path,
                index or make_drive_fit_product_index(data),
                context.config,
            )
