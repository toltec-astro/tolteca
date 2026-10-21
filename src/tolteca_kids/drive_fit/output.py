"""Product publication step shared by drive-fit and commit workflows."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 -- Pydantic field.
from typing import TYPE_CHECKING

from pydantic import PrivateAttr
from tollan.pipeline import Step, StepContext

from tolteca_kids.config import DataProdOutputConfig

from .product import _write_product, make_drive_fit_product_index

if TYPE_CHECKING:
    import xarray as xr


class DriveFitOutputStepContext(
    StepContext["DriveFitOutputStep", DataProdOutputConfig]
):
    """Publication path and registration state for a sequence or commit."""

    data: Path | None = None
    registered: bool = False
    _publication = PrivateAttr(default=None)
    _index = PrivateAttr(default=None)


class DriveFitOutputStep(Step[DataProdOutputConfig, DriveFitOutputStepContext]):
    """Publish the prepared artifacts of a drive-fit or commit collection.

    Parameters
    ----------
    *args, **kwargs
        Output configuration, as accepted by Step.
    publication : DataProdPublication or None, optional
        Staging area already reserved by the workflow before computation.
    index : DataProdIndex or None, optional
        Product identity and associations. None describes a drive-fit sequence.
    """

    def __init__(self, *args, publication=None, index=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._publication = publication
        self._index = index

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
        context._publication = self._publication
        context._index = self._index
        return context

    @classmethod
    def run(cls, data: xr.DataTree, context: DriveFitOutputStepContext) -> None:
        """Write artifacts and commit a complete product inventory.

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
        pending, context._publication = context._publication, None
        index, context._index = context._index, None
        context.data = _write_product(
            data,
            context.config.path,
            index or make_drive_fit_product_index(data),
            context.config,
            publication=pending,
        )
