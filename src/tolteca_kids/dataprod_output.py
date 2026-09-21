"""Configured publication of completed KIDs data products."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tollan.pipeline import Step, StepContext

from .config import DataProdOutputConfig
from .product import write_sweep_product

if TYPE_CHECKING:
    import xarray as xr

__all__ = [
    "DataProdOutput",
    "DataProdOutputConfig",
    "DataProdOutputContext",
    "DataProdOutputView",
]


class DataProdOutputContext(StepContext["DataProdOutput", DataProdOutputConfig]):
    """Output configuration and the location successfully published by the step."""

    product_path: Path | None = None


class DataProdOutput(Step[DataProdOutputConfig, DataProdOutputContext]):
    """Export a completed reduction using the configured output policy.

    The working tree carries the input record in
    ``data['tolteca_kids'].attrs['input']``. Numerical serialization is
    delegated to the product writer; database registration is independent.
    """

    @classmethod
    def view(cls, data: xr.DataTree) -> DataProdOutputView:
        """Return the data-backed publication view."""
        return DataProdOutputView(data, context_key=cls.context_key)

    def create_context(self, data: xr.DataTree) -> DataProdOutputContext:
        """Create publication state without changing the source dataset."""
        return DataProdOutputContext(config=self.config)

    @classmethod
    def run(cls, data: xr.DataTree, context: DataProdOutputContext) -> bool:
        """Publish reduced arrays and record the completed product location."""
        info = data["tolteca_kids"].attrs["input"]
        name = context.config.name.format(**info, stem=Path(info["source"]).stem)
        if not name or name in {".", ".."} or Path(name).name != name:
            msg = f"Output name must be a single directory component: {name!r}"
            raise ValueError(msg)
        context.product_path = write_sweep_product(
            data,
            context.config.path / name,
            input_info=info,
            output_config=context.config.model_dump(mode="json"),
        )
        return True

    @classmethod
    def set_context(cls, data: xr.DataTree, context: DataProdOutputContext) -> None:
        """Store publication metadata on the execution node."""
        data["tolteca_kids"].attrs[cls.context_key] = context.model_dump(mode="json")

    @classmethod
    def get_context(cls, data: xr.DataTree) -> DataProdOutputContext:
        """Read the recorded publication context."""
        return cls.view(data).context

    @classmethod
    def has_context(cls, data: xr.DataTree) -> bool:
        """Return whether the execution node records this step."""
        return "tolteca_kids" in data and cls.context_key in data["tolteca_kids"].attrs


class DataProdOutputView:
    """Publication metadata attached to a working data tree."""

    def __init__(self, data: xr.DataTree, *, context_key: str | None = None) -> None:
        self.data_source = data["tolteca_kids"]
        self._context_key = context_key or DataProdOutput.context_key

    @property
    def context(self) -> DataProdOutputContext:
        """The output step's configuration, completion state, and product path."""
        return DataProdOutputContext.model_validate(
            self.data_source.attrs[self._context_key]
        )
