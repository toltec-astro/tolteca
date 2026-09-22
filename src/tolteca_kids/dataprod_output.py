"""Configured publication of completed KIDs data products."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import xarray as xr
from tollan.pipeline import Step, StepContext

from tolteca_datamodels.file import SourceInfoModel

from .config import DataProdOutputConfig
from .product import write_sweep_product

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
        return DataProdOutputView(
            data,
            context_key=cls.context_key,
            context_cls=cast("type[DataProdOutputContext]", cls.context_cls),
        )

    def create_context(self, data: xr.DataTree) -> DataProdOutputContext:
        """Attach a fresh publication invocation without changing source arrays."""
        if "tolteca_kids" not in data:
            data["tolteca_kids"] = xr.DataTree()
        context_cls = cast("type[DataProdOutputContext]", self.context_cls)
        context = context_cls(config=self.config)
        data["tolteca_kids"].attrs[self.context_key] = context
        return context

    @classmethod
    def run(cls, data: xr.DataTree, context: DataProdOutputContext) -> bool:
        """Publish reduced arrays and record the completed product location."""
        info = SourceInfoModel.model_validate(data["tolteca_kids"].attrs["input"])
        fields = info.model_dump(
            mode="json",
            context={"flat": True},
            exclude={"filepath_info", "filepath_info_orig"},
        )
        name = context.config.name.format(**fields, stem=info.source.path.stem)
        if not name or name in {".", ".."} or Path(name).name != name:
            msg = f"Output name must be a single directory component: {name!r}"
            raise ValueError(msg)
        context.product_path = write_sweep_product(
            data,
            context.config.path / name,
            input_info=info,
            output_config=context.config,
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

    def __init__(
        self,
        data: xr.DataTree,
        *,
        context_key: str | None = None,
        context_cls: type[DataProdOutputContext] = DataProdOutputContext,
    ) -> None:
        self.data_source = data["tolteca_kids"]
        self._context_key = context_key or DataProdOutput.context_key
        self._context_cls = context_cls

    @property
    def context(self) -> DataProdOutputContext:
        """The output step's configuration, completion state, and product path."""
        return self._context_cls.model_validate(
            self.data_source.attrs[self._context_key]
        )
