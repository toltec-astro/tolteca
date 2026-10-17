"""Data-model capabilities for preparing workflow inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, field_validator
from tollan.config.types import StringSelector
from tollan.utils.registry import Registry

from .file import FileAccessor
from .file_store import FileStoreBase
from .types import FrozenDataModel

if TYPE_CHECKING:
    from .types import DataModelInfo

__all__ = ["DataModelSet", "DataModelSpec", "data_models"]


class DataModelSpec(FrozenDataModel):
    """Capabilities provided by a data model.

    Parameters
    ----------
    file_accessor_cls : type of FileAccessor
        Concrete accessor providing file-table enrichment, queries and readers.
    file_store_types : tuple of FileStoreBase subclasses, optional
        Stores to search for this data model, including companion data.
        Defaults to no stores. Explicit sources do not require a store.

    Notes
    -----
    Call `register` to make these capabilities available to generic file access
    and input discovery. Construction alone does not register it.
    """

    model_config = ConfigDict(extra="forbid")

    file_accessor_cls: type[FileAccessor]
    file_store_types: tuple[type[FileStoreBase], ...] = ()

    @field_validator("file_accessor_cls")
    @classmethod
    def _require_concrete_accessor(
        cls, value: type[FileAccessor]
    ) -> type[FileAccessor]:
        if not hasattr(value, "data_model_cls"):
            msg = "Expected a concrete FileAccessor with a data-model binding"
            raise ValueError(msg)
        return value

    @property
    def data_model_cls(self) -> type[DataModelInfo]:
        """Data-model definition associated with these capabilities."""
        return self.file_accessor_cls.data_model_cls

    @property
    def name(self) -> str:
        """Name used in record memberships and data-model selection."""
        return self.data_model_cls.name

    def register(self) -> None:
        """Enable these capabilities for generic file access and discovery.

        Raises
        ------
        ValueError
            The name is already registered.
        """
        _data_models.register(self.name, self)


_data_models: Registry[str, DataModelSpec] = Registry()
data_models = _data_models.view
"""Live read-only lookup of capabilities registered with `DataModelSpec.register`."""


class DataModelSet(BaseModel):
    """Selected data-model capabilities used together.

    Parameters
    ----------
    data_models : dict of str to DataModelSpec
        Definitions available for table enrichment and observation queries.
        Use `from_select` to choose registered capabilities.

    Notes
    -----
    Pass this set to ``InputDiscovery.discover(data_models=...)`` to restrict a
    workflow. Configured data-model selections can narrow the set further.
    Later registrations do not change the selected handlers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_models: dict[str, DataModelSpec]

    @property
    def file_accessor_types(self) -> tuple[type[FileAccessor], ...]:
        """File-table handlers provided by the selected data models."""
        return tuple(
            dict.fromkeys(
                model.file_accessor_cls for model in self.data_models.values()
            )
        )

    @property
    def file_store_types(self) -> tuple[type[FileStoreBase], ...]:
        """Store classes required by the selected data models."""
        return tuple(
            dict.fromkeys(
                store_cls
                for model in self.data_models.values()
                for store_cls in model.file_store_types
            )
        )

    @classmethod
    def from_select(
        cls,
        selection: StringSelector | str | list[str] | dict[str, Any] = ".*",
    ) -> DataModelSet:
        """Choose registered capabilities for a workflow.

        Parameters
        ----------
        selection : StringSelector or compatible input, optional
            Data-model names to select. Defaults to all registered names.

        Returns
        -------
        DataModelSet
            Selected capabilities in registration order.
        """
        selector = StringSelector.model_validate(selection)
        registered = data_models
        return cls.model_validate(
            {
                "data_models": {
                    name: registered[name] for name in selector.select(registered)
                },
            }
        )

    def select(self, selector: StringSelector) -> DataModelSet:
        """Narrow the available data-model capabilities.

        Parameters
        ----------
        selector : StringSelector
            Selection applied only to names already in this set.

        Returns
        -------
        DataModelSet
            Selected data-model capabilities.
        """
        return type(self)(
            data_models={
                name: self.data_models[name]
                for name in selector.select(self.data_models)
            },
        )
