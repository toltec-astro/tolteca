"""Data-model declarations and explicit capability selections.

This module does not load or register data models. Named selection of assembled
support is provided by ``tolteca_datamodels.data_model``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Self

from pydantic import ConfigDict

from .file import FileAccessorBase
from .file_store import FileStoreBase
from .types import DataModelInfo, FrozenDataModel

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tollan.config.types import StringSelector

__all__ = ["DataModelSet", "DataModelSpec"]


class DataModelSpec(FrozenDataModel):
    """Identity and capabilities provided by one data model.

    Parameters
    ----------
    info_cls : type of DataModelInfo
        Named identity associated with these capabilities.
    file_accessor_cls : type of FileAccessorBase
        File-table accessor to use for this data model. The same
        accessor may be associated with more than one data-model identity.
    file_store_types : tuple of FileStoreBase subclasses, optional
        Stores to search for this data model, including companion data.
        Defaults to no stores. Explicit sources do not require a store.

    Notes
    -----
    Construction does not register the spec. Pass an explicit set to a workflow,
    or enable an extension with
    ``tolteca_datamodels.data_model.data_models.register(spec)``.
    """

    model_config = ConfigDict(extra="forbid")

    info_cls: type[DataModelInfo]
    file_accessor_cls: type[FileAccessorBase]
    file_store_types: tuple[type[FileStoreBase], ...] = ()

    @property
    def name(self) -> str:
        """Name used in record compatibility and data-model selection."""
        return self.info_cls.name


@dataclass(frozen=True, slots=True)
class DataModelSet:
    """An explicit selection of data-model capabilities.

    Parameters
    ----------
    data_models : mapping of str to DataModelSpec
        Prepared specs available to a workflow. The mapping is copied and made
        read-only at construction. Later registrations do not enlarge this selection.

    Attributes
    ----------
    file_accessor_types : tuple of FileAccessorBase subclasses
        Unique accessor classes in first-selected order.
    file_store_types : tuple of FileStoreBase subclasses
        Unique required store classes in first-selected order.

    Notes
    -----
    A set selects capabilities, not a new data-model identity or workflow.
    Store and file-table operations consume its capability lists. Import
    ``DataModelSet`` from ``tolteca_datamodels.data_model`` to select assembled
    support by name with ``from_select()``.
    """

    data_models: Mapping[str, DataModelSpec]
    file_accessor_types: tuple[type[FileAccessorBase], ...] = field(
        init=False, repr=False
    )
    file_store_types: tuple[type[FileStoreBase], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "data_models", MappingProxyType(dict(self.data_models))
        )
        object.__setattr__(
            self,
            "file_accessor_types",
            tuple(
                dict.fromkeys(
                    model.file_accessor_cls for model in self.data_models.values()
                )
            ),
        )
        object.__setattr__(
            self,
            "file_store_types",
            tuple(
                dict.fromkeys(
                    store_cls
                    for model in self.data_models.values()
                    for store_cls in model.file_store_types
                )
            ),
        )

    def select(self, selector: StringSelector) -> Self:
        """Narrow this selection without consulting registrations.

        Parameters
        ----------
        selector : StringSelector
            Selection applied only to names already in this set.

        Returns
        -------
        DataModelSet
            Selected capabilities, retaining this set's concrete type.
        """
        return type(self)(
            data_models={
                name: self.data_models[name]
                for name in selector.select(self.data_models)
            },
        )
