"""Registered data models and workflow selection.

Look up built-in capabilities with ``data_models.toltec`` or
``data_models.lmt``. Use ``data_models[name]`` for any registered name and
`DataModelSet.from_select` to select capabilities for a workflow.
Enable optional extensions with ``data_models.register(spec)``.
"""

from dataclasses import dataclass
from typing import Any, Self

from tollan.config.types import StringSelector
from tollan.utils.registry import Registry, RegistryView

from .core.data_model import DataModelSet as CoreDataModelSet
from .core.data_model import DataModelSpec
from .lmt.data_model import data_model_spec as lmt_data_model_spec
from .toltec.data_model import data_model_spec as toltec_data_model_spec

__all__ = ["DataModelSet", "data_models"]

_data_models = Registry[str, DataModelSpec](
    {
        lmt_data_model_spec.name: lmt_data_model_spec,
        toltec_data_model_spec.name: toltec_data_model_spec,
    }
)


class _DataModels(RegistryView[str, DataModelSpec]):
    """Named data-model lookup with validated registration."""

    def __init__(self, registry: Registry[str, DataModelSpec]) -> None:
        self._registry: Registry[str, DataModelSpec] = registry

    @property
    def lmt(self) -> DataModelSpec:
        """Registered LMT telescope capabilities."""
        return self[lmt_data_model_spec.name]

    @property
    def toltec(self) -> DataModelSpec:
        """Registered TolTEC capabilities, including telescope companions."""
        return self[toltec_data_model_spec.name]

    def register(self, spec: DataModelSpec) -> None:
        """Enable an extension's declared capabilities by name.

        Parameters
        ----------
        spec : DataModelSpec
            Identity, file accessor and store requirements supplied by an extension.

        Raises
        ------
        pydantic.ValidationError
            The supplied value is not a valid data-model specification.
        ValueError
            The name already exists, including a built-in or the identical spec.

        Notes
        -----
        Register before selecting capabilities. Existing sets retain their selected
        specs; subsequent named selections include the new registration.
        """
        spec = DataModelSpec.model_validate(spec)
        self._registry.register(spec.name, spec)


data_models = _DataModels(_data_models)
"""Data-model lookup with controlled registration; no item assignment or deletion."""


@dataclass(frozen=True, slots=True)
class DataModelSet(CoreDataModelSet):
    """Select assembled data-model capabilities for a workflow.

    Use `from_select` to select built-ins and activated extensions by name,
    or `from_specs` to supply explicit definitions without consulting registrations.
    """

    @classmethod
    def from_select(
        cls,
        selection: StringSelector | str | list[str] | dict[str, Any] = ".*",
    ) -> Self:
        """Choose assembled capabilities by name.

        Parameters
        ----------
        selection : StringSelector or compatible input, optional
            Names to select. Defaults to all available names. A list names exact
            entries; a string is a full-match regular expression.

        Returns
        -------
        DataModelSet
            Selected specs in registration order, independent of later
            registrations. An unmatched selector produces an empty set.

        Examples
        --------
        >>> models = DataModelSet.from_select(["lmt", "toltec"])
        >>> selected = models.select(StringSelector.model_validate(["toltec"]))
        >>> list(selected.data_models)
        ['toltec']
        >>> list(models.data_models)
        ['lmt', 'toltec']
        """
        selector = StringSelector.model_validate(selection)
        return cls.from_specs(
            {name: data_models[name] for name in selector.select(data_models)}
        )
