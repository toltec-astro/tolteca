"""Input-discovery configuration, registration, and runtime handler."""

from __future__ import annotations

from functools import reduce
from inspect import isabstract
from operator import or_
from typing import TYPE_CHECKING, Annotated, Literal

import pandas as pd
from pydantic import (
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    GetPydanticSchema,
    Tag,
    TypeAdapter,
)
from tollan.config import ConfigHandler, FrozenBaseModel, SubConfigKeyTransformer
from tollan.config.types import StringSelector
from tollan.utils.log import logger
from tollan.utils.registry import Registry

from ..data_model import DataModelSet
from ..file import FileAccessor
from .entries import InputDiscoveryEntry

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import CoreSchema

    from ..file import SourceErrorPolicyT

__all__ = ["InputDiscovery", "InputDiscoveryConfig"]

_input_discovery_adapter = TypeAdapter(
    dict[str, type[InputDiscoveryEntry]], config=ConfigDict(strict=True)
)
_input_discovery_models: Registry[str, type[InputDiscoveryEntry]] = Registry()


def _require_registration(value: object) -> object:
    msg = "No input-discovery entry types are registered"
    raise ValueError(msg)


def _input_discovery_schema(
    source_type: object, handler: GetCoreSchemaHandler
) -> CoreSchema:
    if not _input_discovery_models:
        return handler.generate_schema(
            Annotated[InputDiscoveryEntry, BeforeValidator(_require_registration)]
        )
    # Each compiled schema keeps its own registration snapshot.
    models = dict(_input_discovery_models)

    def input_discovery_key(value: object) -> str | None:
        if isinstance(value, dict):
            matches = models.keys() & value.keys()
        else:
            matches = {key for key, model in models.items() if type(value) is model}
        return next(iter(matches)) if len(matches) == 1 else None

    # Runtime annotations are assembled from this schema's registered models.
    variants = tuple(
        Annotated[model, Tag(key)]  # ty: ignore[invalid-type-form]
        for key, model in models.items()
    )
    discriminator = Discriminator(
        input_discovery_key,
        custom_error_type="input_discovery_key",
        custom_error_message="Expected exactly one discovery key from {keys}",
        custom_error_context={"keys": ", ".join(models)},
    )
    return handler.generate_schema(
        Annotated[reduce(or_, variants), discriminator]  # ty: ignore[invalid-type-form]
    )


class InputDiscoveryConfig(FrozenBaseModel):
    """Settings for discovering input files.

    Parameters
    ----------
    inputs : list, optional
        Registered discovery settings. Each dictionary must contain exactly one
        identifying key, such as ``data_lmt_path``, ``data_toltec_path`` or
        ``sources``. Typed discovery entries are also accepted. Defaults to an
        empty list.
    data_model_select : StringSelector, optional
        Data-model capabilities enabled for every entry. Defaults to all
        available names. Entries may narrow this selection further.
    """

    model_config = ConfigDict(extra="forbid")
    data_model_select: StringSelector = Field(default=".*", validate_default=True)
    inputs: list[
        Annotated[InputDiscoveryEntry, GetPydanticSchema(_input_discovery_schema)]
    ] = Field(default_factory=list)

    @classmethod
    def register_input(cls, key: str, model_cls: type[InputDiscoveryEntry]) -> None:
        """Enable a discovery model in input configuration.

        Parameters
        ----------
        key : str
            Required model field identifying this kind of discovery entry.
        model_cls : type of InputDiscoveryEntry
            Concrete settings model with a ``discover()`` method. Register
            extensions before loading configuration or defining models that
            contain ``InputDiscoveryConfig``.

        Raises
        ------
        pydantic.ValidationError
            If key is not a string or model_cls is not an InputDiscoveryEntry subclass.
        ValueError
            If the model is abstract, the key is not a required unaliased model
            field, or the key or model is already registered, including the same pair.
        """
        global _input_discovery_models  # noqa: PLW0603 -- publish only a usable schema's registry.
        ((key, model_cls),) = _input_discovery_adapter.validate_python(
            {key: model_cls}
        ).items()
        if isabstract(model_cls):
            msg = "Expected a concrete InputDiscoveryEntry subclass"
            raise ValueError(msg)
        field = model_cls.model_fields.get(key)
        if (
            field is None
            or not field.is_required()
            or field.validation_alias
            or field.serialization_alias
        ):
            msg = f"Discovery key {key!r} must be a required unaliased field"
            raise ValueError(msg)
        candidate = Registry[str, type[InputDiscoveryEntry]]()
        for registered_key, registered_model in _input_discovery_models.items():
            candidate.register(registered_key, registered_model)
        candidate.register(key, model_cls)
        if model_cls in _input_discovery_models.values():
            msg = f"Discovery model is already registered: {model_cls!r}"
            raise ValueError(msg)
        previous = _input_discovery_models
        _input_discovery_models = candidate
        try:
            cls.model_rebuild(force=True)
        except Exception:
            _input_discovery_models = previous
            cls.model_rebuild(force=True)
            raise


class InputDiscovery(
    SubConfigKeyTransformer[Literal["input_discovery"]],
    ConfigHandler[InputDiscoveryConfig],
):
    """Discover configured observation inputs.

    Parameters
    ----------
    runtime_context : tollan.config.RuntimeContext
        Runtime containing ``input_discovery.inputs`` settings.

    Examples
    --------
    >>> from tolteca_datamodels.input_discovery import InputDiscovery
    >>> from tollan.config import RuntimeContext
    >>> discovery = InputDiscovery(RuntimeContext({"input_discovery": {"inputs": [
    ...     {"sources": ["https://archive.example/data.nc"], "mode": "guess"},
    ... ]}}))
    >>> len(discovery.discover())
    1
    """

    def discover(
        self,
        *,
        data_models: DataModelSet | None = None,
        on_error: SourceErrorPolicyT = "record",
    ) -> pd.DataFrame:
        """Combine independently selected discovery entries.

        Parameters
        ----------
        data_models : DataModelSet, optional
            Capabilities available to this workflow, defaulting to all registered
            support. Global and per-entry ``data_model_select`` settings narrow
            this scope; they cannot enable handlers absent from this set.
            This argument does not change the configuration.
        on_error : {"raise", "record"}, optional
            Whether source-information failures raise or remain in
            the returned table's ``source_trace`` column. Passed to each entry;
            configuration and selection errors still raise. Defaults to "record".

        Returns
        -------
        pandas.DataFrame
            Rows in entry order. Duplicate locations and local symlink targets
            keep the first selected row. Entries without matches contribute no
            rows. No selected candidates yields an empty table.
        """
        if data_models is None:
            data_models = DataModelSet.from_select()
        data_models = data_models.select(self.config.data_model_select)
        tables = []
        logger.debug("Input discovery: {} configured entries", len(self.config.inputs))
        for index, entry in enumerate(self.config.inputs):
            logger.debug(
                "Input discovery inputs[{}] ({}): {}",
                index,
                type(entry).__name__,
                entry,
            )
            table = entry.discover(data_models=data_models, on_error=on_error)
            logger.debug(
                "Input discovery inputs[{}]: {} selected candidates", index, len(table)
            )
            if table.empty:
                continue
            tables.append(table)
        if not tables:
            logger.debug("Input discovery: no selected candidates")
            files = FileAccessor.partial(accessor_types=data_models.file_accessor_types)
            return files.from_sources((), on_error=on_error)
        combined = pd.concat(tables, ignore_index=True)
        seen = set()
        positions = []
        for i, location in enumerate(combined.source_loc):
            key = str(location.url) if location.is_remote() else location.path.resolve()
            if key not in seen:
                seen.add(key)
                positions.append(i)
        logger.debug(
            "Input discovery: {} nonempty entries, {} selected rows -> {} distinct "
            "files ({} duplicate locations or symlink targets removed)",
            len(tables),
            len(combined),
            len(positions),
            len(combined) - len(positions),
        )
        return combined.iloc[positions].reset_index(drop=True)
