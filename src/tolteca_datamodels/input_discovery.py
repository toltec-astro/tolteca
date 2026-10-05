"""Configured discovery of observation inputs."""

from __future__ import annotations

import glob
from functools import reduce
from inspect import isabstract
from operator import or_
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import pandas as pd
from pydantic import (
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    GetPydanticSchema,
    Tag,
)
from tollan.config import ConfigHandler, FrozenBaseModel, SubConfigKeyTransformer
from tollan.config.types import AnyPath, FileLocField
from tollan.utils.fileloc import FileLoc, fileloc
from tollan.utils.log import logger

from ._input_discovery import InputDiscoveryEntry
from .lmt.input_discovery import LmtInputDiscovery
from .lmt_base.obs_spec import ObsSpecField
from .toltec.file import FileTableModeT, ToltecFileAccessor
from .toltec.input_discovery import ToltecInputDiscovery

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import CoreSchema

    from .file import SourceInfoErrorPolicyT

__all__ = [
    "GlobSource",
    "InputDiscovery",
    "InputDiscoveryConfig",
    "InputDiscoveryEntry",
    "LmtInputDiscovery",
    "SourcesInputDiscovery",
    "ToltecInputDiscovery",
]


def _local_pattern(value: object) -> object:
    location = FileLoc.model_validate(value)
    if location.is_remote() or location.root.url is not None:
        msg = "Glob patterns must be local filesystem paths"
        raise ValueError(msg)
    return value


class GlobSource(FrozenBaseModel):
    """Expand a local filesystem pattern.

    Parameters
    ----------
    glob : str or pathlib.Path
        Pattern relative to the runtime's rootpath. ``*`` matches within one
        directory; ``**`` searches recursively. Only files are returned, in glob
        enumeration order. Remote patterns are unsupported.
        Without a rootpath, relative patterns use the working directory when
        discovery runs. Wildcards and parent-directory components are preserved.
    """

    model_config = ConfigDict(extra="forbid")
    glob: Annotated[AnyPath, BeforeValidator(_local_pattern)]

    def resolve(self) -> list[FileLoc]:
        """Expand the pattern to file locations.

        Returns
        -------
        list of FileLoc
            Matched files in glob enumeration order, excluding directories.
            An unmatched pattern returns an empty list.
        """
        paths = [
            Path(path)
            for path in glob.iglob(str(self.glob), recursive=True)  # noqa: PTH207
            if Path(path).is_file()
        ]
        logger.debug("Input glob {!r}: {} files", str(self.glob), len(paths))
        return [fileloc(path) for path in paths]


class SourcesInputDiscovery(InputDiscoveryEntry):
    """Discover inputs from exact locations and explicit glob patterns.

    Parameters
    ----------
    sources : list
        Each item is an exact FileLoc-compatible location, or ``{"glob": pattern}``.
        FileLoc dictionaries such as ``{"path": "data.nc"}`` are accepted.
        Wildcard characters in exact locations are literal. Dictionaries matching
        neither form are rejected.
    obs_spec : str, int or ObsSpec, optional
        Observation selector applied after table population, independently of
        other discovery entries. Omitted by default.
    mode : {"guess", "guess_or_read", "read"}, optional
        Default ``"guess_or_read"`` reads supported, openable sources whose
        filenames are unrecognized. ``"guess"`` uses filenames only; ``"read"``
        reads all supported, openable sources.
    """

    sources: list[GlobSource | FileLocField] = Field(min_length=1)
    obs_spec: ObsSpecField | None = None
    mode: FileTableModeT = "guess_or_read"

    def discover(self, *, on_error: SourceInfoErrorPolicyT = "raise") -> pd.DataFrame:
        """Build a file table from these locations and patterns.

        Parameters
        ----------
        on_error : {"raise", "record"}, optional
            Whether source-information failures raise (default) or remain in
            the table's ``diagnostics`` column before ObsSpec selection.

        Returns
        -------
        pandas.DataFrame
            Rows in source and glob enumeration order, with optional ObsSpec
            selection applied after population. No matches yields an empty table.
        """
        locations = [
            location
            for source in self.sources
            for location in (
                source.resolve() if isinstance(source, GlobSource) else [source]
            )
        ]
        n_resolved = len(locations)
        # Retain distinct alias spellings until selection has used filename info.
        locations = list(
            {str(location.url): location for location in locations}.values()
        )
        table = ToltecFileAccessor.from_sources(
            locations, mode=self.mode, on_error=on_error
        )
        selected = (
            table
            if self.obs_spec is None
            else table.toltec_file.query_obs_spec(self.obs_spec)
        )
        logger.debug(
            "Source discovery: {} sources -> {} resolved locations -> {} distinct "
            "candidates -> {} selected; mode={}, obs_spec={}",
            len(self.sources),
            n_resolved,
            len(table),
            len(selected),
            self.mode,
            self.obs_spec,
        )
        return selected


_input_discovery_models: dict[str, type[InputDiscoveryEntry]] = {
    "data_lmt_path": LmtInputDiscovery,
    "data_toltec_path": ToltecInputDiscovery,
    "sources": SourcesInputDiscovery,
}


def _input_discovery_schema(
    source_type: object, handler: GetCoreSchemaHandler
) -> CoreSchema:
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
    """

    model_config = ConfigDict(extra="forbid")
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
            contain ``InputDiscoveryConfig``. Re-registering the same pair
            is harmless.

        Raises
        ------
        TypeError
            If the model is not a concrete discovery entry.
        ValueError
            If the key is not a required model field, or the key or model is
            already registered with a different counterpart.
        """
        if (
            not isinstance(model_cls, type)
            or not issubclass(model_cls, InputDiscoveryEntry)
            or isabstract(model_cls)
        ):
            msg = "Expected a concrete InputDiscoveryEntry subclass"
            raise TypeError(msg)
        field = model_cls.model_fields.get(key)
        if (
            field is None
            or not field.is_required()
            or field.validation_alias
            or field.serialization_alias
        ):
            msg = f"Discovery key {key!r} must be a required unaliased field"
            raise ValueError(msg)
        existing = _input_discovery_models.get(key)
        if existing is model_cls:
            return
        if existing is not None or model_cls in _input_discovery_models.values():
            msg = f"Discovery key or model is already registered: {key!r}"
            raise ValueError(msg)
        _input_discovery_models[key] = model_cls
        try:
            cls.model_rebuild(force=True)
        except Exception:
            del _input_discovery_models[key]
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
    >>> from tollan.config import RuntimeContext
    >>> discovery = InputDiscovery(RuntimeContext({"input_discovery": {"inputs": [
    ...     {"sources": ["https://archive.example/data.nc"], "mode": "guess"},
    ... ]}}))
    >>> len(discovery.discover())
    1
    """

    def discover(self, *, on_error: SourceInfoErrorPolicyT = "raise") -> pd.DataFrame:
        """Combine independently selected discovery entries.

        Parameters
        ----------
        on_error : {"raise", "record"}, optional
            Whether source-information failures raise (default) or remain in
            the returned table's ``diagnostics`` column. Passed to each entry;
            configuration and selection errors still raise.

        Returns
        -------
        pandas.DataFrame
            Rows in entry order. Duplicate locations and local symlink targets
            keep the first selected row. Entries without matches contribute no
            rows. No selected candidates yields an empty table.
        """
        tables = []
        logger.debug("Input discovery: {} configured entries", len(self.config.inputs))
        for index, entry in enumerate(self.config.inputs):
            logger.debug(
                "Input discovery inputs[{}] ({}): {}",
                index,
                type(entry).__name__,
                entry,
            )
            table = entry.discover(on_error=on_error)
            logger.debug(
                "Input discovery inputs[{}]: {} selected candidates", index, len(table)
            )
            if table.empty:
                continue
            tables.append(table)
        if not tables:
            logger.debug("Input discovery: no selected candidates")
            return ToltecFileAccessor.from_sources([], on_error=on_error)
        combined = pd.concat(tables, ignore_index=True)
        seen = set()
        positions = []
        for i, location in enumerate(combined.toltec_file.source_loc):
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
