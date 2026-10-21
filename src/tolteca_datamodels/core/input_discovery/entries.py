"""Shared models for configured input discovery."""

from __future__ import annotations

import glob
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from pydantic import BeforeValidator, ConfigDict, Field
from tollan.config import FrozenBaseModel
from tollan.config.types import AnyPath, FileLocField, StringSelector
from tollan.utils.fileloc import FileLoc, fileloc
from tollan.utils.log import logger

from ..data_model import DataModelSet
from ..file import FileAccessor, FileTableModeT
from ..obs_spec import ObsSpecBase
from .query import query_file_store

if TYPE_CHECKING:
    import pandas as pd

    from ..file import SourceErrorPolicyT
    from ..file_store import FileStoreBase


class InputDiscoveryEntry(FrozenBaseModel, ABC):
    """Settings for one input-discovery operation.

    Parameters
    ----------
    data_model_select : StringSelector, optional
        Data-model capabilities to use within the enclosing workflow's scope.
        Defaults to all available capabilities.
    """

    model_config = ConfigDict(extra="forbid")
    data_model_select: StringSelector = Field(default=".*", validate_default=True)

    def _select_data_models(self, data_models: DataModelSet | None) -> DataModelSet:
        if data_models is None:
            data_models = DataModelSet.from_select()
        selected = data_models.select(self.data_model_select)
        logger.debug(
            "{} data models: {}", type(self).__name__, list(selected.data_models)
        )
        return selected

    @abstractmethod
    def discover(
        self,
        *,
        data_models: DataModelSet | None = None,
        on_error: SourceErrorPolicyT = "record",
    ) -> pd.DataFrame:
        """Discover inputs using these settings.

        Parameters
        ----------
        data_models : DataModelSet, optional
            Available data-model capabilities.
            Defaults to all registered support. The entry's data-model selection
            narrows this scope before candidates are resolved.
        on_error : {"raise", "record"}, optional
            Whether source-information failures raise or remain in
            the returned table's ``source_trace`` column. Invalid settings raise.
            Default is "record".

        Returns
        -------
        pandas.DataFrame
            Selected file rows in discovery order. No matches yields an empty
            table. Includes the ``source`` and ``source_loc`` columns supplied
            by ``FileAccessor.from_sources()``.
        """


class FileStoreInputDiscovery[ObsSpecT: ObsSpecBase](InputDiscoveryEntry):
    """Build a selected file table from a directory-backed store.

    Parameters
    ----------
    obs_spec : ObsSpecT, optional
        Observation selector used for candidate discovery and table selection.
        Defaults to ``"*"``, validated through the bound selector field type.
    mode : {"guess", "guess_or_read", "read"}, optional
        How to populate candidate information. Default ``"guess"`` uses
        filenames only. ``"guess_or_read"`` reads supported, openable files
        whose filenames are unrecognized; ``"read"`` also reads recognized files
        supported by the selected handlers. Recognized foreign filenames retain
        their guesses without being opened.

    Notes
    -----
    Bind ``ObsSpecT`` to an annotated selector type that accepts selector text
    and serializes it in the desired configuration format. Implement `_store`
    with a file store accepting that selector type.
    """

    obs_spec: ObsSpecT = Field(default="*", validate_default=True)
    mode: FileTableModeT = "guess"

    @property
    @abstractmethod
    def _store(self) -> FileStoreBase[ObsSpecT]:
        """File store for this discovery entry."""

    def discover(
        self,
        *,
        data_models: DataModelSet | None = None,
        on_error: SourceErrorPolicyT = "record",
    ) -> pd.DataFrame:
        """Build and select the candidate file table.

        Parameters
        ----------
        data_models : DataModelSet, optional
            Store requirements and file-table support for population and
            selector interpretation. Defaults to all registered data models.
        on_error : {"raise", "record"}, optional
            Whether source-information failures raise or remain in
            the table's ``source_trace`` column before ObsSpec selection.
            Default is "record".

        Returns
        -------
        pandas.DataFrame
            Candidates populated using ``mode`` and filtered with ``obs_spec``,
            retaining discovery order. No matches yields an empty table.
        """
        data_models = self._select_data_models(data_models)
        return query_file_store(
            self._store,
            self.obs_spec,
            mode=self.mode,
            on_error=on_error,
            data_models=data_models,
        )


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
        Pattern relative to the runtime's root_path. ``*`` matches within one
        directory; ``**`` searches recursively. Only files are returned, in glob
        enumeration order. Remote patterns are unsupported.
        Without a root_path, relative patterns use the working directory when
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


class SourcesInputDiscovery[ObsSpecT: ObsSpecBase](InputDiscoveryEntry):
    """Discover inputs from exact locations and explicit glob patterns.

    Parameters
    ----------
    sources : list
        Each item is an exact FileLoc-compatible location, or ``{"glob": pattern}``.
        FileLoc dictionaries such as ``{"path": "data.nc"}`` are accepted.
        Wildcard characters in exact locations are literal. Dictionaries matching
        neither form are rejected.
    obs_spec : ObsSpecT, optional
        Observation selector applied after table population, independently of
        other discovery entries. Omitted by default.
    mode : {"guess", "guess_or_read", "read"}, optional
        Default ``"guess_or_read"`` reads supported, openable sources whose
        filenames are unrecognized. ``"guess"`` uses filenames only; ``"read"``
        also reads recognized files supported by the selected handlers.
        Recognized foreign filenames retain their guesses without being opened.

    Notes
    -----
    Bind ``ObsSpecT`` to the selector field type used by the application.
    That type controls accepted selector inputs and their serialization.
    """

    sources: list[GlobSource | FileLocField] = Field(min_length=1)
    obs_spec: ObsSpecT | None = None
    mode: FileTableModeT = "guess_or_read"

    def discover(
        self,
        *,
        data_models: DataModelSet | None = None,
        on_error: SourceErrorPolicyT = "record",
    ) -> pd.DataFrame:
        """Build a file table from these locations and patterns.

        Parameters
        ----------
        data_models : DataModelSet, optional
            File-table support used for population and selector interpretation.
            Defaults to all registered support. Explicit locations and globs
            are resolved regardless of the selected models' store requirements.
        on_error : {"raise", "record"}, optional
            Whether source-information failures raise or remain in
            the table's ``source_trace`` column before ObsSpec selection.
            Default is "record".

        Returns
        -------
        pandas.DataFrame
            Rows in source and glob enumeration order, with optional ObsSpec
            selection applied after population. No matches yields an empty table.
        """
        data_models = self._select_data_models(data_models)
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
        files = FileAccessor.partial(accessor_types=data_models.file_accessor_types)
        selected = files.from_sources(locations, mode=self.mode, on_error=on_error)
        if self.obs_spec is not None and not selected.empty:
            selected = files(selected).query_obs_spec(self.obs_spec)
        logger.debug(
            "Source discovery: {} sources -> {} resolved locations -> {} distinct "
            "candidates -> {} selected; mode={}, obs_spec={}",
            len(self.sources),
            n_resolved,
            len(locations),
            len(selected),
            self.mode,
            self.obs_spec,
        )
        return selected
