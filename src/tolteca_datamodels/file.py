"""Filename identification and file tables for LMT and TolTEC sources."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Literal, overload

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    computed_field,
    field_validator,
    model_validator,
)
from tollan.utils.fileloc import FileLoc, fileloc
from tollan.utils.log import logger

from ._filepath import FilepathInfo
from .lmt import _filepath as _lmt_filepath  # noqa: F401 -- register LMT filenames.
from .model_table import (
    ModelColumn,
    ModelColumnMapper,
    ModelTableAccessor,
    ScalarColumnRule,
)
from .obs_spec import ObsSpecContext
from .source import SourceReaderFactory, identify_source, open_source
from .toltec import (
    _filepath as _toltec_filepath,  # noqa: F401 -- register TolTEC filenames.
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from contextlib import AbstractContextManager
    from pathlib import Path

    from .obs_spec import ObsSpecBase

__all__ = [
    "FileAccessor",
    "FileTableModeT",
    "FilepathInfo",
    "SourceInfoDiagnostic",
    "SourceInfoErrorPolicyT",
]

type FileTableModeT = Literal["guess", "guess_or_read", "read"]
_validate_file_table_mode = TypeAdapter(FileTableModeT).validate_python
type SourceInfoErrorPolicyT = Literal["raise", "record"]
_validate_source_info_error_policy = TypeAdapter(SourceInfoErrorPolicyT).validate_python
type _SourceInfoStage = Literal["filepath", "open", "metadata"]


class SourceInfoDiagnostic(BaseModel):
    """An issue encountered while collecting information about a source.

    Parameters
    ----------
    stage : {"filepath", "open", "metadata"}
        Operation that reported the issue.
    message : str
        Description of the issue.
    exception_type : str, optional
        Exception class name, or None for a diagnostic without an exception.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    stage: _SourceInfoStage
    message: str
    exception_type: str | None = None


class _SourceInfoDiagnostics:
    """Collect stage outcomes before updating a table's diagnostics."""

    def __init__(self, on_error: SourceInfoErrorPolicyT) -> None:
        self._on_error = _validate_source_info_error_policy(on_error)
        self._outcomes: dict[
            int, dict[_SourceInfoStage, SourceInfoDiagnostic | None]
        ] = {}

    def record(
        self,
        position: int,
        stage: _SourceInfoStage,
        message: str,
        exception_type: str | None = None,
    ) -> None:
        """Record an identification miss or exception for a stage."""
        self._outcomes.setdefault(position, {})[stage] = SourceInfoDiagnostic(
            stage=stage, message=message, exception_type=exception_type
        )

    @contextmanager
    def capture(
        self,
        position: int,
        stage: _SourceInfoStage,
        *,
        exceptions: tuple[type[Exception], ...] = (OSError, ValueError, KeyError),
    ) -> Iterator[None]:
        self._outcomes.setdefault(position, {})[stage] = None
        try:
            yield
        except exceptions as error:
            if self._on_error == "raise":
                raise
            self.record(position, stage, str(error), type(error).__name__)

    def update_table(self, table: pd.DataFrame) -> None:
        name = FileAccessor.DIAGNOSTICS_COLUMN
        if (
            name not in table
            and self._on_error != "record"
            and not any(
                diagnostic is not None
                for outcomes in self._outcomes.values()
                for diagnostic in outcomes.values()
            )
        ):
            return
        previous = table[name] if name in table else [[] for _ in range(len(table))]
        diagnostics = []
        for position, cell in enumerate(previous):
            # Concatenating with a table without diagnostics fills these cells.
            items = [] if not isinstance(cell, list) and pd.isna(cell) else cell
            outcomes = self._outcomes.get(position, {})
            for diagnostic in outcomes.values():
                if diagnostic is not None:
                    logger.debug(
                        "Source {} (row {}): {}: {}{}",
                        table.source_loc.iloc[position].url,
                        position,
                        diagnostic.stage,
                        f"{diagnostic.exception_type}: "
                        if diagnostic.exception_type is not None
                        else "",
                        diagnostic.message,
                    )
            # Replace lists rather than mutate cells shared by DataFrame copies.
            diagnostics.append(
                [item for item in items if item.stage not in outcomes]
                + [item for item in outcomes.values() if item is not None]
            )
        table[name] = pd.Series(diagnostics, dtype=object).array


class _SourceSourceLoc(BaseModel):
    """A resource and the location used to identify its filename."""

    model_config = ConfigDict(extra="forbid")

    source: object
    source_loc: FileLoc

    @model_validator(mode="before")
    @classmethod
    def _normalize_source(cls, data: Any) -> Any:
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            data = {"source": data}
        if data.get("source_loc") is None and "source" in data:
            data = data | {"source_loc": data["source"]}
        return data

    @field_validator("source")
    @classmethod
    def _validate_source(cls, source: object) -> object:
        if source is None:
            msg = "A source is required"
            raise ValueError(msg)
        return source

    @computed_field
    @property
    def source_loc_is_local(self) -> bool:
        """Whether the identification location is a local file location."""
        return self.source_loc.is_local()

    @computed_field
    @property
    def source_loc_host(self) -> str:
        """Identification hostname, empty for local file locations."""
        return self.source_loc.host

    @computed_field
    @property
    def source_loc_scheme(self) -> str:
        """Identification URL scheme."""
        return self.source_loc.scheme


class FileAccessor(ModelTableAccessor):
    """Build and query file tables with source locations.

    Parameters
    ----------
    data : pandas.DataFrame
        Table containing ``source`` and ``source_loc`` columns. The table is
        used directly, without copying. Resources are stored in ``source``;
        ``source_loc`` contains the FileLoc used for filename identification.

    Raises
    ------
    AttributeError
        If either required column is missing.
    """

    IO_OBJ_COLUMN: ClassVar[str] = "io_obj"
    DIAGNOSTICS_COLUMN: ClassVar[str] = "diagnostics"

    _source_columns: ClassVar[dict[str, ModelColumn]] = {
        "source": ModelColumn("source", dtype=object),
        "source_loc": ModelColumn("source_loc", dtype=object),
    } | ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[ScalarColumnRule()],
        include_computed=True,
    )(_SourceSourceLoc)
    _source_adapter: ClassVar = TypeAdapter(list[_SourceSourceLoc])
    _filepath_info_types: ClassVar[tuple[type[FilepathInfo], ...]] = ()
    _filepath_info_columns: ClassVar[dict[str, ModelColumn]] = {}
    _io_obj_types: ClassVar[tuple[type, ...]] = ()
    _source_opener: ClassVar[Callable[..., Any] | None] = None
    _source_readers: ClassVar[tuple[SourceReaderFactory, ...]] = ()
    _metadata_file_extensions: ClassVar[tuple[str, ...]] = ()
    _metadata_columns: ClassVar[dict[str, ModelColumn]] = {}
    _obs_spec_cls: ClassVar[type[ObsSpecBase] | None] = None
    _obs_spec_context: ClassVar[ObsSpecContext] = ObsSpecContext()

    def __init__(self, data: pd.DataFrame) -> None:
        if not _SourceSourceLoc.model_fields.keys() <= set(data):
            msg = "A file model table requires source and source_loc columns"
            raise AttributeError(msg)
        super().__init__(data)

    @classmethod
    def from_sources(
        cls,
        sources: Iterable[Any],
        *,
        mode: FileTableModeT = "guess",
        on_error: SourceInfoErrorPolicyT = "raise",
    ) -> pd.DataFrame:
        """Build a file table with optional metadata reads.

        Parameters
        ----------
        sources : iterable
            Resources, or dicts containing ``source`` and ``source_loc``.
            Resources are retained unchanged. An omitted or None ``source_loc``
            uses ``source`` and must be convertible to FileLoc. For an open
            object, supply its filename identification location explicitly.
            Only ``source`` is opened; ``source_loc`` supplies the filename
            used for guessing.
        mode : {"guess", "guess_or_read", "read"}, optional
            Default is "guess", using filenames only. "guess_or_read" also
            attempts metadata reads when no accepted filepath type matches;
            missing values in a recognized filename do not trigger a read.
            "read" attempts metadata reads for every supported source.
        on_error : {"raise", "record"}, optional
            Default is "raise". "record" retains rows with filename-validation,
            opening or metadata-reading failures in ``diagnostics``. Identification
            misses are recorded in either mode without raising. Invalid arguments
            always raise.

        Returns
        -------
        pandas.DataFrame
            One row per input, with object-valued ``source`` and ``source_loc``
            columns in input order. ``source_loc_is_local``, ``source_loc_host``
            and ``source_loc_scheme`` describe the identification location,
            without checking availability. Header metadata replaces guessed
            observation information, preserving resource locations and
            filename-derived ``file_*`` columns. Unidentified information remains
            null. When present, ``diagnostics`` contains SourceInfoDiagnostic lists;
            an empty list means no reported issues. A filepath diagnostic may
            remain after header metadata is successfully read.

        Raises
        ------
        pydantic.ValidationError
            If the mode or resource-location data is invalid.
        TypeError
            If a location has a type that FileLoc cannot convert.

        Notes
        -----
        The base accessor only stores resources and locations. File-type-specific
        accessors add filename identification and supported metadata readers.
        Unstaged remote resources and unsupported formats retain their guessed
        information. Files opened here are closed before returning; supplied
        Dataset and DataTree objects remain open. Recording handles I/O errors
        (OSError) and invalid or missing data (ValueError or KeyError). Other
        exception types propagate.

        Examples
        --------
        >>> table = FileAccessor.from_sources([
        ...     {"source": "cached.nc", "source_loc": "archive:/observation.nc"},
        ...     "local.nc",
        ... ])
        >>> FileAccessor(table).source_loc.iloc[0].as_rsync_arg()
        'archive:/observation.nc'
        """
        mode = _validate_file_table_mode(mode)
        diagnostics = _SourceInfoDiagnostics(on_error)
        table = cls.from_models(
            cls._source_adapter.validate_python(sources),
            columns=cls._source_columns,
        )
        accessor = cls(table)
        unidentified = accessor._guess_columns(diagnostics=diagnostics)
        logger.debug(
            "Building {}: {} sources, {} unidentified filenames, mode={!r}",
            cls.__name__,
            len(table),
            len(unidentified),
            mode,
        )
        if mode != "guess":
            accessor._read_metadata(
                (
                    (position, row)
                    for position, row in enumerate(table.itertuples(index=False))
                    if mode == "read" or position in unidentified
                ),
                diagnostics=diagnostics,
            )
        diagnostics.update_table(table)
        return table

    def _guess_columns(self, *, diagnostics: _SourceInfoDiagnostics) -> set[int]:
        """Populate filename guesses in place; return unidentified row positions."""
        if not self._filepath_info_types:
            return set(range(len(self._obj)))
        parsed = []
        for position, location in enumerate(self.source_loc):
            info = None
            with diagnostics.capture(position, "filepath"):
                info = FilepathInfo.from_source(
                    location,
                    accepted_types=self._filepath_info_types,
                    raise_on_error=False,
                )
                if info is None:
                    diagnostics.record(
                        position, "filepath", "No accepted filename pattern matched"
                    )
            parsed.append(info)
        self.update_from_models(
            parsed, columns=self._filepath_info_columns, inplace=True
        )
        return {position for position, info in enumerate(parsed) if info is None}

    def query_obs_spec(self, spec: ObsSpecBase | str | int) -> pd.DataFrame:
        """Query rows with an observation specification.

        Parameters
        ----------
        spec : ObsSpecBase, str or int
            Observation selector using this accessor's field layout and
            label interpretations. Value constraints precede positional selection.

        Returns
        -------
        pandas.DataFrame
            Matching rows, retaining their order and index.

        Raises
        ------
        NotImplementedError
            If the accessor has no observation-selector class configured.
        """
        if self._obs_spec_cls is None:
            msg = "No observation-selector class configured for this accessor"
            raise NotImplementedError(msg)
        return self._obs_spec_cls.parse(spec).select(
            self._obj, context=self._obs_spec_context
        )

    @overload
    def read_metadata(
        self,
        *,
        inplace: Literal[False] = False,
        on_error: SourceInfoErrorPolicyT = "raise",
    ) -> pd.DataFrame: ...

    @overload
    def read_metadata(
        self, *, inplace: Literal[True], on_error: SourceInfoErrorPolicyT = "raise"
    ) -> None: ...

    @overload
    def read_metadata(
        self, *, inplace: bool, on_error: SourceInfoErrorPolicyT = "raise"
    ) -> pd.DataFrame | None: ...

    def read_metadata(
        self, *, inplace: bool = False, on_error: SourceInfoErrorPolicyT = "raise"
    ) -> pd.DataFrame | None:
        """Read metadata into table columns.

        Parameters
        ----------
        inplace : bool, optional
            Whether to update this table. Default is False.
        on_error : {"raise", "record"}, optional
            Default is "raise". "record" retains previously collected values and
            records opening or metadata-reading failures in ``diagnostics``.
            Identification misses are recorded in either mode. Successful retries
            clear the corresponding stage's diagnostics; skipped stages retain them.

        Returns
        -------
        pandas.DataFrame or None
            Updated table, or None if ``inplace=True``. Header values replace
            guessed observation information. Filename columns, source
            locations, row order, index and unrelated columns are preserved.

        Notes
        -----
        Resource support and ownership follow `from_sources`. Unstaged remote
        resources and unsupported formats retain their existing information.
        In raising mode, read errors propagate without updating the table.
        Assignment errors during an in-place update can leave the table
        partially updated.
        """
        diagnostics = _SourceInfoDiagnostics(on_error)
        result = self._obj if inplace else self._obj.copy()
        accessor = self if inplace else type(self)(result)
        accessor._read_metadata(
            enumerate(result.itertuples(index=False)), diagnostics=diagnostics
        )
        diagnostics.update_table(result)
        return None if inplace else result

    def _read_metadata(
        self, rows: Iterable[tuple[int, Any]], *, diagnostics: _SourceInfoDiagnostics
    ) -> None:
        """Populate metadata in place for the supplied row positions and records."""
        if not self._source_readers:
            return
        metadata = {}
        for position, row in rows:
            obj = getattr(row, self.IO_OBJ_COLUMN, None)
            if obj is None and isinstance(row.source, self._io_obj_types):
                obj = row.source
            if obj is None and (
                row.source_loc.path.suffix not in self._metadata_file_extensions
                or fileloc(row.source).is_remote()
            ):
                logger.debug("Skipping metadata read for {}", row.source_loc)
                continue
            with self._open_metadata_source(
                position, row, diagnostics=diagnostics
            ) as data:
                if data is None:
                    continue
                with diagnostics.capture(position, "metadata"):
                    reader = identify_source(data, readers=self._source_readers)
                    if reader is None:
                        diagnostics.record(
                            position, "metadata", "No supported dataset recognized"
                        )
                        continue
                    metadata[position] = reader.read_metadata()
        logger.debug("Read metadata for {} table rows", len(metadata))
        self.update_from_models(metadata, columns=self._metadata_columns, inplace=True)

    @contextmanager
    def open(
        self,
        *,
        io_objs: Iterable[object | None] | None = None,
        on_error: SourceInfoErrorPolicyT = "record",
    ) -> Iterator[pd.DataFrame]:
        """Open table resources and yield a new, header-enriched table.

        Parameters
        ----------
        io_objs : iterable of object or None, optional
            Open objects supported by this accessor, one per table row.
            None entries use the corresponding source. If omitted, use the
            table's ``io_obj`` column when present, otherwise open its sources.
            Supplied objects are borrowed and are not closed by this scope.
        on_error : {"raise", "record"}, optional
            Default is "record", retaining failed rows and storing their
            diagnostics in ``diagnostics``. Failed opens and unrecognized payloads
            have ``io_obj=None`` and do not attempt metadata reads. Metadata
            failures retain the recognized open object for inspection. Filename
            diagnostics do not prevent opening or header reads. Use "raise" to
            stop on the first opening, identification or metadata-reading failure.

        Yields
        ------
        pandas.DataFrame
            A new table with updated metadata and an ``io_obj`` column of open
            resources. The input table is unchanged.

        Raises
        ------
        ValueError
            If ``io_objs`` has the wrong length. With ``on_error="raise"``, also
            raised for unstaged remote sources and payloads not recognized by
            the accessor's readers, including supplied open objects.
        TypeError
            If a supplied IO object is not supported by this accessor.
        NotImplementedError
            If opening a file with an accessor that has no source opener.

        Notes
        -----
        Each distinct local path is opened once per scope. Files opened here
        are closed on exit, including on errors. Load any file-backed arrays
        needed afterward while the scope is open. Selecting rows does not
        extend resource lifetimes. The yielded table retains metadata after
        exit, but its temporary ``io_obj`` column is removed.
        """
        diagnostics = _SourceInfoDiagnostics(on_error)
        if io_objs is None:
            io_objs = self._obj.get(self.IO_OBJ_COLUMN)
        objects = [None] * len(self._obj) if io_objs is None else list(io_objs)
        if len(objects) != len(self._obj):
            msg = "Provide one IO object (or None) per table row"
            raise ValueError(msg)
        if any(
            obj is not None and not isinstance(obj, self._io_obj_types)
            for obj in objects
        ):
            msg = "IO objects must have a supported type or be None"
            raise TypeError(msg)
        with ExitStack() as stack:
            owned = {}
            for i, source in enumerate(self.source):
                obj = objects[i]
                objects[i] = None
                with diagnostics.capture(i, "open"):
                    if obj is None and isinstance(source, self._io_obj_types):
                        obj = source
                    if obj is not None:
                        logger.debug(
                            "Borrowing table resource for {}", self.source_loc.iloc[i]
                        )
                    else:
                        location = fileloc(source)
                        key = (
                            location.path.resolve()
                            if location.is_local()
                            else location.url
                        )
                        if key not in owned:
                            logger.debug("Opening table resource {}", location)
                            owned[key] = stack.enter_context(
                                self._open_source(location)
                            )
                        obj = owned[key]
                    self._validate_open_source(obj)
                    objects[i] = obj
            result = self._obj.copy()
            result[self.IO_OBJ_COLUMN] = pd.Series(objects, dtype=object).array
            try:
                type(self)(result)._read_metadata(
                    (
                        (position, row)
                        for position, row in enumerate(result.itertuples(index=False))
                        if objects[position] is not None
                    ),
                    diagnostics=diagnostics,
                )
                diagnostics.update_table(result)
                yield result
            finally:
                if self.IO_OBJ_COLUMN in result:
                    del result[self.IO_OBJ_COLUMN]
                logger.debug("Closing table scope: {} owned resources", len(owned))

    @classmethod
    def _validate_open_source(cls, data_source: object) -> None:
        """Require a payload recognized by the accessor's configured readers."""
        if (
            cls._source_readers
            and identify_source(data_source, readers=cls._source_readers) is None
        ):
            msg = f"No supported dataset recognized by {cls.__name__}"
            raise ValueError(msg)

    @contextmanager
    def _open_metadata_source(
        self, position: int, row: Any, *, diagnostics: _SourceInfoDiagnostics
    ) -> Iterator[Any]:
        """Borrow or open a metadata source, yielding None when opening fails."""
        obj = getattr(row, self.IO_OBJ_COLUMN, None)
        if obj is None and isinstance(row.source, self._io_obj_types):
            obj = row.source
        with ExitStack() as stack:
            with diagnostics.capture(
                position, "open", exceptions=(OSError, ValueError)
            ):
                if obj is None:
                    obj = stack.enter_context(self._open_source(row.source))
            yield obj

    @classmethod
    def _open_source(cls, source: str | Path | FileLoc) -> AbstractContextManager[Any]:
        """Open an owned resource with the configured backend."""
        if cls._source_opener is None:
            msg = f"No source opener configured for {cls.__name__}"
            raise NotImplementedError(msg)
        return open_source(source, opener=cls._source_opener)

    @property
    def source(self) -> pd.Series:
        """Resource objects in row order."""
        return self._obj["source"]

    @property
    def source_loc(self) -> pd.Series:
        """Filename identification locations in row order."""
        return self._obj["source_loc"]
