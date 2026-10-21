"""Shared file-table machinery."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, overload

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    computed_field,
    field_validator,
    model_validator,
)
from tollan.accessor.pandas.model_table import (
    ModelColumn,
    ModelColumnMapper,
    ModelTableAccessor,
    RecordRuleMapper,
    ScalarColumnRule,
)
from tollan.utils.fileloc import FileLoc, fileloc
from tollan.utils.log import logger
from tollan.utils.typing import (
    GenericBindingRequirement,
    resolve_concrete_generic_bindings,
)

from .. import filepath as _filepath
from ..filepath import FilepathInfo
from ..obs_spec import ObsSpecBase, ObsSpecContext
from ..source import SourceOpener, SourceReaderRules, source_opener
from ..types import DataModelInfo, FrozenDataModel
from ._capabilities import _AccessorTypes, _FileCapabilities
from ._resolution import _IdentifiedSource
from .trace import SourceErrorPolicyT, _SourceTraceCollector, _summarize_traces

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping
    from enum import Flag

    from ..source import SourceReader

__all__ = [
    "FileAccessor",
    "FileAccessorView",
    "FileTableModeT",
    "SourceErrorPolicyT",
]

type FileTableModeT = Literal["guess", "guess_or_read", "read"]
_validate_file_table_mode = TypeAdapter(FileTableModeT).validate_python


def _borrowed_source(row: Any) -> object | None:
    obj = getattr(row, FileAccessor.IO_OBJ_COLUMN, None)
    if obj is not None:
        return obj
    if not isinstance(row.source, (str, Path, FileLoc)):
        return row.source
    return None


class _SourceSourceLoc(FrozenDataModel):
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


class _SourceIdentification(BaseModel):
    """Reader selection for one resource and one set of routing choices."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    source: object
    source_loc: FileLoc
    choices: dict[type, type | None]
    accessor_cls: type
    handlers: tuple[type, ...]

    def applies_to(self, row: Any, choices: dict) -> bool:
        same_source = self.source is row.source or (
            isinstance(self.source, (str, Path, FileLoc))
            and isinstance(row.source, (str, Path, FileLoc))
            and self.source == row.source
        )
        return (
            same_source
            and self.source_loc == row.source_loc
            and self.choices == choices
        )


class FileAccessor[DataModelT: DataModelInfo](ModelTableAccessor):
    """Build and query file tables with source locations.

    Parameters
    ----------
    data : pandas.DataFrame
        Table containing ``source`` and ``source_loc`` columns. The table is
        used directly, without copying. Resources are stored in ``source``;
        ``source_loc`` contains the FileLoc used for filename identification.
    accessor_types : tuple of FileAccessor subclasses, optional
        Accessor classes for enrichment and observation queries. Defaults to all
        registered classes. Concrete accessors use their own support and do not
        accept this argument. Duplicate classes are applied once, in first-occurrence
        order.

    Raises
    ------
    AttributeError
        If either required column is missing.
    ValueError
        If accessor_types is supplied to a concrete accessor or contains a class
        without a concrete data-model binding.

    Notes
    -----
    Importing `tolteca_datamodels.file` enables LMT and TolTEC support. Optional
    data models may expose an explicit registration function. The base
    accessor defaults to all registered support; concrete accessors retain their
    declared scope. Registrations must be in place before constructing an accessor.

    Bind ``DataModelT`` to a `DataModelInfo` subclass when defining a concrete
    accessor. Its `data_model_cls` class attribute contains that class.
    Concrete subclasses can be used directly without registration. Register a
    `DataModelSpec` definition to include an accessor in generic workflows.
    """

    data_model_cls: ClassVar[type[DataModelInfo]]
    """Data-model definition handled by this concrete accessor."""

    FILEPATH_INFO_COLUMN: ClassVar[str] = "filepath_info"
    IO_OBJ_COLUMN: ClassVar[str] = "io_obj"
    ACCESSOR_OBJ_COLUMN: ClassVar[str] = "accessor_obj"
    METADATA_READ_COLUMN: ClassVar[str] = "metadata_read"
    _IDENTIFICATION_COLUMN: ClassVar[str] = "_source_identification"
    TRACE_COLUMN: ClassVar[str] = "source_trace"

    _source_columns: ClassVar[dict[str, ModelColumn]] = {
        "source": ModelColumn("source", dtype=object),
        "source_loc": ModelColumn("source_loc", dtype=object),
    } | ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[ScalarColumnRule()],
        include_computed=True,
    )(_SourceSourceLoc)
    _source_adapter: ClassVar = TypeAdapter(list[_SourceSourceLoc])
    _data_models: ClassVar[frozenset[str]] = frozenset()
    _reader_rules: ClassVar[SourceReaderRules] = SourceReaderRules(rules=())
    _source_opener: ClassVar[SourceOpener] = source_opener
    _metadata_columns: ClassVar[dict[str, ModelColumn]] = {}
    _obs_spec_cls: ClassVar[type[ObsSpecBase] | None] = None
    _obs_spec_context: ClassVar[ObsSpecContext] = ObsSpecContext()
    _data_kind_type: ClassVar[type[Flag] | None] = None
    _data_kind_column: ClassVar[str] = ""
    _data_kind_mapper: ClassVar[RecordRuleMapper[int] | None] = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        bindings = resolve_concrete_generic_bindings(
            cls,
            FileAccessor,
            DataModelT=GenericBindingRequirement("data_model_cls", bound=DataModelInfo),
        )
        if bindings is None:
            return
        data_model_cls = bindings["DataModelT"]
        if not isinstance(getattr(data_model_cls, "name", None), str):
            msg = f"{cls.__name__}'s data-model definition must declare a string name"
            raise TypeError(msg)
        cls.data_model_cls = data_model_cls
        cls._data_models = frozenset([data_model_cls.name])

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        accessor_types: tuple[type[FileAccessor], ...] | None = None,
    ) -> None:
        if accessor_types is not None and type(self) is not FileAccessor:
            msg = "Concrete file accessors do not accept accessor_types"
            raise ValueError(msg)
        if not _SourceSourceLoc.model_fields.keys() <= set(data):
            msg = "A file model table requires source and source_loc columns"
            raise AttributeError(msg)
        super().__init__(data)
        if accessor_types is None:
            if type(self) is FileAccessor:
                # Specs depend on FileAccessor; lookup is needed only at runtime.
                from ..data_model import data_models  # noqa: PLC0415

                accessor_types = tuple(
                    spec.file_accessor_cls for spec in data_models.values()
                )
            else:
                accessor_types = (type(self),)
        self._capabilities = _FileCapabilities[FileAccessor](
            accessor_types=accessor_types, default_accessor=type(self)
        )

    @classmethod
    def partial(
        cls, *, accessor_types: tuple[type[FileAccessor], ...]
    ) -> FileAccessorView:
        """Bind file-table operations to selected accessor classes.

        Parameters
        ----------
        accessor_types : tuple of FileAccessor subclasses
            Concrete classes for table enrichment, source access, and queries.
            They need not be registered. An empty tuple enables filename
            information only. A concrete accessor can select only itself.

        Returns
        -------
        FileAccessorView
            Immutable generic-accessor factory. Use its ``from_sources()`` to
            build a table and call the view with a table to access it using the
            same accessor classes. No files are opened by creating the view.
        """
        view = FileAccessorView(accessor_types=accessor_types)
        if cls is FileAccessor:
            return view
        return FileAccessorView(accessor_types=(cls,)).partial(
            accessor_types=view.accessor_types
        )

    @classmethod
    def from_sources(
        cls,
        sources: Iterable[Any],
        *,
        accessor_types: tuple[type[FileAccessor], ...] | None = None,
        mode: FileTableModeT = "guess",
        on_error: SourceErrorPolicyT = "record",
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
        accessor_types : tuple of FileAccessor subclasses, optional
            Enrichment and observation-query classes for the generic accessor.
            Defaults to all registered classes. Filename parsing remains
            independent of this choice. Duplicate classes are applied once, in
            first-occurrence order. Omit for concrete accessors.
        mode : {"guess", "guess_or_read", "read"}, optional
            Default is "guess", using filenames only. "guess_or_read" also
            attempts metadata reads when no registered filename parser matches;
            missing values in a recognized filename do not trigger a read.
            "read" attempts reads for recognized filenames supported by the
            selected accessor classes, and for unrecognized filenames. Recognized
            foreign filenames retain their guesses without being opened in either mode.
            Filename support is determined by shared ``data_models`` memberships.
        on_error : {"raise", "record"}, optional
            Default is "record", retaining rows with filename-validation,
            opening, identification, metadata-reading or projection failures
            in ``source_trace``.
            Use "raise" to stop on failure. An unrecognized filename
            alone is recorded without raising in either mode. Invalid arguments
            always raise.

        Returns
        -------
        pandas.DataFrame
            One row per input, with object-valued ``source`` and ``source_loc``
            columns in input order. ``source_loc_is_local``, ``source_loc_host``
            and ``source_loc_scheme`` describe the identification location,
            without checking availability. ``filepath_info`` contains the
            FilepathInfo parsed from ``source_loc``, or None if parsing did not
            succeed. It remains attached to its row after selection and metadata
            updates. ``data_models`` holds current memberships: None means
            unknown, and an empty collection matches no data-model scope.
            Successful filename parsing supplies a collection, possibly empty.
            Header metadata replaces guessed observation information,
            preserving resource locations and
            filename-derived ``file_*`` columns. Unidentified information remains
            null. ``source_trace`` contains SourceTraceEvent lists describing
            successes, skips and failures. A filename miss remains in the
            history after header metadata is successfully read. ``metadata_read``
            is True only after a successful source metadata read and table update.

        Raises
        ------
        pydantic.ValidationError
            If the mode or resource-location data is invalid.
        ValueError
            If accessor_types is supplied to a concrete accessor, or an opened object
            is unrecognized with ``on_error="raise"``.
        TypeError
            If a location has a type that FileLoc cannot convert.
        RuntimeError
            If matching filename or reader registrations are ambiguous.

        Notes
        -----
        Every accessor projects all registered filename types and uses its
        selected accessor classes for derived columns and metadata. The public
        `tolteca_datamodels.file` module enables LMT and TolTEC support; optional
        instruments opt in through their registration entry points. Concrete
        accessors restrict enrichment and readers, not filename information.
        Classifications use separate integer columns such as ``data_kind_toltec`` and
        ``data_kind_dummy_instru``.
        Each enabled classifier supplies its own unknown flag for unmatched
        rows, including foreign and unidentified inputs.
        Bind subsequent generic accessors with the same accessor_types; the returned
        DataFrame does not retain that selection.
        The selected opener receives the original source. Built-in openers
        require local files; custom policies may support other locations.
        Unsupported sources follow on_error, retaining guesses in record mode.
        Files opened here are closed before returning; supplied
        objects remain open. Recording handles I/O errors
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
        trace = _SourceTraceCollector(on_error)
        table = cls.from_models(
            cls._source_adapter.validate_python(sources),
            columns=cls._source_columns,
        )
        table[cls.METADATA_READ_COLUMN] = False
        accessor = (
            cls(table, accessor_types=accessor_types)
            if accessor_types is not None
            else cls(table)
        )
        accessor._guess_columns(trace=trace)
        accessor._record_scope(trace)
        logger.debug(
            "Building {}: {} sources, {} unidentified filenames, mode={!r}",
            cls.__name__,
            len(table),
            accessor.filepath_info.isna().sum(),
            mode,
        )
        if mode != "guess":
            for position, memberships in enumerate(accessor._iter_row_memberships()):
                if mode == "guess_or_read" and memberships is not None:
                    trace.record(
                        position,
                        "metadata",
                        "Data-model memberships known",
                        outcome="skipped",
                    )
            accessor._read_metadata(
                (
                    (position, row)
                    for position, row in enumerate(table.itertuples(index=False))
                    if row.data_models is None
                    or (
                        mode == "read"
                        and not accessor._capabilities._data_models.isdisjoint(
                            cast("Iterable[str]", row.data_models)
                        )
                    )
                ),
                trace=trace,
            )
        else:
            for position in range(len(table)):
                trace.record(
                    position,
                    "metadata",
                    "Filename guesses requested",
                    outcome="skipped",
                )
        trace.update_table(table)
        return table

    def _guess_columns(self, *, trace: _SourceTraceCollector) -> None:
        """Store filename parse results and project their query columns in place."""
        parsed = []
        models = {}
        for position, location in enumerate(self.source_loc):
            info = None
            with trace.capture(position, "filepath"):
                info = FilepathInfo.from_source(location, raise_on_error=False)
                if info is None:
                    trace.record(
                        position,
                        "filepath",
                        "No accepted filename pattern matched",
                        outcome="unrecognized",
                    )
                if info is not None:
                    models.setdefault(type(info), {})[position] = info
            parsed.append(info)
        self._obj[self.FILEPATH_INFO_COLUMN] = pd.Series(parsed, dtype=object).array
        model_classes = dict.fromkeys(
            [parser.filepath_info_cls for parser in _filepath._filepath_parsers]
            + list(models)
        )
        self._reset_projection_columns(
            self._obj,
            (model._columns for model in model_classes),
            list(range(len(self._obj))),
        )
        for model, rows in models.items():
            self.update_from_models(rows, columns=model._columns, inplace=True)
        self._obj["data_models"] = pd.Series(
            [
                memberships if memberships is not None or info is None else []
                for info, memberships in zip(
                    parsed, self._iter_row_memberships(), strict=True
                )
            ],
            dtype=object,
        ).array
        self._guess_data_kinds()
        groups = {}
        for position, memberships in enumerate(
            self._obj.get("data_models", [None] * len(self._obj))
        ):
            data_models = (
                frozenset(cast("Iterable[str]", memberships))
                if pd.api.types.is_list_like(memberships)
                else frozenset()
            )
            groups.setdefault(data_models, []).append(position)
        computed = self._capabilities._guess_derived_columns(self._obj, groups)
        self._obj[computed.columns] = computed

    def _iter_row_memberships(self) -> Iterator[Iterable[str] | None]:
        """Yield current memberships, or None when scope is unknown."""
        for data_models in self._obj.get("data_models", [None] * len(self._obj)):
            if pd.api.types.is_list_like(data_models):
                yield cast("Iterable[str]", data_models)
            else:
                yield None

    def _record_scope(self, trace: _SourceTraceCollector) -> None:
        message = "Allowed data models: " + ", ".join(
            sorted(self._capabilities._data_models)
        )
        for position, memberships in enumerate(self._iter_row_memberships()):
            if memberships is None:
                outcome = "unrecognized"
            elif self._capabilities._data_models.isdisjoint(memberships):
                outcome = "out_of_scope"
            else:
                outcome = "in_scope"
            trace.record(position, "scope", message, outcome=outcome)

    def source_trace_summary(self) -> pd.DataFrame:
        """Summarize the latest processing outcome at each stage.

        Returns
        -------
        pandas.DataFrame
            Columns ``stage``, ``outcome``, ``count`` and ``total``. Count is the
            number of rows with that latest stage outcome; total is this table's
            row count. Stages not reached by a row do not contribute a count.
            Per-row ``source_trace`` lists retain earlier attempts.
        """
        return _summarize_traces(
            self._obj.get(self.TRACE_COLUMN, []), total=len(self._obj)
        )

    def _get_rule_column_names(self) -> dict[str, str]:
        """Use file-only attributes where no current query column exists."""
        return {
            name.removeprefix("file_"): name
            for name in self._obj
            if isinstance(name, str)
            and name.startswith("file_")
            and name.removeprefix("file_") not in self._obj
        }

    @staticmethod
    def _reset_projection_columns(
        table: pd.DataFrame,
        projections: Iterable[Mapping[str, ModelColumn]],
        positions: list[int],
    ) -> None:
        """Prepare destinations and clear the rows replaced by a population pass."""
        names = set()
        for columns in projections:
            for name, column in columns.items():
                if name in names:
                    continue
                names.add(name)
                if name not in table:
                    table[name] = pd.Series(
                        None,
                        index=table.index,
                        dtype=object if column.dtype is None else column.dtype,
                    ).array
                else:
                    column_position = cast("int", table.columns.get_loc(name))
                    table.iloc[positions, column_position] = None

    @staticmethod
    def _guess_derived_columns(table: pd.DataFrame) -> pd.DataFrame:
        """Compute columns while preserving the input index and row order."""
        return pd.DataFrame(index=table.index)

    def _guess_data_kinds(self) -> None:
        """Populate each instrument's classification column independently."""
        names = self._get_rule_column_names()
        for name, cls in self._capabilities._data_kind_accessors.items():
            kind_type = cls._data_kind_type
            assert kind_type is not None
            values: list[int | None] = [kind_type.__members__["unknown"].value] * len(
                self._obj
            )
            if cls._data_kind_mapper is not None and "data_models" in self._obj:
                positions = [
                    position
                    for position, data_models in enumerate(self._obj.data_models)
                    if pd.api.types.is_list_like(data_models)
                    and not cls._data_models.isdisjoint(data_models)
                ]
                if positions:
                    table = (
                        self._obj
                        if len(positions) == len(self._obj)
                        else self._obj.iloc[positions]
                    )
                    kinds = cls._data_kind_mapper.map_table(table, column_names=names)
                    for position, kind in zip(positions, kinds, strict=True):
                        values[position] = kind
            self._obj[name] = pd.array(values, dtype="Int64")

    def query_obs_spec(self, spec: ObsSpecBase | str | int) -> pd.DataFrame:
        """Query rows with an observation specification.

        Parameters
        ----------
        spec : ObsSpecBase, str or int
            Observation selector using this accessor's field layout and
            label interpretations. A parsed instance can also be used without
            registered handlers; then labels have no numeric interpretations.
            Value constraints precede positional selection.

        Returns
        -------
        pandas.DataFrame
            Matching rows, retaining their order and index.

        Raises
        ------
        NotImplementedError
            If text or an integer is supplied without a configured selector class.
            A parsed ObsSpecBase instance supplies its own field layout.
        """
        if self._capabilities._obs_spec_cls is None:
            if not isinstance(spec, ObsSpecBase):
                msg = "No observation-selector class configured for this accessor"
                raise NotImplementedError(msg)
        else:
            spec = self._capabilities._obs_spec_cls.parse(spec)
        return spec.select(self._obj, context=self._capabilities._obs_spec_context)

    def _get_query_namespace(self) -> dict[str, object]:
        return self._query_namespace | self._capabilities._query_namespace

    def _with_table(self, table: pd.DataFrame) -> FileAccessor:
        """Bind the same data-model support to another table."""
        if type(self) is FileAccessor:
            return FileAccessor(table, accessor_types=self._capabilities.accessor_types)
        return type(self)(table)

    @overload
    def update_from_source_metadata(
        self,
        *,
        inplace: Literal[False] = False,
        force: bool = False,
        on_error: SourceErrorPolicyT = "record",
    ) -> pd.DataFrame: ...

    @overload
    def update_from_source_metadata(
        self,
        *,
        inplace: Literal[True],
        force: bool = False,
        on_error: SourceErrorPolicyT = "record",
    ) -> None: ...

    @overload
    def update_from_source_metadata(
        self,
        *,
        inplace: bool,
        force: bool = False,
        on_error: SourceErrorPolicyT = "record",
    ) -> pd.DataFrame | None: ...

    def update_from_source_metadata(
        self,
        *,
        inplace: bool = False,
        force: bool = False,
        on_error: SourceErrorPolicyT = "record",
    ) -> pd.DataFrame | None:
        """Read source metadata and update the table's observation columns.

        Parameters
        ----------
        inplace : bool, optional
            Whether to update this table. Default is False.
        force : bool, optional
            Reread rows whose ``metadata_read`` is already True. Default is False,
            reading only rows without a successful source metadata update.
        on_error : {"raise", "record"}, optional
            Default is "record", retaining previously collected values and
            recording opening, identification, metadata-reading or projection
            failures in ``source_trace``. Use "raise" to stop on failure.
            Successful retries resolve earlier issues without deleting history.

        Returns
        -------
        pandas.DataFrame or None
            Updated table, or None if ``inplace=True``. Header values replace
            guessed observation information. Filename columns, source
            locations, row order, index and unrelated columns are preserved.
            Successful updates set ``metadata_read=True``. Failed or skipped
            reads leave that flag and previously collected values unchanged.
            Readers returning no metadata may update classifications without
            marking metadata complete or replacing observation values.

        Raises
        ------
        ValueError
            If an opened object is unrecognized with ``on_error="raise"``.
        RuntimeError
            If opener choices or matching readers are ambiguous, regardless of on_error.

        Notes
        -----
        Current ``data_models`` memberships restrict reads to the selected
        handlers. Null memberships allow content identification; an empty
        collection matches no handler. Original ``filepath_info`` does not
        determine the current scope.
        Current identity columns select readers; original ``file_*`` snapshots
        do not override them. A failed selected reader is not retried as another
        data type.
        Inside `open`, reuse the live ``accessor_obj`` without opening or
        identifying the source again. Otherwise, open and identify eligible
        sources as needed, closing newly owned resources after each read.
        Resource support and ownership follow `from_sources`. Unsupported
        sources follow on_error, retaining existing information in record mode.
        Each row's metadata and classifications are updated together. Failed
        rows retain their collected values; record mode adds an error event and
        continues with the remaining rows. In raising mode, a failure leaves
        the table unchanged, including with ``inplace=True``.
        """
        trace = _SourceTraceCollector(on_error)
        result = self._obj if inplace else self._obj.copy()
        accessor = self if inplace else self._with_table(result)
        accessor._record_scope(trace)
        completed = (
            result.get(
                self.METADATA_READ_COLUMN, pd.Series(data=False, index=result.index)
            )
            .fillna(value=False)
            .astype(bool)
        )
        if not force:
            for position in completed.to_numpy().nonzero()[0]:
                trace.record(
                    int(position),
                    "metadata",
                    "Metadata already populated",
                    outcome="skipped",
                )
        accessor._read_metadata(
            (
                (position, row)
                for position, row in enumerate(result.itertuples(index=False))
                if force or not completed.iloc[position]
            ),
            trace=trace,
        )
        trace.update_table(result)
        return None if inplace else result

    def _read_metadata(
        self,
        rows: Iterable[tuple[int, Any]],
        *,
        trace: _SourceTraceCollector,
    ) -> None:
        """Acquire live accessors, read their metadata and publish complete rows."""
        if not self._capabilities._reader_resolver:
            return
        choices = self._get_reader_choices()
        live = self.ACCESSOR_OBJ_COLUMN in self._obj
        selected_rows = {}
        for position, row in rows:
            if not choices[position]:
                trace.record(
                    position,
                    "metadata",
                    "Outside allowed data models",
                    outcome="skipped",
                )
            elif live and getattr(row, self.ACCESSOR_OBJ_COLUMN) is None:
                trace.record(
                    position, "metadata", "No live accessor", outcome="skipped"
                )
            else:
                selected_rows[position] = row
        identifications = {}

        def live_accessors():
            for position, data in self._iter_metadata_sources(
                selected_rows.items(), trace=trace
            ):
                result = self._identify_source(
                    position,
                    selected_rows[position],
                    data,
                    choices[position],
                    trace=trace,
                )
                if result is not None:
                    identified, identifications[position] = result
                    yield position, identified

        updated = self._update_from_live_accessors(live_accessors(), trace=trace)
        self._save_identifications(
            {position: identifications[position] for position in updated}
        )
        completed = [
            position for position, has_metadata in updated.items() if has_metadata
        ]
        if completed:
            status = self._obj.get(
                self.METADATA_READ_COLUMN,
                pd.Series(data=False, index=self._obj.index),
            ).copy()
            status.iloc[completed] = True
            self._obj[self.METADATA_READ_COLUMN] = status.array

    def _update_from_live_accessors(
        self,
        readers: Iterable[tuple[int, _IdentifiedSource]],
        *,
        trace: _SourceTraceCollector,
    ) -> dict[int, bool]:
        """Project successful rows and report which supplied metadata."""
        metadata: dict[type[FileAccessor], dict[int, object]] = {}
        metadata_positions = set()
        kinds = {}
        for position, identified in readers:
            with trace.capture(position, "metadata", record_success=False):
                reader, handlers = identified
                meta = reader.meta
                handlers = cast("tuple[type[FileAccessor], ...]", handlers)
                memberships = getattr(meta, "data_models", None)
                if memberships is not None:
                    # The read result can correct affiliation without another read.
                    handlers += tuple(
                        cls
                        for cls in self._capabilities.accessor_types
                        if cls not in handlers
                        and type(reader) in cls._reader_rules.accessor_classes
                        and not cls._data_models.isdisjoint(memberships)
                    )
                classifications = {
                    cls._data_kind_column: cls._classify_metadata(reader, meta)
                    for cls in handlers
                    if cls._data_kind_type is not None
                    and (
                        memberships is None
                        or not cls._data_models.isdisjoint(memberships)
                    )
                }
                if meta is not None:
                    metadata_positions.add(position)
                    for cls in handlers:
                        metadata.setdefault(cls, {})[position] = meta
                kinds[position] = classifications
        if kinds:
            destinations = {
                name
                for cls in self._capabilities.accessor_types
                for name in cls._metadata_columns
            } | self._capabilities._data_kind_accessors.keys()
            prepared = self._prepare_metadata_columns(
                self._obj[[name for name in self._obj if name in destinations]],
                metadata,
                kinds,
                trace=trace,
            )
            self._obj[prepared.columns] = prepared
        logger.debug("Read metadata for {} table rows", len(metadata_positions))
        updated = {
            position: position in metadata_positions
            for position in kinds
            if not trace.failed(position, "metadata")
        }
        for position, has_metadata in updated.items():
            if has_metadata:
                trace.record(position, "metadata", "Read metadata and updated table")
            else:
                trace.record(
                    position,
                    "metadata",
                    "Reader supplies no metadata",
                    outcome="skipped",
                )
        return updated

    def _identify_source(
        self,
        position: int,
        row: Any,
        data: object,
        choices: dict[type[FileAccessor], type | None],
        *,
        trace: _SourceTraceCollector,
    ) -> tuple[_IdentifiedSource, _SourceIdentification] | None:
        """Identify a live source and prepare its trace and reusable selection."""
        with trace.capture(position, "identify"):
            identified = self._resolve_reader(position, row, data, choices)
            if identified is not None:
                selection = _SourceIdentification(
                    source=row.source,
                    source_loc=row.source_loc,
                    choices=cast("dict[type, type | None]", choices),
                    accessor_cls=type(identified.reader),
                    handlers=identified.handlers,
                )
                trace.record(
                    position,
                    "identify",
                    f"Reader: {type(identified.reader).__name__}",
                )
                return identified, selection
        return None

    def _resolve_reader(
        self,
        position: int,
        row: Any,
        data: object,
        choices: dict[type[FileAccessor], type | None],
    ) -> _IdentifiedSource | None:
        """Reuse a compatible selection, binding it to the current live payload."""
        cached = self._obj.get(self._IDENTIFICATION_COLUMN)
        selection = None if cached is None else cached.iloc[position]
        if (
            isinstance(selection, _SourceIdentification)
            and selection.applies_to(row, choices)
            and all(
                selection.accessor_cls in cls._reader_rules.accessor_classes
                for cls in selection.handlers
            )
        ):
            reader = (
                getattr(row, self.ACCESSOR_OBJ_COLUMN, None)
                if getattr(row, self.IO_OBJ_COLUMN, None) is data
                else None
            )
            if reader is None:
                reader = selection.accessor_cls(data)
            return _IdentifiedSource(reader, selection.handlers)
        identified = self._capabilities._reader_resolver(data, choices=choices)
        if self._capabilities._reader_resolver and identified is None:
            msg = f"No supported source recognized by {type(self).__name__}"
            raise ValueError(msg)
        return identified

    def _save_identifications(
        self, identifications: dict[int, _SourceIdentification]
    ) -> None:
        if not identifications:
            return
        previous = self._obj.get(self._IDENTIFICATION_COLUMN)
        values = [None] * len(self._obj) if previous is None else previous.tolist()
        for position, selection in identifications.items():
            values[position] = selection
        self._obj[self._IDENTIFICATION_COLUMN] = pd.Series(values, dtype=object).array

    def _prepare_metadata_columns(
        self,
        table: pd.DataFrame,
        metadata: dict[type[FileAccessor], dict[int, object]],
        kinds: dict[int, dict[str, Flag | None]],
        *,
        trace: _SourceTraceCollector,
    ) -> pd.DataFrame:
        """Prepare complete row updates, isolating failed projections by position."""
        try:
            staged = table.copy()
            projected = ModelTableAccessor(staged)
            if metadata:
                self._reset_projection_columns(
                    staged,
                    (
                        cls._metadata_columns
                        for cls in self._capabilities.accessor_types
                    ),
                    list(
                        dict.fromkeys(
                            pos for batch in metadata.values() for pos in batch
                        )
                    ),
                )
                for cls, batch in metadata.items():
                    projected.update_from_models(
                        batch, columns=cls._metadata_columns, inplace=True
                    )
            for name, cls in self._capabilities._data_kind_accessors.items():
                kind_type = cls._data_kind_type
                assert kind_type is not None
                values = {
                    position: (
                        kind
                        if isinstance(kind := row_kinds.get(name), kind_type)
                        else kind_type.__members__["unknown"]
                    )
                    for position, row_kinds in kinds.items()
                }
                projected.update_from_models(
                    values,
                    columns={name: ModelColumn("value", dtype="Int64")},
                    inplace=True,
                )
        except (OSError, ValueError, KeyError):
            if len(kinds) == 1:
                with trace.capture(next(iter(kinds)), "metadata"):
                    raise
                return table
            # Retry projection only. Successful reads and owned sources are not
            # revisited, and each row's handlers and classifications stay together.
            positions = list(kinds)
            middle = len(positions) // 2
            for part in (positions[:middle], positions[middle:]):
                selected = set(part)
                batches = {
                    cls: {pos: model for pos, model in batch.items() if pos in selected}
                    for cls, batch in metadata.items()
                    if selected.intersection(batch)
                }
                table = self._prepare_metadata_columns(
                    table,
                    batches,
                    {pos: kinds[pos] for pos in part},
                    trace=trace,
                )
            return table
        return staged

    def _iter_metadata_sources(
        self, rows: Iterable[tuple[int, Any]], *, trace: _SourceTraceCollector
    ) -> Iterator[tuple[int, Any]]:
        """Yield supported sources while keeping each owned resource open."""
        choices = None
        for position, row in rows:
            if choices is None and _borrowed_source(row) is None:
                choices = self._get_opener_choices()
            with ExitStack() as stack:
                data = self._acquire_source(
                    position,
                    row,
                    None,
                    owned={},
                    stack=stack,
                    choices={} if choices is None else choices[position],
                    trace=trace,
                )
                if data is not None:
                    yield position, data

    def _get_opener_choices(
        self,
    ) -> list[dict[type[FileAccessor], Callable[..., Any] | None]]:
        """Per-data-model opener choices from current table values."""
        table = self._obj
        names = self._get_rule_column_names()
        ext_name = names.get("ext", "ext")
        extensions = table.get(
            ext_name, pd.Series(None, index=table.index, dtype=object)
        )
        # A declared current extension, including null, is authoritative.
        missing = extensions.isna().to_numpy().nonzero()[0]
        if "ext" not in table and len(missing):
            extensions = extensions.copy()
            extensions.iloc[missing] = [
                fileloc(location).path.suffix.removeprefix(".")
                for location in table.source_loc.iloc[missing]
            ]
            table = table.assign(**{ext_name: extensions})
        return self._scope_rule_choices(
            self._capabilities._opener_resolver.match_table(table, column_names=names)
        )

    def _get_reader_choices(
        self,
    ) -> list[dict[type[FileAccessor], type | None]]:
        """Prepare per-row rules within current data-model memberships."""
        matched = self._capabilities._reader_resolver.match_table(
            self._obj, column_names=self._get_rule_column_names()
        )
        return self._scope_rule_choices(matched)

    def _scope_rule_choices[ChoiceT](
        self, matched: Mapping[type[FileAccessor], list[ChoiceT]]
    ) -> list[dict[type[FileAccessor], ChoiceT]]:
        """Restrict choices to current data-model memberships."""
        return [
            {
                cls: matched[cls][position]
                for cls in matched
                if cls is FileAccessor
                or data_models is None
                or not cls._data_models.isdisjoint(data_models)
            }
            for position, data_models in enumerate(self._iter_row_memberships())
        ]

    @staticmethod
    def _classify_metadata(
        reader: SourceReader, metadata: object | None
    ) -> Flag | None:
        """Classify a successful read in this data model, if applicable."""
        return None

    @contextmanager
    def open(
        self,
        *,
        io_objs: Iterable[object | None] | None = None,
        update_from_metadata: bool = False,
        force: bool = False,
        on_error: SourceErrorPolicyT = "record",
    ) -> Iterator[pd.DataFrame]:
        """Open and identify sources for use within a resource scope.

        Parameters
        ----------
        io_objs : iterable of object or None, optional
            Open objects supported by this accessor, one per table row.
            None entries use the corresponding source. If omitted, use the
            table's ``io_obj`` column when present, otherwise open its sources.
            Supplied objects are borrowed and are not closed by this scope.
        update_from_metadata : bool, optional
            Read metadata from identified accessors and update the yielded table
            before entering the scope. Default is False, stopping after
            identification. Rows with ``metadata_read=True`` are skipped unless
            ``force=True``.
        force : bool, optional
            Reread previously populated metadata when ``update_from_metadata=True``.
            Default is False. Does not request a metadata update on its own.
        on_error : {"raise", "record"}, optional
            Default is "record", retaining failed rows and storing their
            events in ``source_trace``. Failed opens have ``io_obj=None``;
            unrecognized payloads retain ``io_obj`` but have ``accessor_obj=None``.
            Use "raise" to stop on an opening, identification, or requested
            metadata update failure.

        Yields
        ------
        pandas.DataFrame
            A new table with ``io_obj`` containing opened resources and
            ``accessor_obj`` containing identified data accessors. Current
            metadata columns are unchanged unless ``update_from_metadata=True``.
            The input table is unchanged. Known foreign
            rows are skipped, retaining None in both temporary columns.

        Raises
        ------
        ValueError
            If ``io_objs`` has the wrong length. With ``on_error="raise"``, also
            raised when no opener matches, a built-in opener receives a remote
            source, or the readers do not recognize a payload, including a
            supplied open object.
        RuntimeError
            If opener choices or matching readers are ambiguous, regardless of on_error.

        Notes
        -----
        Each local path and selected opener pair is opened once per scope.
        Files opened here are closed on exit, including on errors. Load
        file-backed arrays needed afterward while the scope is open. Selecting
        rows does not extend resource lifetimes. The yielded table retains
        metadata after exit, but its temporary ``io_obj`` and ``accessor_obj``
        columns are removed. Call `update_from_source_metadata` to read
        and project metadata explicitly. Reader selections are retained for
        reopening; live accessors are not retained after this scope closes.
        """
        trace = _SourceTraceCollector(on_error)
        if io_objs is None:
            io_objs = self._obj.get(self.IO_OBJ_COLUMN)
        objects = [None] * len(self._obj) if io_objs is None else list(io_objs)
        if len(objects) != len(self._obj):
            msg = "Provide one IO object (or None) per table row"
            raise ValueError(msg)
        opener_choices = self._get_opener_choices()
        choices = self._get_reader_choices()
        self._record_scope(trace)
        with ExitStack() as stack:
            owned = {}
            accessors = [None] * len(self._obj)
            identifications = {}
            for i, row in enumerate(
                cast("Iterable[Any]", self._obj.itertuples(index=False))
            ):
                obj = objects[i]
                objects[i] = None
                if self._capabilities._reader_resolver and not choices[i]:
                    trace.record(
                        i, "open", "Outside allowed data models", outcome="skipped"
                    )
                    continue
                obj = self._acquire_source(
                    i,
                    row,
                    obj,
                    owned=owned,
                    stack=stack,
                    choices=opener_choices[i],
                    trace=trace,
                )
                objects[i] = obj
                if objects[i] is not None:
                    result = self._identify_source(i, row, obj, choices[i], trace=trace)
                    if result is not None:
                        identified, identifications[i] = result
                        accessors[i] = identified.reader
            result = self._obj.copy()
            result[self.IO_OBJ_COLUMN] = pd.Series(objects, dtype=object).array
            result[self.ACCESSOR_OBJ_COLUMN] = pd.Series(accessors, dtype=object).array
            try:
                accessor = self._with_table(result)
                accessor._save_identifications(identifications)
                trace.update_table(result)
                if update_from_metadata:
                    accessor.update_from_source_metadata(
                        inplace=True, force=force, on_error=on_error
                    )
                yield result
            finally:
                for name in (self.IO_OBJ_COLUMN, self.ACCESSOR_OBJ_COLUMN):
                    if name in result:
                        del result[name]
                logger.debug("Closing table scope: {} owned resources", len(owned))

    def _acquire_source(  # noqa: PLR0913 -- caller supplies the row trace and resource scope.
        self,
        position: int,
        row: Any,
        supplied: object | None,
        *,
        owned: dict,
        stack: ExitStack,
        choices: Mapping[type[FileAccessor], Callable[..., Any] | None],
        trace: _SourceTraceCollector,
    ) -> object | None:
        """Borrow a supplied payload or acquire a scope-owned format resource."""
        with trace.capture(position, "open"):
            borrowed = _borrowed_source(row) if supplied is None else supplied
            if borrowed is not None:
                trace.record(position, "open", "Reused live resource")
                return borrowed
            resolved = self._capabilities._opener_resolver(choices)
            opener = None if resolved is None else resolved[0]
            location = fileloc(row.source)
            key = (
                location.path.resolve() if location.is_local() else location.url,
                id(opener),
            )
            if key not in owned:
                logger.debug("Opening table resource {}", location)
                owned[key] = stack.enter_context(
                    self._open_source(
                        row.source, source_loc=row.source_loc, opener=opener
                    )
                )
            return owned[key]
        return None

    @staticmethod
    @contextmanager
    def _open_source(
        source: str | Path | FileLoc,
        *,
        source_loc: FileLoc,
        opener: Callable[..., Any] | None,
    ) -> Iterator[Any]:
        """Pass the source to its opener and manage the returned resource."""
        if opener is None:
            msg = f"No source opener matches {source_loc}"
            raise ValueError(msg)
        resource = opener(source)
        with ExitStack() as stack:
            if hasattr(resource, "__enter__"):
                resource = stack.enter_context(resource)
            elif callable(close := getattr(resource, "close", None)):
                stack.callback(close)
            yield resource

    @property
    def source(self) -> pd.Series:
        """Resource objects in row order."""
        return self._obj["source"]

    @property
    def source_loc(self) -> pd.Series:
        """Filename identification locations in row order."""
        return self._obj["source_loc"]

    @property
    def filepath_info(self) -> pd.Series:
        """Parse results for source_loc in row order; None for unsuccessful parses."""
        return self._obj[self.FILEPATH_INFO_COLUMN]


class FileAccessorView(BaseModel):
    """File-table operations restricted to selected accessor classes.

    Parameters
    ----------
    accessor_types : tuple of FileAccessor subclasses
        Concrete classes, in application order. Duplicate classes are kept
        once. An empty tuple enables filename information only.

    Notes
    -----
    Obtain a view through ``FileAccessor.partial(accessor_types=...)``. Filename
    parsing still uses all available registrations. The view does not remove
    rows or existing columns, and later registrations do not add classes to it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    accessor_types: _AccessorTypes[FileAccessor]

    def partial(
        self, *, accessor_types: tuple[type[FileAccessor], ...]
    ) -> FileAccessorView:
        """Narrow the enabled accessor classes without changing this view.

        Parameters
        ----------
        accessor_types : tuple of FileAccessor subclasses
            Classes to retain. Classes absent from this view are not enabled.

        Returns
        -------
        FileAccessorView
            Restricted view, preserving the existing class order.
        """
        selected = type(self)(accessor_types=accessor_types)
        return type(self)(
            accessor_types=tuple(
                cls for cls in self.accessor_types if cls in selected.accessor_types
            )
        )

    def __call__(self, table: pd.DataFrame) -> FileAccessor:
        """Access a file table with these accessor classes.

        Parameters
        ----------
        table : pandas.DataFrame
            File table containing source and source_loc columns.

        Returns
        -------
        FileAccessor
            Accessor bound to the supplied table without copying or enriching it.
        """
        return FileAccessor(table, accessor_types=self.accessor_types)

    def from_sources(
        self,
        sources: Iterable[Any],
        *,
        mode: FileTableModeT = "guess",
        on_error: SourceErrorPolicyT = "record",
    ) -> pd.DataFrame:
        """Build a file table using these accessor classes.

        Parameters
        ----------
        sources : iterable
            Resources or source/source_loc dictionaries accepted by
            `FileAccessor.from_sources`.
        mode : {"guess", "guess_or_read", "read"}, optional
            Filename guessing and metadata-read behavior, default "guess".
            Follows `FileAccessor.from_sources` without enabling additional classes.
        on_error : {"raise", "record"}, optional
            Source-information failure policy, default "record".

        Returns
        -------
        pandas.DataFrame
            File information in source order. Owned resources are closed before
            returning; caller-supplied objects remain open. Bind the result with
            this view to continue using the same accessor classes.
        """
        return FileAccessor.from_sources(
            sources, accessor_types=self.accessor_types, mode=mode, on_error=on_error
        )
