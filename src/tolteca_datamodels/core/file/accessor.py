"""Source validation and filename projection for file tables."""

from __future__ import annotations

from collections import Counter
from functools import cache, cached_property
from typing import TYPE_CHECKING, ClassVar, Self, cast

import pandas as pd
from pydantic import TypeAdapter
from tollan.accessor.pandas.model_table import (
    ExplicitColumnRule,
    ModelColumn,
    ModelColumnMapper,
    ModelTableAccessor,
    ScalarColumnRule,
)
from tollan.utils.log import logger
from tollan.utils.typing import (
    GenericBindingRequirement,
    resolve_concrete_generic_bindings,
)

from ..filepath import FilepathInfo, MetadataField
from .source import _SourceLocation
from .trace import (
    SourceErrorPolicyT,
    SourceTrace,
    SourceTraceKind,
    _SourceTraceCollector,
)
from .workflow import (
    FileAccessorWorkflow,
    FileTableModeT,
    _validate_file_table_mode,
    _validate_source_error_policy,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from contextlib import AbstractContextManager

    from ..obs_spec import ObsSpecBase

__all__ = ["FileAccessorBase"]

_validate_sources_argument = TypeAdapter(list[object]).validate_python


class _FileWorkflowProxy[WorkflowT: FileAccessorWorkflow]:
    """Delegate file operations to the owning accessor or view's workflow."""

    _workflow: WorkflowT

    def open(
        self,
        *,
        on_error: SourceErrorPolicyT = "record",
    ) -> AbstractContextManager[pd.DataFrame]:
        """Open and identify matching and unidentified sources within a context.

        Parameters
        ----------
        on_error : {"record", "raise"}, optional
            Record opening and reader-creation failures in
            ``source_trace`` by default, or propagate the first failure with
            "raise".

        Yields
        ------
        pandas.DataFrame
            Shared table, updated in place with ``io_obj`` holding each
            opened resource and ``accessor_obj`` holding its data accessor.
            Skipped rows hold None. If opening succeeds but reader creation
            fails, ``io_obj`` remains available and ``accessor_obj`` is None.
            On exit, columns from an enclosing open scope are restored;
            otherwise both columns are removed.

        Raises
        ------
        ValueError, TypeError, KeyError, OSError
            Resource opening or reader creation fails with
            ``on_error="raise"``.

        Notes
        -----
        Matching and unidentified rows are opened; recognized foreign rows are
        skipped. Current values select
        the opener (``ext``, or ``file_ext`` when ``ext`` is absent) and reader
        (for example, ``data_models``, ``interface`` and ``interface_kind``).
        Rows whose values select no reader are identified from the opened
        contents. If no opener matches, the row records ``opener_rule_unmatched``
        without raising. Failure to obtain a live reader records
        ``reader_creation_failed`` and follows ``on_error``. This includes no
        reader being found, content inspection errors and construction errors.
        A source that is itself an open object is borrowed. Each local file is
        opened once per opener; resources opened here are closed on exit, including on
        exceptions. Load data needed afterward while the scope is open. Use
        `read_metadata` to read source metadata into the table.

        Examples
        --------
        With the reference observation under ``data_lmt`` in the working directory:

        >>> from tolteca_datamodels.toltec.file import ToltecFileAccessor
        >>> file_accessor = ToltecFileAccessor.from_sources([
        ...     "data_lmt/tel/tel_toltec_2025-10-31_145609_00_0002.nc",
        ...     "data_lmt/toltec/tcs/toltec0/"
        ...     "toltec0_145609_000_0002_2025_10_31_09_27_57.nc",
        ... ])
        >>> with file_accessor.open() as table:
        ...     [type(accessor).__name__ for accessor in table.accessor_obj]
        ['LmtTelAccessor', 'ToltecKidsAccessor']
        >>> "io_obj" in file_accessor.table
        False
        """
        return self._workflow.open(on_error=on_error)

    def read_metadata(self, *, on_error: SourceErrorPolicyT = "record") -> Self:
        """Read source metadata and update this accessor's table in place.

        Parameters
        ----------
        on_error : {"record", "raise"}, optional
            Record source opening, identification and metadata-reading failures
            by default, or propagate the first failure with "raise".

        Returns
        -------
        Self
            This accessor or view. Successful reads replace current metadata columns.
            Filename snapshots, unrelated
            columns and failed rows' values are retained. Readers without
            metadata may supply additional columns and receive a
            ``metadata_unavailable`` trace.

        Raises
        ------
        TypeError, ValueError, AttributeError, OverflowError
            Metadata projection or reader-column preparation fails. These
            errors always propagate and leave destination columns unchanged.
            Source errors propagate only with ``on_error="raise"``.

        Notes
        -----
        Each call reads matching and unidentified rows, including rows populated
        by an earlier call or by the factory. Inside `open`, it uses live
        ``accessor_obj`` values without
        opening or identifying again. Otherwise it opens the required sources
        and closes owned resources afterward. Metadata is obtained through each
        reader's ``meta`` property. Metadata updates do not run filename
        derivation again.

        Examples
        --------
        >>> from tolteca_datamodels.lmt.file import LmtFileAccessor
        >>> file_accessor = LmtFileAccessor.from_sources([
        ...     "data_lmt/tel/tel_toltec_2025-10-31_145609_00_0002.nc",
        ... ])
        >>> _ = file_accessor.read_metadata()
        >>> file_accessor.query("obsnum == 145609").interface.tolist()
        ['tel']
        """
        self._workflow.read_metadata(on_error=on_error)
        return self

    def query_obs_spec(self, spec: ObsSpecBase | str | int) -> pd.DataFrame:
        """Select rows with an observation specification.

        Parameters
        ----------
        spec : ObsSpecBase, str or int
            Observation selector. Text and integers require a common selector
            class among the selected policies. A parsed selector must be
            supported by the workflow. Value constraints apply before
            one positional selection across all rows.

        Returns
        -------
        pandas.DataFrame
            Matching rows, retaining their index and order. The selection
            applies to all rows, not only rows matched by the selected policies.
            Numeric interface constraints use their combined label numbering.

        Raises
        ------
        ValueError
            Invalid selector text, no common selector class, conflicting label
            numbering, or the table lacks a constrained field.
        TypeError
            A parsed selector of another class.

        Examples
        --------
        >>> from tolteca_datamodels.toltec.file import ToltecFileAccessor
        >>> file_accessor = ToltecFileAccessor.from_sources([
        ...     "toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc",
        ...     "toltec7_145606_000_0001_2025_10_31_09_22_14_targsweep.nc",
        ...     "tel_toltec_2025-10-31_145606_00_0001.nc",
        ... ])
        >>> file_accessor.query_obs_spec("145606/7").interface.tolist()
        ['toltec7']
        """
        return self._workflow.query_obs_spec(spec)


class FileAccessorBase[WorkflowT: FileAccessorWorkflow = FileAccessorWorkflow](
    _FileWorkflowProxy[WorkflowT], ModelTableAccessor
):
    """Build and query a file table using a configured workflow.

    Parameters
    ----------
    data : pandas.DataFrame
        Table to access without copying or modifying it. No columns are required
        at construction. Use `from_sources` to create a populated accessor.

    Notes
    -----
    Define extensions as ``FileAccessorBase[YourWorkflow]``. Omitting the workflow
    uses `FileAccessorWorkflow`, which has no matching rules and retains only
    filename-derived information.
    """

    _workflow_cls: ClassVar[type[FileAccessorWorkflow]] = FileAccessorWorkflow

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        bindings = resolve_concrete_generic_bindings(
            cls,
            FileAccessorBase,
            WorkflowT=GenericBindingRequirement("_workflow_cls", FileAccessorWorkflow),
        )
        if bindings is not None:
            cls._workflow_cls = bindings["WorkflowT"]

    @cached_property
    def _workflow(self) -> WorkflowT:
        return cast("type[WorkflowT]", self._workflow_cls)(self)

    _SOURCE_COLUMN: ClassVar[str] = "source"
    _SOURCE_LOC_COLUMN: ClassVar[str] = "source_loc"
    _FILEPATH_INFO_COLUMN: ClassVar[str] = "filepath_info"
    _DATA_MODELS_COLUMN: ClassVar[str] = "data_models"
    _SOURCE_TRACE_COLUMN: ClassVar[str] = "source_trace"
    _IO_OBJ_COLUMN: ClassVar[str] = "io_obj"
    _ACCESSOR_OBJ_COLUMN: ClassVar[str] = "accessor_obj"

    _source_column_mapper: ClassVar[ModelColumnMapper] = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[
            ExplicitColumnRule(
                columns={
                    _SOURCE_COLUMN: ModelColumn(_SOURCE_COLUMN, dtype=object),
                    _SOURCE_LOC_COLUMN: ModelColumn(_SOURCE_LOC_COLUMN, dtype=object),
                }
            ),
            ScalarColumnRule(),
        ],
        include_computed=True,
    )
    _source_columns: ClassVar[dict[str, ModelColumn]] = _source_column_mapper(
        _SourceLocation
    )

    _filepath_column_mapper: ClassVar[ModelColumnMapper] = ModelColumnMapper(
        name_rules=[(r".+", r"file_\g<0>")],
        type_rules=[
            ExplicitColumnRule(
                columns={
                    _DATA_MODELS_COLUMN: ModelColumn(_DATA_MODELS_COLUMN, dtype=object),
                }
            ),
            ScalarColumnRule(),
        ],
        default=None,
    )

    @classmethod
    @cache
    def _filepath_columns(cls, model_cls: type[FilepathInfo]) -> dict[str, ModelColumn]:
        """Prepare columns once per accessor and filename model type."""
        return cls._filepath_column_mapper(model_cls)

    @classmethod
    @cache
    def _filepath_metadata_aliases(
        cls, model_cls: type[FilepathInfo]
    ) -> dict[str, str]:
        """Names initialized from snapshots of marked metadata fields."""
        return {
            name.removeprefix("file_"): name
            for name, column in cls._filepath_columns(model_cls).items()
            if MetadataField in model_cls.model_fields[column.attr].metadata
        }

    @property
    def table(self) -> pd.DataFrame:
        """Underlying DataFrame, without copying."""
        return self._obj

    @property
    def source(self) -> pd.Series:
        """Original resource objects."""
        return self._obj[self._SOURCE_COLUMN]

    @property
    def source_loc(self) -> pd.Series:
        """Validated locations used for filename parsing."""
        return self._obj[self._SOURCE_LOC_COLUMN]

    @property
    def filepath_info(self) -> pd.Series:
        """Parsed filename information; raises KeyError if the column is absent."""
        return self._obj[self._FILEPATH_INFO_COLUMN]

    @property
    def data_models(self) -> pd.Series:
        """Current data-model memberships; raises KeyError if the column is absent."""
        return self._obj[self._DATA_MODELS_COLUMN]

    @property
    def source_trace(self) -> pd.Series:
        """Per-row source traces; raises KeyError if the column is absent."""
        return self._obj[self._SOURCE_TRACE_COLUMN]

    def _make_source_trace_collector(
        self, on_error: SourceErrorPolicyT
    ) -> _SourceTraceCollector:
        """Create a collector over table traces, initializing the column if absent."""
        if self._SOURCE_TRACE_COLUMN not in self._obj:
            self._obj[self._SOURCE_TRACE_COLUMN] = pd.array(
                [SourceTrace() for _ in range(len(self._obj))], dtype=object
            )
        return _SourceTraceCollector(self.source_trace, on_error=on_error)

    @property
    def io_obj(self) -> pd.Series:
        """Opened resources, available within an `open` context."""
        return self.table[self._IO_OBJ_COLUMN]

    @property
    def accessor_obj(self) -> pd.Series:
        """Live data accessors, available within an `open` context."""
        return self.table[self._ACCESSOR_OBJ_COLUMN]

    def _get_query_namespace(self) -> dict[str, object]:
        return super()._get_query_namespace() | self._workflow._get_query_namespace()

    @classmethod
    def from_sources(
        cls,
        sources: Iterable[object],
        *,
        mode: FileTableModeT = "guess",
        on_error: SourceErrorPolicyT = "record",
    ) -> Self:
        """Create a file table and populate it using the configured workflow.

        Parameters
        ----------
        sources : finite iterable
            Resources or dicts with ``source`` and optional ``source_loc``.
            Wrap a single entry in a list.
            An omitted or None ``source_loc`` uses ``source`` and must validate
            as FileLoc. Supply it explicitly for an already-open resource or a
            different original filename. Locations need not exist; resource
            objects are retained unchanged.
        mode : {"guess", "guess_or_read", "read"}, optional
            Default "guess" prepares filename information and derived columns.
            "guess_or_read" reads unidentified sources; "read" also reads sources
            matched by the workflow. Recognized foreign sources remain unopened.
        on_error : {"record", "raise"}, optional
            Record per-entry validation and filename failures in ``source_trace``
            by default, or propagate them with "raise". Invalid outer collections,
            ambiguous registrations and projection failures always raise.

        Returns
        -------
        Self
            Accessor with one row per input in iteration order, including
            duplicates and invalid entries. Invalid source/location pairs have
            missing information and a failure trace. Filename failures retain
            the valid source and location.

            ``source`` and ``source_loc`` retain the resource and validated
            location. ``source_loc_is_local``, ``source_loc_host`` and
            ``source_loc_scheme`` describe that location. ``filepath_info``
            holds the parsed model, or None when unavailable. Supported filename
            fields provide ``file_*`` snapshots; fields marked with MetadataField
            also initialize unprefixed columns. Unfamiliar names provide an
            extension and empty ``data_models``. ``source_trace`` contains each
            row's recorded issues.

        Raises
        ------
        pydantic.ValidationError
            Invalid collection or error policy, or invalid entries in raise mode.
        RuntimeError
            Filename registrations match ambiguously.
        TypeError, ValueError, OverflowError
            A projected value or shared column has an incompatible dtype, or a
            filename converter fails in raise mode.

        Examples
        --------
        >>> file_accessor = FileAccessorBase.from_sources(["example.nc"])
        >>> file_accessor.table.file_ext.tolist()
        ['nc']

        >>> file_accessor = FileAccessorBase.from_sources([
        ...     {"source": "cached.nc", "source_loc": "archive:/observation.nc"},
        ...     "local.nc",
        ... ])
        >>> file_accessor.source_loc.iloc[0].as_rsync_arg()
        'archive:/observation.nc'
        >>> file_accessor.query("source_loc_is_local").source.tolist()
        ['local.nc']
        """
        mode = _validate_file_table_mode(mode)
        on_error = _validate_source_error_policy(on_error)
        sources = _validate_sources_argument(sources)
        file_accessor = cls(pd.DataFrame(index=pd.RangeIndex(len(sources))))
        collector = file_accessor._make_source_trace_collector(on_error)
        models: list[_SourceLocation | None] = []
        for position, source in enumerate(sources):
            model = None
            with collector.capture(position, SourceTraceKind.source_validation_failed):
                model = _SourceLocation.model_validate(source)
            models.append(model)
        file_accessor.update_from_models(
            models, columns=cls._source_columns, inplace=True
        )
        file_accessor._parse_filepath_info(on_error=on_error)
        file_accessor._populate_from_filepath_info()
        file_accessor._workflow.prepare(mode=mode, on_error=on_error)
        file_accessor._log_traces()
        return file_accessor

    def _parse_filepath_info(self, *, on_error: SourceErrorPolicyT) -> None:
        """Parse validated source locations and retain the results."""
        on_error = _validate_source_error_policy(on_error)
        locations = self.source_loc
        collector = self._make_source_trace_collector(on_error)
        parsed = []
        for position, (valid, location) in enumerate(
            zip(locations.notna(), locations, strict=True)
        ):
            info = None
            if valid:
                with collector.capture(position, SourceTraceKind.filepath_parse_failed):
                    info = FilepathInfo.from_source(location, raise_on_error=False)
                    if info is None:
                        collector.record(
                            position, SourceTraceKind.filepath_rule_unmatched
                        )
            parsed.append(info)
        self.table[self._FILEPATH_INFO_COLUMN] = pd.array(parsed, dtype=object)

    def _populate_from_filepath_info(self) -> None:
        """Project retained parses and initialize absent marked metadata columns."""
        filepath_infos = self.table[self._FILEPATH_INFO_COLUMN]
        model_types = filepath_infos.map(type)
        columns = {
            cls: self._filepath_columns(cls)
            for cls in model_types.unique()
            if cls is not type(None)
        }
        self.update_from_models(filepath_infos, columns=columns, inplace=True)
        aliases = {
            name: snapshot
            for cls in columns
            for name, snapshot in self._filepath_metadata_aliases(cls).items()
        }
        for name, snapshot in aliases.items():
            if name in self.table:
                continue
            self.table[name] = self.table[snapshot]

    def source_trace_summary(self) -> pd.DataFrame:
        """Count rows containing each recorded issue kind.

        Returns
        -------
        pandas.DataFrame
            Columns ``kind``, ``count`` and ``total``. Each current row counts at
            most once per issue kind; total is the table's row count. Empty when
            there are no issues or no trace column. Inspect ``source_trace`` for
            individual entries.
        """
        counts = Counter()
        if self._SOURCE_TRACE_COLUMN in self._obj:
            for trace in self.source_trace:
                counts.update(trace.kinds)
        return pd.DataFrame(
            [
                (kind.value, count, len(self._obj))
                for kind, count in sorted(counts.items())
            ],
            columns=["kind", "count", "total"],
        )

    def _log_traces(self) -> None:
        """Report failures and counts with source locations from the file table."""
        for position, trace in enumerate(self.source_trace):
            if not trace.has_error:
                continue
            for entry in trace.iter_error_entries():
                logger.debug(
                    "Source {} (row {}): {}: {} ({})",
                    getattr(self.source_loc.iloc[position], "url", None),
                    position,
                    entry.kind,
                    entry.message,
                    entry.context,
                )
        logger.opt(lazy=True).debug(
            "Source processing: {}/{} filenames parsed.\nSource processing trace:\n{}",
            lambda: self._obj[self._FILEPATH_INFO_COLUMN].notna().sum(),
            lambda: len(self._obj),
            lambda: self.source_trace_summary().to_string(index=False),
        )
