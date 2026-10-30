"""Source validation and filename projection for file tables."""

from __future__ import annotations

from collections import Counter
from functools import cache
from typing import TYPE_CHECKING, ClassVar, Self

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

from ..filepath import FilepathInfo, MetadataField
from .source import _SourceLocation
from .trace import (
    SourceErrorPolicyT,
    SourceTrace,
    SourceTraceKind,
    _SourceTraceCollector,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["FileAccessorBase"]

_validate_sources_argument = TypeAdapter(list[object]).validate_python
_validate_source_error_policy = TypeAdapter(SourceErrorPolicyT).validate_python


class FileAccessorBase(ModelTableAccessor):
    """Build and query a table of file information without opening resources.

    Parameters
    ----------
    data : pandas.DataFrame
        Table to access without copying or modifying it. No columns are required
        at construction. Use `from_sources` to create a populated accessor.
    """

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

    @classmethod
    def from_sources(
        cls, sources: Iterable[object], *, on_error: SourceErrorPolicyT = "record"
    ) -> Self:
        """Create a file table from resource identifiers without opening them.

        Parameters
        ----------
        sources : finite iterable
            Resources or dicts with ``source`` and optional ``source_loc``.
            Wrap a single entry in a list.
            An omitted or None ``source_loc`` uses ``source`` and must validate
            as FileLoc. Supply it explicitly for an already-open resource or a
            different original filename. Locations need not exist; resource
            objects are retained unchanged.
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
        return file_accessor._guess_from_sources(on_error=on_error)

    def _guess_from_sources(self, *, on_error: SourceErrorPolicyT = "record") -> Self:
        """Populate filename information from prepared source locations."""
        self._parse_filepath_info(on_error=on_error)
        self._populate_from_filepath_info()
        self._log_traces()
        return self

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
                counts.update(trace._kinds)
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
