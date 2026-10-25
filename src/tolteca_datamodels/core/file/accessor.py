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

from ..filepath import FilepathInfo
from .source import SourceSourceLoc, _SourceLocation
from .trace import (
    SourceErrorPolicyT,
    SourceTrace,
    SourceTraceKind,
    _SourceTraceCollector,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["FileAccessorBase"]


class FileAccessorBase(ModelTableAccessor):
    """Build and query a table of file information without opening resources.

    Parameters
    ----------
    data : pandas.DataFrame
        Table to access, including an empty DataFrame. Binding does not copy
        or modify it and requires no columns. Use `guess_from_sources` to
        populate it, or `from_sources` to create a populated accessor directly.
    """

    _SOURCE_COLUMN: ClassVar[str] = "source"
    _SOURCE_LOC_COLUMN: ClassVar[str] = "source_loc"
    _FILEPATH_INFO_COLUMN: ClassVar[str] = "filepath_info"
    _DATA_MODELS_COLUMN: ClassVar[str] = "data_models"
    _SOURCE_TRACE_COLUMN: ClassVar[str] = "source_trace"

    _sources_adapter: ClassVar = TypeAdapter(list[object])
    _error_policy_adapter: ClassVar = TypeAdapter(SourceErrorPolicyT)
    _source_columns: ClassVar[dict[str, ModelColumn]] = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[
            ExplicitColumnRule(
                columns={
                    "source": ModelColumn("source", dtype=object),
                    "source_loc": ModelColumn("source_loc", dtype=object),
                }
            ),
            ScalarColumnRule(),
        ],
        include_computed=True,
    )(_SourceLocation)

    _filepath_column_mapper: ClassVar[ModelColumnMapper] = ModelColumnMapper(
        name_rules=[(r".+", r"file_\g<0>"), (r"data_models", _DATA_MODELS_COLUMN)],
        type_rules=[
            ExplicitColumnRule(
                columns={
                    "data_models": ModelColumn(
                        "data_models", dtype=object, converter=list
                    ),
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

    @property
    def table(self) -> pd.DataFrame:
        """Underlying DataFrame, without copying."""
        return self._obj

    @property
    def source(self) -> pd.Series:
        """Original-resource column, available after guessing."""
        return self._obj[self._SOURCE_COLUMN]

    @property
    def source_loc(self) -> pd.Series:
        """Validated-location column, available after guessing."""
        return self._obj[self._SOURCE_LOC_COLUMN]

    @property
    def source_trace(self) -> pd.Series:
        """Trace column with the current index; raises KeyError if absent."""
        return self._obj[self._SOURCE_TRACE_COLUMN]

    def _get_or_create_source_collector(
        self, on_error: SourceErrorPolicyT
    ) -> _SourceTraceCollector:
        """Bind a collector to current rows, creating missing trace storage."""
        if self._SOURCE_TRACE_COLUMN not in self._obj:
            self._obj[self._SOURCE_TRACE_COLUMN] = pd.array(
                [SourceTrace() for _ in range(len(self._obj))], dtype=object
            )
        return _SourceTraceCollector(self.source_trace, on_error=on_error)

    @classmethod
    def from_sources(
        cls, sources: Iterable[object], *, on_error: SourceErrorPolicyT = "record"
    ) -> Self:
        """Create an accessor with source locations and filename information.

        Parameters
        ----------
        sources : finite iterable
            Resources, prepared SourceSourceLoc objects, or dicts containing
            ``source`` and ``source_loc``. See `guess_from_sources` for accepted
            forms and projected columns. No resources are opened.
        on_error : {"record", "raise"}, optional
            Default is "record": retain invalid entries with a failure trace.
            Use "raise" to propagate source-location and filename parsing
            failures. Invalid outer collections, ambiguous registrations and
            projection failures always raise.

        Returns
        -------
        Self
            New accessor whose `table` has one row per input in iteration
            order, including duplicates and invalid entries.

        See Also
        --------
        guess_from_sources : Populate an existing accessor in place.

        Examples
        --------
        >>> files = FileAccessorBase.from_sources(["example.nc"])
        >>> files.table.file_ext.tolist()
        ['nc']
        """
        return cls(pd.DataFrame()).guess_from_sources(sources, on_error=on_error)

    def guess_from_sources(
        self, sources: Iterable[object], *, on_error: SourceErrorPolicyT = "record"
    ) -> Self:
        """Populate source locations and filename information in place.

        Parameters
        ----------
        sources : finite iterable
            Collection of resources, prepared SourceSourceLoc objects, or dicts
            containing ``source`` and ``source_loc``. Lists, tuples and generators
            are accepted and consumed in iteration order. Wrap a single resource
            or source/location dict in a list; strings, mappings and non-iterables
            are not accepted as the outer collection.
            An omitted or None ``source_loc`` uses ``source`` and must validate
            as FileLoc. Supply an explicit location for an already-open object.
            The location need not exist. Only ``source_loc`` is used for parsing;
            the original ``source`` object is retained unchanged.
            A table with no rows or columns acquires one row per input.
            Otherwise the input length must match the table's row count;
            entries map to rows by position, preserving the index.
        on_error : {"record", "raise"}, optional
            Default is "record": retain rows with invalid source-location pairs
            or filename parsing failures, recording the failure in ``source_trace``.
            Use "raise" to propagate those failures. Unrecognized filenames are
            recorded in either mode. Invalid collections or policies, ambiguous
            registrations and column projection failures always raise.
            Source-location validation precedes filename parsing, so in raise
            mode a source-location error is reported before any filename error.

        Returns
        -------
        Self
            This accessor, with projected columns updated in place. Other
            columns and existing trace objects are retained. A missing
            ``source_trace`` column is initialized with an empty trace per row.
            Invalid source-location pairs have missing information columns
            and a failure trace. A filename
            parsing failure retains the validated source and location.
            ``source`` and ``source_loc`` have object
            dtype; ``source_loc_is_local``, ``source_loc_host`` and
            ``source_loc_scheme`` describe the location. ``filepath_info`` holds
            the parsed FilepathInfo object, or None when parsing fails or the
            location has no filename. Unfamiliar filenames provide ``file_ext``
            with an empty ``data_models`` list.
            Scalar filename fields have ``file_*`` snapshot columns. Entry-point
            specializations may add unprefixed current-value columns; values are
            missing on rows whose models do not provide them. ``source_trace`` holds
            a SourceTrace object for each row. Use ``iter_entries()`` to inspect
            recorded issues or ``iter_error_entries()`` to inspect failures only;
            ``has_error`` identifies traces containing a failure. Successful
            guessing adds no issues; parsed information is in ``filepath_info``.
            Empty input retains the source-location
            columns, ``filepath_info`` and ``source_trace``.

        Raises
        ------
        pydantic.ValidationError
            If the error policy or outer collection is invalid, iteration fails,
            or a source-location pair or parsed filename fails validation with
            ``on_error="raise"``.
        ValueError, KeyError
            If the input length does not match the existing rows, or a
            registered parser rejects a value with ``on_error="raise"``.
        RuntimeError
            If matching filename registrations are ambiguous.
        TypeError, ValueError, OverflowError
            If projected fields require incompatible dtypes for a shared column,
            or a parsed value cannot be projected into its declared column dtype.

        Examples
        --------
        >>> import pandas as pd
        >>> files = FileAccessorBase(pd.DataFrame())
        >>> _ = files.guess_from_sources([
        ...     {"source": "cached.nc", "source_loc": "archive:/observation.nc"},
        ...     "local.nc",
        ... ])
        >>> files.source_loc.iloc[0].as_rsync_arg()
        'archive:/observation.nc'
        >>> files.query("source_loc_is_local").source.tolist()
        ['local.nc']
        """
        on_error = self._error_policy_adapter.validate_python(on_error)
        sources = self._sources_adapter.validate_python(sources)
        if len(self._obj.index) == 0 and len(self._obj.columns) == 0:
            self._obj.index = pd.RangeIndex(len(sources))
        elif len(sources) != len(self._obj):
            msg = "The number of sources must match the table's row count."
            raise ValueError(msg)
        collector = self._get_or_create_source_collector(on_error)

        models: list[_SourceLocation | None] = []
        for position, source in enumerate(sources):
            model = None
            with collector.capture(position, SourceTraceKind.source_validation_failed):
                model = _SourceLocation.model_validate(source)
            models.append(model)
        self.update_from_models(models, columns=self._source_columns, inplace=True)

        filepath_infos: list[FilepathInfo | None] = [None] * len(models)
        for position, model in enumerate(models):
            if model is not None:
                with collector.capture(position, SourceTraceKind.filepath_parse_failed):
                    info = (
                        model.filepath_info
                        if isinstance(model, SourceSourceLoc)
                        else FilepathInfo.from_source(
                            model.source_loc, raise_on_error=False
                        )
                    )
                    if info is None:
                        collector.record(
                            position, SourceTraceKind.filepath_unrecognized
                        )
                    else:
                        filepath_infos[position] = info
        self._obj[self._FILEPATH_INFO_COLUMN] = pd.array(filepath_infos, dtype=object)
        columns = {
            model_cls: self._filepath_columns(model_cls)
            for model_cls in dict.fromkeys(
                type(info) for info in filepath_infos if info is not None
            )
        }
        self.update_from_models(filepath_infos, columns=columns, inplace=True)
        self._log_traces()
        return self

    def source_trace_summary(self) -> pd.DataFrame:
        """Count rows containing each recorded issue kind.

        Returns
        -------
        pandas.DataFrame
            Columns ``kind``, ``count`` and ``total``. Each row contributes at most
            once to a kind's count; total is this table's row count. Counts use
            the current rows, including any filtering or duplication. Full
            per-row issue history remains in ``source_trace``. The summary is
            empty if there are no recorded issues or no trace column.
            Count parsed filenames with
            ``table.filepath_info.notna().sum()``.
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
