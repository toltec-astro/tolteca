"""Single-workflow source handling and shared column preparation."""

from __future__ import annotations

from contextlib import contextmanager
from functools import cache
from typing import TYPE_CHECKING, ClassVar, Literal, cast

import pandas as pd
from pydantic import TypeAdapter
from tollan.accessor.pandas.model_table import Contains, ModelColumn, RecordRuleMapper

from ..obs_spec import ObsSpecBase, ObsSpecContext
from ..source import SourceOpener, SourceReaderRules, source_opener
from ._metadata import _read_source_metadata
from ._open import _open_workflow_sources
from .trace import SourceErrorPolicyT, SourceTraceEntry, SourceTraceKind

if TYPE_CHECKING:
    from collections.abc import Collection, Hashable, Iterable, Iterator, Mapping

    from ..source import SourceReader
    from .accessor import FileAccessorBase

__all__ = ["FileAccessorWorkflow", "FileTableModeT"]

type FileTableModeT = Literal["guess", "guess_or_read", "read"]
_validate_file_table_mode = TypeAdapter(FileTableModeT).validate_python
_validate_source_error_policy = TypeAdapter(SourceErrorPolicyT).validate_python


class FileAccessorWorkflow:
    """Match, open and read sources using one workflow's rules.

    Parameters
    ----------
    file_accessor : FileAccessorBase
        Accessor whose table receives derived columns and metadata. Construction
        does not change the table or open resources.

    Notes
    -----
    Set ``_file_match_rules`` to choose matching rows; an empty tuple matches
    none, while ``({},)`` matches every valid source. Configure
    ``_source_opener``, ``_source_reader_rules`` and ``_metadata_columns`` for
    resource opening, reader selection and metadata projection.
    The preparation hooks return column Series without modifying the table:
    ``_prepare_derived_columns(positions)`` covers matching rows;
    ``_prepare_derived_column_defaults()`` covers all rows;
    ``_prepare_reader_columns(reads)`` covers row-position keys in ``reads``.
    Set ``_obs_spec_cls`` and ``_obs_spec_context`` for observation queries.
    ``_supersedes_workflow_types`` gives this workflow precedence over other
    selected workflows during composition.
    """

    _file_match_rules: ClassVar[tuple[dict[str, Contains | Hashable], ...]] = ()
    _supersedes_workflow_types: ClassVar[tuple[type[FileAccessorWorkflow], ...]] = ()
    _query_namespace: ClassVar[dict[str, object]] = {}
    _query_symbols: ClassVar[dict[str, object]] = {}
    _source_opener: ClassVar[SourceOpener] = source_opener
    _source_reader_rules: ClassVar[SourceReaderRules] = SourceReaderRules(rules=())
    _metadata_columns: ClassVar[dict[type, dict[str, ModelColumn]]] = {}
    _obs_spec_cls: ClassVar[type[ObsSpecBase] | None] = None
    _obs_spec_context: ClassVar[ObsSpecContext] = ObsSpecContext()

    def __init__(self, file_accessor: FileAccessorBase) -> None:
        self.file_accessor = file_accessor

    @property
    def table(self) -> pd.DataFrame:
        """The accessor's table, without copying."""
        return self.file_accessor.table

    @staticmethod
    @cache
    def _compile_file_match_rules(
        accessor_type: type[FileAccessorBase],
    ) -> RecordRuleMapper[type[FileAccessorBase]]:
        return RecordRuleMapper(
            rules=[
                (constraints, accessor_type)
                for constraints in accessor_type._workflow_cls._file_match_rules
            ],
            default=None,
        )

    def _match_positions(self) -> list[int]:
        """Match this workflow's rules and record issues for valid unmatched rows."""
        if not self._file_match_rules:
            return []
        accessor_type = type(self.file_accessor)
        selected = self._compile_file_match_rules(accessor_type).map_table(self.table)
        groups = _group_matched_rows_and_record_issues(
            self.file_accessor, selected, (accessor_type,)
        )
        return groups.get(accessor_type, [])

    def prepare(self, *, mode: FileTableModeT, on_error: SourceErrorPolicyT) -> None:
        """Derive initial columns and read metadata requested by the factory.

        Parameters
        ----------
        mode : {"guess", "guess_or_read", "read"}
            Whether to read no sources, unidentified sources, or matching and
            unidentified sources after deriving filename-based information.
        on_error : {"record", "raise"}
            Source-error handling for metadata reads.
        """
        if not self._file_match_rules:
            return
        positions = self._match_positions()
        _apply_derived_column_updates(
            self.table,
            [(positions, self)],
            defaults=self._prepare_derived_column_defaults(),
        )
        if mode != "guess":
            _read_source_metadata(
                self.file_accessor,
                [(positions, self)],
                candidates=(self,),
                unidentified_only=mode == "guess_or_read",
                on_error=_validate_source_error_policy(on_error),
            )

    @contextmanager
    def open(
        self, *, on_error: SourceErrorPolicyT = "record"
    ) -> Iterator[pd.DataFrame]:
        """Open matching and unidentified sources using this workflow's readers.

        Parameters
        ----------
        on_error : {"record", "raise"}, optional
            Record source failures by default, or propagate them.

        Yields
        ------
        pandas.DataFrame
            Shared table with live resource and data-accessor columns. Owned
            resources close and temporary columns are removed on exit.
        """
        on_error = _validate_source_error_policy(on_error)
        with _open_workflow_sources(
            self.file_accessor,
            [(self._match_positions(), self)],
            candidates=(self,) if self._file_match_rules else (),
            on_error=on_error,
        ):
            yield self.table
        self.file_accessor._log_traces()

    def read_metadata(self, *, on_error: SourceErrorPolicyT = "record") -> None:
        """Read source metadata and update the shared table.

        Each call reads metadata, reusing live readers when available.

        Parameters
        ----------
        on_error : {"record", "raise"}, optional
            Record source failures by default, or propagate them. Projection
            failures always raise before changing destination columns.
        """
        on_error = _validate_source_error_policy(on_error)
        _read_source_metadata(
            self.file_accessor,
            [(self._match_positions(), self)],
            candidates=(self,) if self._file_match_rules else (),
            on_error=on_error,
        )
        self.file_accessor._log_traces()

    def query_obs_spec(self, spec: ObsSpecBase | str | int) -> pd.DataFrame:
        """Select rows using this workflow's observation vocabulary.

        Parameters
        ----------
        spec : ObsSpecBase, str or int
            Selector accepted by this workflow's selector class. When no class
            is declared, supply a parsed selector.

        Returns
        -------
        pandas.DataFrame
            Matching rows from the full table, retaining index and order.

        Raises
        ------
        ValueError
            Invalid selector or text supplied without a declared selector class.
        TypeError
            A parsed selector is incompatible with the declared selector class.
        """
        return _select_obs_spec(
            self.table, spec, self._obs_spec_cls, self._obs_spec_context
        )

    def _get_query_namespace(self) -> dict[str, object]:
        return self._query_namespace | self._query_symbols

    def _prepare_derived_columns(
        self, positions: list[int]
    ) -> dict[Hashable, pd.Series]:
        """Prepare this policy's values in selected row order without mutation."""
        return {}

    def _prepare_derived_column_defaults(self) -> dict[Hashable, pd.Series]:
        """Prepare default Series covering every table row without mutation."""
        return {}

    def _prepare_reader_columns(
        self, reads: Mapping[int, tuple[SourceReader, object | None]]
    ) -> dict[str, pd.Series]:
        """Prepare extra columns indexed by row position, without table mutation."""
        return {}


def _select_obs_spec(
    table: pd.DataFrame,
    spec: ObsSpecBase | str | int,
    spec_cls: type[ObsSpecBase] | None,
    context: ObsSpecContext,
) -> pd.DataFrame:
    """Validate the selector against the supplied layout and select once."""
    if spec_cls is not None:
        spec = spec_cls.parse(spec)
    elif not isinstance(spec, ObsSpecBase):
        msg = "Pass a parsed ObsSpec when no common selector class is available"
        raise ValueError(msg)
    return spec.select(table, context=context)


def _apply_derived_column_updates(
    table: pd.DataFrame,
    groups: Iterable[tuple[list[int], FileAccessorWorkflow]],
    *,
    defaults: dict[Hashable, pd.Series] | None = None,
) -> None:
    """Stage destination columns for all supplied groups before assigning any."""
    updates = dict(defaults or {})
    for positions, workflow in groups:
        if not positions:
            continue
        for name, values in workflow._prepare_derived_columns(positions).items():
            if name not in updates:
                updates[name] = (
                    table[name].copy()
                    if name in table
                    else pd.Series(None, index=table.index, dtype=values.dtype)
                )
            updates[name].iloc[positions] = values.array
    for name, values in updates.items():
        table[name] = values


def _group_matched_rows_and_record_issues(
    file_accessor: FileAccessorBase,
    selected: Iterable[type[FileAccessorBase] | None],
    accessor_types: tuple[type[FileAccessorBase], ...],
) -> dict[type[FileAccessorBase], list[int]]:
    """Group rule results and record compatibility issues on valid unmatched rows."""
    matches: dict[type[FileAccessorBase], list[int]] = {}
    locations = file_accessor.table.get(file_accessor._SOURCE_LOC_COLUMN)
    collector = file_accessor._make_source_trace_collector("record")
    memberships = file_accessor.table.get(file_accessor._DATA_MODELS_COLUMN)
    present = (
        memberships.notna()
        if memberships is not None
        else [False] * len(file_accessor.table)
    )
    valid = (
        locations.notna()
        if locations is not None
        else [True] * len(file_accessor.table)
    )
    values = (
        memberships if memberships is not None else [None] * len(file_accessor.table)
    )
    accessor_names = [f"{cls.__module__}.{cls.__qualname__}" for cls in accessor_types]
    for position, (is_valid, cls, has_value, value, trace) in enumerate(
        zip(valid, selected, present, values, collector.traces, strict=True)
    ):
        if not is_valid:
            continue
        if cls is not None:
            matches.setdefault(cls, []).append(position)
            continue
        current = cast("Collection[str]", value) if has_value else ()
        trace.record(
            SourceTraceEntry(
                kind=SourceTraceKind.file_accessor_rule_unmatched,
                context={
                    "accessor_types": list(accessor_names),
                    "data_models": sorted(current),
                },
            )
        )
    return matches
