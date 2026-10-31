"""Issues recorded while collecting source information."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum, auto
from functools import cached_property
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Iterator, KeysView

    import pandas as pd

__all__ = ["SourceErrorPolicyT", "SourceTrace", "SourceTraceEntry", "SourceTraceKind"]

type SourceErrorPolicyT = Literal["raise", "record"]


class SourceTraceKind(StrEnum):
    """Issues recorded while preparing rows of a file table.

    Kinds ending in ``_rule_unmatched`` report no applicable rule. These are
    informational; kinds ending in ``_failed`` report processing errors.
    """

    source_validation_failed = auto()
    filepath_rule_unmatched = auto()
    filepath_parse_failed = auto()
    file_accessor_rule_unmatched = auto()
    opener_rule_unmatched = auto()
    source_open_failed = auto()
    reader_creation_failed = auto()
    metadata_read_failed = auto()
    metadata_unavailable = auto()

    @cached_property
    def is_error(self) -> bool:
        """Whether this kind represents a processing failure."""
        return self.value.endswith("_failed")


@dataclass(frozen=True, slots=True)
class SourceTraceEntry:
    """One issue recorded while preparing a source.

    Parameters
    ----------
    kind : SourceTraceKind
        Issue represented by this entry.
    message : str, optional
        Additional detail, such as an exception message. Defaults to None when
        the kind alone describes the result.
    context : dict, optional
        Supporting details for inspection and logging. Defaults to None.
        Failures include an ``exception_type``; live resources and exceptions
        are not retained.
    """

    kind: SourceTraceKind
    message: str | None = None
    context: dict[str, object] | None = None

    @property
    def is_error(self) -> bool:
        """Whether this entry records a processing failure."""
        return self.kind.is_error


@dataclass(slots=True)
class SourceTrace:
    """Latest issue of each kind recorded for one source.

    A new trace is empty. Use `record` to add or replace an issue of the same
    kind, `iter_entries` to inspect issues, or `iter_error_entries` to inspect
    failures only. Successful operations do not clear recorded issues.

    Examples
    --------
    >>> trace = SourceTrace()
    >>> trace.record(SourceTraceEntry(kind=SourceTraceKind.filepath_rule_unmatched))
    >>> trace.has_error
    False
    >>> trace.record(SourceTraceEntry(kind=SourceTraceKind.filepath_parse_failed))
    >>> trace.has_error
    True
    >>> [entry.kind.value for entry in trace.iter_error_entries()]
    ['filepath_parse_failed']
    """

    _entries_by_kind: dict[SourceTraceKind, SourceTraceEntry] = field(
        default_factory=dict, init=False, repr=False
    )
    _error_entries_by_kind: dict[SourceTraceKind, SourceTraceEntry] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def kinds(self) -> KeysView[SourceTraceKind]:
        """Recorded issue kinds, in order of first occurrence."""
        return self._entries_by_kind.keys()

    def iter_entries(self) -> Iterator[SourceTraceEntry]:
        """Iterate over the latest issue of each recorded kind.

        Yields
        ------
        SourceTraceEntry
            One issue per kind, in order of each kind's first occurrence.
        """
        return iter(self._entries_by_kind.values())

    def iter_error_entries(self) -> Iterator[SourceTraceEntry]:
        """Iterate over the latest failure of each recorded error kind.

        Yields
        ------
        SourceTraceEntry
            One failure per kind, in order of each kind's first occurrence.
        """
        return iter(self._error_entries_by_kind.values())

    @property
    def has_error(self) -> bool:
        """Whether any recorded entry is a processing failure."""
        return bool(self._error_entries_by_kind)

    def record(self, entry: SourceTraceEntry) -> None:
        """Retain an issue, replacing any previous entry of the same kind.

        Parameters
        ----------
        entry : SourceTraceEntry
            Latest details for this kind. Other kinds remain unchanged.
        """
        self._entries_by_kind[entry.kind] = entry
        if entry.is_error:
            self._error_entries_by_kind[entry.kind] = entry


@dataclass(slots=True)
class _SourceTraceCollector:
    """Record issues by row position in an existing table trace column."""

    traces: pd.Series
    on_error: SourceErrorPolicyT

    def record(
        self,
        position: int,
        kind: SourceTraceKind,
        message: str | None = None,
        *,
        context: dict[str, object] | None = None,
    ) -> None:
        entry = SourceTraceEntry(kind=kind, message=message, context=context)
        cast("SourceTrace", self.traces.iloc[position]).record(entry)

    @contextmanager
    def capture(
        self,
        position: int,
        failure_kind: SourceTraceKind,
        *,
        exceptions: tuple[type[Exception], ...] = (ValueError, KeyError, OSError),
    ) -> Iterator[None]:
        try:
            yield
        except exceptions as error:
            self.record(
                position,
                failure_kind,
                str(error),
                context={"exception_type": type(error).__name__},
            )
            if self.on_error == "raise":
                raise
