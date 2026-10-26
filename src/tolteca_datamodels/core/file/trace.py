"""Issues recorded while collecting source information."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum, auto
from functools import cached_property
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd
    from pydantic import JsonValue

__all__ = ["SourceErrorPolicyT", "SourceTrace", "SourceTraceEntry", "SourceTraceKind"]

type SourceErrorPolicyT = Literal["raise", "record"]


class SourceTraceKind(StrEnum):
    """Issues recorded while preparing rows of a file table."""

    source_validation_failed = auto()
    filepath_unrecognized = auto()
    filepath_parse_failed = auto()
    data_model_unmatched = auto()
    data_model_unidentified = auto()

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
        JSON-compatible supporting details. Defaults to None. Failures include
        an ``exception_type``; live resources and exceptions are not retained.
    """

    kind: SourceTraceKind
    message: str | None = None
    context: dict[str, JsonValue] | None = None

    @property
    def is_error(self) -> bool:
        """Whether this entry records a processing failure."""
        return self.kind.is_error


@dataclass(slots=True)
class SourceTrace:
    """Append-only issues recorded for one source.

    A new trace is empty. Use `append` to record an issue, `iter_entries` to
    inspect all issues, or `iter_error_entries` to inspect failures only.

    Examples
    --------
    >>> trace = SourceTrace()
    >>> trace.append(SourceTraceEntry(kind=SourceTraceKind.filepath_unrecognized))
    >>> trace.has_error
    False
    >>> trace.append(SourceTraceEntry(kind=SourceTraceKind.filepath_parse_failed))
    >>> trace.has_error
    True
    >>> [entry.kind.value for entry in trace.iter_error_entries()]
    ['filepath_parse_failed']
    """

    _entries: list[SourceTraceEntry] = field(
        default_factory=list, init=False, repr=False
    )
    _error_entries: list[SourceTraceEntry] = field(
        default_factory=list, init=False, repr=False
    )
    _kinds: set[SourceTraceKind] = field(default_factory=set, init=False, repr=False)

    def iter_entries(self) -> Iterator[SourceTraceEntry]:
        """Iterate over all recorded issues in occurrence order.

        Yields
        ------
        SourceTraceEntry
            Each recorded issue, including failures.
        """
        return iter(self._entries)

    def iter_error_entries(self) -> Iterator[SourceTraceEntry]:
        """Iterate over recorded failures in occurrence order.

        Yields
        ------
        SourceTraceEntry
            Each entry whose kind represents a processing failure.
        """
        return iter(self._error_entries)

    @property
    def has_error(self) -> bool:
        """Whether any recorded entry is a processing failure."""
        return bool(self._error_entries)

    def append(self, entry: SourceTraceEntry) -> None:
        """Record an issue after the existing entries.

        Parameters
        ----------
        entry : SourceTraceEntry
            Issue to append. Existing entries retain their order.
        """
        self._entries.append(entry)
        self._kinds.add(entry.kind)
        if entry.is_error:
            self._error_entries.append(entry)


@dataclass(slots=True)
class _SourceTraceCollector:
    """Append issues by row position to an existing table trace column."""

    traces: pd.Series
    on_error: SourceErrorPolicyT

    def record(
        self,
        position: int,
        kind: SourceTraceKind,
        message: str | None = None,
        *,
        context: dict[str, JsonValue] | None = None,
    ) -> None:
        entry = SourceTraceEntry(kind=kind, message=message, context=context)
        cast("SourceTrace", self.traces.iloc[position]).append(entry)

    @contextmanager
    def capture(self, position: int, failure_kind: SourceTraceKind) -> Iterator[None]:
        try:
            yield
        except (ValueError, KeyError, OSError) as error:
            self.record(
                position,
                failure_kind,
                str(error),
                context={"exception_type": type(error).__name__},
            )
            if self.on_error == "raise":
                raise
