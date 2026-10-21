"""Source-processing events and tabular summaries."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import pandas as pd
from pydantic import ConfigDict, TypeAdapter
from tollan.utils.log import logger

from ..types import FrozenDataModel

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = ["SourceTraceEvent", "source_trace_issues"]

type SourceErrorPolicyT = Literal["raise", "record"]
_validate_error_policy = TypeAdapter(SourceErrorPolicyT).validate_python
type _Stage = Literal["filepath", "scope", "open", "identify", "metadata"]
type _Outcome = Literal[
    "success", "error", "unrecognized", "in_scope", "out_of_scope", "skipped"
]
_TRACE_COLUMN = "source_trace"


class SourceTraceEvent(FrozenDataModel):
    """One outcome while preparing or accessing a source.

    Parameters
    ----------
    stage : {"filepath", "scope", "open", "identify", "metadata"}
        Operation that produced the event.
    outcome : str
        Success, error, unrecognized input, scope membership, or skipped work.
    message : str
        Description of the outcome.
    exception_type : str, optional
        Exception class name for a failed operation; otherwise None.
    """

    model_config = ConfigDict(extra="forbid")

    stage: _Stage
    outcome: _Outcome
    message: str
    exception_type: str | None = None


def source_trace_issues(events: Iterable[SourceTraceEvent]) -> list[SourceTraceEvent]:
    """Find unresolved failures and identification misses in a source trace.

    Parameters
    ----------
    events : iterable of SourceTraceEvent
        Events in processing order.

    Returns
    -------
    list of SourceTraceEvent
        Latest error or unrecognized outcome per stage. A successful retry
        resolves an earlier issue; skipping a stage does not resolve it.
        The original event history is unchanged.
    """
    latest = {event.stage: event for event in events if event.outcome != "skipped"}
    return [
        event
        for event in latest.values()
        if event.outcome in {"error", "unrecognized"} and event.stage != "scope"
    ]


def _summarize_traces(cells: Iterable, *, total: int) -> pd.DataFrame:
    counts = Counter()
    for events in cells:
        if not isinstance(events, list):
            continue
        latest = {event.stage: event for event in events}
        counts.update((event.stage, event.outcome) for event in latest.values())
    return pd.DataFrame(
        [(*key, count, total) for key, count in sorted(counts.items())],
        columns=["stage", "outcome", "count", "total"],
    )


@lru_cache(maxsize=256)
def _event(stage, outcome, message, exception_type):
    # Frozen events for repeated batch outcomes can be shared between rows.
    return SourceTraceEvent(
        stage=stage, outcome=outcome, message=message, exception_type=exception_type
    )


class _SourceTraceCollector:
    """Collect events without mutating existing table cells."""

    def __init__(self, on_error: SourceErrorPolicyT) -> None:
        self._on_error = _validate_error_policy(on_error)
        self._events: dict[int, list[SourceTraceEvent]] = {}
        self._outcomes: dict[int, dict[_Stage, SourceTraceEvent | None]] = {}

    def record(
        self,
        position: int,
        stage: _Stage,
        message: str,
        exception_type: str | None = None,
        *,
        outcome: _Outcome = "success",
    ) -> None:
        event = _event(stage, outcome, message, exception_type)
        self._events.setdefault(position, []).append(event)
        self._outcomes.setdefault(position, {})[stage] = event

    def failed(self, position: int, stage: _Stage) -> bool:
        event = self._outcomes.get(position, {}).get(stage)
        return event is not None and event.outcome == "error"

    @contextmanager
    def capture(
        self,
        position: int,
        stage: _Stage,
        *,
        exceptions: tuple[type[Exception], ...] = (OSError, ValueError, KeyError),
        record_success: bool = True,
    ) -> Iterator[None]:
        self._outcomes.setdefault(position, {})[stage] = None
        try:
            yield
        except exceptions as error:
            if self._on_error == "raise":
                raise
            self.record(
                position, stage, str(error), type(error).__name__, outcome="error"
            )
        else:
            if record_success and self._outcomes[position][stage] is None:
                self.record(position, stage, "Completed")

    def update_table(self, table: pd.DataFrame) -> None:
        previous = table.get(_TRACE_COLUMN, [[] for _ in range(len(table))])
        cells = []
        for position, cell in enumerate(previous):
            events = self._events.get(position, [])
            cells.append((cell if isinstance(cell, list) else []) + events)
            for event in events:
                if event.outcome == "error":
                    logger.debug(
                        "Source {} (row {}): {}: {}: {}",
                        table.source_loc.iloc[position].url,
                        position,
                        event.stage,
                        event.exception_type,
                        event.message,
                    )
        table[_TRACE_COLUMN] = pd.Series(cells, dtype=object).array
        logger.opt(lazy=True).debug(
            "Source processing outcomes:\n{}",
            lambda: _summarize_traces(
                self._events.values(), total=len(table)
            ).to_string(index=False),
        )
