"""Acquire resources and create data accessors for file-table rows."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd
from tollan.utils.fileloc import fileloc

from .source import _enter_opened_resource, _is_source_identifier
from .trace import SourceTraceKind

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
    from pathlib import Path

    from tollan.utils.fileloc import FileLoc

    from ..source import SourceReaderRules
    from .accessor import FileAccessorBase
    from .trace import SourceErrorPolicyT, _SourceTraceCollector
    from .workflow import FileAccessorWorkflow

# Reader creation can reject the object's type, layout or routing information.
_READER_CREATION_EXCEPTIONS = (TypeError, ValueError, KeyError, OSError)


@contextmanager
def _open_workflow_sources(
    file_accessor: FileAccessorBase,
    groups: Iterable[tuple[list[int], FileAccessorWorkflow]],
    *,
    candidates: Sequence[FileAccessorWorkflow],
    unidentified_only: bool = False,
    on_error: SourceErrorPolicyT = "record",
) -> Iterator[
    tuple[list[tuple[list[int], FileAccessorWorkflow]], _SourceTraceCollector]
]:
    """Open eligible rows and yield their live groups and trace collector."""
    collector = file_accessor._make_source_trace_collector(on_error)
    groups = list(groups)
    unidentified = _unidentified_positions(file_accessor)
    if unidentified_only:
        unidentified_set = set(unidentified)
        groups = [
            ([p for p in positions if p in unidentified_set], workflow)
            for positions, workflow in groups
        ]
    matched = {p for positions, _ in groups for p in positions}
    unknown = [p for p in unidentified if p not in matched]
    with _open_sources(
        file_accessor,
        groups,
        unidentified=unknown if candidates else (),
        candidates=candidates,
        collector=collector,
    ) as live_groups:
        yield live_groups, collector


def _unidentified_positions(file_accessor: FileAccessorBase) -> list[int]:
    """Find valid sources with no current data-model membership."""
    table = file_accessor.table
    locations = table.get(file_accessor._SOURCE_LOC_COLUMN)
    valid = locations.notna() if locations is not None else [True] * len(table)
    memberships = table.get(file_accessor._DATA_MODELS_COLUMN)
    if memberships is None:
        return [p for p, ok in enumerate(valid) if ok]
    return [
        p
        for p, (ok, present, values) in enumerate(
            zip(valid, memberships.notna(), memberships, strict=True)
        )
        if ok and (not present or not len(values))
    ]


@contextmanager
def _open_sources(
    file_accessor: FileAccessorBase,
    groups: Sequence[tuple[list[int], FileAccessorWorkflow]],
    *,
    unidentified: Sequence[int] = (),
    candidates: Sequence[FileAccessorWorkflow] = (),
    collector: _SourceTraceCollector,
) -> Iterator[list[tuple[list[int], FileAccessorWorkflow]]]:
    """Bind sources for disjoint policies and expose their live objects."""
    table = file_accessor.table
    column_names = _get_rule_column_names(table)
    existing_resources = (
        list(table[file_accessor._IO_OBJ_COLUMN])
        if file_accessor._IO_OBJ_COLUMN in table
        else [None] * len(table)
    )
    existing_readers = (
        list(table[file_accessor._ACCESSOR_OBJ_COLUMN])
        if file_accessor._ACCESSOR_OBJ_COLUMN in table
        else [None] * len(table)
    )
    resources: list[object | None] = [None] * len(table)
    readers: list[object | None] = [None] * len(table)
    resolved = {
        type(workflow.file_accessor): (list(positions), workflow)
        for positions, workflow in groups
    }
    with ExitStack() as stack:
        opened = _OpenedSources(stack)
        for positions, workflow in groups:
            pending_positions = []
            reader_types = workflow._source_reader_rules.accessor_classes
            for position in positions:
                if type(existing_readers[position]) in reader_types:
                    resources[position] = existing_resources[position]
                    readers[position] = existing_readers[position]
                else:
                    pending_positions.append(position)
            results = _open_and_create_readers(
                workflow,
                pending_positions,
                [existing_resources[p] for p in pending_positions],
                column_names=column_names,
                opened=opened,
                collector=collector,
            )
            for position, (resource, reader) in zip(
                pending_positions, results, strict=True
            ):
                resources[position], readers[position] = resource, reader
        for position in unidentified:
            resource, reader, workflow = _open_unidentified_source(
                file_accessor,
                position,
                candidates,
                column_names=column_names,
                existing_resource=existing_resources[position],
                existing_reader=existing_readers[position],
                opened=opened,
                collector=collector,
            )
            resources[position], readers[position] = resource, reader
            if workflow is not None:
                resolved.setdefault(type(workflow.file_accessor), ([], workflow))[
                    0
                ].append(position)
        live_groups = list(resolved.values())
        previous = {
            name: table[name]
            for name in (
                file_accessor._IO_OBJ_COLUMN,
                file_accessor._ACCESSOR_OBJ_COLUMN,
            )
            if name in table
        }
        table[file_accessor._IO_OBJ_COLUMN] = pd.Series(
            resources, index=table.index, dtype=object
        )
        table[file_accessor._ACCESSOR_OBJ_COLUMN] = pd.Series(
            readers, index=table.index, dtype=object
        )
        try:
            yield live_groups
        finally:
            for name in (
                file_accessor._IO_OBJ_COLUMN,
                file_accessor._ACCESSOR_OBJ_COLUMN,
            ):
                if name in previous:
                    table[name] = previous[name]
                else:
                    del table[name]


def _open_unidentified_source(  # noqa: PLR0913 -- row identity plus shared resource/trace scope.
    file_accessor: FileAccessorBase,
    position: int,
    candidates: Sequence[FileAccessorWorkflow],
    *,
    column_names: dict[str, str],
    existing_resource: object | None,
    existing_reader: object | None,
    opened: _OpenedSources,
    collector: _SourceTraceCollector,
) -> tuple[object | None, object | None, FileAccessorWorkflow | None]:
    """Try policies in precedence order until a reader is created or raises."""
    row = file_accessor.table.iloc[position]
    record = row.to_dict()
    record.update({name: row[column] for name, column in column_names.items()})
    source = row[file_accessor._SOURCE_COLUMN]
    resource = existing_resource
    attempted_openers: dict[int, object | None] = {}
    reader_attempted = False
    for workflow in candidates:
        if type(existing_reader) in workflow._source_reader_rules.accessor_classes:
            return existing_resource, existing_reader, workflow
        if existing_resource is not None or not _is_source_identifier(source):
            resource = existing_resource if existing_resource is not None else source
        else:
            opener = workflow._source_opener(record)
            if opener is None:
                continue
            key = id(opener)
            if key not in attempted_openers:
                attempted_openers[key] = _open_source(
                    position, source, opener, opened, collector
                )
            resource = attempted_openers[key]
        if resource is None:
            continue
        reader_attempted = True
        rules = workflow._source_reader_rules
        with collector.capture(
            position,
            SourceTraceKind.reader_creation_failed,
            exceptions=_READER_CREATION_EXCEPTIONS,
        ):
            reader = _create_reader(resource, rules(record), rules)
            if reader is None:
                continue
            return resource, reader, workflow
        return resource, None, None
    if resource is None and _is_source_identifier(source) and not attempted_openers:
        _open_source(position, source, None, opened, collector)
    elif reader_attempted:
        with collector.capture(position, SourceTraceKind.reader_creation_failed):
            msg = "No selected reader recognizes the opened source"
            raise ValueError(msg)
    return resource, None, None


def _get_rule_column_names(table: pd.DataFrame) -> dict[str, str]:
    """Use filename snapshots for fields that have no current metadata column."""
    return {
        name.removeprefix("file_"): name
        for name in table.columns
        if isinstance(name, str)
        and name.startswith("file_")
        and name.removeprefix("file_") not in table
    }


@dataclass(slots=True)
class _OpenedSources:
    """Resources owned by one scope, opened once per location and opener."""

    stack: ExitStack
    resources: dict[tuple[Hashable, int], Any] = field(default_factory=dict)

    def open(self, source: str | Path | FileLoc, opener: Callable[..., Any]) -> Any:
        location = fileloc(source)
        key = (
            location.path.resolve() if location.is_local() else location.url,
            id(opener),
        )
        if key not in self.resources:
            self.resources[key] = _enter_opened_resource(self.stack, opener(source))
        return self.resources[key]


def _open_and_create_readers(  # noqa: PLR0913 -- matched rows and shared resource scope.
    workflow: FileAccessorWorkflow,
    positions: Sequence[int],
    existing_resources: Sequence[object | None],
    *,
    column_names: dict[str, str],
    opened: _OpenedSources,
    collector: _SourceTraceCollector,
) -> list[tuple[object | None, object | None]]:
    """Return ``(io_obj, accessor_obj)`` per position; failures leave None."""
    if not positions:
        return []
    rows = workflow.table.iloc[list(positions)]
    openers = workflow._source_opener.map_table(rows, column_names=column_names)
    rules = workflow._source_reader_rules
    reader_classes = rules.map_table(rows, column_names=column_names)
    results = []
    for position, source, existing_resource, opener, reader_cls in zip(
        positions,
        rows[workflow.file_accessor._SOURCE_COLUMN],
        existing_resources,
        openers,
        reader_classes,
        strict=True,
    ):
        if existing_resource is not None:
            io_obj = existing_resource
        elif _is_source_identifier(source):
            io_obj = _open_source(position, source, opener, opened, collector)
        else:
            io_obj = source
        accessor_obj = None
        if io_obj is not None:
            with collector.capture(
                position,
                SourceTraceKind.reader_creation_failed,
                exceptions=_READER_CREATION_EXCEPTIONS,
            ):
                accessor_obj = _create_reader(io_obj, reader_cls, rules)
                if accessor_obj is None:
                    msg = "No reader recognizes the opened source"
                    raise ValueError(msg)
        results.append((io_obj, accessor_obj))
    return results


def _open_source(
    position: int,
    source: str | Path | FileLoc,
    opener: Callable[..., Any] | None,
    opened: _OpenedSources,
    collector: _SourceTraceCollector,
) -> object | None:
    """Open a row's source with the opener selected by its current values."""
    if opener is None:
        collector.record(position, SourceTraceKind.opener_rule_unmatched)
        return None
    with collector.capture(position, SourceTraceKind.source_open_failed):
        return opened.open(source, opener)
    return None


def _create_reader(
    data: object,
    reader_cls: type | None,
    rules: SourceReaderRules,
) -> object | None:
    """Select and construct a data accessor, or return None if nothing matches."""
    if reader_cls is None:
        reader_cls = rules.identify(data)
    return None if reader_cls is None else reader_cls(data)
