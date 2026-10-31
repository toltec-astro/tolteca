"""Read live accessors and stage metadata columns for a file table."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd
from tollan.accessor.pandas.model_table import ModelTableAccessor

from ._open import _open_workflow_sources
from .trace import SourceTraceKind

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ..source import SourceReader
    from .accessor import FileAccessorBase
    from .trace import SourceErrorPolicyT, _SourceTraceCollector
    from .workflow import FileAccessorWorkflow


def _read_source_metadata(
    file_accessor: FileAccessorBase,
    groups: Iterable[tuple[list[int], FileAccessorWorkflow]],
    *,
    candidates: Sequence[FileAccessorWorkflow],
    unidentified_only: bool = False,
    on_error: SourceErrorPolicyT = "record",
) -> None:
    """Read eligible sources within one shared resource and projection scope."""
    with _open_workflow_sources(
        file_accessor,
        groups,
        candidates=candidates,
        unidentified_only=unidentified_only,
        on_error=on_error,
    ) as (live_groups, collector):
        _update_from_live_accessors(file_accessor, live_groups, collector=collector)


def _update_from_live_accessors(
    file_accessor: FileAccessorBase,
    groups: Sequence[tuple[list[int], FileAccessorWorkflow]],
    *,
    collector: _SourceTraceCollector,
) -> None:
    """Read each policy's rows, stage destination columns, then assign once."""
    prepared = []
    destinations = set()
    readers = file_accessor.table[file_accessor._ACCESSOR_OBJ_COLUMN].array
    for positions, workflow in groups:
        reads: dict[int, tuple[SourceReader, object | None]] = {}
        for position in positions:
            reader = readers[position]
            if reader is None:
                continue
            with collector.capture(position, SourceTraceKind.metadata_read_failed):
                metadata = reader.meta
                reads[position] = (reader, metadata)
                if metadata is None:
                    collector.record(
                        position,
                        SourceTraceKind.metadata_unavailable,
                        context={"reader": type(reader).__qualname__},
                    )
        metadata = {p: meta for p, (_, meta) in reads.items() if meta is not None}
        reader_columns = workflow._prepare_reader_columns(reads) if reads else {}
        if metadata:
            destinations.update(
                name
                for columns in workflow._metadata_columns.values()
                for name in columns
            )
        destinations.update(reader_columns)
        prepared.append((workflow, metadata, reader_columns))
    if not any(meta or columns for _, meta, columns in prepared):
        return
    staged = file_accessor.table[
        [name for name in file_accessor.table if name in destinations]
    ].copy()
    staged_accessor = ModelTableAccessor(staged)
    for workflow, metadata, reader_columns in prepared:
        staged_accessor.update_from_models(
            metadata, columns=workflow._metadata_columns, inplace=True
        )
        for name, values in reader_columns.items():
            if name not in staged:
                staged[name] = pd.Series(None, index=staged.index, dtype=values.dtype)
            column_position = cast("int", staged.columns.get_loc(name))
            staged.iloc[values.index.to_numpy(), column_position] = values.array
    file_accessor.table[staged.columns] = staged
