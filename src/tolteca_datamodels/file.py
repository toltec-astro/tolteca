"""Filename-based file tables with LMT and TolTEC support.

Importing this module loads the built-in filename parsers. Additional registered
parsers are also used. Observation columns supplement the ``file_*`` snapshots;
resources are opened when metadata is requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from . import filepath as _filepath  # noqa: F401 -- load built-in filename parsers.
from .core.file import (
    CompositeFileAccessor,
    CompositeFileWorkflow,
    FileAccessorBase,
    FileTableModeT,
    SourceErrorPolicyT,
)
from .data_model import DataModelSet

if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = ["FileAccessor", "file_table_from_sources"]


class _RegisteredFileWorkflow(CompositeFileWorkflow):
    """File-handling rules and column preparation for FileAccessor."""

    @classmethod
    def _get_component_accessor_types(cls) -> tuple[type[FileAccessorBase], ...]:
        """All concrete capabilities available through registered data models."""
        return DataModelSet.from_select().file_accessor_types


@pd.api.extensions.register_dataframe_accessor("file")
class FileAccessor(CompositeFileAccessor[_RegisteredFileWorkflow]):
    """Build file tables using registered data-model capabilities.

    Filename fields retain their ``file_*`` snapshots. Marked metadata fields
    also initialize unprefixed columns on every parsed row. Selected concrete
    accessors add UIDs and classifications on matching rows.
    Instrument classifications occupy separate flag columns, such as
    ``data_kind_toltec``. Query them with qualified enum names, for example
    ``bits_any_set(data_kind_toltec, ToltecDataKind.raw_sweep)``.

    Parameters
    ----------
    data : pandas.DataFrame
        Table to access, including an empty DataFrame. Binding requires no
        columns and does not copy or modify the table. Use `from_sources` to
        create a populated accessor.

    Examples
    --------
    >>> file_accessor = FileAccessor.from_sources([
    ...     "tel_toltec_2025-10-31_145609_00_0002.nc",
    ... ])
    >>> file_accessor.table[["interface", "obsnum", "file_obsnum"]].to_dict("records")
    [{'interface': 'tel', 'obsnum': 145609, 'file_obsnum': 145609}]
    """


def file_table_from_sources(
    sources: Iterable[object],
    *,
    mode: FileTableModeT = "guess",
    on_error: SourceErrorPolicyT = "record",
) -> pd.DataFrame:
    """Build a DataFrame of source locations and filename information.

    Parameters
    ----------
    sources : finite iterable
        Resources or dicts containing
        ``source`` and ``source_loc``. See `FileAccessor.from_sources` for the
        accepted forms and projected columns.
    mode : {"guess", "guess_or_read", "read"}, optional
        Default "guess" uses filenames. "guess_or_read" reads unidentified
        sources; "read" also reads recognized sources handled by selected policies.
    on_error : {"record", "raise"}, optional
        Default is "record": retain invalid entries with a failure trace.
        Use "raise" to propagate source-location and filename parsing failures.
        Invalid outer collections, ambiguous registrations and projection
        failures always raise.

    Returns
    -------
    pandas.DataFrame
        One row per input in iteration order, with source locations, filename
        information and row-local traces. This is the table created by
        `FileAccessor.from_sources`, without an additional copy.

    See Also
    --------
    FileAccessor.from_sources : Create an accessor over the populated table.

    Examples
    --------
    >>> table = file_table_from_sources(["example.nc"])
    >>> table.file_ext.tolist()
    ['nc']
    """
    return FileAccessor.from_sources(sources, mode=mode, on_error=on_error).table
