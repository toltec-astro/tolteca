"""TolTEC file tables and companion telescope information."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pandas as pd
import xarray as xr
from tollan.accessor.pandas.model_table import Contains
from tollan.accessor.xarray import unwrap_data_source

from ..core.file import FileAccessorBase
from ..core.source import SourceReaderRules
from ..lmt._obs_spec import _obs_spec_context as _lmt_obs_spec_context
from ..lmt.file import LmtFileWorkflow
from ..lmt.metadata import LmtTelMetadata
from ..lmt.tel import LmtTelAccessor, _get_tel_reader_record
from ..lmt_base.file import LmtBaseFileWorkflow, _metadata_column_mapper
from . import filepath as _filepath  # noqa: F401 -- load this instrument's parsers.
from ._data_kind import _data_kind_mapper
from ._obs_spec import _obs_spec_context
from .hwpr import ToltecHwprAccessor, ToltecHwprIOMapper, ToltecHwprMetadata
from .kids.core import ToltecKidsAccessor, ToltecKidsIOMapper
from .metadata import (
    ToltecRawObsBlockMetadata,
    ToltecRawObsMetadata,
    ToltecSweepMetadata,
    ToltecTimestreamMetadata,
)
from .types import ToltecDataKind, ToltecDataModel

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping

    from tollan.utils.fileloc import FileLoc

    from ..core.file import FileTableModeT, SourceErrorPolicyT
    from ..core.source import SourceReader

__all__ = [
    "ToltecFileAccessor",
    "ToltecFileWorkflow",
    "get_info_from_source",
    "guess_info_from_sources",
    "guess_or_read_info_from_sources",
    "read_info_from_sources",
]


def _get_reader_record(data_source: object) -> object | None:
    """Routing values recorded in an opened TolTEC or telescope dataset."""
    if not isinstance(data_source, (xr.Dataset, xr.DataTree)):
        return None
    if (record := _get_tel_reader_record(data_source)) is not None:
        return record
    ds = cast("xr.Dataset", unwrap_data_source(data_source))
    hwpr = ToltecHwprIOMapper.from_data_source(ds)
    if all(
        hwpr.has_arr(ds, field)
        for field in (hwpr.schema.enc_ts, hwpr.schema.enc_val, hwpr.schema.enc_uts)
    ):
        return {"data_models": [ToltecDataModel.name], "interface": "hwpr"}
    kids = ToltecKidsIOMapper.from_data_source(ds)
    if kids._get_data_kind(ds) != ToltecDataKind.unknown:
        return {"data_models": [ToltecDataModel.name], "interface_kind": "roach"}
    return None


class ToltecFileWorkflow(LmtBaseFileWorkflow):
    """File-handling rules and column preparation for ToltecFileAccessor.

    Parameters
    ----------
    file_accessor : FileAccessorBase
        Accessor over the shared table. Construction does not process rows.
    """

    _file_match_rules = ({"data_models": Contains(value=ToltecDataModel.name)},)
    _supersedes_workflow_types = (LmtFileWorkflow,)
    _source_reader_rules: ClassVar = SourceReaderRules(
        prepare_record=_get_reader_record,
        rules=(
            (
                {
                    "data_models": Contains(value=ToltecDataModel.name),
                    "interface": "tel",
                },
                LmtTelAccessor,
            ),
            (
                {
                    "data_models": Contains(value=ToltecDataModel.name),
                    "interface_kind": "roach",
                },
                ToltecKidsAccessor,
            ),
            (
                {
                    "data_models": Contains(value=ToltecDataModel.name),
                    "interface": "hwpr",
                },
                ToltecHwprAccessor,
            ),
        ),
    )
    _metadata_columns: ClassVar = {
        model: _metadata_column_mapper(model)
        for model in (
            LmtTelMetadata,
            ToltecHwprMetadata,
            ToltecRawObsMetadata,
            ToltecTimestreamMetadata,
            ToltecRawObsBlockMetadata,
            ToltecSweepMetadata,
        )
    }
    _obs_spec_context: ClassVar = _lmt_obs_spec_context | _obs_spec_context
    _query_namespace: ClassVar = dict(ToltecDataKind.__members__)
    _query_symbols: ClassVar = {"ToltecDataKind": ToltecDataKind}
    _data_kind_rule_mapper: ClassVar = _data_kind_mapper

    def _prepare_derived_column_defaults(self) -> dict[Hashable, pd.Series]:
        return {
            "data_kind_toltec": pd.Series(
                self._data_kind_rule_mapper.default,
                index=self.table.index,
                dtype="Int64",
            )
        }

    def _prepare_derived_columns(
        self, positions: list[int]
    ) -> dict[Hashable, pd.Series]:
        derived = super()._prepare_derived_columns(positions)
        derived["data_kind_toltec"] = self._map_derived_column_from_current_values(
            positions,
            derived,
            self._data_kind_rule_mapper,
            dtype="Int64",
            fallback_columns=("prefix", "suffix", "proc_suffix", "ext"),
        )
        return derived

    def _prepare_reader_columns(
        self, reads: Mapping[int, tuple[SourceReader, object | None]]
    ) -> dict[str, pd.Series]:
        """Classify from the reader, including telescope layout variants."""
        kinds = {}
        for position, (reader, metadata) in reads.items():
            if (kind := getattr(reader, "data_kind", None)) is not None:
                kinds[position] = kind.value
            elif isinstance(metadata, LmtTelMetadata):
                kinds[position] = (
                    ToltecDataKind.unknown
                    if ToltecDataModel.name not in metadata.data_models
                    else ToltecDataKind.lmt_tel2
                    if Path(metadata.filepath_orig).name.startswith("tel_toltec2_")
                    else ToltecDataKind.lmt_tel
                ).value
        return {"data_kind_toltec": pd.Series(kinds, dtype="Int64")}


@pd.api.extensions.register_dataframe_accessor("toltec_file")
class ToltecFileAccessor(FileAccessorBase[ToltecFileWorkflow]):
    """Collect filename information with TolTEC observation and roach columns.

    All supplied rows and available filename registrations are retained.
    Rows whose current ``data_models`` contain ``"toltec"`` receive UIDs and
    classification, including telescope companions. Other rows retain filename
    snapshots, marked metadata and an informational compatibility trace.
    `from_sources` guesses without opening resources. Use `open` to access
    their contents and `read_metadata` to replace guessed values
    with metadata.
    When selected together with LmtFileAccessor, this accessor handles shared
    telescope rows once; LMT remains available for other telescope files.
    ``data_kind_toltec`` stores integer ToltecDataKind flags, with ``unknown``
    for unsupported filenames, foreign files and invalid sources. Queries accept
    member names such as ``raw_sweep`` and qualified names such as
    ``ToltecDataKind.raw_sweep`` with ``bits_any_set`` or ``bits_all_set``.
    Observation selectors number ``toltec*`` interfaces by their ROACH index.

    Parameters
    ----------
    data : pandas.DataFrame
        Table to access, including an empty DataFrame. Binding does not copy or
        modify it. Use `from_sources` to build a populated accessor.

    Examples
    --------
    >>> file_accessor = ToltecFileAccessor.from_sources([
    ...     "toltec8_145605_000_0001_2025_10_31_09_20_06_tune.nc",
    ...     "example.nc",
    ... ])
    >>> selected = file_accessor.query("bits_any_set(data_kind_toltec, raw_sweep)")
    >>> selected.interface.tolist()
    ['toltec8']
    >>> file_accessor.query("bits_any_set(data_kind_toltec, unknown)").source.tolist()
    ['example.nc']
    """


def get_info_from_source(
    source: object,
    *,
    source_loc: str | Path | FileLoc | None = None,
    mode: FileTableModeT = "guess",
    on_error: SourceErrorPolicyT = "record",
) -> Any:
    """Return a single file-table row with attribute access to its columns.

    Parameters
    ----------
    source : object
        File location or an already opened resource.
    source_loc : str, pathlib.Path or FileLoc, optional
        Location used for filename identification. Defaults to ``source``;
        required when ``source`` cannot be converted to FileLoc. For example,
        a cached file can be identified by its remote location.
    mode : {"guess", "guess_or_read", "read"}, optional
        How to populate the row; see `ToltecFileAccessor.from_sources`.
        Default is "guess".
    on_error : {"raise", "record"}, optional
        Failure handling; see `ToltecFileAccessor.from_sources`. Default is "record".

    Returns
    -------
    namedtuple
        Available file information, accessible by column name such as
        ``row.obsnum``. Columns depend on the recognized filename and metadata;
        unidentified sources may lack observation columns.
    """
    file_accessor = ToltecFileAccessor.from_sources(
        [{"source": source, "source_loc": source_loc}], mode=mode, on_error=on_error
    )
    return next(file_accessor.table.itertuples(index=False))


def guess_info_from_sources(
    sources: Iterable[Any], *, on_error: SourceErrorPolicyT = "record"
) -> pd.DataFrame:
    """Build a file table from filename information only.

    Parameters
    ----------
    sources : finite iterable
        Resources or source/location dicts; see `ToltecFileAccessor.from_sources`.
    on_error : {"raise", "record"}, optional
        Failure handling; see `ToltecFileAccessor.from_sources`. Default is "record".

    Returns
    -------
    pandas.DataFrame
        One row per source, including unrecognized filenames. No files are opened.
    """
    return ToltecFileAccessor.from_sources(
        sources, mode="guess", on_error=on_error
    ).table


def guess_or_read_info_from_sources(
    sources: Iterable[Any], *, on_error: SourceErrorPolicyT = "record"
) -> pd.DataFrame:
    """Build a file table, reading metadata for unrecognized filenames.

    Parameters
    ----------
    sources : finite iterable
        Resources or source/location dicts; see `ToltecFileAccessor.from_sources`.
    on_error : {"raise", "record"}, optional
        Failure handling; see `ToltecFileAccessor.from_sources`. Default is "record".

    Returns
    -------
    pandas.DataFrame
        Guessed information, supplemented with metadata for supported sources
        whose filenames are unrecognized. See `ToltecFileAccessor.from_sources`
        for resource ownership and read-error behavior.
    """
    return ToltecFileAccessor.from_sources(
        sources, mode="guess_or_read", on_error=on_error
    ).table


def read_info_from_sources(
    sources: Iterable[Any], *, on_error: SourceErrorPolicyT = "record"
) -> pd.DataFrame:
    """Build a file table with metadata reads for TolTEC and unidentified files.

    Parameters
    ----------
    sources : finite iterable
        Resources or source/location dicts; see `ToltecFileAccessor.from_sources`.
    on_error : {"raise", "record"}, optional
        Failure handling; see `ToltecFileAccessor.from_sources`. Default is "record".

    Returns
    -------
    pandas.DataFrame
        Guessed information updated with read metadata. Recognized foreign
        filenames are not opened. Unsupported sources retain their guesses.
        See `ToltecFileAccessor.from_sources` for resource ownership and
        read-error behavior.
    """
    return ToltecFileAccessor.from_sources(
        sources, mode="read", on_error=on_error
    ).table
