"""TolTEC filename identification and file tables."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, cast

import pandas as pd
import xarray as xr
from tollan.accessor.pandas.model_table import (
    Contains,
    ExplicitColumnRule,
    ModelColumn,
    ModelColumnMapper,
    QuantityColumnRule,
    RecordRuleMapper,
    ScalarColumnRule,
    SkyCoordColumnRule,
    TimeColumnRule,
)
from tollan.accessor.xarray import unwrap_data_source

from ..core.file import FileAccessor, FileTableModeT, SourceErrorPolicyT
from ..core.source import SourceReaderRules
from ..lmt._obs_spec import _obs_spec_context as _lmt_obs_spec_context
from ..lmt.metadata import LmtTelMetadata
from ..lmt.tel import LmtTelAccessor, _get_tel_reader_record
from ..lmt_base.funcs import make_uid_columns
from ..lmt_base.obs_spec import ObsSpec
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
    from collections.abc import Iterable

    from tollan.utils.fileloc import FileLoc

    from ..core.source import SourceReader


__all__ = [
    "FileTableModeT",
    "ToltecFileAccessor",
    "get_info_from_source",
    "guess_info_from_sources",
    "guess_or_read_info_from_sources",
    "read_info_from_sources",
]


_metadata_column_mapper = ModelColumnMapper(
    name_rules=[(r".+", r"\g<0>")],
    type_rules=[
        ExplicitColumnRule(
            columns={
                "data_models": ModelColumn("data_models", dtype=object, converter=list),
            }
        ),
        QuantityColumnRule.model_validate({"physical_type": "time", "unit": "s"}),
        QuantityColumnRule.model_validate({"physical_type": "frequency", "unit": "Hz"}),
        TimeColumnRule(format="mjd"),
        SkyCoordColumnRule.model_validate({"attrs": ["ra", "dec"], "unit": "deg"}),
        ScalarColumnRule(),
    ],
    include_computed=True,
    default=None,
)


def _get_reader_record(data_source: object) -> object | None:
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


class _DataKindRule(NamedTuple):
    """Filename constraints; ellipsis leaves a value unconstrained."""

    prefix: object = ...
    interface: object = ...
    instru: object = ...
    interface_kind: object = ...
    suffix: object = ...
    proc_suffix: object = ...
    ext: object = ...


# Insertion order defines priority: the first matching rule wins.
_data_kind_mapper = RecordRuleMapper[int](
    rules=(
        (rule._asdict(), kind.value)
        for rule, kind in (
            # NetCDF suffixes identify raw and reduced sweep/timestream data.
            {
                _DataKindRule(
                    instru="toltec", interface_kind="roach", suffix=suffix, ext="nc"
                ): kind
                for suffix, kind in (
                    ("vnasweep", ToltecDataKind.vnasweep),
                    ("targsweep", ToltecDataKind.targsweep),
                    ("targetsweep", ToltecDataKind.targsweep),
                    ("tune", ToltecDataKind.tune),
                    ("vnasweep_processed", ToltecDataKind.reduced_vnasweep),
                    ("targsweep_processed", ToltecDataKind.reduced_targsweep),
                    ("tune_processed", ToltecDataKind.reduced_targsweep),
                    ("sweep", ToltecDataKind.reduced_sweep),
                    ("timestream_processed", ToltecDataKind.solved_timestream),
                    ("timestream", ToltecDataKind.raw_timestream),
                    (None, ToltecDataKind.raw_timestream),
                )
            }
            | {
                _DataKindRule(
                    instru="toltec", interface_kind="roach", suffix=suffix, ext="txt"
                ): ToltecDataKind.kids_model_params_table
                for suffix in ("vnasweep", "targsweep", "tune")
            }
            # Processing tags identify tables independently of the input sweep kind.
            | {
                _DataKindRule(
                    instru="toltec",
                    interface_kind="roach",
                    proc_suffix="targamps",
                    ext="dat",
                ): ToltecDataKind.targ_amps_dat,
            }
            | {
                _DataKindRule(
                    instru="toltec",
                    interface_kind="roach",
                    proc_suffix=proc_suffix,
                    ext="ecsv",
                ): ToltecDataKind.kids_prop_table
                for proc_suffix in ("kids_find", "kids_fit", "chan_prop", "adrv")
            }
            | {
                _DataKindRule(
                    instru="toltec", prefix="hwpr", ext="nc"
                ): ToltecDataKind.hwpr,
                _DataKindRule(
                    instru="toltec", prefix="toltec_hk", ext="nc"
                ): ToltecDataKind.housekeeping,
                _DataKindRule(
                    instru="toltec", prefix="wyatt", ext="nc"
                ): ToltecDataKind.wyatt,
                _DataKindRule(
                    instru="lmt", prefix="tel_toltec", ext="nc"
                ): ToltecDataKind.lmt_tel,
                _DataKindRule(
                    instru="lmt", prefix="tel_toltec2", ext="nc"
                ): ToltecDataKind.lmt_tel2,
            }
        ).items()
    ),
    default=ToltecDataKind.unknown.value,
)


@pd.api.extensions.register_dataframe_accessor("toltec_file")
class ToltecFileAccessor(FileAccessor[ToltecDataModel]):
    """Build, query, and open TolTEC file tables.

    Parameters
    ----------
    data : pandas.DataFrame
        File table, as returned by `from_sources`.

    Notes
    -----
    Queries accept ToltecDataKind member names such as ``raw_sweep`` without an
    import or ``.value``. Use ``bits_any_set(data_kind_toltec, raw_sweep)`` to select
    any raw sweep kind; combine flag constants with ``|``, ``&``, and ``~``.
    Header reads replace guessed observation identifiers and data kinds.
    Filename information uses all registered parsers. TolTEC classifications
    cover instrument files and telescope companions with prefixes ``tel_toltec``
    or ``tel_toltec2``. Other inputs retain their filename information with
    an unknown TolTEC data kind.
    Within an `open` scope, metadata reads reuse existing IO objects without
    closing them. Open objects may be Dataset or DataTree instances; local
    files are opened as xarray Datasets. Current membership and identity columns
    select TolTEC KIDs, HWPR or telescope readers. Without a matching rule,
    content identification accepts telescope data whose receiver header
    identifies TolTEC. Successful metadata reads correct guessed affiliation.
    Numeric interface selectors refer to ROACH numbers;
    telescope and auxiliary interface labels are unordered.

    Examples
    --------
    >>> table = guess_info_from_sources(
    ...     ["tel_toltec_2025-10-31_145609_00_0002.nc"]
    ... )
    >>> table.toltec_file.query('interface == "tel"')["interface"].tolist()
    ['tel']
    >>> table.toltec_file.query('bits_any_set(data_kind_toltec, lmt_tel)').shape[0]
    1

    For local KIDs and telescope files listed in ``paths``::

        table = guess_info_from_sources(paths)
        table = table.toltec_file.update_from_source_metadata()
        selected = table.toltec_file.query("atten_drive == 7.25")
        with table.toltec_file.open() as opened:
            selected = opened.toltec_file.query('interface == "tel"')
            az = [ds.lmt_tel.az.compute() for ds in selected.io_obj]
    """

    _data_kind_type: ClassVar = ToltecDataKind
    _data_kind_column: ClassVar = "data_kind_toltec"
    _data_kind_mapper: ClassVar = _data_kind_mapper
    _reader_rules: ClassVar = SourceReaderRules(
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
    _obs_spec_cls: ClassVar = ObsSpec
    _obs_spec_context: ClassVar = _lmt_obs_spec_context | _obs_spec_context

    _query_namespace: ClassVar[dict[str, object]] = (
        FileAccessor._query_namespace | ToltecDataKind.__members__
    )

    _metadata_columns: ClassVar[dict[str, ModelColumn]] = {
        name: column
        for model in (
            LmtTelMetadata,
            ToltecHwprMetadata,
            ToltecRawObsMetadata,
            ToltecTimestreamMetadata,
            ToltecRawObsBlockMetadata,
            ToltecSweepMetadata,
        )
        for name, column in _metadata_column_mapper(model).items()
    }

    @staticmethod
    def _guess_derived_columns(table: pd.DataFrame) -> pd.DataFrame:
        return make_uid_columns(
            table.master, table.obsnum, table.subobsnum, table.scannum, table.interface
        )

    @staticmethod
    def _classify_metadata(
        reader: SourceReader, metadata: object | None
    ) -> ToltecDataKind:
        if (kind := getattr(reader, "data_kind", None)) is not None:
            return kind
        # Preserve the existing telescope-variant classification from its header.
        return (
            ToltecDataKind.lmt_tel2
            if Path(cast("LmtTelMetadata", metadata).filepath_orig).name.startswith(
                "tel_toltec2_"
            )
            else ToltecDataKind.lmt_tel
        )


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
        File location or open xarray object.
    source_loc : str, pathlib.Path or FileLoc, optional
        Location used for filename identification. Defaults to ``source``;
        required when ``source`` cannot be converted to FileLoc. For example,
        a cached file can be identified by its remote location.
    mode : {"guess", "guess_or_read", "read"}, optional
        How to populate the row; see `ToltecFileAccessor.from_sources`.
        Default is "guess".
    on_error : {"raise", "record"}, optional
        Failure handling; see `FileAccessor.from_sources`. Default is "record".

    Returns
    -------
    namedtuple
        File information, including null values for unidentified information.
        Values are accessible by column name, such as ``row.obsnum``.
    """
    table = ToltecFileAccessor.from_sources(
        [{"source": source, "source_loc": source_loc}], mode=mode, on_error=on_error
    )
    return next(table.itertuples(index=False))


def guess_info_from_sources(
    sources: Iterable[Any], *, on_error: SourceErrorPolicyT = "record"
) -> pd.DataFrame:
    """Build a file table from filename information only.

    Parameters
    ----------
    sources : iterable
        Resources or source/location dicts; see `FileAccessor.from_sources`.
    on_error : {"raise", "record"}, optional
        Failure handling; see `FileAccessor.from_sources`. Default is "record".

    Returns
    -------
    pandas.DataFrame
        One row per source, including unrecognized filenames. No files are opened.
    """
    return ToltecFileAccessor.from_sources(sources, mode="guess", on_error=on_error)


def guess_or_read_info_from_sources(
    sources: Iterable[Any], *, on_error: SourceErrorPolicyT = "record"
) -> pd.DataFrame:
    """Build a file table, reading metadata for unrecognized filenames.

    Parameters
    ----------
    sources : iterable
        Resources or source/location dicts; see `FileAccessor.from_sources`.
    on_error : {"raise", "record"}, optional
        Failure handling; see `FileAccessor.from_sources`. Default is "record".

    Returns
    -------
    pandas.DataFrame
        Guessed information, supplemented with metadata for supported sources
        whose filenames are unrecognized. See `ToltecFileAccessor.from_sources`
        for resource ownership and read-error behavior.
    """
    return ToltecFileAccessor.from_sources(
        sources, mode="guess_or_read", on_error=on_error
    )


def read_info_from_sources(
    sources: Iterable[Any], *, on_error: SourceErrorPolicyT = "record"
) -> pd.DataFrame:
    """Build a file table with metadata reads for TolTEC and unidentified files.

    Parameters
    ----------
    sources : iterable
        Resources or source/location dicts; see `FileAccessor.from_sources`.
    on_error : {"raise", "record"}, optional
        Failure handling; see `FileAccessor.from_sources`. Default is "record".

    Returns
    -------
    pandas.DataFrame
        Guessed information updated with read metadata. Recognized foreign
        filenames are not opened. Unsupported sources retain their guesses.
        See `ToltecFileAccessor.from_sources` for resource ownership and
        read-error behavior.
    """
    return ToltecFileAccessor.from_sources(sources, mode="read", on_error=on_error)
