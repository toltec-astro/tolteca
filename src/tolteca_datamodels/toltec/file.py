"""Source identification and in-memory TolTEC file catalogs."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from functools import cached_property
from typing import TYPE_CHECKING, Self

import pandas as pd
import pyarrow as pa
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    field_validator,
)
from tollan.utils.fileloc import FileLoc, fileloc
from tollan.utils.log import logger

from .types import ToltecDataKind

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from pydantic import SerializationInfo, SerializerFunctionWrapHandler

__all__ = [
    "FilepathInfo",
    "SourceInfoModel",
    "ToltecFileAccessor",
    "guess_info_from_source",
    "guess_info_from_sources",
]

_FILENAME_PATTERNS = (
    (
        re.compile(
            r"(?P<interface>tel_[a-z0-9]+|wyatt)_(?P<ut>\d{4}-\d{2}-\d{2})"
            r"_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)"
            r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>\w+)"
        ),
        "%Y-%m-%d",
    ),
    (
        re.compile(
            r"(?P<interface>toltec(?P<roach>\d+)|hwpr|tel_[a-z0-9]+|toltec_hk|toltec|hk)"
            r"_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)"
            r"(?:_(?P<ut>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}))?"
            r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>\w+)"
        ),
        "%Y_%m_%d_%H_%M_%S",
    ),
)

# Exact suffixes, grouped by extension. The empty suffix denotes no suffix.
_KIDS_DATA_KINDS = {
    "nc": {
        "vnasweep": ToltecDataKind.VnaSweep,
        "targsweep": ToltecDataKind.TargetSweep,
        "targetsweep": ToltecDataKind.TargetSweep,
        "tune": ToltecDataKind.Tune,
        "vnasweep_processed": ToltecDataKind.ReducedVnaSweep,
        "targsweep_processed": ToltecDataKind.ReducedTargetSweep,
        "tune_processed": ToltecDataKind.ReducedTargetSweep,
        "sweep": ToltecDataKind.ReducedSweep,
        "timestream_processed": ToltecDataKind.SolvedTimeStream,
        "timestream": ToltecDataKind.RawTimeStream,
        "": ToltecDataKind.RawTimeStream,
    },
    "txt": dict.fromkeys(
        ("vnasweep", "targsweep", "tune"), ToltecDataKind.KidsModelParamsTable
    ),
    "dat": dict.fromkeys(
        ("vnasweep_targamps", "targsweep_targamps", "tune_targamps"),
        ToltecDataKind.TargAmpsDat,
    ),
    "ecsv": dict.fromkeys(
        (
            f"{sweep}_{product}"
            for sweep in ("vnasweep", "targsweep", "tune")
            for product in ("kids_find", "kids_fit", "chan_prop", "adrv")
        ),
        ToltecDataKind.KidsPropTable,
    ),
}
_AUXILIARY_DATA_KINDS = {
    "tel_toltec2": ToltecDataKind.LmtTel2,
    "hwpr": ToltecDataKind.Hwpr,
    "hk": ToltecDataKind.HouseKeeping,
    "toltec_hk": ToltecDataKind.HouseKeeping,
    "wyatt": ToltecDataKind.Wyatt,
}
_TELESCOPE_INTERFACE = re.compile(r"tel_[a-z0-9]+")

_FILEPATH_INFO_DTYPE = pd.ArrowDtype(
    pa.struct(
        [
            ("interface", pa.string()),
            ("roach", pa.int64()),
            ("master_str", pa.string()),
            ("obsnum", pa.int64()),
            ("subobsnum", pa.int64()),
            ("scannum", pa.int64()),
            ("ut", pa.timestamp("us", tz="UTC")),
            ("suffix", pa.string()),
            ("ext", pa.string()),
        ]
    )
)

_FILEPATH_ALIASES = {
    f"file_{name}{qualifier}": (column, name)
    for column, qualifier in (("filepath_info", ""), ("filepath_info_orig", "_orig"))
    for name in _FILEPATH_INFO_DTYPE.pyarrow_dtype.names
}


class FilepathInfo(BaseModel):
    """Identifiers encoded in a TolTEC or telescope acquisition path.

    ``ut`` is the filename's UTC date/time, not filesystem modification time.
    ``interface`` is the filename token, such as ``tel_toltec``. The suffix and
    timestamp can be absent; ``ext`` does not include the leading dot.
    ``master_str`` identifies the acquisition master in the directory layout
    ``toltec/<master>/<interface>/<filename>`` (``tel`` for telescope files).
    Names need not have a database record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    interface: str
    roach: int | None = Field(ge=0, le=12)
    master_str: str | None
    obsnum: int = Field(ge=0)
    subobsnum: int = Field(ge=0)
    scannum: int = Field(ge=0)
    ut: AwareDatetime | None
    suffix: str | None
    ext: str

    @classmethod
    def from_path(
        cls, source: str | Path | FileLoc, *, raise_on_error: bool = True
    ) -> Self | None:
        """Parse an acquisition path without opening the local or remote source.

        An unrecognized name raises ValueError, or returns None when
        ``raise_on_error=False``. Invalid dates always raise ValueError.
        """
        source = fileloc(source)
        for pattern, ut_format in _FILENAME_PATTERNS:
            match = pattern.fullmatch(source.path.name)
            if match is not None:
                fields = match.groupdict()
                ut = fields.pop("ut")
                ut_parsed = (
                    datetime.strptime(ut, ut_format).replace(tzinfo=UTC)
                    if ut is not None
                    else None
                )
                break
        else:
            if not raise_on_error:
                return None
            msg = f"Unrecognized TolTEC filename: {source}"
            raise ValueError(msg)
        # Require the acquisition layout: an unrelated ancestor named "toltec"
        # does not make the following directory an acquisition master.
        master_str = None
        match source.path.parent.parts:
            case (*_, "toltec", master, interface) if interface in (
                fields["interface"],
                fields["interface"].partition("_")[0],
            ):
                master_str = master
        return cls.model_validate(
            {
                "roach": None,
                **fields,
                "ut": ut_parsed,
                "master_str": master_str,
            }
        )

    def guess_data_kind(self) -> ToltecDataKind:
        """Guess the data kind from interface, suffix, and extension."""
        if self.interface.startswith("toltec") and self.interface[6:].isdecimal():
            kinds = _KIDS_DATA_KINDS.get(self.ext)
            if kinds is not None:
                return kinds.get(self.suffix or "", ToltecDataKind.Unknown)
        elif self.ext == "nc":
            kind = _AUXILIARY_DATA_KINDS.get(self.interface)
            if kind is not None:
                return kind
            if _TELESCOPE_INTERFACE.fullmatch(self.interface):
                return ToltecDataKind.LmtTel
        return ToltecDataKind.Unknown


class SourceInfoModel(BaseModel):
    """Serializable discovery metadata for one local or remote file.

    Missing identifiers remain ``None`` until supplied by filename or header
    metadata. This record contains neither open resources nor numerical arrays.

    Pass ``context={"flat": True}`` to ``model_dump`` to serialize source
    locations as resolved strings and data kinds as names. The default retains
    FileLoc's structured representation. ``round_trip=True`` omits derived
    fields for subsequent model validation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    source: FileLoc
    filepath_orig: str | None = Field(
        default=None, description="Original acquisition path recorded in the header"
    )
    interface: str | None = None
    roach: int | None = Field(default=None, ge=0, le=12)
    master: str | None = Field(
        default=None,
        description="Acquisition master name, with or without a database record",
    )
    obsnum: int | None = Field(default=None, ge=0)
    subobsnum: int | None = Field(default=None, ge=0)
    scannum: int | None = Field(default=None, ge=0)
    filepath_info: FilepathInfo | None = Field(
        default=None,
        description="Effective path information from source identification",
    )
    data_kind: ToltecDataKind = ToltecDataKind.Unknown

    @computed_field
    @property
    def file_ut(self) -> datetime | None:
        """Return the effective filename's UTC date/time, if available."""
        return None if self.filepath_info is None else self.filepath_info.ut

    @computed_field
    @property
    def file_suffix(self) -> str | None:
        """Return the effective filename suffix, if available."""
        return None if self.filepath_info is None else self.filepath_info.suffix

    @computed_field
    @property
    def file_ext(self) -> str | None:
        """Return the effective filename extension without the dot, if available."""
        return None if self.filepath_info is None else self.filepath_info.ext

    @computed_field(repr=False)
    @cached_property
    def filepath_info_orig(self) -> FilepathInfo | None:
        """Parsed original acquisition path, or None if absent or unrecognized.

        The result is cached on this frozen record. It is never inferred from
        the current source location. Use ``round_trip=True`` when dumping a
        record for subsequent model validation, to omit computed fields.
        """
        if self.filepath_orig is None:
            return None
        return FilepathInfo.from_path(self.filepath_orig, raise_on_error=False)

    @property
    def filepath(self) -> Path:
        """Return the local filesystem path; reject remote locations."""
        if self.source.is_remote():
            msg = "A remote source has no local filepath; stage it before opening"
            raise ValueError(msg)
        return self.source.path

    @field_serializer("source", mode="wrap")
    def _serialize_source(
        self,
        value: FileLoc,
        handler: SerializerFunctionWrapHandler,
        info: SerializationInfo,
    ) -> object:
        if isinstance(info.context, dict) and info.context.get("flat"):
            return str(value.path) if value.is_local() else value.url
        return handler(value)

    @field_serializer("data_kind", mode="wrap")
    def _serialize_data_kind(
        self,
        value: ToltecDataKind,
        handler: SerializerFunctionWrapHandler,
        info: SerializationInfo,
    ) -> object:
        if isinstance(info.context, dict) and info.context.get("flat"):
            return value.name if value.name is not None else value.value
        return handler(value)

    @field_validator("data_kind", mode="before")
    @classmethod
    def _validate_data_kind(cls, value):
        # Catalog/config projections use names; normal JSON dumps use integers.
        if not isinstance(value, str):
            return value
        kind = ToltecDataKind(0)
        try:
            for name in value.split("|"):
                kind |= ToltecDataKind[name]
        except KeyError as exc:
            msg = f"Unknown data kind: {value}"
            raise ValueError(msg) from exc
        return kind


def guess_info_from_source(
    source: str | Path | FileLoc,
    *,
    filepath_orig: str | None = None,
    raise_on_error: bool = True,
) -> SourceInfoModel | None:
    """Identify a filename without opening or fetching the source.

    If the current filename is unrecognized, parse ``filepath_orig`` when
    supplied. The returned record retains the current source location and
    records the original path separately. Recognizable current filenames take
    precedence, including their suffix and timestamp. A master identified by
    the original acquisition path takes precedence over the current directory.
    """
    source = fileloc(source)
    parsed = FilepathInfo.from_path(source, raise_on_error=False)
    parsed_orig = (
        FilepathInfo.from_path(filepath_orig, raise_on_error=False)
        if filepath_orig is not None
        else None
    )
    if parsed is None and parsed_orig is not None:
        parsed = parsed_orig
        logger.debug(
            "Identify unrecognized filename {} using recorded path {}",
            source,
            filepath_orig,
        )
    if parsed is None:
        if not raise_on_error:
            return None
        msg = f"Unrecognized TolTEC filename: {source}"
        raise ValueError(msg)
    if parsed_orig is not None and parsed_orig.master_str is not None:
        parsed = parsed.model_copy(update={"master_str": parsed_orig.master_str})
    return SourceInfoModel(
        source=source,
        filepath_orig=filepath_orig,
        filepath_info=parsed,
        interface=parsed.interface,
        roach=parsed.roach,
        master=parsed.master_str,
        obsnum=parsed.obsnum,
        subobsnum=parsed.subobsnum,
        scannum=parsed.scannum,
        data_kind=parsed.guess_data_kind(),
    )


def guess_info_from_sources(sources: Iterable[str | Path | FileLoc]) -> pd.DataFrame:
    """Build a filename-only catalog, retaining unknown names for header discovery."""
    return ToltecFileAccessor.from_info_list(
        guess_info_from_source(source, raise_on_error=False)
        or SourceInfoModel(source=fileloc(source))
        for source in sources
    )


class _StructFields:
    """Resolve attribute names to Arrow child columns on the current frame."""

    def __init__(self, series: pd.Series) -> None:
        self._series = series

    def __getattr__(self, name: str) -> pd.Series | _StructFields:
        dtype = self._series.dtype
        assert isinstance(dtype, pd.ArrowDtype)
        if name.startswith("_") or name not in dtype.pyarrow_dtype.names:
            raise AttributeError(name)
        field = self._series.struct.field(name)
        dtype = field.dtype
        if isinstance(dtype, pd.ArrowDtype) and pa.types.is_struct(dtype.pyarrow_dtype):
            return _StructFields(field)
        return field


class _FilepathAliases(dict):
    """Project a path field only when its shorthand is used in a query."""

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__()
        self._data = data

    def __missing__(self, name: str) -> pd.Series:
        column, field = _FILEPATH_ALIASES[name]
        # Pandas rewrites resolved entries during evaluation. Keep those values
        # in this query-local dictionary, never on the catalog accessor.
        self[name] = value = self._data[column].struct.field(field)
        return value

    def __contains__(self, name: object) -> bool:
        try:
            self[name]
        except KeyError:
            return False
        return True


@pd.api.extensions.register_dataframe_accessor("toltec_file")
class ToltecFileAccessor:
    """Select and reconstruct source records from a metadata-only catalog."""

    def __init__(self, data: pd.DataFrame) -> None:
        if "source" not in data:
            msg = "A file catalog requires a source column"
            raise AttributeError(msg)
        self._obj = data

    @staticmethod
    def from_info_list(records: Iterable[SourceInfoModel]) -> pd.DataFrame:
        """Build a file catalog from source records without opening files.

        Accept any iterable, including an empty one. Parsed path information
        uses Arrow struct columns with UTC timestamps. Source locations become
        resolved local paths or remote URLs; data kinds use their enum names.
        Select rows with ``catalog.toltec_file.query(expression)`` and restore
        models with ``catalog.toltec_file.to_info_list()``.
        """
        columns = [*SourceInfoModel.model_fields, "filepath_info_orig"]
        rows = [
            record.model_dump(
                context={"flat": True},
                exclude={"file_ut", "file_suffix", "file_ext"},
            )
            for record in records
        ]
        catalog = pd.DataFrame.from_records(rows, columns=columns)
        for name in ("filepath_info", "filepath_info_orig"):
            catalog[name] = pd.Series(catalog[name], dtype=_FILEPATH_INFO_DTYPE)
        return catalog

    def query(self, expression: str) -> pd.DataFrame:
        """Select rows using pandas expressions, including dotted struct fields.

        For example, ``file_suffix_orig in ['vnasweep', 'tune']`` selects by
        the original filename. ``filepath_info.suffix`` and
        ``filepath_info_orig.suffix`` also accept the aliases ``file_suffix``
        and ``file_suffix_orig``. The same rule applies to every path field.
        Comparisons, membership, and nulls follow pandas query semantics.
        Only evaluate trusted expressions.
        """
        resolvers = {
            name: _StructFields(self._obj[name])
            for name, dtype in self._obj.dtypes.items()
            if isinstance(dtype, pd.ArrowDtype)
            and pa.types.is_struct(dtype.pyarrow_dtype)
        }
        return self._obj.query(
            expression,
            engine="python",
            resolvers=[resolvers, _FilepathAliases(self._obj)],
        )

    def to_info_list(self, query: str | None = None) -> list[SourceInfoModel]:
        """Return records, optionally selected with a pandas query expression."""
        selected = self._obj if query is None else self.query(query)
        # Only the effective path struct is an input; the original is computed.
        columns = [key for key in SourceInfoModel.model_fields if key in selected]
        result = []
        for row in selected[columns].to_dict(orient="records"):
            parsed = row.pop("filepath_info", None)
            fields = {
                key: None if pd.isna(value) else value for key, value in row.items()
            }
            if isinstance(parsed, dict):
                fields["filepath_info"] = {
                    key: parsed[key] for key in FilepathInfo.model_fields
                }
            result.append(SourceInfoModel.model_validate(fields))
        return result
