"""Filename identification and in-memory acquisition file catalogs."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import cache, cached_property, reduce
from operator import or_
from typing import TYPE_CHECKING, Annotated, ClassVar, NamedTuple

import pandas as pd
import pyarrow as pa
from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SerializeAsAny,
    TypeAdapter,
    WrapSerializer,
    computed_field,
    field_serializer,
    field_validator,
    model_serializer,
)
from tollan.utils.dict import dict_from_regex_match
from tollan.utils.fileloc import FileLoc, fileloc
from tollan.utils.log import logger

from .lmt_base.metadata import LmtRawObsMetadata
from .toltec.types import ToltecDataKind

if TYPE_CHECKING:
    import re
    from collections.abc import Callable, Iterable
    from pathlib import Path
    from types import EllipsisType

    from pydantic import (
        SerializationInfo,
        SerializerFunctionWrapHandler,
        ValidationInfo,
    )

    from .lmt_base.metadata import LmtInstruMixin

__all__ = [
    "DataKindKey",
    "FilepathInfo",
    "SourceInfoModel",
    "ToltecFileAccessor",
    "guess_data_kind",
    "guess_info_from_source",
    "guess_info_from_sources",
]


class DataKindKey(NamedTuple):
    """A normalized dictionary key for source classification.

    Use ``...`` for ignored fields on both the table and lookup keys.
    ``None`` represents an absent value, not an ignored field.
    """

    prefix: str | EllipsisType | None = ...
    interface: str | EllipsisType | None = ...
    instru: str | EllipsisType | None = ...
    instru_component: str | EllipsisType | None = ...
    suffix: str | EllipsisType | None = ...
    ext: str | EllipsisType | None = ...


# Domain modules populate these tables below, after the shared models exist.
filename_patterns: dict[re.Pattern[str], type[FilepathInfo]] = {}
filename_field_types: dict[str, Callable[[str], object]] = {}
prefix_interface: dict[str, str] = {}
data_kind_map: dict[DataKindKey, ToltecDataKind] = {}
# Retain these fields in the lookup key for each instrument/component pair.
# Other families use (prefix, ext); ignored positions are always Ellipsis.
data_kind_fields: dict[tuple[str | None, str | None], tuple[str, ...]] = {}


_FILEPATH_INFO_DTYPE = pd.ArrowDtype(
    pa.struct(
        [
            ("prefix", pa.string()),
            ("interface", pa.string()),
            ("instru", pa.string()),
            ("instru_component", pa.string()),
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


def _validate_data_kind(value):
    # Catalog/config projections use names; normal JSON dumps use integers.
    if not isinstance(value, str):
        return value
    # Arrow string fields represent the unnamed zero flag as "0".
    if value == "0":
        return ToltecDataKind(0)
    kind = ToltecDataKind(0)
    try:
        for name in value.split("|"):
            kind |= ToltecDataKind[name]
    except KeyError as exc:
        msg = f"Unknown data kind: {value}"
        raise ValueError(msg) from exc
    return kind


def _serialize_data_kind(
    value: ToltecDataKind,
    handler: SerializerFunctionWrapHandler,
    info: SerializationInfo,
) -> object:
    if isinstance(info.context, dict) and info.context.get("flat"):
        return value.name if value.name is not None else value.value
    return handler(value)


type _DataKindField = Annotated[
    ToltecDataKind,
    BeforeValidator(_validate_data_kind),
    WrapSerializer(_serialize_data_kind),
]


class FilepathInfo(BaseModel):
    """Identifiers encoded in an acquisition path, independent of file contents.

    ``prefix`` preserves the literal filename prefix. ``ut`` is the writer's
    file-creation time encoded in the filename, in UTC at the filename's
    precision. It is independent of observation time and filesystem mtime.
    The suffix may be absent; ``ext`` excludes the leading dot.
    All derived fields are stored when the record is constructed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    @staticmethod
    def _parse_ut(value: str) -> datetime:
        fmt = "%Y_%m_%d_%H_%M_%S" if "_" in value else "%Y-%m-%d"
        return datetime.strptime(value, fmt).replace(tzinfo=UTC)

    prefix: str
    interface: str
    instru: str | None
    instru_component: str | None
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
    ) -> FilepathInfo | None:
        """Parse a local or remote path without opening the source.

        Return the registered record type with all filename-derived fields
        populated. Calling on a subclass limits parsing to that record type.
        Unrecognized names raise ValueError, or return None when
        ``raise_on_error=False``. Invalid dates always raise ValueError.

        Examples
        --------
        >>> info = FilepathInfo.from_path("tel_toltec_2025-10-31_145609_00_0002.nc")
        >>> info.prefix, info.interface
        ('tel_toltec', 'tel')
        """
        source = fileloc(source)
        path = source.path
        for pattern, info_cls in filename_patterns.items():
            if not issubclass(info_cls, cls):
                continue
            fields = dict_from_regex_match(pattern, path.name, filename_field_types)
            if fields is not None:
                return info_cls.model_validate(cls._derive_fields(path, fields))
        if not raise_on_error:
            return None
        msg = f"Unrecognized acquisition filename: {source}"
        raise ValueError(msg)

    @staticmethod
    def _derive_fields(path: Path, fields: dict) -> dict:
        """Complete parsed fields before constructing the record."""
        prefix = fields["prefix"]
        interface = fields["interface"] = prefix_interface.get(prefix, prefix)
        # Numeric ROACH captures recognize alternate numeric spellings without
        # changing the literal filename prefix or interface.
        interface_key = ToltecInfo.roach_interface.get(fields.get("roach"), interface)
        identity = interface_instru.get(interface_key)
        fields.update(
            instru=None if identity is None else identity.instru,
            instru_component=None if identity is None else identity.instru_component,
            master_str=None,
        )
        # Only the acquisition directory layout supplies a master. An arbitrary
        # ancestor named "toltec" is not sufficient evidence.
        match path.parent.parts:
            case (*_, "toltec", master, directory_interface) if directory_interface in (
                prefix,
                prefix.partition("_")[0],
            ):
                fields["master_str"] = master
        return fields


def guess_data_kind(  # noqa: PLR0913 -- one argument per classification key field
    *,
    prefix: str | None = None,
    interface: str | None = None,
    instru: str | None = None,
    instru_component: str | None = None,
    suffix: str | None = None,
    ext: str | None = None,
) -> ToltecDataKind:
    """Classify source metadata without opening a file.

    Return ``ToltecDataKind.Unknown`` when the supplied identifiers have no
    registered classification. For ROACH data, instrument, component, suffix,
    and extension determine the kind; other families use prefix and extension.
    """
    fields = {
        "prefix": prefix,
        "interface": interface,
        "instru": instru,
        "instru_component": instru_component,
        "suffix": suffix,
        "ext": ext,
    }
    names = data_kind_fields.get((instru, instru_component), ("prefix", "ext"))
    key = DataKindKey(**{name: fields[name] for name in names})
    return data_kind_map.get(key, ToltecDataKind.Unknown)


@cache
def _get_filepath_adapter() -> TypeAdapter[FilepathInfo]:
    models = dict.fromkeys(filename_patterns.values())
    return TypeAdapter(reduce(or_, models))


def _restore_filepath_info(value: object) -> object:
    if isinstance(value, FilepathInfo):
        return value
    return _get_filepath_adapter().validate_python(value)


type _FilepathInfoField = Annotated[
    SerializeAsAny[FilepathInfo], BeforeValidator(_restore_filepath_info)
]


@cache
def _get_metadata_adapters() -> dict[
    ToltecDataKind, tuple[type[LmtRawObsMetadata], TypeAdapter[LmtRawObsMetadata]]
]:
    # Resolve domain classes after the root file/reader imports have completed.
    from .lmt.metadata import LmtTelMetadata  # noqa: PLC0415
    from .toltec.metadata import (  # noqa: PLC0415
        ToltecSweepMetadataHeader,
        ToltecTimeStreamMetadata,
    )

    models: tuple[tuple[ToltecDataKind, type[LmtRawObsMetadata]], ...] = (
        (ToltecDataKind.RawSweep, ToltecSweepMetadataHeader),
        (ToltecDataKind.RawTimeStream, ToltecTimeStreamMetadata),
        (ToltecDataKind.LmtTel | ToltecDataKind.LmtTel2, LmtTelMetadata),
    )
    adapters = {}
    for kinds, cls in models:
        adapter = cls, TypeAdapter[LmtRawObsMetadata](cls)
        adapters.update((kind, adapter) for kind in (*kinds, kinds))
    return adapters


class SourceInfoModel(BaseModel):
    """Serializable discovery metadata for one local or remote file.

    Missing identifiers remain ``None`` until supplied by filename or header
    metadata. This record contains neither open resources nor numerical arrays.
    ``metadata`` holds a telescope metadata record or raw KIDs ``meta_header``;
    ``data_kind`` determines its type when restoring serialized records.

    Pass ``context={"flat": True}`` to ``model_dump`` to serialize source
    locations as resolved strings and data kinds as names, and expose scalar
    metadata fields as top-level aliases. Existing source fields take precedence
    over metadata aliases. The default retains FileLoc's structured
    representation. ``round_trip=True`` omits aliases and computed properties;
    stored filename fields and typed metadata are retained.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    source: FileLoc
    filepath_orig: str | None = Field(
        default=None, description="Original acquisition path recorded in the header"
    )
    interface: str | None = None
    roach: int | None = None
    master: str | None = Field(
        default=None,
        description="Acquisition master name, with or without a database record",
    )
    obsnum: int | None = Field(default=None, ge=0)
    subobsnum: int | None = Field(default=None, ge=0)
    scannum: int | None = Field(default=None, ge=0)
    filepath_info: _FilepathInfoField | None = Field(
        default=None,
        description="Effective path information from source identification",
    )
    # Field validation uses this discriminator before restoring metadata.
    data_kind: _DataKindField = ToltecDataKind.Unknown
    metadata: SerializeAsAny[LmtRawObsMetadata] | None = Field(
        default=None,
        description="Typed acquisition headers; absent for filename-only discovery",
    )

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object, info: ValidationInfo) -> object:
        if value is None:
            return None
        kind = info.data.get("data_kind", ToltecDataKind.Unknown)
        try:
            metadata_cls, adapter = _get_metadata_adapters()[kind]
        except KeyError as exc:
            msg = f"No acquisition metadata model for {kind}"
            raise ValueError(msg) from exc
        if isinstance(value, LmtRawObsMetadata) and type(value) is not metadata_cls:
            msg = f"Source metadata for {kind} requires {metadata_cls.__name__}"
            raise ValueError(msg)
        # Serialized dataclasses are dictionaries, not strict Python instances.
        return adapter.validate_python(value, strict=False)

    @model_serializer(mode="wrap")
    def _serialize_record(  # noqa: ANN202 -- preserve the wrapped model's JSON schema
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ):
        # Leave the return schema with the wrapped model; only flat projections
        # add aliases, while normal serialization retains the declared fields.
        result = handler(self)
        if (
            not info.round_trip
            and isinstance(info.context, dict)
            and info.context.get("flat")
        ):
            for name, value in (result.get("metadata") or {}).items():
                if name not in result and not isinstance(value, (dict, list)):
                    result[name] = value
        return result

    @computed_field
    @property
    def file_prefix(self) -> str | None:
        """Return the effective filename prefix, if available."""
        return None if self.filepath_info is None else self.filepath_info.prefix

    @computed_field
    @property
    def file_ut(self) -> datetime | None:
        """Writer's file-creation time encoded in the effective filename, in UTC."""
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
    def filepath_info_orig(self) -> SerializeAsAny[FilepathInfo] | None:
        """Parsed original acquisition path, or None if absent or unrecognized."""
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
        msg = f"Unrecognized acquisition filename: {source}"
        raise ValueError(msg)
    if parsed_orig is not None and parsed_orig.master_str is not None:
        parsed = parsed.model_copy(update={"master_str": parsed_orig.master_str})
    return SourceInfoModel(
        source=source,
        filepath_orig=filepath_orig,
        filepath_info=parsed,
        interface=parsed.interface,
        roach=getattr(parsed, "roach", None),
        master=parsed.master_str,
        obsnum=parsed.obsnum,
        subobsnum=parsed.subobsnum,
        scannum=parsed.scannum,
        data_kind=guess_data_kind(
            prefix=parsed.prefix,
            interface=parsed.interface,
            instru=parsed.instru,
            instru_component=parsed.instru_component,
            suffix=parsed.suffix,
            ext=parsed.ext,
        ),
    )


def guess_info_from_sources(sources: Iterable[str | Path | FileLoc]) -> pd.DataFrame:
    """Build a filename-only catalog, retaining unknown names for header discovery."""
    return ToltecFileAccessor.from_info_list(
        guess_info_from_source(source, raise_on_error=False)
        or SourceInfoModel(source=fileloc(source))
        for source in sources
    )


def _get_filepath_fields(
    info: SourceInfoModel, filepath_orig: str | None
) -> tuple[dict, FilepathInfo | None]:
    """Fill discovery fields from filenames, preserving the current source."""
    fields = info.model_dump(
        round_trip=True, exclude={"source", "filepath_info", "metadata"}
    )
    fields["source"] = info.source
    fields["filepath_info"] = info.filepath_info
    fields["metadata"] = info.metadata
    fields["filepath_orig"] = filepath_orig
    guessed = guess_info_from_source(
        info.source, filepath_orig=filepath_orig, raise_on_error=False
    )
    if guessed is not None:
        for name, value in guessed.model_dump(
            round_trip=True, exclude={"source", "filepath_info", "metadata"}
        ).items():
            if fields[name] is None or fields[name] is ToltecDataKind.Unknown:
                fields[name] = value
        if fields["filepath_info"] is None:
            fields["filepath_info"] = guessed.filepath_info
    filepath_info_orig = guessed.filepath_info_orig if guessed is not None else None
    if filepath_info_orig is not None and filepath_info_orig.master_str is not None:
        fields["master"] = filepath_info_orig.master_str
        if fields["filepath_info"] is not None:
            fields["filepath_info"] = fields["filepath_info"].model_copy(
                update={"master_str": filepath_info_orig.master_str}
            )
    return fields, filepath_info_orig


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


class _CatalogAliases(dict):
    """Project a struct field only when its shorthand is used in a query."""

    _filepath_aliases: ClassVar = {
        f"file_{name}{qualifier}": (column, name)
        for column, qualifier in (
            ("filepath_info", ""),
            ("filepath_info_orig", "_orig"),
        )
        for name in _FILEPATH_INFO_DTYPE.pyarrow_dtype.names
    }

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__()
        self._data = data
        self._aliases = self._filepath_aliases.copy()
        if "metadata" in data:
            dtype = data["metadata"].dtype
            if isinstance(dtype, pd.ArrowDtype) and pa.types.is_struct(
                dtype.pyarrow_dtype
            ):
                self._aliases.update(
                    (field.name, ("metadata", field.name))
                    for field in dtype.pyarrow_dtype
                    if field.name not in data
                    and field.name not in self._aliases
                    and not pa.types.is_nested(field.type)
                )

    def __missing__(self, name: str) -> pd.Series:
        column, field = self._aliases[name]
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
        and acquisition metadata use Arrow struct columns. Filename timestamps
        use UTC; observation timestamps use Unix seconds. Quantities and sky
        coordinates retain their model's parseable string representation.
        Source locations become resolved paths or URLs; data kinds use names.
        Select rows with ``catalog.toltec_file.query(expression)`` and restore
        models with ``catalog.toltec_file.to_info_list()``.
        """
        columns = [*SourceInfoModel.model_fields, "filepath_info_orig"]
        rows = [
            record.model_dump(
                context={"flat": True},
                exclude={"file_prefix", "file_ut", "file_suffix", "file_ext"},
            )
            for record in records
        ]
        catalog = pd.DataFrame.from_records(rows, columns=columns)
        for name in ("filepath_info", "filepath_info_orig"):
            catalog[name] = pd.Series(catalog[name], dtype=_FILEPATH_INFO_DTYPE)
        metadata = pa.array(catalog["metadata"].tolist())
        catalog["metadata"] = pd.Series(metadata, dtype=pd.ArrowDtype(metadata.type))
        return catalog

    def query(self, expression: str) -> pd.DataFrame:
        """Select rows using pandas expressions, including dotted struct fields.

        For example, ``file_suffix_orig in ['vnasweep', 'tune']`` selects by
        the original filename. ``filepath_info.suffix`` and
        ``filepath_info_orig.suffix`` also accept the aliases ``file_suffix``
        and ``file_suffix_orig``. The same rule applies to every path field.
        Acquisition fields also support ``metadata.atten_drive`` or the scalar
        shorthand ``atten_drive``. Source columns and filename aliases take
        precedence over metadata shorthands; ``metadata.master`` retains the
        raw code while ``master`` is the source's readable label. Metadata
        fields are available only when records include that metadata type.
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
            resolvers=[resolvers, _CatalogAliases(self._obj)],
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
                # Arrow pads subtype-only fields with null in other rows.
                fields["filepath_info"] = {
                    key: value
                    for key, value in parsed.items()
                    if key in FilepathInfo.model_fields or value is not None
                }
            result.append(SourceInfoModel.model_validate(fields))
        return result


filename_field_types.update(
    obsnum=int, subobsnum=int, scannum=int, ut=FilepathInfo._parse_ut
)

# Readers and metadata import the shared models above. Import domain modules
# only after defining them; each module registers its declarations exactly once.
from .lmt import file as _lmt_file  # noqa: E402, F401
from .lmt.types import LmtInfo  # noqa: E402
from .toltec import file as _toltec_file  # noqa: E402, F401
from .toltec.types import ToltecInfo  # noqa: E402

interface_instru: dict[str, LmtInstruMixin] = {
    **LmtInfo.interface_instru,
    **ToltecInfo.interface_instru,
}
