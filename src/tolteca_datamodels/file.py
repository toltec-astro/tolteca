"""Filename identification and in-memory acquisition file catalogs."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 -- Pydantic resolves computed-field types.
from functools import cache, cached_property, reduce
from operator import or_
from typing import TYPE_CHECKING, Annotated, ClassVar, NamedTuple

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import (
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

from .lmt_base.file import FilepathInfo
from .lmt_base.metadata import LmtRawObsMetadata
from .toltec.types import ToltecDataKind

if TYPE_CHECKING:
    import re
    from collections.abc import Callable, Iterable
    from enum import Enum
    from pathlib import Path
    from types import EllipsisType

    from pydantic import (
        SerializationInfo,
        SerializerFunctionWrapHandler,
        ValidationInfo,
    )

__all__ = [
    "FileIdentifier",
    "FilenameParser",
    "FilepathInfo",
    "FilepathInfoMatch",
    "SourceInfoModel",
    "ToltecFileAccessor",
    "guess_info_from_source",
    "guess_info_from_sources",
]


class FilepathInfoMatch(NamedTuple):
    """Field constraints for matching parsed path information.

    Use ``...`` to leave a field unconstrained. ``None`` matches an absent
    value; an empty string matches only an empty string.
    """

    prefix: str | EllipsisType | None = ...
    interface: str | EllipsisType | None = ...
    instru: str | EllipsisType | None = ...
    instru_component: str | EllipsisType | None = ...
    suffix: str | EllipsisType | None = ...
    proc_suffix: str | EllipsisType | None = ...
    ext: str | EllipsisType | None = ...


class FilenameParser(NamedTuple):
    """Record type and ordered parsers for one filename pattern.

    Each parser receives the current fields and source path, and returns a
    partial dictionary without modifying its inputs. Updates are applied before
    the next parser runs; later values replace earlier ones. Order parsers after
    any parsers supplying their required fields. Each runs once. Pydantic
    validates the completed record and converts ordinary numeric captures.
    """

    filepath_info_cls: type[FilepathInfo]
    parsers: tuple[Callable[[dict, Path], dict[str, object]], ...]


class FileIdentifier:
    """Identify local or remote sources from paths without opening files.

    Domain modules register filename patterns, ordered parsers, and
    classification rules on this class.
    """

    filename_pattern_parser: ClassVar[dict[re.Pattern[str], FilenameParser]] = {}
    # Insertion order defines priority: the first matching rule wins.
    filepath_info_data_kind: ClassVar[dict[FilepathInfoMatch, Enum]] = {}

    @classmethod
    def parse_filepath(
        cls,
        source: str | Path | FileLoc,
        *,
        info_cls: type[FilepathInfo] = FilepathInfo,
        raise_on_error: bool = True,
    ) -> FilepathInfo | None:
        """Parse a path into its registered filename-information model.

        Limit matching to subclasses of ``info_cls``. Unrecognized names raise
        ValueError, or return None with ``raise_on_error=False``. Invalid field
        values, including malformed dates, always raise. No file is opened.
        """
        source = fileloc(source)
        path = source.path
        for pattern, parser in cls.filename_pattern_parser.items():
            if not issubclass(parser.filepath_info_cls, info_cls):
                continue
            fields = dict_from_regex_match(pattern, path.name)
            if fields is None:
                continue
            for parse in parser.parsers:
                fields.update(parse(fields, path))
            return parser.filepath_info_cls.model_validate(fields)
        if not raise_on_error:
            return None
        msg = f"Unrecognized acquisition filename: {source}"
        raise ValueError(msg)

    @classmethod
    def guess_data_kinds(cls, fields: pd.DataFrame | pa.Table) -> list[Enum]:
        """Classify rows of filename-derived fields without opening files.

        Columns use the names in :class:`FilepathInfoMatch`; missing columns
        are treated as absent values. Extra columns are ignored. Return the
        first matching registered enum member for each row, in input row order, or
        ``ToltecDataKind.Unknown`` if no rule matches. All conditions operate on
        columns; matching does not short-circuit condition evaluation per row.

        Examples
        --------
        >>> fields = pd.DataFrame({"prefix": ["tel_toltec"], "ext": ["nc"]})
        >>> FileIdentifier.guess_data_kinds(fields) == [ToltecDataKind.LmtTel]
        True
        """
        size = len(fields)
        if not size:
            return []
        rules = tuple(cls.filepath_info_data_kind.items())
        if not rules:
            return [ToltecDataKind.Unknown] * size
        if isinstance(fields, pd.DataFrame):
            names = [name for name in FilepathInfoMatch._fields if name in fields]
            fields = pa.Table.from_pandas(fields[names], preserve_index=False)
        columns = {name: fields[name] for name in fields.column_names}
        comparisons = {}
        conditions = []
        condition_all = pa.repeat(value=True, size=size)
        column_absent = pa.scalar(None, type=pa.string())
        for key, _ in rules:
            condition = condition_all
            for name, value in zip(key._fields, key, strict=True):
                if value is ...:
                    continue
                # Share repeated column comparisons within this batch only.
                if (name, value) not in comparisons:
                    column = columns.get(name, column_absent)
                    # Arrow installs these compute functions dynamically.
                    comparisons[name, value] = (
                        pc.is_null(column)  # ty: ignore[unresolved-attribute]
                        if value is None
                        else pc.fill_null(
                            pc.equal(column, value),  # ty: ignore[unresolved-attribute]
                            fill_value=False,
                        )
                    )
                condition = pc.and_(condition, comparisons[name, value])  # ty: ignore[unresolved-attribute]
            conditions.append(condition)
        indices = pc.case_when(  # ty: ignore[unresolved-attribute]
            pc.make_struct(  # ty: ignore[unresolved-attribute]
                *conditions, field_names=[str(i) for i in range(len(rules))]
            ),
            *range(len(rules)),
            len(rules),
        )
        # Rule positions preserve enum identity even across enums with equal values.
        kinds = [kind for _, kind in rules]
        kinds.append(ToltecDataKind.Unknown)
        return [kinds[i] for i in indices.to_pylist()]

    @classmethod
    def guess_data_kind(cls, info: FilepathInfo) -> Enum:
        """Classify parsed filepath information without opening a file.

        Return the first matching registered enum member, or
        ``ToltecDataKind.Unknown`` when no rule matches. For multiple records, use
        :meth:`guess_data_kinds`. ROACH NetCDF and model-table files use the full
        ``suffix``; DAT/ECSV products use the parsed ``proc_suffix``, such as
        ``targamps`` or ``kids_find``. Other families use prefix and extension.
        """
        fields = pa.Table.from_pylist(
            [info.model_dump(include=set(FilepathInfoMatch._fields))]
        )
        return cls.guess_data_kinds(fields)[0]

    @classmethod
    def _guess_source_fields(
        cls,
        source: str | Path | FileLoc,
        *,
        filepath_orig: str | None = None,
        raise_on_error: bool = True,
    ) -> dict | None:
        """Collect source fields and effective path information without classifying."""
        source = fileloc(source)
        parsed = cls.parse_filepath(source, raise_on_error=False)
        parsed_orig = (
            cls.parse_filepath(filepath_orig, raise_on_error=False)
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
        return {
            "source": source,
            "filepath_orig": filepath_orig,
            "filepath_info": parsed,
            "interface": parsed.interface,
            "roach": getattr(parsed, "roach", None),
            "master": parsed.master_str,
            "obsnum": parsed.obsnum,
            "subobsnum": parsed.subobsnum,
            "scannum": parsed.scannum,
        }


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
            ("proc_suffix", pa.string()),
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


@cache
def _get_filepath_adapter() -> TypeAdapter[FilepathInfo]:
    models = dict.fromkeys(
        parser.filepath_info_cls
        for parser in FileIdentifier.filename_pattern_parser.values()
    )
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
    ToltecDataKind | str, tuple[type[LmtRawObsMetadata], TypeAdapter[LmtRawObsMetadata]]
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
    # Telescope metadata identifies the interface even when an unrecognized
    # original filename cannot distinguish the tel/tel2 stream kind.
    adapters["tel"] = adapters[ToltecDataKind.LmtTel]
    return adapters


class SourceInfoModel(BaseModel):
    """Serializable discovery metadata for one local or remote file.

    Missing identifiers remain ``None`` until supplied by filename or header
    metadata. This record contains neither open resources nor numerical arrays.
    ``metadata`` holds a telescope metadata record or raw KIDs ``meta_header``;
    ``data_kind`` determines its type when restoring serialized records. A
    telescope record with unknown stream kind is identified by ``interface``.

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
        key = (
            (info.data.get("interface") or kind)
            if kind is ToltecDataKind.Unknown
            else kind
        )
        try:
            metadata_cls, adapter = _get_metadata_adapters()[key]
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
    fields = FileIdentifier._guess_source_fields(
        source, filepath_orig=filepath_orig, raise_on_error=raise_on_error
    )
    if fields is None:
        return None
    fields["data_kind"] = FileIdentifier.guess_data_kind(fields["filepath_info"])
    return SourceInfoModel.model_validate(fields)


def guess_info_from_sources(sources: Iterable[str | Path | FileLoc]) -> pd.DataFrame:
    """Build a filename-only catalog, retaining unknown names for header discovery."""
    return ToltecFileAccessor.from_sources(sources)


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
    """Build, select, and reconstruct source records in a metadata-only catalog."""

    def __init__(self, data: pd.DataFrame) -> None:
        if "source" not in data:
            msg = "A file catalog requires a source column"
            raise AttributeError(msg)
        self._obj = data

    @classmethod
    def from_sources(cls, sources: Iterable[str | Path | FileLoc]) -> pd.DataFrame:
        """Build a filename-only catalog from local or remote paths.

        Classify the collection in one batch without opening files. Retain
        unrecognized names with unknown identifiers; invalid dates in recognized
        names raise ValueError. Accept generators and empty inputs.

        Examples
        --------
        >>> catalog = ToltecFileAccessor.from_sources(
        ...     ["tel_toltec_2025-10-31_145609_00_0002.nc"]
        ... )
        >>> catalog["interface"].tolist()
        ['tel']
        """
        rows = [
            FileIdentifier._guess_source_fields(source, raise_on_error=False)
            or {"source": fileloc(source)}
            for source in sources
        ]
        columns = pa.table(
            {
                name: pa.array(
                    [getattr(row.get("filepath_info"), name, None) for row in rows],
                    type=pa.string(),
                )
                for name in FilepathInfoMatch._fields
            }
        )
        kinds = FileIdentifier.guess_data_kinds(columns)
        return cls.from_info_list(
            SourceInfoModel.model_validate({**row, "data_kind": kind})
            for row, kind in zip(rows, kinds, strict=True)
        )

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


# Domain modules populate the registries after the shared declarations exist.
from .lmt import file as _lmt_file  # noqa: E402, F401
from .toltec import file as _toltec_file  # noqa: E402, F401
