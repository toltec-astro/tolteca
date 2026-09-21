"""Enrich file records with scalar KIDs header metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tollan.utils.log import logger

from ..file import SourceInfoModel, guess_info_from_source
from ..toltecdb import ToltecDBRawObsMaster
from ..types import ToltecDataKind
from .core import ToltecKidsAccessor

if TYPE_CHECKING:
    import xarray as xr

    from ..file import FilepathInfo

__all__ = ["identify_source"]


def _get_filepath_fields(
    info: SourceInfoModel, filepath_orig: str | None
) -> tuple[dict, FilepathInfo | None]:
    """Fill discovery fields from filenames, preserving the current source."""
    fields = info.model_dump(round_trip=True, exclude={"source", "filepath_info"})
    fields["source"] = info.source
    fields["filepath_info"] = info.filepath_info
    fields["filepath_orig"] = filepath_orig
    guessed = guess_info_from_source(
        info.source, filepath_orig=filepath_orig, raise_on_error=False
    )
    if guessed is not None:
        for name, value in guessed.model_dump(
            round_trip=True, exclude={"source", "filepath_info"}
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


def identify_source(
    data: xr.Dataset,
    info: SourceInfoModel,
    *,
    check_filepath_orig: bool = False,
) -> SourceInfoModel:
    """Return a fresh discovery record enriched with KIDs header metadata.

    The recorded original filename supplies guesses when the current filename
    is unrecognized. Explicit header identifiers replace guesses, and missing
    headers leave them unchanged. A known header kind replaces conflicting
    guesses but retains a compatible, more specific kind. The current source
    location is preserved. With ``check_filepath_orig=True``, conflicting
    identifiers or master between explicit headers and the recorded original
    path raise ValueError. This consistency check is off by default. No signal
    arrays are read.
    """
    accessor = ToltecKidsAccessor(data)
    mapper = accessor.mapper
    filepath_orig = info.filepath_orig
    filename = mapper.schema.filename_orig
    if filename in mapper and mapper.get_name(filename) is not None:
        filepath_orig = mapper.get_str(data, filename)
    fields, filepath_info_orig = _get_filepath_fields(info, filepath_orig)
    for name in ("roach", "obsnum", "subobsnum", "scannum", "master"):
        mapping = getattr(mapper.schema, name)
        if mapping not in mapper or mapper.get_name(mapping) is None:
            continue
        value = mapper.get_scalar(data, mapping)
        if name == "master":
            value = ToltecDBRawObsMaster.get_master_type(ToltecDBRawObsMaster(value))
        if check_filepath_orig and filepath_info_orig is not None:
            value_orig = (
                filepath_info_orig.master_str
                if name == "master"
                else getattr(filepath_info_orig, name)
            )
            if value_orig is not None and value_orig != value:
                msg = (
                    f"Conflicting {name} for {info.source}: "
                    f"{mapper.get_name(mapping)}={value!r}, "
                    f"filepath_orig={fields['filepath_orig']!r} "
                    f"identifies {value_orig!r}"
                )
                raise ValueError(msg)
        if fields[name] != value:
            logger.debug(
                "Update {} for {} from {}: {!r} -> {!r}",
                name,
                info.source,
                mapper.get_name(mapping),
                fields[name],
                value,
            )
        fields[name] = value
    kind = accessor.data_kind
    if kind != ToltecDataKind.Unknown:
        # Keep a specific filename kind when the header only identifies a family.
        kind_guessed = fields["data_kind"]
        fields["data_kind"] = kind & kind_guessed or kind
        logger.debug(
            "Identify data_kind for {} from KIDs headers: {} -> {}",
            info.source,
            kind_guessed,
            fields["data_kind"],
        )
    if fields["roach"] is not None:
        fields["interface"] = f"toltec{fields['roach']}"
    return SourceInfoModel.model_validate(fields)
