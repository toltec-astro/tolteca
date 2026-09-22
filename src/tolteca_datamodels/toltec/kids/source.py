"""Identify KIDs source records from scalar headers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tollan.utils.log import logger

from ...file import SourceInfoModel, _get_filepath_fields
from ..toltecdb import ToltecDBRawObsMaster
from ..types import ToltecDataKind
from .core import ToltecKidsAccessor

if TYPE_CHECKING:
    import xarray as xr

__all__ = ["identify_source"]


def identify_source(  # noqa: C901 -- both read modes share one identifier/consistency policy
    data: xr.Dataset,
    info: SourceInfoModel,
    *,
    read_metadata: bool = False,
    check_filepath_orig: bool = False,
) -> SourceInfoModel:
    """Return a fresh discovery record with KIDs header identifiers.

    The recorded original filename supplies guesses when the current filename
    is unrecognized. Explicit header identifiers replace guesses, and missing
    headers leave them unchanged. A known header kind replaces conflicting
    guesses but retains a compatible, more specific kind. The current source
    location is preserved. With ``check_filepath_orig=True``, conflicting
    identifiers or master between explicit headers and the recorded original
    path raise ValueError. This consistency check is off by default. No signal
    arrays are read. By default only identifiers are read and any previous
    metadata snapshot is cleared. With ``read_metadata=True``, raw acquisitions
    require and include complete ``meta_header`` records. Processed and unknown
    inputs still receive identifiers only.
    """
    accessor = ToltecKidsAccessor(data)
    mapper = accessor.mapper
    kind = accessor.data_kind
    meta = (
        accessor.meta_header
        if read_metadata and kind & ToltecDataKind.RawKidsData
        else None
    )
    filepath_orig = info.filepath_orig
    filepath = mapper.schema.filepath_orig
    if meta is not None:
        filepath_orig = meta.filepath_orig
    elif filepath in mapper and mapper.get_name(filepath) is not None:
        filepath_orig = mapper.get_str(data, filepath)
    fields, filepath_info_orig = _get_filepath_fields(info, filepath_orig)
    fields["metadata"] = meta
    for name in ("roach", "obsnum", "subobsnum", "scannum", "master"):
        mapping = getattr(mapper.schema, name)
        if mapping not in mapper or mapper.get_name(mapping) is None:
            continue
        value = (
            getattr(meta, name)
            if meta is not None
            else mapper.get_scalar(data, mapping)
        )
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
    if meta is not None:
        logger.debug(
            "Read KIDs header metadata for {}: uid_raw_obs_file={}",
            info.source,
            meta.uid_raw_obs_file,
        )
    return SourceInfoModel.model_validate(fields)
