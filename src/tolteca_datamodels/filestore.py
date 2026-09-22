"""Shared file-access configuration and source identification."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import TYPE_CHECKING, Literal

from pydantic import ConfigDict
from tollan.config import ConfigHandler, FrozenBaseModel, SubConfigKeyTransformer
from tollan.config.types import AbsAnyPath, AbsDirectoryPath
from tollan.utils.log import logger

from .file import SourceInfoModel, _get_filepath_fields, guess_data_kind
from .lmt.tel import LmtTelIOMapper
from .toltec.kids.source import identify_source as identify_kids_source
from .toltec.types import ToltecDataKind

if TYPE_CHECKING:
    import xarray as xr

__all__ = ["FileAccess", "FileAccessConfig", "HeaderReadMode", "identify_source"]


class HeaderReadMode(StrEnum):
    """Choose when a file catalog opens sources to identify their metadata.

    ``never`` uses filename guesses only. ``fallback`` opens sources whose
    current filename is unrecognized. ``always`` opens all supported candidates,
    including recognized filenames, to read authoritative header identifiers.
    """

    never = auto()
    fallback = auto()
    always = auto()


def identify_source(
    data: xr.Dataset,
    info: SourceInfoModel,
    *,
    check_filepath_orig: bool = False,
) -> SourceInfoModel:
    """Read source identifiers from an open telescope or KIDs dataset.

    Return a new source record, preserving its current location. Telescope
    structure is recognized through its timestamp mapping; its strict metadata
    supplies observation identifiers and the recorded path. This reads scalar
    headers, two timestamp endpoints and the two target-coordinate pairs, not
    pointing streams. KIDs identification reads only its scalar headers.

    Recognized but incomplete telescope files raise their metadata errors.
    Unsupported data retains filename guesses. Set ``check_filepath_orig=True``
    to compare header identifiers with the recorded acquisition path. It does
    not compare the current filename or filesystem timestamps.
    """
    mapper = LmtTelIOMapper.from_data_source(data)
    if not mapper.has_arr(data, mapper.schema.tel_time):
        return identify_kids_source(data, info, check_filepath_orig=check_filepath_orig)
    meta = mapper.get_metadata(data, check_filepath_orig=check_filepath_orig)
    fields, filepath_info_orig = _get_filepath_fields(info, meta.filepath_orig)
    fields.update(
        interface=meta.interface,
        roach=None,
        master=meta.master_str,
        obsnum=meta.obsnum,
        subobsnum=meta.subobsnum,
        scannum=meta.scannum,
    )
    # The recorded prefix distinguishes the telescope stream. A time variable
    # alone cannot tell which stream produced it.
    kind = (
        guess_data_kind(prefix=filepath_info_orig.prefix, ext=filepath_info_orig.ext)
        if filepath_info_orig is not None
        else ToltecDataKind.Unknown
    )
    if kind == ToltecDataKind.Unknown:
        kind = info.data_kind
    fields["data_kind"] = (
        kind & (ToltecDataKind.LmtTel | ToltecDataKind.LmtTel2)
        or ToltecDataKind.Unknown
    )
    logger.debug(
        "Identified telescope source {}: uid_raw_obs_file={}, filepath_orig={}",
        info.source,
        meta.uid_raw_obs_file,
        meta.filepath_orig,
    )
    return SourceInfoModel.model_validate(fields)


class FileAccessConfig(FrozenBaseModel):
    """Paths and catalog selection shared by file-based applications.

    Explicit input files do not require ``data_lmt_path``. A product root need
    not exist yet; validation does not create it or enable product output.
    ``select`` is a pandas expression evaluated on the input metadata catalog.
    """

    model_config = ConfigDict(extra="forbid")
    data_lmt_path: AbsDirectoryPath | None = None
    dataprod_toltec_path: AbsAnyPath | None = None
    select: str | None = None


class FileAccess(
    SubConfigKeyTransformer[Literal["files"]], ConfigHandler[FileAccessConfig]
):
    """Read shared file-access settings from a runtime's ``files`` section.

    Examples
    --------
    >>> from tollan.config import RuntimeContext
    >>> files = FileAccess(RuntimeContext({"files": {"select": "roach < 7"}}))
    >>> files.config.select
    'roach < 7'

    Configure these settings unconditionally: discovery and selection happen
    before per-input configuration is evaluated.
    """
