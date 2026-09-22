"""Register TolTEC acquisition filenames and source classifications."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ..file import FileIdentifier, FilenameParser, FilepathInfoMatch
from ..lmt_base.file import FilepathInfo
from .types import ToltecDataKind, ToltecInfo

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["RoachFilepathInfo"]


class RoachFilepathInfo(FilepathInfo):
    """Filename-derived observation identifiers for a TolTEC ROACH readout."""

    roach: int


def _parse_prefix(fields: dict, path: Path) -> dict[str, object]:
    """Use the TolTEC filename prefix as the interface name."""
    return {"interface": fields["prefix"]}


def _parse_interface(fields: dict, path: Path) -> dict[str, object]:
    """Identify the instrument and component for a TolTEC interface."""
    identity = ToltecInfo.interface_instru[fields["interface"]]
    return {"instru": identity.instru, "instru_component": identity.instru_component}


def _parse_ut_datetime(fields: dict, path: Path) -> dict[str, object]:
    """Parse a ``YYYY_MM_DD_HH_MM_SS`` file timestamp as UTC."""
    return {
        "ut": datetime.strptime(fields["ut"], "%Y_%m_%d_%H_%M_%S").replace(tzinfo=UTC)
    }


def _parse_ut_date_only(fields: dict, path: Path) -> dict[str, object]:
    """Parse a ``YYYY-MM-DD`` file date as midnight UTC."""
    return {"ut": datetime.strptime(fields["ut"], "%Y-%m-%d").replace(tzinfo=UTC)}


def _parse_master_str(fields: dict, path: Path) -> dict[str, object]:
    """Read the master from a ``toltec/<master>/<interface>`` parent path.

    Return ``master_str=None`` when the parent path does not match the parsed
    interface. No filesystem access is performed.
    """
    interface = fields["interface"]
    match path.parent.parts:
        case (*_, "toltec", master, directory_interface) if (
            directory_interface == interface
        ):
            return {"master_str": master}
    return {"master_str": None}


_PARSERS_COMMON = (_parse_prefix, _parse_interface, _parse_master_str)

FileIdentifier.filename_pattern_parser.update(
    {
        re.compile(
            r"^(?P<prefix>toltec(?P<roach>\d+))_"
            r"(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)_"
            r"(?P<ut>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})"
            r"(?:_(?P<suffix>(?:vnasweep|targsweep|tune)_"
            r"(?P<proc_suffix>[^/.]+)|[^/.]+))?\.(?P<ext>.+)$"
        ): FilenameParser(
            filepath_info_cls=RoachFilepathInfo,
            parsers=(*_PARSERS_COMMON, _parse_ut_datetime),
        ),
        re.compile(
            r"^(?P<prefix>hwpr|toltec_hk)_"
            r"(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)_"
            r"(?P<ut>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})"
            r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>.+)$"
        ): FilenameParser(
            filepath_info_cls=FilepathInfo,
            parsers=(*_PARSERS_COMMON, _parse_ut_datetime),
        ),
        re.compile(
            r"^(?P<prefix>wyatt)_(?P<ut>\d{4}-\d{2}-\d{2})"
            r"_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)"
            r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>.+)$"
        ): FilenameParser(
            filepath_info_cls=FilepathInfo,
            parsers=(*_PARSERS_COMMON, _parse_ut_date_only),
        ),
    }
)

# TolTEC data kinds.
FileIdentifier.filepath_info_data_kind.update(
    # NetCDF suffixes identify raw and reduced sweep/timestream data.
    {
        FilepathInfoMatch(
            instru="toltec", instru_component="roach", suffix=suffix, ext="nc"
        ): kind
        for suffix, kind in (
            ("vnasweep", ToltecDataKind.VnaSweep),
            ("targsweep", ToltecDataKind.TargetSweep),
            ("targetsweep", ToltecDataKind.TargetSweep),
            ("tune", ToltecDataKind.Tune),
            ("vnasweep_processed", ToltecDataKind.ReducedVnaSweep),
            ("targsweep_processed", ToltecDataKind.ReducedTargetSweep),
            ("tune_processed", ToltecDataKind.ReducedTargetSweep),
            ("sweep", ToltecDataKind.ReducedSweep),
            ("timestream_processed", ToltecDataKind.SolvedTimeStream),
            ("timestream", ToltecDataKind.RawTimeStream),
            (None, ToltecDataKind.RawTimeStream),
        )
    }
    | {
        FilepathInfoMatch(
            instru="toltec", instru_component="roach", suffix=suffix, ext="txt"
        ): ToltecDataKind.KidsModelParamsTable
        for suffix in ("vnasweep", "targsweep", "tune")
    }
    # Processing tags identify tables independently of the input sweep kind.
    | {
        FilepathInfoMatch(
            instru="toltec", instru_component="roach", proc_suffix="targamps", ext="dat"
        ): ToltecDataKind.TargAmpsDat,
    }
    | {
        FilepathInfoMatch(
            instru="toltec",
            instru_component="roach",
            proc_suffix=proc_suffix,
            ext="ecsv",
        ): ToltecDataKind.KidsPropTable
        for proc_suffix in ("kids_find", "kids_fit", "chan_prop", "adrv")
    }
    | {
        FilepathInfoMatch(prefix="hwpr", ext="nc"): ToltecDataKind.Hwpr,
        FilepathInfoMatch(prefix="toltec_hk", ext="nc"): ToltecDataKind.HouseKeeping,
        FilepathInfoMatch(prefix="wyatt", ext="nc"): ToltecDataKind.Wyatt,
        FilepathInfoMatch(prefix="tel_toltec", ext="nc"): ToltecDataKind.LmtTel,
        FilepathInfoMatch(prefix="tel_toltec2", ext="nc"): ToltecDataKind.LmtTel2,
    }
)
