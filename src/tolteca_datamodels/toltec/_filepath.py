"""TolTEC acquisition filepath information."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar

from .._filepath import FilepathInfo, register_filepath_parser
from ..lmt._filepath import LmtFilepathInfo
from ..lmt_base._filepath import FilepathInfoMixin
from .types import ToltecInfo

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["ToltecFilepathInfo", "ToltecLmtFilepathInfo"]


class ToltecLmtFilepathInfo(LmtFilepathInfo):
    """Telescope filepath information for TolTEC observations."""


class ToltecFilepathInfo(FilepathInfo, FilepathInfoMixin):
    """Information encoded in a TolTEC acquisition filepath."""

    instru: ToltecInfo.InstruT
    producer: ToltecInfo.InstruT = ToltecInfo.instru
    proc_suffix: str | None = None
    roach: ToltecInfo.RoachT | None = None

    @staticmethod
    def _parse_prefix(filepath_data: dict, path: Path) -> dict[str, object]:
        """Use the TolTEC filename prefix as the interface name."""
        return {"interface": filepath_data["prefix"]}

    @staticmethod
    def _parse_interface(filepath_data: dict, path: Path) -> dict[str, object]:
        """Identify the instrument, producer, and class of a TolTEC interface."""
        identity = ToltecInfo.interface_interface_info[filepath_data["interface"]]
        return {
            "instru": identity.instru,
            "producer": identity.producer,
            "interface_kind": identity.interface_kind,
        }

    @staticmethod
    def _parse_ut_datetime(filepath_data: dict, path: Path) -> dict[str, object]:
        """Parse a ``YYYY_MM_DD_HH_MM_SS`` file timestamp as UTC."""
        return {
            "ut": datetime.strptime(filepath_data["ut"], "%Y_%m_%d_%H_%M_%S").replace(
                tzinfo=UTC
            )
        }

    @staticmethod
    def _parse_ut_date_only(filepath_data: dict, path: Path) -> dict[str, object]:
        """Parse a ``YYYY-MM-DD`` file date as midnight UTC."""
        return {
            "ut": datetime.strptime(filepath_data["ut"], "%Y-%m-%d").replace(tzinfo=UTC)
        }

    @staticmethod
    def _parse_master(filepath_data: dict, path: Path) -> dict[str, object]:
        """Read the master from a ``toltec/<master>/<interface>`` parent path.

        Return ``master=None`` when the parent path does not match the parsed
        interface. No filesystem access is performed.
        """
        interface = filepath_data["interface"]
        match path.parent.parts:
            case (*_, "toltec", master, directory_interface) if (
                directory_interface == interface
            ):
                return {"master": master}
        return {"master": None}

    _parsers_common: ClassVar = (_parse_prefix, _parse_interface, _parse_master)


register_filepath_parser(
    pattern=LmtFilepathInfo._make_filename_pattern(r"tel_toltec2?"),
    parsers=(*LmtFilepathInfo._parsers, ToltecFilepathInfo._parse_interface),
    filepath_info_cls=ToltecLmtFilepathInfo,
)
register_filepath_parser(
    pattern=re.compile(
        r"^(?P<prefix>toltec(?P<roach>\d+))_"
        r"(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)_"
        r"(?P<ut>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})"
        r"(?:_(?P<suffix>(?:vnasweep|targsweep|tune)_"
        r"(?P<proc_suffix>[^/.]+)|[^/.]+))?\.(?P<ext>.+)$"
    ),
    parsers=(
        *ToltecFilepathInfo._parsers_common,
        ToltecFilepathInfo._parse_ut_datetime,
    ),
    filepath_info_cls=ToltecFilepathInfo,
)
register_filepath_parser(
    pattern=re.compile(
        r"^(?P<prefix>hwpr|toltec_hk)_"
        r"(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)_"
        r"(?P<ut>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})"
        r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>.+)$"
    ),
    parsers=(
        *ToltecFilepathInfo._parsers_common,
        ToltecFilepathInfo._parse_ut_datetime,
    ),
    filepath_info_cls=ToltecFilepathInfo,
)
register_filepath_parser(
    pattern=re.compile(
        r"^(?P<prefix>wyatt)_(?P<ut>\d{4}-\d{2}-\d{2})"
        r"_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)"
        r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>.+)$"
    ),
    parsers=(
        *ToltecFilepathInfo._parsers_common,
        ToltecFilepathInfo._parse_ut_date_only,
    ),
    filepath_info_cls=ToltecFilepathInfo,
)
