"""TolTEC acquisition filepath information."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar

from .._filepath import FilepathInfo
from ..lmt_base._filepath import FilepathInfoMixin
from .types import RoachT, ToltecInfo

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["ToltecFilepathInfo"]


class ToltecFilepathInfo(FilepathInfo, FilepathInfoMixin):
    """Information encoded in a TolTEC acquisition filepath."""

    instru: ToltecInfo.InstruT
    proc_suffix: str | None = None
    roach: RoachT | None = None

    @staticmethod
    def _parse_prefix(filepath_data: dict, path: Path) -> dict[str, object]:
        """Use the TolTEC filename prefix as the interface name."""
        return {"interface": filepath_data["prefix"]}

    @staticmethod
    def _parse_interface(filepath_data: dict, path: Path) -> dict[str, object]:
        """Identify the instrument and component for a TolTEC interface."""
        identity = ToltecInfo.interface_instru[filepath_data["interface"]]
        return {
            "instru": identity.instru,
            "instru_component": identity.instru_component,
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

    _filename_patterns: ClassVar = {
        re.compile(
            r"^(?P<prefix>toltec(?P<roach>\d+))_"
            r"(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)_"
            r"(?P<ut>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})"
            r"(?:_(?P<suffix>(?:vnasweep|targsweep|tune)_"
            r"(?P<proc_suffix>[^/.]+)|[^/.]+))?\.(?P<ext>.+)$"
        ): (*_parsers_common, _parse_ut_datetime),
        re.compile(
            r"^(?P<prefix>hwpr|toltec_hk)_"
            r"(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)_"
            r"(?P<ut>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})"
            r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>.+)$"
        ): (*_parsers_common, _parse_ut_datetime),
        re.compile(
            r"^(?P<prefix>wyatt)_(?P<ut>\d{4}-\d{2}-\d{2})"
            r"_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)"
            r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>.+)$"
        ): (*_parsers_common, _parse_ut_date_only),
    }
