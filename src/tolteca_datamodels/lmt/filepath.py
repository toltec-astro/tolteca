"""LMT acquisition filepath information."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar

from ..core.filepath import FilepathInfo, register_filepath_parser
from ..lmt_base.filepath import FilepathInfoMixin
from ..lmt_base.types import LabelT
from .types import LmtInfo

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["LmtFilepathInfo"]


class LmtFilepathInfo(FilepathInfo, FilepathInfoMixin):
    """Information encoded in an LMT acquisition filepath.

    Telescope filename example (the bracketed suffix is optional)::

        {prefix}_{ut}_{obsnum}_{subobsnum}_{scannum}[_{suffix}].{ext}
        tel_toltec_2025-10-31_145609_00_0002.nc

    ``ut`` records the writer's file-creation date, represented as midnight UTC.
    The full ``tel_*`` prefix is preserved; the interface is ``tel``.
    Telescope files belong to the LMT data model. Registered instrument-specific
    parsers can identify additional data-model memberships.

    """

    instru: LmtInfo.InstruT
    master: LabelT

    @staticmethod
    def _parse_prefix(filepath_data: dict, path: Path) -> dict[str, object]:
        """Identify the telescope interface for a matched telescope filename."""
        return {"interface": "tel"}

    @staticmethod
    def _parse_interface(filepath_data: dict, path: Path) -> dict[str, object]:
        """Identify the instrument and class of the telescope interface."""
        identity = LmtInfo.interface_interface_info[filepath_data["interface"]]
        return {
            "instru": identity.instru,
            "data_models": identity.data_models,
            "interface_kind": identity.interface_kind,
        }

    @staticmethod
    def _parse_ut(filepath_data: dict, path: Path) -> dict[str, object]:
        """Parse the UTC file-creation date in a telescope filename."""
        return {
            "ut": datetime.strptime(filepath_data["ut"], "%Y-%m-%d").replace(tzinfo=UTC)
        }

    @staticmethod
    def _parse_master(filepath_data: dict, path: Path) -> dict[str, object]:
        """Identify TCS as the telescope acquisition master, regardless of directory."""
        return {"master": "tcs"}

    _parsers: ClassVar = (_parse_prefix, _parse_interface, _parse_ut, _parse_master)

    @staticmethod
    def _make_filename_pattern(prefix: str) -> re.Pattern[str]:
        """Compile a telescope filename pattern with the given prefix expression."""
        return re.compile(
            rf"^(?P<prefix>{prefix})_"
            r"(?P<ut>\d{4}-\d{2}-\d{2})"
            r"_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)"
            r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>.+)$"
        )


register_filepath_parser(
    pattern=LmtFilepathInfo._make_filename_pattern(r"tel_[A-Za-z0-9_]+"),
    parsers=LmtFilepathInfo._parsers,
    filepath_info_cls=LmtFilepathInfo,
)
