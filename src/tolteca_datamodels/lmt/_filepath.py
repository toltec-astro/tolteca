"""LMT acquisition filepath information."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar

from .._filepath import FilepathInfo
from ..lmt_base._filepath import FilepathInfoMixin
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

    Examples
    --------
    >>> info = LmtFilepathInfo.from_source("tel_toltec_2025-10-31_145609_00_0002.nc")
    >>> info.prefix, info.ext
    ('tel_toltec', 'nc')
    """

    instru: LmtInfo.InstruT
    master: LabelT

    @staticmethod
    def _parse_prefix(filepath_data: dict, path: Path) -> dict[str, object]:
        """Identify the telescope interface for a matched telescope filename."""
        return {"interface": "tel"}

    @staticmethod
    def _parse_interface(filepath_data: dict, path: Path) -> dict[str, object]:
        """Identify the instrument and component for the telescope interface."""
        identity = LmtInfo.interface_instru[filepath_data["interface"]]
        return {
            "instru": identity.instru,
            "instru_component": identity.instru_component,
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

    _filename_patterns: ClassVar = {
        re.compile(
            r"^(?P<prefix>tel_[A-Za-z0-9_]+)_"
            r"(?P<ut>\d{4}-\d{2}-\d{2})"
            r"_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)"
            r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>.+)$"
        ): (
            _parse_prefix,
            _parse_interface,
            _parse_ut,
            _parse_master,
        ),
    }
