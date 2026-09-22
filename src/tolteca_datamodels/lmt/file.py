"""Register telescope acquisition filename patterns and interface names."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ..file import FileIdentifier, FilenameParser
from ..lmt_base.file import FilepathInfo
from .types import LmtInfo

if TYPE_CHECKING:
    from pathlib import Path


def _parse_prefix(fields: dict, path: Path) -> dict[str, object]:
    """Identify the telescope interface for a matched telescope filename."""
    return {"interface": "tel"}


def _parse_interface(fields: dict, path: Path) -> dict[str, object]:
    """Identify the instrument and component for the telescope interface."""
    identity = LmtInfo.interface_instru[fields["interface"]]
    return {"instru": identity.instru, "instru_component": identity.instru_component}


def _parse_ut(fields: dict, path: Path) -> dict[str, object]:
    """Parse the UTC file-creation date in a telescope filename."""
    return {"ut": datetime.strptime(fields["ut"], "%Y-%m-%d").replace(tzinfo=UTC)}


def _parse_master_str(fields: dict, path: Path) -> dict[str, object]:
    """Identify TCS as the telescope acquisition master, regardless of directory."""
    return {"master_str": "tcs"}


FileIdentifier.filename_pattern_parser.update(
    {
        re.compile(
            r"^(?P<prefix>tel_toltec|tel_toltec2)_"
            r"(?P<ut>\d{4}-\d{2}-\d{2})"
            r"_(?P<obsnum>\d+)_(?P<subobsnum>\d+)_(?P<scannum>\d+)"
            r"(?:_(?P<suffix>[^/.]+))?\.(?P<ext>.+)$"
        ): FilenameParser(
            filepath_info_cls=FilepathInfo,
            parsers=(
                _parse_prefix,
                _parse_interface,
                _parse_ut,
                _parse_master_str,
            ),
        ),
    }
)
