"""LMT telescope data models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._filepath import LmtFilepathInfo
from .tel import LmtTelAccessor, identify_lmt_tel, open_lmt_tel
from .types import LmtInfo

if TYPE_CHECKING:
    from .tel import LmtTelDataset as LmtTelDataset
    from .tel import LmtTelDataTree as LmtTelDataTree

__all__ = [
    "LmtFilepathInfo",
    "LmtInfo",
    "LmtTelAccessor",
    "identify_lmt_tel",
    "open_lmt_tel",
]
