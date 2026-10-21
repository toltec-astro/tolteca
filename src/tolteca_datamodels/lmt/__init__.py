"""LMT telescope data models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import filepath as filepath
from .tel import LmtTelAccessor, open_lmt_tel
from .types import LmtInfo

if TYPE_CHECKING:
    from .tel import LmtTelDataset as LmtTelDataset
    from .tel import LmtTelDataTree as LmtTelDataTree

__all__ = [
    "LmtInfo",
    "LmtTelAccessor",
    "open_lmt_tel",
]
