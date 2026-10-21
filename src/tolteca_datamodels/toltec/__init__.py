"""TolTEC-specific data models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import filepath as filepath
from .hwpr import open_toltec_hwpr
from .kids.core import ToltecKidsAccessor, open_toltec_kids
from .types import ToltecInfo

if TYPE_CHECKING:
    from .kids.core import ToltecKidsDataset as ToltecKidsDataset
    from .kids.core import ToltecKidsDataTree as ToltecKidsDataTree


__all__ = [
    "ToltecInfo",
    "ToltecKidsAccessor",
    "open_toltec_hwpr",
    "open_toltec_kids",
]
