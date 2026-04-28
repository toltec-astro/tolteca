"""Reference xarray accessor implementations for generic KIDs data processing.

This module provides reference implementations of schema and mapper classes
using tollan's accessor framework. These serve as templates for instrument-specific
implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import kids  # noqa: F401
from .dataset import make_kids_dataset, open_datatree

if TYPE_CHECKING:
    from .dataset import KidsDataset, KidsDataTree  # noqa: F401

__all__ = [
    "make_kids_dataset",
    "open_datatree",
]
