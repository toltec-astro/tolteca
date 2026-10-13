"""LMT telescope data models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..data_model import DataModel as _DataModel
from . import _filepath as _filepath
from .file import LmtFileAccessor as _LmtFileAccessor
from .tel import LmtTelAccessor, identify_lmt_tel, open_lmt_tel
from .types import LmtInfo

if TYPE_CHECKING:
    from .tel import LmtTelDataset as LmtTelDataset
    from .tel import LmtTelDataTree as LmtTelDataTree

__all__ = [
    "LmtInfo",
    "LmtTelAccessor",
    "data_model",
    "identify_lmt_tel",
    "open_lmt_tel",
]

data_model = _DataModel(name=LmtInfo.data_model, file_accessor_cls=_LmtFileAccessor)
"""LMT telescope data-model definition."""

data_model.register()
