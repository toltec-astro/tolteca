"""LMT telescope capabilities for input workflows."""

from ..core.data_model import DataModelSpec
from .file import LmtFileAccessor
from .file_store import LmtFileStore
from .types import LmtDataModel

__all__ = ["data_model_spec"]

data_model_spec = DataModelSpec(
    info_cls=LmtDataModel,
    file_accessor_cls=LmtFileAccessor,
    file_store_types=(LmtFileStore,),
)
"""LMT telescope workflow capabilities."""
