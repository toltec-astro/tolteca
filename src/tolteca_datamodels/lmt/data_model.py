"""LMT telescope capabilities for input workflows."""

from ..core.data_model import DataModelSpec
from .file import LmtFileAccessor
from .file_store import LmtFileStore

__all__ = ["data_model"]

data_model = DataModelSpec(
    file_accessor_cls=LmtFileAccessor,
    file_store_types=(LmtFileStore,),
)
"""LMT telescope workflow capabilities."""
