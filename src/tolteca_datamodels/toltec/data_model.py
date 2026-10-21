"""TolTEC capabilities for input workflows."""

from ..core.data_model import DataModelSpec
from ..lmt.file_store import LmtFileStore
from .file import ToltecFileAccessor
from .file_store import ToltecFileStore

__all__ = ["data_model"]

data_model = DataModelSpec(
    file_accessor_cls=ToltecFileAccessor,
    file_store_types=(LmtFileStore, ToltecFileStore),
)
"""TolTEC workflow capabilities, including telescope companions."""
