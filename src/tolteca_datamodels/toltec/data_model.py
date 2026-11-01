"""TolTEC capabilities for input workflows."""

from ..core.data_model import DataModelSpec
from ..lmt.file_store import LmtFileStore
from .file import ToltecFileAccessor
from .file_store import ToltecFileStore
from .types import ToltecDataModel

__all__ = ["data_model_spec"]

data_model_spec = DataModelSpec(
    info_cls=ToltecDataModel,
    file_accessor_cls=ToltecFileAccessor,
    file_store_types=(LmtFileStore, ToltecFileStore),
)
"""TolTEC workflow capabilities, including telescope companions."""
