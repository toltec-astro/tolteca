"""Dummy-instrument capabilities for input workflows."""

from tolteca_datamodels.core.data_model import DataModelSpec
from tolteca_datamodels.lmt.file_store import LmtFileStore

from .file import DummyInstruFileAccessor
from .file_store import DummyInstruFileStore
from .types import DummyInstruDataModel

__all__ = ["data_model_spec"]

data_model_spec = DataModelSpec(
    info_cls=DummyInstruDataModel,
    file_accessor_cls=DummyInstruFileAccessor,
    file_store_types=(LmtFileStore, DummyInstruFileStore),
)
"""Dummy-instrument workflow capabilities, including telescope companions."""
