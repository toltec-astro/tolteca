"""Dummy-instrument capabilities for input workflows."""

from tolteca_datamodels.core.data_model import DataModelSpec
from tolteca_datamodels.lmt.file_store import LmtFileStore

from .file import DummyInstruFileAccessor
from .file_store import DummyInstruFileStore

__all__ = ["data_model"]

data_model = DataModelSpec(
    file_accessor_cls=DummyInstruFileAccessor,
    file_store_types=(LmtFileStore, DummyInstruFileStore),
)
"""Dummy-instrument workflow capabilities, including telescope companions."""
