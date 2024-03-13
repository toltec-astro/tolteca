from pydantic import Field
from tollan.config.types import ImmutableBaseModel, ModelListBase
from tollan.utils.fileloc import FileLoc

from .base import FileIOBase


class DataItemModel(ImmutableBaseModel):
    """A data item."""

    meta: dict = Field(description="The meta data")
    file_loc: FileLoc | None = Field(descriptioin="The data item file location")
    io_obj: FileIOBase | None = Field(description="The data io object.")


class DataPoolModel(ModelListBase[DataItemModel]):
    """A class to manage a collection of data items."""

    def to_attr_records(self):  # noqa: D102
        pass

    def to_pandas(self, cols_method="outer"):  # noqa: D102
        pass


class DataCollector:
    """A base class to discover data items for data pool."""


class DataProd:
    """A base container class for data products."""

    def __init__(self, index=None, index_table=None, data_items=None, meta=None):
        pass
