import pandas as pd
from collections import UserList
from tollan.config.types import ImmutableBaseModel, ModelListBase
from tollan.utils.fileloc import FileLocStr
from pydantic import Field


class DataItemModel(ImmutableBaseModel):
    """A data item."""

    meta: dict = Field(description="The meta data")
    file_loc: FileLocStr | None = Field(descriptioin="The source location")
    io_obj: object | None = Field(description="The data io object.")


class DataItemListModel(ModelListBase[DataItemModel]):
    """A class to manage a list of data items."""

    def to_attr_records():
        pass

    def to_data_frame(cols_method="outer"):
        pass


class DataProd:
    """A base container class for data products."""

    def __init__(index=None, index_table=None, data_items=None, meta=None):
        pass
