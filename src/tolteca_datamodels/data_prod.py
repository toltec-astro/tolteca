from enum import auto
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field
from strenum import StrEnum
from tollan.config.types import ImmutableBaseModel
from tollan.utils.fileloc import FileLoc


class DataProdMetaBase(BaseModel):
    """A base class for data product metadata."""

    name: str = Field(
        description="Data product name.",
    )
    source: FileLoc = Field(
        description="Data source.",
    )
    filepath: None | Path = Field(
        description="Local filepath if the source can be resolved.",
    )


DataProdType = TypeVar("DataProdType", bound=str)
DataProdMetaType = TypeVar("DataProdMetaType", bound=DataProdMetaBase)


class DataProdModelGeneric(ImmutableBaseModel, Generic[DataProdType, DataProdMetaType]):
    """A data product."""

    type: DataProdType = Field(description="Data product type.")
    meta: DataProdMetaType = Field(description="Meta data.")
    source: FileLoc = Field(
        description="Data source",
    )
    data: Any = Field(
        description="Data object.",
    )
    data_items: list["DataProdModelGeneric"] = Field(
        description="Child data products.",
    )

    def to_attr_records(self):  # noqa: D102
        pass

    def to_pandas(self, cols_method="outer"):  # noqa: D102
        pass


class GeneralDataProdType(StrEnum):
    """General data product types."""

    dp_file = auto()
    dp_named_group = auto()


class DPM_File(DataProdMetaBase):  # noqa: N801
    """Metadata class for ``dp_file``."""

    io_cls_name: str = Field(description="IO class name.")


class DP_File(DataProdModelGeneric[GeneralDataProdType, DPM_File]):  # noqa: N801
    """Data product that maps to a single file."""

    type: Literal[GeneralDataProdType.dp_file] = GeneralDataProdType.dp_file


class LmtDataProdType(StrEnum):
    """LMT data product types."""

    dp_raw_obs = auto()


class ToltecSpecificDataProdType(StrEnum):
    """Toltec specific data product types."""

    dp_basic_reduced_obs = auto()


ToltecDataProdType = GeneralDataProdType | LmtDataProdType | ToltecSpecificDataProdType


class ToltecDataProdModelGeneric(
    DataProdModelGeneric[ToltecDataProdType, DataProdMetaType],
    Generic[DataProdMetaType],
):
    """The TolTEC data product model."""


class DataProd:
    """A base container class for data products."""

    def __init__(self, index=None, index_table=None, data_items=None, meta=None):
        pass
