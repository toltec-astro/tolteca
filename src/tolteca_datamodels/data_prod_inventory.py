"""Portable identity and contents of an independently maintained inventory."""

from uuid import UUID

from pydantic import ConfigDict, Field
from tollan.config import FrozenBaseModel

from .data_prod import DataProdIndex
from .data_prod_namespace import DataProdNamespaceIndex

__all__ = ["DataProdInventory", "DataProdInventoryIndex"]


class DataProdInventory(FrozenBaseModel):
    """Identity of one independently maintained set of product records."""

    model_config = ConfigDict(extra="forbid")
    uid: UUID = Field(strict=False, description="Portable inventory UUID.")
    name: str | None = Field(
        default=None, description="Display qualifier, for example taco or unity."
    )


class DataProdInventoryIndex(FrozenBaseModel):
    """Product and namespace indexes contributed by one inventory."""

    model_config = ConfigDict(extra="forbid")
    inventory: DataProdInventory
    entries: list[DataProdIndex | DataProdNamespaceIndex] = Field(default_factory=list)
