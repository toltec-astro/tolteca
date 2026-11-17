"""Portable contents of a TolTECA product inventory."""

from pydantic import ConfigDict, Field
from tollan.config import FrozenBaseModel

from ..core.data_prod import DataProdInventory, DataProdNamespaceIndex
from .index import DataProdIndex

__all__ = ["DataProdInventoryIndex"]


class DataProdInventoryIndex(FrozenBaseModel):
    """Product and namespace indexes reported by one inventory.

    Use this envelope to preserve reporting ownership when transferring records.
    The contained indexes retain their product identities and source locations.
    An empty contribution is valid; constructing it does not register an
    inventory or inspect its artifacts.
    """

    model_config = ConfigDict(extra="forbid")
    inventory: DataProdInventory = Field(description="Reporting inventory identity.")
    entries: list[DataProdIndex | DataProdNamespaceIndex] = Field(
        default_factory=list,
        description="Product and namespace indexes reported by this inventory.",
    )
