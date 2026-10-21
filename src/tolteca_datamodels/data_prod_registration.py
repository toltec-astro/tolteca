"""Portable product and execution records for one registration transaction."""

from pydantic import ConfigDict, Field

from .data_prod import DataProdIndex
from .data_prod_inventory import DataProdInventoryIndex
from .data_prod_namespace import DataProdNamespaceIndex
from .types import FrozenDataModel

__all__ = ["DataProdRegistration"]


class DataProdRegistration(FrozenDataModel):
    """Product inventories and namespace execution records to register together.

    Parameters
    ----------
    entries : list, optional
        Product indexes, namespace indexes, or inventories containing those indexes.
        An inventory retains its reporting identity when imported; unwrapped
        indexes belong to the receiving database's local inventory.
        Relationship targets may appear anywhere
        in this batch or already exist in the destination. Omitted records and
        relationships are retained. An empty batch makes no changes.

    Notes
    -----
    Serialize with ``model_dump_json()`` and restore with
    ``model_validate_json()``. The batch contains metadata and source references;
    registering it does not read or copy artifact payloads.
    """

    model_config = ConfigDict(extra="forbid")
    entries: list[DataProdIndex | DataProdNamespaceIndex | DataProdInventoryIndex] = (
        Field(
            default_factory=list,
            description="Product and namespace indexes, with reporting inventories.",
        )
    )
