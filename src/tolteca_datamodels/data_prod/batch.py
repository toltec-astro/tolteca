"""Portable indexes prepared and submitted together."""

from typing import Any

from pydantic import ConfigDict, Field

from ..core.data_prod import DataProdNamespaceIndex
from ..core.data_prod.actor import _ActorsByUid
from ..core.types import FrozenDataModel
from .index import DataProdIndex
from .inventory import DataProdInventoryIndex
from .review import DataProdReview

__all__ = ["DataProdBatch"]


class DataProdBatch(FrozenDataModel):
    """Indexes and submission metadata prepared or registered together.

    Parameters
    ----------
    items : list, optional
        Indexes in submission order, empty by default. Namespace indexes can
        include runs. Inventory indexes retain the reporting inventory's identity.
    meta : dict of str to Any, optional
        Submission metadata, empty by default. Product metadata and run
        configuration remain in their own records.
    actors : dict of UUID to Actor, optional
        Actor records carried with this submission, empty by default. Each key
        must match its actor's UUID.
    reviews : list of DataProdReview, optional
        Product assessments, empty by default. Reviews can be submitted without
        product items when their subjects already exist at the destination.

    Notes
    -----
    Raw-file preparation returns this same container. Construction validates the
    records without opening artifacts, creating runs or registering products.
    Related products, actors and predecessor reviews may be supplied together
    or already exist at the destination; registration checks those references.

    Serialize with ``model_dump_json()`` and restore with
    ``model_validate_json()``. Each index owns its source locations. Wrapping a
    loaded index retains its resolution base in memory; JSON does not retain that
    directory. An empty batch is valid.
    """

    model_config = ConfigDict(extra="forbid")

    meta: dict[str, Any] = Field(
        default_factory=dict, description="Submission metadata."
    )
    actors: _ActorsByUid = Field(
        default_factory=dict, description="Actor records keyed by their UUIDs."
    )
    reviews: list[DataProdReview] = Field(
        default_factory=list,
        description="Assessments of products already registered or supplied in items.",
    )
    items: list[DataProdIndex | DataProdNamespaceIndex | DataProdInventoryIndex] = (
        Field(
            default_factory=list,
            description="Product, namespace and inventory indexes.",
        )
    )
