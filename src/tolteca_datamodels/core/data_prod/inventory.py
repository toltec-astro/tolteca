"""Identity of an independently maintained product inventory."""

from uuid import UUID

from pydantic import ConfigDict, Field
from tollan.config import FrozenBaseModel

__all__ = ["DataProdInventory"]


class DataProdInventory(FrozenBaseModel):
    """Identity of a provider's independently maintained product records.

    An inventory records which products, items, runs and tags a provider reports,
    including its namespace paths. It has no storage root of its own. The same
    product can appear in several inventories with different source locations
    or availability.

    For example, Taco's live inventory and Unity's copied-tree inventory remain
    distinct when their records are imported into a unified DPDB. Their product
    identities are preserved. ``uid`` identifies the inventory across relocation;
    ``name`` supplies a display qualifier such as ``taco`` or ``unity-copy``.
    """

    model_config = ConfigDict(extra="forbid")
    uid: UUID = Field(strict=False, description="Portable inventory UUID.")
    name: str | None = Field(
        default=None, description="Display qualifier, for example taco or unity."
    )
