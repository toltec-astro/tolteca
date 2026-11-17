"""Exact fitted-model references within recorded sweep bindings."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from ..core.types import FrozenDataModel
from ..data_prod import (
    DataProdIndex,
    DataProdItem,
    DataProdSource,
)
from ..data_prod.identity import make_raw_obs_product_uid
from ..data_prod.metadata import RawObsProcMeta
from .types import RawObsProcRef, RawObsRef

__all__ = ["KidsModelBinding"]


class KidsModelBinding(FrozenDataModel):
    """One fitted-model artifact in an exact processed-acquisition realization.

    Parameters
    ----------
    product : RawObsProcRef
        Namespaced realization reference, including the producing run UID.
    producer, name : str
        Exact artifact metadata selectors. The enclosing sweep item supplies
        interface and block scope.
    source : DataProdSource or None, optional
        Exact source chosen by an artifact-table query. None requires the
        metadata selection to be unique across storage copies.

    Notes
    -----
    Construction validates the reference shape. Use resolve to check product
    ownership and item membership, and the model preparation operation to check
    the report against the actual sweep. Payloads are not embedded in a context.
    """

    model_config = ConfigDict(extra="forbid")
    product: RawObsProcRef
    producer: str = Field(min_length=1)
    name: str = Field(min_length=1)
    source: DataProdSource | None = None

    def resolve(
        self, index: DataProdIndex, *, raw_obs: RawObsRef, interface: str, block: int
    ) -> DataProdItem:
        """Resolve an exact model item without opening its payload.

        Parameters
        ----------
        index : DataProdIndex
            Pinned processed-product index with its producing run record.
        raw_obs : RawObsRef
            Recorded sweep acquisition owning the calibration.
        interface : str
            Calibration network, such as toltec0.
        block : int
            Acquired sweep block. Native artifacts use scope ``blocks/<block>``.

        Returns
        -------
        DataProdItem
            The unique ECSV artifact matching producer, name and sweep scope.
            Resolve its location through the supplied index when opening it.

        Raises
        ------
        ValueError
            Product/run identity or raw ownership differs, or the selection is
            absent, ambiguous or attributed to another producing run.
        """
        product, run = index.product, index.run
        if (
            product.reference != self.product
            or not isinstance(product.meta, RawObsProcMeta)
            or run is None
            or not product.uid.endswith(f"/{run.uid}")
        ):
            msg = "Model binding requires its exact processed-product realization"
            raise ValueError(msg)
        primary = [assoc for assoc in product.assocs if assoc.is_primary]
        identity = make_raw_obs_product_uid(
            product.meta.master,
            product.meta.obsnum,
            product.meta.subobsnum,
            product.meta.scannum,
        )
        if (
            identity != raw_obs.uid
            or len(primary) != 1
            or primary[0].data_prod_assoc_type != "dpa_raw_obs_proc_raw_obs"
            or (primary[0].uid, primary[0].namespace_uid)
            != (raw_obs.uid, raw_obs.namespace_uid)
        ):
            msg = "Model product does not derive from the recorded sweep acquisition"
            raise ValueError(msg)
        expected = {
            "type": "artifact",
            "format": "ecsv",
            "interface": interface,
            "scope": f"blocks/{block}",
            "producer": self.producer,
            "name": self.name,
        }
        items = [
            item
            for item in product.data_items
            if all(item.meta.get(key) == value for key, value in expected.items())
            and (self.source is None or item.source == self.source)
        ]
        if len(items) != 1:
            msg = "Model selection must match exactly one artifact at the sweep scope"
            raise ValueError(msg)
        [item] = items
        if item.meta.get("run_uid") != str(run.uid):
            msg = "Model artifact belongs to a different producing run"
            raise ValueError(msg)
        if item.meta.get("namespace_uid") not in (None, str(product.namespace_uid)):
            msg = "Model artifact belongs to a different namespace"
            raise ValueError(msg)
        return item
