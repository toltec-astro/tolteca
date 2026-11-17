"""Portable TolTECA data products and filesystem publication.

Read an index from a YAML file or a product directory::

    index = DataProdIndex.read("dataprod_toltec/scratch/145609-0-1/dp_raw_obs.yaml")
    items = index.product.data_items

``index.product`` supplies typed metadata, data items and associations. Each item
carries its own metadata and a source reference.
``index`` supplies locations and supporting namespace, actor and run records.
Index paths,
storage locations and namespace directory names do not change product identity.
"""

from ..core.data_prod import (
    Actor,
    CounterRunNameFormat,
    CounterRunNameGenerator,
    CounterRunNameState,
    DataProdAssoc,
    DataProdAssocRun,
    DataProdAudit,
    DataProdInventory,
    DataProdItem,
    DataProdNamespace,
    DataProdNamespaceIndex,
    DataProdRef,
    DataProdSource,
    DataProdTag,
    Location,
    PublicationRequest,
    RunName,
    RunUid,
    StoreRun,
    WorkflowRun,
)
from .batch import (
    DataProdBatch,
)
from .calibration import (
    DataProdCalRef,
    RawObsBlockHistory,
    RawObsBlockRelation,
    RawObsFileCalObsFileMeta,
    RawObsHistoryBlock,
)
from .identity import (
    make_obs_product_uid,
    make_raw_obs_product_uid,
    normalize_primary_uid,
)
from .index import (
    DataProd,
    DataProdIndex,
)
from .inventory import (
    DataProdInventoryIndex,
)
from .model_table import artifact_table_from_indexes
from .raw_obs import build_raw_obs_batch, read_raw_obs_index
from .review import DataProdReview, current_reviews
from .store import (
    DataProdStore,
    DataProdTagBusyError,
)

__all__ = [
    "Actor",
    "CounterRunNameFormat",
    "CounterRunNameGenerator",
    "CounterRunNameState",
    "DataProd",
    "DataProdAssoc",
    "DataProdAssocRun",
    "DataProdAudit",
    "DataProdBatch",
    "DataProdCalRef",
    "DataProdIndex",
    "DataProdInventory",
    "DataProdInventoryIndex",
    "DataProdItem",
    "DataProdNamespace",
    "DataProdNamespaceIndex",
    "DataProdRef",
    "DataProdReview",
    "DataProdSource",
    "DataProdStore",
    "DataProdTag",
    "DataProdTagBusyError",
    "Location",
    "PublicationRequest",
    "RawObsBlockHistory",
    "RawObsBlockRelation",
    "RawObsFileCalObsFileMeta",
    "RawObsHistoryBlock",
    "RunName",
    "RunUid",
    "StoreRun",
    "WorkflowRun",
    "artifact_table_from_indexes",
    "build_raw_obs_batch",
    "current_reviews",
    "make_obs_product_uid",
    "make_raw_obs_product_uid",
    "normalize_primary_uid",
    "read_raw_obs_index",
]
