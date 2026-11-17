"""Metadata-only artifact selection from portable product indexes."""

from collections import defaultdict
from collections.abc import Iterable  # noqa: TC003 -- public signature
from dataclasses import dataclass

import pandas as pd
from tollan.accessor.pandas.model_table import ModelColumn, ModelTableAccessor

from ..core.data_prod import (
    DataProdInventory,  # noqa: TC001 -- public row annotation
    DataProdItem,  # noqa: TC001 -- public row annotation
    DataProdNamespaceIndex,  # noqa: TC001 -- public signature
    DataProdRef,  # noqa: TC001 -- public row annotation
)
from .identity import normalize_primary_uid
from .index import DataProdIndex
from .inventory import DataProdInventoryIndex

__all__ = ["artifact_table_from_indexes"]


@dataclass
class _ArtifactRow:
    index: DataProdIndex
    item: DataProdItem
    inventory: DataProdInventory | None
    group: DataProdRef | None = None
    tag: str | None = None

    @property
    def observation(self):
        """Observation coordinates from metadata or an explicit logical primary."""
        meta = self.index.product.meta
        fields = ("master", "obsnum", "subobsnum", "scannum")
        values = {key: getattr(meta, key, None) for key in fields}
        if values["obsnum"] is not None:
            return values
        primary = [a for a in self.index.product.assocs if a.is_primary]
        if len(primary) != 1 or primary[0].namespace_uid is not None:
            return values
        try:
            uid = normalize_primary_uid(primary[0].uid)
        except ValueError:
            return values  # An intermediate derived product is not a logical identity.
        parts = uid.split("-")
        if len(parts) not in (2, 4):
            return values  # Named groups do not identify one observation.
        return dict(zip(fields, [parts[0], *(int(v) for v in parts[1:])], strict=False))


_columns = {
    "product": ModelColumn("index.product.reference", dtype=object),
    "product_uid": ModelColumn("index.product.uid", dtype="string"),
    "namespace_uid": ModelColumn(
        "index.product.namespace_uid", dtype="string", converter=str
    ),
    "data_prod_type": ModelColumn("index.product.meta.data_prod_type", dtype="string"),
    "inventory_uid": ModelColumn(
        "inventory.uid", default=None, dtype="string", converter=str
    ),
    "inventory_name": ModelColumn("inventory.name", default=None, dtype="string"),
    "group": ModelColumn("group", dtype=object),
    "tag": ModelColumn("tag", dtype="string"),
    "run_uid": ModelColumn(
        "index.run.uid", default=None, dtype="string", converter=str
    ),
    "run_name": ModelColumn("index.store_run.name", default=None, dtype="string"),
    **{
        name: ModelColumn(
            "observation",
            dtype="Int64",
            converter=lambda value, key=name: value.get(key),
        )
        for name in ("obsnum", "subobsnum", "scannum")
    },
    "master": ModelColumn(
        "observation", dtype="string", converter=lambda value: value.get("master")
    ),
    **{
        name: ModelColumn(
            "item.meta", dtype="string", converter=lambda meta, key=name: meta.get(key)
        )
        for name in (
            "name",
            "producer",
            "format",
            "kind",
            "interface",
            "scope",
            "array",
        )
    },
    **{
        name: ModelColumn(f"item.source.{name}", default=None, dtype="string")
        for name in ("location", "role", "availability_state", "checksum")
    },
    "source_path": ModelColumn("item.source.path", dtype="string", converter=str),
    "size": ModelColumn("item.source.size", default=None, dtype="Int64"),
    "index": ModelColumn("index", dtype=object),
    "item": ModelColumn("item", dtype=object),
}


def artifact_table_from_indexes(
    indexes: Iterable[DataProdIndex | DataProdInventoryIndex | DataProdNamespaceIndex],
) -> pd.DataFrame:
    """Build a queryable artifact table without opening artifact payloads.

    Parameters
    ----------
    indexes : iterable
        Loaded DataProdIndex, DataProdInventoryIndex or DataProdNamespaceIndex
        records, including version groups when querying tags. Embedded
        members are included. ``DataProdBatch.items`` is accepted; namespace
        descriptors contribute no artifacts. Read YAML with DataProdIndex.read
        first so relative storage roots resolve against that document.

    Returns
    -------
    pandas.DataFrame
        One row per artifact source and selecting group/tag, or one row with a
        null tag when untagged. Inventory reports remain separate. Sources marked
        unavailable are excluded; unset availability is retained. ``product`` is
        a portable reference; ``index`` and ``item`` retain the exact records.
        ``source`` is the resolved URI. Use ModelTableAccessor(table).query(...)
        for the same expression syntax as file selection. Repeated tag matches
        are not different artifacts; distinguish sources by product, location
        and source_path. No filename parsing or payload I/O occurs.
        Observation columns use product metadata, or its explicitly associated
        global observation/acquisition primary UID when metadata omits them.
        Calibration associations are not used for these columns; named-group
        primaries do not imply one observation.
        ``name``, ``producer``, ``format``, ``kind``, ``interface``, ``scope`` and
        ``array`` expose the producer's artifact metadata as nullable strings.
    """
    records = list(_product_indexes(indexes))
    tags = _tags(records)
    rows = []
    for index, inventory in records:
        key = (None if inventory is None else inventory.uid, index.product.reference)
        for item in index.product.data_items:
            if item.meta.get(
                "type"
            ) != "artifact" or item.source.availability_state not in (
                None,
                "available",
            ):
                continue
            for group, tag in tags.get(key, [(None, None)]):
                rows.append(_ArtifactRow(index, item, inventory, group, tag))
    return _artifact_table(rows)


def _artifact_table(rows):
    """Project portable or relational artifact rows with the same column contract."""
    table = ModelTableAccessor.from_models(rows, columns=_columns)
    table["source"] = pd.Series(
        [row.index.get_source_uri(row.item.source) for row in rows], dtype="string"
    )
    return table.drop_duplicates(
        [
            "inventory_uid",
            "product",
            "location",
            "source_path",
            "source",
            "group",
            "tag",
        ]
    ).reset_index(drop=True)


def _product_indexes(entries, inventory=None):
    for entry in entries:
        if isinstance(entry, DataProdInventoryIndex):
            yield from _product_indexes(entry.entries, entry.inventory)
        elif isinstance(entry, DataProdIndex):
            yield entry, inventory
            yield from _product_indexes(entry.products, inventory)


def _tags(records):
    selections = {}
    tags = defaultdict(list)
    for index, inventory in records:
        inventory_uid = None if inventory is None else inventory.uid
        for label, tag in index.product.tags.items():
            key = (inventory_uid, index.product.reference, label)
            if key in selections and selections[key] != tag.current:
                msg = f"Conflicting version selections in artifact table: {key}"
                raise ValueError(msg)
            selections[key] = tag.current
    for (inventory_uid, group, label), current in selections.items():
        if current is not None:
            tags[inventory_uid, current].append((group, label))
    return tags
