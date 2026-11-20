"""Resolve exact local acquisition files from their owning product indexes."""

from tollan.utils.fileloc import fileloc


def _source(index, interface):
    items = [
        item
        for item in index.product.data_items
        if item.meta.get("interface") == interface
        and item.meta.get("type") != "artifact"
    ]
    if len(items) != 1:
        msg = f"Expected one raw source for {interface}"
        raise ValueError(msg)
    return _local(index.get_source_uri(items[0].source))


def _local(uri):
    loc = fileloc(uri)
    if loc.is_remote() or not loc.path.is_file():
        msg = f"Model preparation requires an existing local file: {uri}"
        raise ValueError(msg)
    return loc.path
