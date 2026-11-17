"""Write membership records before publishing a version-group header."""

from urllib.parse import quote

from ..core.data_prod._io import _write_yaml
from ._versions import MemberRecord


def _append_version_member(header, directory, name, edge):
    header.version_count += 1
    member = MemberRecord(position=header.version_count, name=name, edge=edge)
    root = directory / ".members"
    _write_yaml(
        member, root / "by_uid" / f"{quote(edge.uid, safe='')}.yaml", overwrite=True
    )
    _write_yaml(member, root / f"{member.position}.yaml", overwrite=True)


def _compact_version_group(header, directory):
    """Convert an embedded projection on its first live update."""
    if not header.products:
        return
    edges = {
        a.uid: a
        for a in header.product.assocs
        if a.data_prod_assoc_type == "dpa_version_group_data_prod"
    }
    if set(edges) != {p.product.uid for p in header.products}:
        msg = "Embedded version membership disagrees with its products"
        raise ValueError(msg)
    for p in header.products:
        if p.run is None or p.store_run is None:
            msg = "Version member lacks its run identity or directory name"
            raise ValueError(msg)
        _append_version_member(
            header, directory, p.store_run.name, edges[p.product.uid]
        )
    header.products.clear()
    header.product.assocs = [a for a in header.product.assocs if a.uid not in edges]
