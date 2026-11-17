"""Traversal of marked primary associations."""


def _resolve_primary(product, get_next, get_identity):
    """Follow primary links, detecting cycles by the supplied product identity."""
    visited = set()
    while True:
        identity = get_identity(product)
        if identity in visited:
            msg = f"Primary association cycle at {identity!r}"
            raise ValueError(msg)
        visited.add(identity)
        target = get_next(product)
        if target is None:
            return product
        product = target


def _select_primary_assoc(product_uid, assocs):
    """Select the single marked association of a nonterminal product."""
    primary = [assoc for assoc in assocs if assoc.is_primary]
    if len(primary) != 1:
        msg = f"Product {product_uid!r} requires exactly one primary association"
        raise ValueError(msg)
    return primary[0]
