"""Step execution shared by KIDs integration tests."""

from tollan.accessor.xarray import ensure_datatree


def apply_step(step, data, *, parent_namespace=None, force=None):
    """Execute a step at a selected scope and retain the working tree."""
    tree = ensure_datatree(data)
    scope = tree[parent_namespace] if parent_namespace is not None else tree
    if force is not None:
        step = type(step)(step.config.model_copy(update={"force": force}))
    step(scope)
    return tree
