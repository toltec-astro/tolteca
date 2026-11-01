"""Explicit file capabilities and reader failure injection for integration tests."""


def patch_metadata_read(monkeypatch, policy, read):
    """Replace metadata reads for all readers selectable by a policy."""
    for accessor_cls in policy.accessor_classes:
        monkeypatch.setattr(accessor_cls, "meta", property(read))
