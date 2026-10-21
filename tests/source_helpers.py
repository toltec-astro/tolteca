"""Metadata failure injection for reader lifecycle tests."""


def patch_metadata_read(monkeypatch, policy, read):
    """Patch the selected accessor classes' metadata property."""
    for accessor_cls in policy.accessor_classes:
        monkeypatch.setattr(accessor_cls, "meta", property(read))
