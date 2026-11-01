"""Configured discovery in a dummy instrument subtree."""

from __future__ import annotations

from functools import cached_property

from tollan.config.types import AbsAnyPath

from tolteca_datamodels.lmt_base.input_discovery import LmtBaseFileStoreInputDiscovery

from .file_store import DummyInstruFileStore

__all__ = ["DummyInstruInputDiscovery"]


class DummyInstruInputDiscovery(LmtBaseFileStoreInputDiscovery):
    """Discover files in a dummy instrument subtree.

    Parameters
    ----------
    data_dummy_instru_path : pathlib.Path
        Directory containing the dummy channel directories. A missing directory
        yields no candidates. Relative paths use the runtime's root_path.
    obs_spec : str, int or ObsSpec, optional
        Observation selector, default ``"*"``. Narrows candidate discovery and
        selects rows after population. Numeric interfaces refer to dummy channels.
    mode : {"guess", "guess_or_read", "read"}, optional
        Default ``"guess"`` uses filenames only. ``"guess_or_read"`` reads
        supported, openable files with unrecognized filenames; ``"read"`` also
        reads recognized files supported by the selected handlers.
    data_model_select : StringSelector, optional
        Data models to use within the workflow's scope. Defaults to all
        available models. This store is searched when dummy support is selected.
    """

    data_dummy_instru_path: AbsAnyPath

    @cached_property
    def _store(self) -> DummyInstruFileStore:
        """File store for this discovery entry."""
        return DummyInstruFileStore(self.data_dummy_instru_path)
