"""Configured discovery in an LMT acquisition tree."""

from __future__ import annotations

from functools import cached_property

from tollan.config.types import AbsAnyPath

from ..lmt_base.input_discovery import LmtBaseFileStoreInputDiscovery
from .file_store import LmtFileStore

__all__ = ["LmtInputDiscovery"]


class LmtInputDiscovery(LmtBaseFileStoreInputDiscovery):
    """Discover candidates in an LMT acquisition tree.

    Parameters
    ----------
    data_lmt_path : pathlib.Path
        Directory containing ``tel`` and instrument subtrees. A missing
        directory yields no candidates.
    data_model_select : StringSelector, optional
        Data models to use within the workflow's scope. Defaults to all
        available models. ``["toltec"]`` searches telescope and TolTEC stores;
        ``["lmt"]`` searches only telescope files.
    obs_spec : str, int or ObsSpec, optional
        Observation selector, default ``"*"``. Used both to narrow discovery
        globs and to select rows after table population.
    mode : {"guess", "guess_or_read", "read"}, optional
        How to populate candidate information before ObsSpec selection.
        Default ``"guess"`` uses filenames only.
    """

    data_lmt_path: AbsAnyPath

    @cached_property
    def _store(self) -> LmtFileStore:
        """File store for this discovery entry."""
        return LmtFileStore(self.data_lmt_path)
