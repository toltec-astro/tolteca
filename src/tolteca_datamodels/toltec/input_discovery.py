"""Configured discovery in a TolTEC acquisition subtree."""

from __future__ import annotations

from functools import cached_property

from tollan.config.types import AbsAnyPath

from ..lmt_base.input_discovery import LmtBaseFileStoreInputDiscovery
from .file_store import ToltecFileStore

__all__ = ["ToltecInputDiscovery"]


class ToltecInputDiscovery(LmtBaseFileStoreInputDiscovery):
    """Discover candidates in a TolTEC acquisition subtree.

    Parameters
    ----------
    data_toltec_path : pathlib.Path
        Directory containing master/interface subdirectories, usually
        ``data_lmt/toltec``. Discovery stays within this directory; it does not
        search the sibling telescope directory. A missing directory yields no
        candidates.
    obs_spec : str, int or ObsSpec, optional
        Observation selector, default ``"*"``. Narrows discovery globs and
        selects rows after table population.
    mode : {"guess", "guess_or_read", "read"}, optional
        How to populate candidate information before ObsSpec selection.
        Default ``"guess"`` uses filenames only.
    data_model_select : StringSelector, optional
        Data models to use within the workflow's scope. Defaults to all
        available models. This store is searched when TolTEC support is selected.
    """

    data_toltec_path: AbsAnyPath

    @cached_property
    def _store(self) -> ToltecFileStore:
        """File store for this discovery entry."""
        return ToltecFileStore(self.data_toltec_path)
