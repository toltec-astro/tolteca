"""File-store support for LMT observation selectors."""

from __future__ import annotations

from ..file_store import FileStoreBase
from .obs_spec import ObsSpec

__all__ = ["LmtFileStoreBase"]


class LmtFileStoreBase(FileStoreBase[ObsSpec]):
    """Candidate discovery with LMT observation selectors.

    Parameters
    ----------
    path : str or pathlib.Path
        Read-only root of the directory tree. The directory need not exist.

    Notes
    -----
    Subclasses supply local glob patterns and instrument-specific selector
    contexts. Candidate searches accept an ObsSpec instance or selector text.
    """
