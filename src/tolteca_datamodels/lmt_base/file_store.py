"""File-store support for LMT observation selectors."""

from __future__ import annotations

from typing import ClassVar

from ..core.file_store import FileStoreBase
from ..core.obs_spec import ObsSpecContext
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
    Subclasses supply local glob patterns and, when needed, numeric
    interpretations of interface labels. These interpretations apply only to
    their own managed locations. Substores interpret selectors independently.
    Candidate searches accept an ObsSpec instance or selector text.
    """

    _obs_spec_context: ClassVar[ObsSpecContext] = ObsSpecContext()
