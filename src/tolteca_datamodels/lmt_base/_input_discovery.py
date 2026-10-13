"""Shared discovery settings for LMT file stores."""

from .._input_discovery import FileStoreInputDiscovery
from .obs_spec import ObsSpecField

__all__ = ["LmtBaseFileStoreInputDiscovery"]


class LmtBaseFileStoreInputDiscovery(FileStoreInputDiscovery[ObsSpecField]):
    """Discover file-store inputs with an LMT observation selector.

    Parameters
    ----------
    obs_spec : str, int or ObsSpec, optional
        Observation selector, default ``"*"``. Narrows candidate discovery
        and selects rows after table population.
    """
