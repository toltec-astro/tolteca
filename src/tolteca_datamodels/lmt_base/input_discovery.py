"""Shared discovery settings for LMT-layout inputs."""

from ..core.input_discovery import FileStoreInputDiscovery
from ..core.input_discovery import SourcesInputDiscovery as _SourcesInputDiscovery
from .obs_spec import ObsSpecField

__all__ = ["LmtBaseFileStoreInputDiscovery", "SourcesInputDiscovery"]


class LmtBaseFileStoreInputDiscovery(FileStoreInputDiscovery[ObsSpecField]):
    """Discover file-store inputs with an LMT observation selector.

    Parameters
    ----------
    obs_spec : str, int or ObsSpec, optional
        Observation selector, default ``"*"``. Narrows candidate discovery
        and selects rows after table population.
    """


class SourcesInputDiscovery(_SourcesInputDiscovery[ObsSpecField]):
    """Discover explicit locations and globs with an LMT observation selector.

    Parameters
    ----------
    obs_spec : str, int or ObsSpec, optional
        Observation selector applied after table population. Omitted by default.
    """
