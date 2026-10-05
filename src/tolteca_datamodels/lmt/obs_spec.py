"""Observation-selector context for telescope interfaces."""

from ..obs_spec import ObsSpecContext
from .types import LmtInfo

__all__ = ["obs_spec_context"]

obs_spec_context = ObsSpecContext(
    labels={"interface": dict.fromkeys(LmtInfo.interfaces)},
)
"""Telescope interface labels."""
