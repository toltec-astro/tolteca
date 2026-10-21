"""Observation-selector context for telescope interfaces."""

from ..core.obs_spec import ObsSpecContext
from .types import LmtInfo

_obs_spec_context = ObsSpecContext(
    field_label_numbers={"interface": dict.fromkeys(LmtInfo.interfaces)},
)
"""Telescope interface labels."""
