"""Observation-selector context for TolTEC interfaces."""

from ..obs_spec import ObsSpecContext
from .types import ToltecInfo

_obs_spec_context = ObsSpecContext(
    field_label_numbers={
        "interface": dict.fromkeys(ToltecInfo.interfaces) | ToltecInfo.interface_roach,
    },
)
"""TolTEC interface labels and their ROACH numbers, where applicable."""
