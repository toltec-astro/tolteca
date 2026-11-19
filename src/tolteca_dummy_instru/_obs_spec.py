"""Observation-selector context for dummy-instrument interfaces."""

from tolteca_datamodels.core.obs_spec import ObsSpecContext
from tolteca_datamodels.lmt.types import LmtInfo

from .types import DummyInstruInfo

_obs_spec_context = ObsSpecContext(
    field_label_numbers={
        "interface": dict.fromkeys(LmtInfo.interfaces)
        | DummyInstruInfo.interface_channel,
    },
)
"""Dummy interface labels and channel numbers."""
