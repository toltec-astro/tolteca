"""Observation-selector context for dummy-instrument interfaces."""

from tolteca_datamodels.obs_spec import ObsSpecContext

from .types import DummyInstruInfo

__all__ = ["obs_spec_context"]

obs_spec_context = ObsSpecContext(
    labels={"interface": DummyInstruInfo.interface_channel},
)
"""Dummy interface labels and channel numbers."""
