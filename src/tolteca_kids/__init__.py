from tollan.config.types import ImmutableBaseModel
from .sweep_check import SweepCheckerConfig

__version__ = "v0.0.1"


class KidsConfig(ImmutableBaseModel):
    sweep_checker: SweepCheckerConfig


class ToltecaConfig(ImmutableBaseModel):
    kids: KidsConfig
