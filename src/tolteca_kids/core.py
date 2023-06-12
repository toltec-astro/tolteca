from pydantic import Field

from tolteca_config.core import ConfigModel, ConfigHandler, SubConfigKeyTransformer

from .sweep_check import SweepChecker

# from .sweep_fitter import SweepFitter
# from .tone_finder import ToneFinder

__all__ = ["Kids"]


class KidsConfig(ConfigModel):
    """The config for KIDs data handling workflow."""

    sweep_checker: SweepChecker = Field(default_factory=dict)
    # tone_finder: ToneFinder = Field(default_factory=dict)
    # sweep_fitter: SweepFitter = Field(default_factory=dict)


class Kids(SubConfigKeyTransformer["kids"], ConfigHandler[KidsConfig]):
    """The class to work with KIDs data."""

    @property
    def sweep_checker(self):
        """The sweep checker."""
        return self.config.sweep_checker

    @property
    def tone_finder(self):
        """The tone finder."""
        return self.config.tone_finder()

    @property
    def sweep_fitter(self):
        """The tone finder."""
        return self.config.sweep_fitter()
