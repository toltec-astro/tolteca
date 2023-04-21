from tollan.config.types import ImmutableBaseModel

__all__ = ["SweepFitterConfig", "SweepFitter"]


class SweepFitterConfig(ImmutableBaseModel):
    """The config class for sweep model fitting."""

    def __call__(self):
        """Return `SweepChecker` instance with this config."""
        return SweepFitter(self)


class SweepFitter:
    """A class to do sweep model fitting."""

    def __init__(self, config):
        self._config = config
