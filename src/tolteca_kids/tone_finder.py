from loguru import logger
from tollan.config.types import ImmutableBaseModel

from .utils import prepare_data_context

__all__ = ["ToneFinderConfig", "ToneFinder"]


class ToneFinderConfig(ImmutableBaseModel):
    """The config class for tone finding."""

    def __call__(self):
        """Return `ToneFinder` instance with this config."""
        return ToneFinder(self)


class ToneFinder:
    """A class to do tone finding."""

    def __init__(self, config):
        self._config = config

    def find_tones(self, swp):
        """Run the checker on `swp`."""
        ctx_data = prepare_data_context(swp, read=True)
        swp_filepath = ctx_data.filepath
        swp = ctx_data.data
        logger.debug(f"loaded {swp_filepath=} {swp}")

        return locals()
