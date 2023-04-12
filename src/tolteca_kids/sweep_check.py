#!/usr/bin/env python

import os
from loguru import logger
from tollan.config.types import ImmutableBaseModel
from .utils import DebugConfig
from ..datamodels.io.toltec import NcFileIO
from kidsproc.kidsdata.Sweep import MultiSweep


class SweepCheckerConfig(ImmutableBaseModel):
    """The config of sweep checker."""

    def __call__(self, cfg):
        return SweepChecker(self)


class SweepChecker(object):
    """A class to handle sweep checking."""

    _debug_plot_options = {
        "gi0": 0,
        "di0": 0,
        "n_rows": 5,
        "n_cols": 5,
        "panel_width": 3,
        "panel_width": 3,
    }

    def __init__(self, config):
        self._config = config

    @classmethod
    def _check_spike(cls, swp):
        pass

    @classmethod
    def _check_stats(cls, swp):
        pass

    @classmethod
    def _check_noise(cls, swp):
        pass

    @classmethod
    def _ensure_data(cls, swp):
        if isinstance(swp, (str, os.PathLike)):
            swp_file = swp
            swp_io = NcFileIO(swp_file)
            swp = swp_io.read()
        elif isinstance(swp, NcFileIO):
            swp_io = swp
            swp_file = swp_io.filepath
            swp = swp_io.read()
        elif isinstance(swp, MultiSweep):
            swp_io = None
            swp_file = swp.meta["filepath"]
        else:
            raise ValueError("invalid input swp")
        return swp_file, swp_io, swp

    def check(self, swp):
        swp_file, swp_io, swp = self._ensure_data(swp)
        logger.debug(f"loaded {swp_file=} {swp}")
