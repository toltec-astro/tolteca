#!/usr/bin/env python

from ..mapping.raster import SkyRasterScanModel
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
from tollan.utils.log import get_logger


def test_sky_raster_scan_model():

    logger = get_logger()
    m = SkyRasterScanModel()

    logger.debug(f"{m=!s}")
    x0, y0 = m(0. << u.s)
    logger.debug(f"t=0 {x0=!s} {y0=!s}")
    t_pattern = m.t_pattern
    logger.debug(f'{t_pattern=}')
    x, y = m(t_pattern)
    logger.debug(f"t={t_pattern} {x=!s} {y=!s}")

    t = np.arange(20).reshape((4, 5)) << u.s
    x, y = m(t)
    logger.debug(f'{t=!s} {x=!s} {y=!s}')
    assert x.shape == y.shape == t.shape
    assert x[0][0] == x0
    assert y[0][0] == y0


def test_targeted_offset_mapping_model():

    logger = get_logger()
    m = SkyRasterScanModel().get_traj_model(
        target=SkyCoord(100 << u.deg, 30 << u.deg, frame='icrs'),
        )

    logger.debug(f"{m=!s}")
    x0, y0 = m(0. << u.s)
    logger.debug(f"t=0 {x0=!s} {y0=!s}")
    t_pattern = m.t_pattern
    logger.debug(f'{t_pattern=}')
    x, y = m(t_pattern)
    logger.debug(f"t={t_pattern} {x=!s} {y=!s}")

    t = np.arange(20).reshape((4, 5)) << u.s
    x, y = m(t)
    logger.debug(f'{t=!s} {x=!s} {y=!s}')
    assert x.shape == y.shape == t.shape
    assert x[0][0] == x0
    assert y[0][0] == y0
