#!/usr/bin/env python


from ..toltec.models import (
    ToltecArrayProjModel, ToltecSkyProjModel)
from tollan.utils.log import get_logger
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import pytest


def test_toltec_array_proj():

    logger = get_logger()
    m = ToltecArrayProjModel()
    logger.debug(f"m: {m}")
    logger.debug(f'input_frame: {m.input_frame!r}')
    logger.debug(f'output_frame: {m.output_frame!r}')
    v = m(0, 0, 0, 1)
    np.testing.assert_almost_equal(v[-1].to_value(u.deg), 135.)
    with pytest.raises(
            ValueError, match='Invalid .* in input'):
        m(0, 0, 0, 4)


def test_toltec_sky_proj():

    logger = get_logger()

    target = SkyCoord(180. << u.deg, 60. << u.deg, frame='icrs')
    t0 = Time('2021-10-29 10:00:00')

    m = ToltecArrayProjModel() | ToltecSkyProjModel(
            origin_coords_icrs=target,
            time_obs=t0)

    logger.debug(f"m: {m}")
    logger.debug(f'input_frame: {ToltecArrayProjModel.input_frame!r}')
    logger.debug(f'output_frame: {ToltecSkyProjModel.output_frame!r}')
    v = m(0, 0, 0, 1)
    np.testing.assert_almost_equal(v[-1].to_value(u.deg), 70.74665876)
    with pytest.raises(
            ValueError, match='Invalid .* in input'):
        m(0, 0, 0, 4)
