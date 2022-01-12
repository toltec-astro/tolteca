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


def test_toltec_sky_proj_multi():

    logger = get_logger()

    t = np.arange(0, 3, 1) << u.min
    t0 = Time('2022-01-01T00:00:00')
    target = SkyCoord(180. << u.deg, 60. << u.deg, frame='icrs')
    time_obs = t0 + t
    observer = ToltecSkyProjModel.observer
    bs_coords_altaz = target.transform_to(
        observer.altaz(time=time_obs))

    print(bs_coords_altaz)

    m = ToltecSkyProjModel(
            origin_coords_icrs=target,
            origin_coords_altaz=bs_coords_altaz,
            time_obs=time_obs)

    logger.debug(f"m: {m}")

    x_t, y_t = np.meshgrid(
        np.arange(0, 10, 1) - 5, np.arange(0, 10, 1) - 5, indexing='ij')
    x_t = x_t.ravel()[np.newaxis, :] << u.arcsec
    y_t = y_t.ravel()[np.newaxis, :] << u.arcsec
    pa_t = np.zeros_like(x_t)

    (ra, dec, pa_icrs), eval_ctx_icrs = m(
        x_t, y_t, pa_t, evaluate_frame='icrs', return_eval_context=True)
    # import pdb
    # pdb.set_trace()

    (az, alt, pa_altaz), eval_ctx_altaz = m(
        x_t, y_t, pa_t, evaluate_frame='altaz', return_eval_context=True)

    assert np.allclose(
        eval_ctx_icrs['coords_altaz'].alt,
        eval_ctx_altaz['coords_altaz'].alt)
    assert np.allclose(
        eval_ctx_icrs['coords_altaz'].az,
        eval_ctx_altaz['coords_altaz'].az)

    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2)
    for i, tt in enumerate(t):
        ax.plot(
            ra[i].arcsec, dec[i].arcsec,
            color=f'C{i % 6}', marker='.', linestyle='none')
        bx.plot(
            az[i].arcsec, alt[i].arcsec,
            color=f'C{i % 6}', marker='.', linestyle='none')
    plt.show()
