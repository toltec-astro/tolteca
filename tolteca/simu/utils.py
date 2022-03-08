#!/usr/bin/env python

import yaml
from collections import UserDict
from dataclasses import dataclass
from typing import ClassVar

import astropy.units as u
import numpy as np
from astropy.coordinates import Longitude, Latitude, Angle, SkyCoord
from astropy.wcs import WCS

from tollan.utils import rupdate
from tollan.utils.fmt import pformat_yaml
from tollan.utils.log import get_logger


__all__ = [
    'PersistentState', 'SkyBoundingBox', 'get_lon_extent', 'make_time_grid']


class PersistentState(UserDict):
    """A class to persist state as YAML file.
    """

    def __init__(self, filepath, init=None, update=None):
        if filepath.exists():
            with open(filepath, 'r') as fo:
                state = yaml.safe_load(fo)
            if update is not None:
                rupdate(state, update)
        elif init is not None:
            state = init
        else:
            raise ValueError("cannot initialize/load persistent state")
        self._filepath = filepath
        super().__init__(state)

    def sync(self):
        """Update the YAML file with the state."""
        with open(self._filepath, 'w') as fo:
            yaml.dump(self.data, fo)
        return self

    def reload(self):
        """Update the state with the YAML file."""
        with open(self._filepath, 'r') as fo:
            state = yaml.load(fo)
        self.data = state

    def __str__(self):
        return pformat_yaml({
            'state': self.data,
            'filepath': self.filepath})

    @property
    def filepath(self):
        return self._filepath


def get_lon_extent(lon):
    """
    Return the extent for longitude array with proper re-wrapping.
    """
    logger = get_logger()

    if not isinstance(lon, Longitude):
        raise ValueError("input has to be instance of Longitude.")
    lon = np.ravel(lon)
    # this will always make sure use the original wrap angle when
    # possible since argmin will return the one with smaller index.
    wrap_angle = Angle([lon.wrap_angle.degree, 360., 180.] << u.deg)
    lon_deg_wrapped = np.vstack(
        [lon.wrap_at(a).degree for a in wrap_angle])
    # calculate the span for each wrap angle and take the smaller one
    west_deg = np.min(lon_deg_wrapped, axis=1)
    east_deg = np.max(lon_deg_wrapped, axis=1)
    dist_deg = east_deg - west_deg
    i_w = np.argmin(dist_deg)
    lon_wrap_angle = wrap_angle[i_w]
    logger.debug(
        f"re-wrapping longitude at {lon_wrap_angle}")
    return (
        Longitude(west_deg[i_w] << u.deg, wrap_angle=lon_wrap_angle),
        Longitude(east_deg[i_w] << u.deg, wrap_angle=lon_wrap_angle),
        lon_wrap_angle)


@dataclass
class SkyBoundingBox:
    """A class to represent a box (Quadrangle) on the sky."""

    w: Longitude
    e: Longitude
    s: Latitude
    n: Latitude
    lon_wrap_angle: Angle

    logger: ClassVar = get_logger()

    @property
    def center(self):
        return Angle([
            (self.w + self.e) * 0.5, (self.s + self.n) * 0.5])

    @property
    def width(self):
        return (self.e - self.w)

    @property
    def height(self):
        return (self.n - self.s)

    def pad_with(self, dx, dy):
        """Return an instance with padding."""
        coslat = np.cos((self.n + self.s) * 0.5)
        dlon = dx / coslat
        # because with the padding, the lon may go across the wrap angle
        # we need to re-compute the extent.
        w, e, lon_wrap_angle = get_lon_extent(
            Longitude(
                [self.w - dlon, self.e + dlon],
                wrap_angle=self.lon_wrap_angle))
        return self.__class__(
            w=w,
            e=e,
            s=Latitude(self.s - dy),
            n=Latitude(self.n + dy),
            lon_wrap_angle=lon_wrap_angle)

    @classmethod
    def from_lonlat(cls, lon, lat):
        """Return the sky footprint defined by `lon`, `lat`."""
        west, east, lon_wrap_angle = get_lon_extent(lon)
        south, north = Latitude(np.min(lat)), Latitude(np.max(lat))
        return cls(
            w=west, e=east, s=south, n=north, lon_wrap_angle=lon_wrap_angle)

    @classmethod
    def from_wcs(cls, wcsobj, data_shape):
        """Return the sky footprint defined by the WCS."""
        ny, nx = data_shape
        # get the corner coords for lon and lat
        lon_c, lat_c = wcsobj.wcs_pix2world(
                np.array([0, 0, nx - 1, nx - 1]),
                np.array([0, ny - 1, 0, ny - 1]),
                0)
        lon_c = Longitude(lon_c << u.deg)
        lat_c = Latitude(lat_c << u.deg)
        return cls.from_lonlat(lon_c, lat_c)

    # def __str__(self):
    #     return (
    #         f"{self.__class__.__name__}"
    #         f"(lon=[{self.w.degree}:{self.e.degree}] deg, "
    #         f"lat=[{self.s.degree}, {self.n.degree}] deg, "
    #         f"lon_wrap_angle={self.lon_wrap_angle})")


def make_time_grid(t, f_smp, chunk_len=None):
    """Return equal-bin time grid from 0 to `t` with `f_smp`.

    When `chunk_len` is set, the time grid is divided to multiple chunks.
    """
    logger = get_logger()
    t_grid = np.arange(
            0, t.to_value(u.s),
            (1 / f_smp).to_value(u.s)) * u.s
    if chunk_len is None:
        return t_grid
    # make chunked grid
    n_times_per_chunk = int((
            chunk_len * f_smp).to_value(
                    u.dimensionless_unscaled))
    n_times = len(t_grid)
    n_chunks = n_times // n_times_per_chunk + bool(
            n_times % n_times_per_chunk)
    t_chunks = []
    for i in range(n_chunks):
        t_chunks.append(
                t_grid[i * n_times_per_chunk:(i + 1) * n_times_per_chunk])
    # merge the last chunk if it is too small
    if n_chunks >= 2:
        if len(t_chunks[-1]) * 10 < len(t_chunks[-2]):
            last_chunk = t_chunks.pop()
            t_chunks[-1] = np.hstack([t_chunks[-1], last_chunk])
    n_chunks = len(t_chunks)
    logger.info(
            f"make time chunks with n_times_per_chunk={n_times_per_chunk}"
            f" n_times={n_times} n_chunks={n_chunks}")
    return t_chunks


def make_wcs(
        sky_bbox, pixscale,
        crval=None,
        n_pix_max=None,
        adaptive_pixscale_factor=1):
    """Return an `astropy.wcs.WCS` for the sky bbox and pixel scale.
    
    When `n_pix_max` is set, the pixscale is adapted so the number of pixels
    is within the limit. In this case, the adopted pixel scale is
    ``i * adpative_pixscale_factor * pixscale``, where ``i`` is some integer.
    """
    logger = get_logger()
    pixsize = (1 << u.pix).to(u.arcsec, equivalencies=pixscale)
    nx = (sky_bbox.width / pixsize).to_value(u.dimensionless_unscaled)
    ny = (sky_bbox.height / pixsize).to_value(u.dimensionless_unscaled)
    nx, ny = int(np.ceil(nx)), int(np.ceil(ny))
    logger.debug(f"image wcs shape: {nx=} {ny=} {pixsize=}")
    if n_pix_max is not None and  nx * ny > n_pix_max:
        s = (nx * ny / n_pix_max) ** 0.5
        # round s (to larger) so that is is int number of apf
        s = np.ceil(s / adaptive_pixscale_factor) * adaptive_pixscale_factor
        snx = nx / s
        sny = ny / s
        sps = pixsize * s
        logger.debug(
                f"adjusted pixel shape: nx={snx} ny={sny} pixsize={sps}")
        nx, ny, pixsize = snx, sny, sps
    wcsobj = WCS(naxis=2)
    wcsobj.pixel_shape = (nx, ny)
    wcsobj.wcs.crpix = [nx / 2, ny / 2]
    wcsobj.wcs.cdelt = np.array([
            -pixsize.to_value(u.deg),
            pixsize.to_value(u.deg),
            ])
    wcsobj.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    if crval is not None:
        if isinstance(crval, SkyCoord):
            crval = [crval.ra.degree, crval.dec.degree]
        else:
            crval = [
                crval[0].to_value(u.degree),
                crval[1].to_value(u.degree),
                ]
    else:
        crval = [v.degree for v in sky_bbox.center]
    wcsobj.wcs.crval = crval 
    return wcsobj