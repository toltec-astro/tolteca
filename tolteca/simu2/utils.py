#!/usr/bin/env python

import yaml
from collections import UserDict
from dataclasses import dataclass
from typing import ClassVar

import astropy.units as u
import numpy as np
from astropy.coordinates import Longitude, Latitude, Angle

from tollan.utils import rupdate
from tollan.utils.fmt import pformat_yaml
from tollan.utils.log import get_logger


__all__ = ['PersistentState', 'SkyBoundingBox', 'make_time_grid']


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
        return u.Quantity([
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
        return self.__class__(
            w=self.w - dlon,
            e=self.e + dlon,
            s=self.s - dy,
            n=self.n + dy,
            lon_wrap_angle=self.lon_wrap_angle)

    @classmethod
    def _get_lon_extent(cls, lon):
        """
        Return the extent for longitude array with proper re-wrapping.
        """
        if not isinstance(lon, Longitude):
            raise ValueError("input has to be instance of Longitude.")
        lon = np.ravel(lon)
        wrap_angle = Angle([180., 360.] << u.deg)
        lon_deg_wrapped = np.vstack(
            [lon.wrap_at(a).degree for a in wrap_angle])
        # calculate the span for each wrap angle and take the smaller one
        west_deg = np.min(lon_deg_wrapped, axis=1)
        east_deg = np.max(lon_deg_wrapped, axis=1)
        dist_deg = east_deg - west_deg
        i_w = np.argmin(dist_deg)
        lon_wrap_angle = wrap_angle[i_w]
        cls.logger.debug(
            f"re-wrapping longitude at {lon_wrap_angle}")
        return (
            Longitude(west_deg[i_w] << u.deg),
            Longitude(east_deg[i_w] << u.deg),
            lon_wrap_angle)

    @classmethod
    def from_lonlat(cls, lon, lat):
        """Return the sky footprint defined by `lon`, `lat`."""
        west, east, lon_wrap_angle = cls._get_lon_extent(lon)
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
