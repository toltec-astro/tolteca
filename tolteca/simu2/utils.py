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


__all__ = ['PersistentState', 'SkyBoundingBox']


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
