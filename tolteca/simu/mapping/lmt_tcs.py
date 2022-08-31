#!/usr/bin/env python

from .base import TrajMappingModel, PatternKind
from ...datamodels.io.toltec.tel import LmtTelFileIO
from ..lmt import lmt_observer
from .utils import resolve_sky_coords_frame

from tollan.utils.log import timeit

from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.interpolate import interp1d


class LmtTcsTrajMappingModel(TrajMappingModel):
    """The class for to describe trajectories loaded from LMT TCS ``tel.nc``
    file."""

    def __init__(self, filepath):
        super().__init__()
        self._teldata = LmtTelFileIO(filepath).read()
        self._observer = lmt_observer
        # build the interp for coords
        t_grid_s = self._teldata.time.to_value(u.s)
        ra_grid_deg = self._teldata.ra.to_value(u.deg)
        dec_grid_deg = self._teldata.dec.to_value(u.deg)
        az_grid_deg = self._teldata.az.to_value(u.deg)
        alt_grid_deg = self._teldata.alt.to_value(u.deg)

        self._interps = dict()
        self._interps['ra'] = interp1d(t_grid_s, ra_grid_deg)
        self._interps['dec'] = interp1d(t_grid_s, dec_grid_deg)
        self._interps['az'] = interp1d(t_grid_s, az_grid_deg)
        self._interps['alt'] = interp1d(t_grid_s, alt_grid_deg)
        self._interps['holdflag'] = interp1d(
                t_grid_s, self._teldata.holdflag, kind='nearest')
        self._name = f'{filepath.stem}'
        self.meta['mapping_type'] = self._teldata.meta['mapping_type']

    @property
    def target(self):
        """The target coordinates."""
        return self._teldata.target

    @property
    def ref_frame(self):
        """The reference frame with respect to which the offsets
        are interpreted."""
        return self._teldata.ref_frame

    @property
    def t0(self):
        """The start time (in UT) of the mapping."""
        return self._teldata.t0

    @property
    def observer(self):
        """The observer of the mapping."""
        return self._observer

    @property
    def t_pattern(self):
        return self._teldata.time[-1]

    @property
    def pattern_kind(self):
        _dispatch_pattern_kind = {
            'Map': PatternKind.raster
            }
        return _dispatch_pattern_kind[self._teldata.meta['mapping_type']]

    @timeit
    def evaluate_coords(self, t):
        """Return the mapping pattern coordinates as evaluated at target.
        """
        time_obs = self.t0 + t
        ref_frame = resolve_sky_coords_frame(
                self.ref_frame,
                observer=self.observer,
                time_obs=time_obs)
        if ref_frame.name == 'altaz':
            lon, lat = 'az', 'alt'
        elif ref_frame.name == 'icrs':
            lon, lat = 'ra', 'dec'
        else:
            raise ValueError(f"invalid ref_frame={ref_frame}")
        lon = self._interps[lon](t.to_value(u.s)) << u.deg
        lat = self._interps[lat](t.to_value(u.s)) << u.deg
        return SkyCoord(lon, lat, frame=ref_frame)

    @timeit
    def evaluate(self, t):
        coords = self.evaluate_coords(t)
        return coords.data.lon.to(u.deg), coords.data.lat.to(u.deg)

    @timeit
    def evaluate_holdflag(self, t):
        return self._interps['holdflag'](t.to_value(u.s))

