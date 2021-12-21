#!/usr/bin/env python


from .base import OffsetMappingModel, PatternKind
from .utils import rotation_matrix_2d

from tollan.utils.log import timeit

import numpy as np
import astropy.units as u
from astropy.modeling import Parameter
from gwcs import coordinate_frames as cf
import functools


__all__ = ['SkyRasterScanModel', ]


class RasterScanModelMeta(OffsetMappingModel.__class__):
    """A generic implementation of raster scan pattern in offset frame."""

    def __new__(meta, name, bases, attrs):
        meta.update_model_attrs(attrs)
        return super().__new__(meta, name, bases, attrs)

    @classmethod
    def update_model_attrs(meta, attrs):
        frame = attrs['frame']
        frame_unit = attrs['frame_unit'] = frame.unit[0]
        attrs.update(
            length=Parameter(
                default=10., unit=frame_unit,
                description='The length of scan.'),
            space=Parameter(
                default=1., unit=frame_unit,
                description='The space between scans.'),
            n_scans=Parameter(
                default=10., unit=u.dimensionless_unscaled,
                description='The number of scans.'),
            rot=Parameter(
                default=0., unit=u.deg,
                description='The rotation angle with respect '
                            'to the +lon direction.'),
            speed=Parameter(
                default=1., unit=frame_unit / u.s,
                description='The scan speed.'),
            t_turnaround=Parameter(
                default=1., unit=u.s,
                description='The time for turning around after each scan.'),
            )
        attrs['pattern_name'] = meta.pattern_name
        attrs['pattern_kind'] = meta.pattern_kind
        attrs['evaluate'] = functools.partial(
            meta._evaluate, return_holdflag_only=False)

        def evaluate_holdflag(self, t):
            return meta._evaluate(
                t,
                length=self.length, space=self.space, n_scans=self.n_scans,
                rot=self.rot, speed=self.speed, t_turnaround=self.t_turnaround,
                return_holdflag_only=True)
        attrs['evaluate_holdflag'] = evaluate_holdflag
        attrs['t_pattern'] = property(meta.calc_t_pattern)
        return attrs

    pattern_name = 'raster'
    pattern_kind = PatternKind.raster

    @staticmethod
    @timeit
    def _evaluate(
            t,
            length, space, n_scans, rot, speed, t_turnaround,
            return_holdflag_only=False):
        """Compute a raster pattern in a generic offset frame.

        This assumes a circular turn around trajectory where the speed
        of the turn around is implicitly controlled by `t_turnaround`.
        """
        lunit = length.unit
        t_per_scan = length / speed
        holdflag = np.zeros(t.shape, dtype=bool)
        xy = np.empty((2, ) + t.shape, dtype=np.float64) << lunit
        if n_scans == 1:
            if return_holdflag_only:
                return holdflag
            # have to make a special case here to compute x and y
            xy[0] = (t / t_per_scan - 0.5) * length
            xy[1] = 0 << lunit
        else:
            n_spaces = n_scans - 1
            ratio_scan_to_si = (
                    t_per_scan / (t_turnaround + t_per_scan))
            ratio_scan_to_turnaround = (t_per_scan / t_turnaround)

            # scan index
            _si = (t / (t_turnaround + t_per_scan))
            si = _si.astype(int)
            si_frac = _si - si

            # get scan and turnaround part
            scan_frac = np.empty_like(si_frac)
            turnaround_frac = np.empty_like(si_frac)
            turnaround = si_frac > ratio_scan_to_si
            if return_holdflag_only:
                holdflag[turnaround] = True
                return holdflag
            # continue to compute x and y
            scan_frac[turnaround] = 1.
            scan_frac[~turnaround] = si_frac[~turnaround] / ratio_scan_to_si
            turnaround_frac[turnaround] = si_frac[turnaround] - (
                    1. - si_frac[turnaround]) * ratio_scan_to_turnaround
            turnaround_frac[~turnaround] = 0.

            xy[0] = (scan_frac - 0.5) * length
            xy[1] = (si / n_spaces - 0.5) * n_spaces * space

            # turnaround part
            radius_t = space / 2
            theta_t = (turnaround_frac[turnaround] * np.pi).to(
                u.rad, u.dimensionless_angles())
            dy = radius_t * (1. - np.cos(theta_t))
            dx = radius_t * np.sin(theta_t)
            xy[0, turnaround] += dx
            xy[1, turnaround] += dy
            # make continuous
            xy[0] *= (-1) ** si

        # apply rotation
        lunit = length.unit
        xx, yy = np.einsum('ij...,j...->i...',
                           rotation_matrix_2d(rot.to_value(u.rad)),
                           xy)
        return xx, yy

    @staticmethod
    def calc_t_pattern(m):
        """Compute the time to execute the full pattern for model."""
        return (
            m.length / m.speed * m.n_scans
            + m.t_turnaround * (m.n_scans - 1.)).to(u.s)


class SkyRasterScanModel(
        OffsetMappingModel, metaclass=RasterScanModelMeta):
    """The raster scan model in sky offset frame."""

    frame = cf.Frame2D(
        name='skyoffset', axes_names=('lon', 'lat'),
        unit=(u.deg, u.deg)
        )
