#!/usr/bin/env python


from .base import OffsetMappingModel, PatternKind
from .utils import rotation_matrix_2d
from .raster import RasterScanModelMeta

from tollan.utils.log import timeit

import numpy as np
import astropy.units as u
from astropy.modeling import Parameter
from gwcs import coordinate_frames as cf


__all__ = ['SkyLissajousModel', 'SkyDoubleLissajousModel', 'SkyRastajousModel']


class LissajousModelMeta(OffsetMappingModel.__class__):
    """A generic implementation of lissajous pattern in offset frame."""

    def __new__(meta, name, bases, attrs):
        meta.update_model_attrs(attrs)
        return super().__new__(meta, name, bases, attrs)

    @classmethod
    def update_model_attrs(meta, attrs):
        frame = attrs['frame']
        frame_unit = attrs['frame_unit'] = frame.unit[0]
        attrs.update(
            x_length=Parameter(
                default=10., unit=frame_unit,
                description='Size of pattern in x direction.'
                ),
            y_length=Parameter(
                default=10., unit=frame_unit,
                description='Size of pattern in y direction.'
                ),
            x_omega=Parameter(
                default=1., unit=u.rad / u.s,
                description='Angular frequency in x direction.'
                ),
            y_omega=Parameter(
                default=1., unit=u.rad / u.s,
                description='Angular frequency in y direction.'
                ),
            delta=Parameter(
                default=0., unit=u.rad,
                description='Phase angle of x with respect to y.'
                ),
            rot=Parameter(
                default=0., unit=u.deg,
                description='The rotation angle with respect '
                            'to the +lon direction.'),
            )
        attrs['pattern_name'] = meta.pattern_name
        attrs['pattern_kind'] = meta.pattern_kind
        attrs['evaluate'] = staticmethod(meta._evaluate)

        def evaluate_holdflag(self, t):
            return np.zeros(t.shape, dtype=bool)

        attrs['evaluate_holdflag'] = evaluate_holdflag
        attrs['t_pattern'] = property(meta.calc_t_pattern)
        return attrs

    pattern_name = 'lissajous'
    pattern_kind = PatternKind.lissajous

    @staticmethod
    @timeit
    def _evaluate(
            t,
            x_length, y_length, x_omega, y_omega, delta, rot):
        """Compute a lissajous pattern in a generic offset frame.

        """
        t = np.asarray(t) * t.unit
        lunit = x_length.unit

        xy = np.empty((2, ) + t.shape, dtype=np.float64) << lunit
        xy[0] = x_length * 0.5 * np.sin(x_omega * t + delta)
        xy[1] = y_length * 0.5 * np.sin(y_omega * t)

        # apply rotation
        xx, yy = np.einsum('ij...,j...->i...',
                           rotation_matrix_2d(rot.to_value(u.rad)),
                           xy)
        return xx, yy

    @staticmethod
    def _calc_t_pattern(x_omega, y_omega):
        t_x = 2 * np.pi * u.rad / x_omega
        t_y = 2 * np.pi * u.rad / y_omega
        r = (t_y / t_x).to_value(u.dimensionless_unscaled)
        # an upscaling factor to round r to the 2nd decimal
        s = 100
        r = np.lcm(int(r * s), s) / s
        return (t_x * r).to(u.s)

    @classmethod
    def calc_t_pattern(cls, m):
        """Compute the time to execute the full pattern for model."""
        return cls._calc_t_pattern(m.x_omega, m.y_omega)


class SkyLissajousModel(
        OffsetMappingModel, metaclass=LissajousModelMeta):
    """The lissajous model in sky offset frame."""

    frame = cf.Frame2D(
        name='skyoffset', axes_names=('lon', 'lat'),
        unit=(u.deg, u.deg)
        )


class DoubleLissajousModelMeta(OffsetMappingModel.__class__):
    """A generic implementation of double lissajous pattern in offset frame."""

    def __new__(meta, name, bases, attrs):
        meta.update_model_attrs(attrs)
        return super().__new__(meta, name, bases, attrs)

    @classmethod
    def update_model_attrs(meta, attrs):
        frame = attrs['frame']
        frame_unit = attrs['frame_unit'] = frame.unit[0]
        attrs.update(
            x_length_0=Parameter(
                default=10., unit=frame_unit,
                description='Size of major pattern in x direction.'
                ),
            y_length_0=Parameter(
                default=10., unit=frame_unit,
                description='Size of major pattern in y direction.'
                ),
            x_omega_0=Parameter(
                default=1., unit=u.rad / u.s,
                description='Angular frequency in x for major pattern.'
                ),
            y_omega_0=Parameter(
                default=1., unit=u.rad / u.s,
                description='Angular frequency in y for major pattern.'
                ),
            delta_0=Parameter(
                default=0., unit=u.rad,
                description=(
                    'Phase angle of x with respect to y for major pattern')
                ),
            x_length_1=Parameter(
                default=5., unit=frame_unit,
                description='Size of minor pattern in x direction.'
                ),
            y_length_1=Parameter(
                default=5., unit=frame_unit,
                description='Size of minor pattern in y direction.'
                ),
            x_omega_1=Parameter(
                default=1., unit=u.rad / u.s,
                description='Angular frequency in x for minor pattern.'
                ),
            y_omega_1=Parameter(
                default=1., unit=u.rad / u.s,
                description='Angular frequency in y for minor pattern.'
                ),
            delta_1=Parameter(
                default=0., unit=u.rad,
                description=(
                    'Phase angle of x with respect to y for minor pattern')
                ),
            delta=Parameter(
                default=0., unit=u.rad,
                description=(
                    'Phase angle of major pattern y with respect '
                    'to minor pattern y')
                ),
            rot=Parameter(
                default=0., unit=u.deg,
                description='The rotation angle with respect '
                            'to the +lon direction.'),
            )
        attrs['pattern_name'] = meta.pattern_name
        attrs['pattern_kind'] = meta.pattern_kind
        attrs['evaluate'] = staticmethod(meta._evaluate)

        def evaluate_holdflag(self, t):
            return np.zeros(t.shape, dtype=bool)

        attrs['evaluate_holdflag'] = evaluate_holdflag
        attrs['t_pattern'] = property(meta.calc_t_pattern)
        return attrs

    pattern_name = 'double_lissajous'
    pattern_kind = PatternKind.lissajous

    @staticmethod
    @timeit
    def _evaluate(
            t,
            x_length_0, y_length_0, x_omega_0, y_omega_0, delta_0,
            x_length_1, y_length_1, x_omega_1, y_omega_1, delta_1,
            delta, rot):
        """Compute a double lissajous pattern in a generic offset frame.

        """
        t = np.asarray(t) * t.unit
        lunit = x_length_0.unit

        x_0 = x_length_0 * 0.5 * np.sin(x_omega_0 * t + delta + delta_0)
        y_0 = y_length_0 * 0.5 * np.sin(y_omega_0 * t + delta)
        x_1 = x_length_1 * 0.5 * np.sin(x_omega_1 * t + delta_1)
        y_1 = y_length_1 * 0.5 * np.sin(y_omega_1 * t)

        xy = np.empty((2, ) + t.shape, dtype=np.float64) << lunit
        xy[0] = x_0 + x_1
        xy[1] = y_0 + y_1

        # apply rotation
        xx, yy = np.einsum('ij...,j...->i...',
                           rotation_matrix_2d(rot.to_value(u.rad)),
                           xy)
        return xx, yy

    @staticmethod
    def calc_t_pattern(m):
        """Compute the time to execute the full pattern for model."""
        t0 = LissajousModelMeta._calc_t_pattern(m.x_omega_0, m.y_omega_0)
        t1 = LissajousModelMeta._calc_t_pattern(m.x_omega_1, m.y_omega_1)
        return t0 if t0 > t1 else t1


class SkyDoubleLissajousModel(
        OffsetMappingModel, metaclass=DoubleLissajousModelMeta):
    """The double lissajous model in sky offset frame."""

    frame = cf.Frame2D(
        name='skyoffset', axes_names=('lon', 'lat'),
        unit=(u.deg, u.deg)
        )


class RastajousModelMeta(OffsetMappingModel.__class__):
    """A generic implementation of rastajous pattern in offset frame."""

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
            x_length_0=Parameter(
                default=10., unit=frame_unit,
                description='Size of major pattern in x direction.'
                ),
            y_length_0=Parameter(
                default=10., unit=frame_unit,
                description='Size of major pattern in y direction.'
                ),
            x_omega_0=Parameter(
                default=1., unit=u.rad / u.s,
                description='Angular frequency in x for major pattern.'
                ),
            y_omega_0=Parameter(
                default=1., unit=u.rad / u.s,
                description='Angular frequency in y for major pattern.'
                ),
            delta_0=Parameter(
                default=0., unit=u.rad,
                description=(
                    'Phase angle of x with respect to y for major pattern')
                ),
            x_length_1=Parameter(
                default=5., unit=frame_unit,
                description='Size of minor pattern in x direction.'
                ),
            y_length_1=Parameter(
                default=5., unit=frame_unit,
                description='Size of minor pattern in y direction.'
                ),
            x_omega_1=Parameter(
                default=1., unit=u.rad / u.s,
                description='Angular frequency in x for minor pattern.'
                ),
            y_omega_1=Parameter(
                default=1., unit=u.rad / u.s,
                description='Angular frequency in y for minor pattern.'
                ),
            delta_1=Parameter(
                default=0., unit=u.rad,
                description=(
                    'Phase angle of x with respect to y for minor pattern')
                ),
            delta=Parameter(
                default=0., unit=u.rad,
                description=(
                    'Phase angle of major pattern y with respect '
                    'to minor pattern y')
                ),
            )
        attrs['pattern_name'] = meta.pattern_name
        attrs['pattern_kind'] = meta.pattern_kind
        attrs['evaluate'] = staticmethod(meta._evaluate)

        def evaluate_holdflag(self, t):
            return np.zeros(t.shape, dtype=bool)

        attrs['evaluate_holdflag'] = evaluate_holdflag
        attrs['t_pattern'] = property(meta.calc_t_pattern)
        return attrs

    pattern_name = 'rastajous'
    pattern_kind = PatternKind.lissajous

    @staticmethod
    @timeit
    def _evaluate(
            t,
            length, space, n_scans, rot, speed, t_turnaround,
            x_length_0, y_length_0, x_omega_0, y_omega_0, delta_0,
            x_length_1, y_length_1, x_omega_1, y_omega_1, delta_1,
            delta):
        """Compute a rastajous pattern in a generic offset frame.

        """
        t = np.asarray(t) * t.unit
        lunit = x_length_0.unit

        x_r, y_r = RasterScanModelMeta._evaluate(
            t=t, length=length, space=space, n_scans=n_scans, rot=rot,
            speed=speed, t_turnaround=t_turnaround,
            return_holdflag_only=False
            )
        x_l, y_l = DoubleLissajousModelMeta._evaluate(
            t=t,
            x_length_0=x_length_0, y_length_0=y_length_0,
            x_omega_0=x_omega_0, y_omega_0=y_omega_0, delta_0=delta_0,
            x_length_1=x_length_1, y_length_1=y_length_1,
            x_omega_1=x_omega_1, y_omega_1=y_omega_1, delta_1=delta_1,
            delta=delta, rot=rot
            )
        xy = np.empty((2, ) + t.shape, dtype=np.float64) << lunit
        xy[0] = x_r + x_l
        xy[1] = y_r + y_l

        # apply rotation
        xx, yy = np.einsum('ij...,j...->i...',
                           rotation_matrix_2d(rot.to_value(u.rad)),
                           xy)
        return xx, yy

    @staticmethod
    def calc_t_pattern(m):
        """Compute the time to execute the full pattern for model."""
        # total pattern time is based on the raster pattern
        return RasterScanModelMeta.calc_t_pattern(m)


class SkyRastajousModel(
        OffsetMappingModel, metaclass=RastajousModelMeta):
    """The rastajous model in sky offset frame."""

    frame = cf.Frame2D(
        name='skyoffset', axes_names=('lon', 'lat'),
        unit=(u.deg, u.deg)
        )
