#! /usr/bin/env python

import numpy as np
import inspect

from astropy.coordinates.baseframe import frame_transform_graph
from astropy.modeling import Model, Parameter
import astropy.units as u
from astropy.modeling import models
from astropy import coordinates as coord

from gwcs import coordinate_frames as cf
from scipy import interpolate

from tollan.utils.log import get_logger
from tollan.utils import getobj
from tollan.utils.log import timeit
from tollan.utils.namespace import NamespaceMixin


def _get_skyoffset_frame(c):
    """This function creates a skyoffset_frame and ensures
    the cached origin frame attribute is the correct instance.
    """
    frame = c.skyoffset_frame()
    frame_transform_graph._cached_frame_attributes['origin'] = \
        frame.frame_attributes['origin']
    return frame


class _Model(Model, NamespaceMixin):

    _namespace_type_key = 'model'

    @classmethod
    def _namespace_from_dict_op(cls, d):
        # we resolve the model here so that we can allow
        # one use only the model class name to specify a model class.
        if cls._namespace_type_key not in d:
            raise ValueError(
                    f'unable to load model: '
                    f'missing required key "{cls._namespace_type_key}"')
        model_cls = cls._resolve_model_cls(d[cls._namespace_type_key])
        return dict(d, **{cls._namespace_type_key: model_cls})

    @staticmethod
    def _resolve_model_cls(arg):
        """Return a template class specified by `arg`.

        If `arg` is string, it is resolved using `tollan.utils.getobj`.
        """
        logger = get_logger()

        _arg = arg  # for logging
        if isinstance(arg, str):
            arg = getobj(arg)
        # check if _resolve_template_cls attribute is present
        if inspect.ismodule(arg):
            raise ValueError(f"cannot resolve model class from {arg}")
        if not isinstance(arg, Model):
            raise ValueError(f"cannot resolve model class from {arg}")
        model_cls = arg
        logger.debug(
                f"resolved model {_arg} as {model_cls}")
        return model_cls


class ProjModel(_Model):
    """Base class for models that transform the detector locations.
    """

    def __init__(self, t0=None, target=None, *args, **kwargs):
        inputs = kwargs.pop('inputs', self.input_frame.axes_names)
        outputs = kwargs.pop('outputs', self.output_frame.axes_names)
        kwargs.setdefault('name', self._name)
        super().__init__(*args, **kwargs)
        self.inputs = inputs
        self.outputs = outputs
        self._t0 = t0
        self._target = target

    def mpl_axes_params(self):
        return dict(aspect='equal')


class SkyMapModel(_Model):
    """A model that describes mapping patterns on the sky.

    It computes the sky coordinates as a function of the time.
    """

    n_inputs = 1
    n_outputs = 2

    def __init__(self, t0=None, target=None, ref_frame=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._t0 = t0
        self._target = target
        self._ref_frame = ref_frame

    def evaluate(self, x, y):
        return NotImplemented

    def evaluate_at(self, ref_coord, *args):
        """Returns the mapping pattern as evaluated at given coordinates.
        """
        frame = _get_skyoffset_frame(ref_coord)
        return coord.SkyCoord(*self(*args), frame=frame).transform_to(
                ref_coord.frame)

    # TODO these three method seems to be better live in a wrapper
    # model rather than this model
    @property
    def t0(self):
        return self._t0

    @property
    def target(self):
        return self._target

    @property
    def ref_frame(self):
        return self._ref_frame


class RasterScanModelMeta(SkyMapModel.__class__):
    """A meta class that defines a raster scan pattern.

    This is implemented as a meta class so that we can reuse it
    in any map model of any coordinate frame.
    """

    def __new__(meta, name, bases, attrs):
        frame = attrs['frame']
        frame_unit = frame.unit[0]

        attrs.update(dict(
            frame_unit=frame_unit,
            length=Parameter(default=10., unit=frame_unit),
            space=Parameter(default=1., unit=frame_unit),
            n_scans=Parameter(default=10., unit=u.dimensionless_unscaled),
            rot=Parameter(default=0., unit=u.deg),
            speed=Parameter(default=1., unit=frame_unit / u.s),
            # accel=Parameter(default=1., unit=cls.frame_unit / u.s ** 2),
            t_turnover=Parameter(default=1., unit=u.s),
                ))

        def get_total_time(self):
            return self.length / self.speed * self.n_scans + \
                    self.t_turnover * (self.n_scans - 1.)

        attrs['get_total_time'] = get_total_time

        @timeit(name)
        def evaluate(
                self, t, length, space, n_scans, rot, speed, t_turnover):
            """This computes a raster patten around the origin.

            This assumes a circular turn over trajectory where the
            speed of the turn over is implicitly controlled by `t_turnover`.
            """
            t = np.asarray(t) * t.unit
            t_per_scan = length / speed

            if n_scans == 1:
                # have to make a special case here
                x = (t / t_per_scan - 0.5) * length
                y = np.zeros(t.shape) << length.unit
            else:
                n_spaces = n_scans - 1

                # bbox_width = length
                # bbox_height = space * n_spaces
                # # (x0, y0, w, h)
                # bbox = (
                #         -bbox_width / 2., -bbox_height / 2.,
                #         bbox_width, bbox_height)
                t_per_scan = length / speed
                ratio_scan_to_si = (
                        t_per_scan / (t_turnover + t_per_scan))
                ratio_scan_to_turnover = (t_per_scan / t_turnover)

                # scan index
                _si = (t / (t_turnover + t_per_scan))
                si = _si.astype(int)
                si_frac = _si - si

                # get scan and turnover part
                scan_frac = np.empty_like(si_frac)
                turnover_frac = np.empty_like(si_frac)

                turnover = si_frac > ratio_scan_to_si
                scan_frac[turnover] = 1.
                scan_frac[~turnover] = si_frac[~turnover] / ratio_scan_to_si
                turnover_frac[turnover] = si_frac[turnover] - (
                        1. - si_frac[turnover]) * ratio_scan_to_turnover
                turnover_frac[~turnover] = 0.

                x = (scan_frac - 0.5) * length
                y = (si / n_spaces - 0.5) * n_spaces * space

                # turnover part
                radius_t = space / 2
                theta_t = turnover_frac[turnover] * np.pi * u.rad
                dy = radius_t * (1 - np.cos(theta_t))
                dx = radius_t * np.sin(theta_t)
                x[turnover] = x[turnover] + dx
                y[turnover] = y[turnover] + dy

                # make continuous
                x = x * (-1) ** si

            # apply rotation
            m_rot = models.AffineTransformation2D(
                models.Rotation2D._compute_matrix(
                    angle=rot.to_value('rad')) * self.frame_unit,
                translation=(0., 0.) * self.frame_unit)
            xx, yy = m_rot(x, y)
            return xx, yy

        attrs['evaluate'] = evaluate
        return super().__new__(meta, name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        inst = super().__call__(*args, **kwargs)
        inst.inputs = ('t', )
        inst.outputs = cls.frame.axes_names
        return inst


class LissajousModelMeta(SkyMapModel.__class__):
    """A meta class that defines a Lissajous scan pattern.

    This is implemented as a meta class so that we can reuse it
    in any map model of any coordinate frame.
    """

    def __new__(meta, name, bases, attrs):
        frame = attrs['frame']
        frame_unit = frame.unit[0]

        attrs.update(dict(
            frame_unit=frame_unit,
            x_length=Parameter(default=10., unit=frame_unit),
            y_length=Parameter(default=10., unit=frame_unit),
            x_omega=Parameter(default=1. * u.rad / u.s),
            y_omega=Parameter(default=1. * u.rad / u.s),
            delta=Parameter(default=0., unit=u.rad),
            rot=Parameter(default=0., unit=u.deg),
                ))

        def get_total_time(self):
            t_x = 2 * np.pi * u.rad / self.x_omega
            t_y = 2 * np.pi * u.rad / self.y_omega
            r = (t_y / t_x).to_value(u.dimensionless_unscaled)
            s = 100
            r = np.lcm(int(r * s), s) / s
            return t_x * r

        attrs['get_total_time'] = get_total_time

        @timeit(name)
        def evaluate(
                self, t, x_length, y_length, x_omega, y_omega, delta, rot):
            """This computes a lissajous pattern around the origin.

            """
            t = np.asarray(t) * t.unit

            x = x_length * 0.5 * np.sin(x_omega * t + delta)
            y = y_length * 0.5 * np.sin(y_omega * t)

            m_rot = models.AffineTransformation2D(
                models.Rotation2D._compute_matrix(
                    angle=rot.to_value('rad')) * self.frame_unit,
                translation=(0., 0.) * self.frame_unit)
            xx, yy = m_rot(x, y)
            return xx, yy

        attrs['evaluate'] = evaluate
        return super().__new__(meta, name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        inst = super().__call__(*args, **kwargs)
        inst.inputs = ('t', )
        inst.outputs = cls.frame.axes_names
        return inst


class TrajectoryModelMeta(SkyMapModel.__class__):
    """A meta class that defines a trajectory.

    This is implemented as a meta class so that we can reuse it
    in any map model of any coordinate frame.
    """

    def __new__(meta, name, bases, attrs):
        frame = attrs['frame']
        frame_unit = frame.unit[0]

        attrs.update(dict(
            frame_unit=frame_unit,
                ))

        def get_total_time(self):
            return self._time[-1]

        attrs['get_total_time'] = get_total_time

        lon_attr, lat_attr = {
                'icrs': ('_ra', '_dec'),
                'altaz': ('_az', '_alt'),
                }[frame.name]

        @property
        def _lon(self):
            return getattr(self, lon_attr)

        attrs['_lon_attr'] = lon_attr
        attrs['_lon'] = _lon

        @property
        def _lat(self):
            return getattr(self, lat_attr)

        attrs['_lat_attr'] = lat_attr
        attrs['_lat'] = _lat

        @timeit(name)
        def evaluate(
                self, t):
            """This computes the position based on interpolation.

            """
            return self._lon_interp(t), self._dec_interp(t)

        attrs['evaluate'] = evaluate
        return super().__new__(meta, name, bases, attrs)

    def __call__(
            cls, *args, **kwargs):
        data = dict()
        for attr in ('time', 'ra', 'dec', 'az', 'alt'):
            data[f'_{attr}'] = kwargs.pop(attr, None)
        inst = super().__call__(*args, **kwargs)
        inst.__dict__.update(data)
        inst.inputs = ('t', )
        inst.outputs = cls.frame.axes_names
        inst._lon_interp = interpolate.interp1d(inst._time, inst._lon)
        inst._lat_interp = interpolate.interp1d(inst._time, inst._lat)
        return inst


class SkyRasterScanModel(SkyMapModel, metaclass=RasterScanModelMeta):
    frame = cf.Frame2D(
            name='skyoffset', axes_names=('lon', 'lat'),
            unit=(u.deg, u.deg))


class SkyLissajousModel(SkyMapModel, metaclass=LissajousModelMeta):
    frame = cf.Frame2D(
            name='skyoffset', axes_names=('lon', 'lat'),
            unit=(u.deg, u.deg))


class SkyICRSTrajModel(SkyMapModel, metaclass=TrajectoryModelMeta):
    frame = cf.CelestialFrame(
            name='icrs',
            reference_frame=coord.ICRS(),
            unit=(u.deg, u.deg)
            )


class SkyAltAzTrajModel(SkyMapModel, metaclass=TrajectoryModelMeta):
    frame = cf.CelestialFrame(
            name='altaz',
            reference_frame=coord.AltAz(),
            unit=(u.deg, u.deg)
            )
