#! /usr/bin/env python

# Author:
#   Zhiyuan Ma
#
# History:
#   2020/04/02 Zhiyuan Ma:
#       - Created.

"""
This recipe defines a TolTEC KIDs data simulator.

"""
from astroplan import Observer
from astropy.wcs.utils import celestial_frame_to_wcs
from astropy.time import Time
from pytz import timezone
from tollan.utils.log import timeit
from tollan.utils.cli.multi_action_argparser import \
        MultiActionArgumentParser
from regions import PixCoord, PolygonPixelRegion, PolygonSkyRegion
from astropy.visualization import quantity_support
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from kidsproc.kidsmodel.simulator import KidsSimulator
import numpy as np
from scipy import signal
import itertools
from astropy.visualization import quantity_support
import pickle
from tollan.utils.log import timeit, Timer
import concurrent
import psutil
from tolteca.recipes.simu_hwp_noise import save_or_show
from astropy import coordinates as coord
from astropy.modeling import Model, Parameter
from astropy.modeling.functional_models import GAUSSIAN_SIGMA_TO_FWHM
from astropy.modeling import models
# from astropy import coordinates as coord
from astropy import units as u
from gwcs import wcs
from gwcs import coordinate_frames as cf
from astropy.table import Table, Column
from tolteca.recipes import get_logger
from abc import ABCMeta
from astropy.coordinates.baseframe import frame_transform_graph


def _get_skyoffset_frame(c):
    """This function creates a skyoffset_frame and ensures
    the cached origin frame attribute is the correct instance.
    """
    frame = c.skyoffset_frame()
    frame_transform_graph._cached_frame_attributes['origin'] = \
        frame.frame_attributes['origin']
    return frame


class SkyMapModel(Model):
    """A model that describes mapping patterns on the sky.

    It takes a time, and computes the sky coords.
    """

    n_inputs = 1
    n_outputs = 2

    def evaluate(self, x, y):
        return NotImplemented


class RasterScanModelMeta(SkyMapModel.__class__):

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

            This assumes a circular turn over trajectory.
            """
            t = np.asarray(t) * t.unit
            n_spaces = n_scans - 1

            # bbox_width = length
            # bbox_height = space * n_spaces
            # # (x0, y0, w, h)
            # bbox = (
            #         -bbox_width / 2., -bbox_height / 2.,
            #         bbox_width, bbox_height)
            t_per_scan = length / speed
            ratio_scan_to_si = (t_per_scan / (t_turnover + t_per_scan)).value
            ratio_scan_to_turnover = (t_per_scan / t_turnover).value

            # scan index
            _si = (t / (t_turnover + t_per_scan)).value
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
            theta_t = turnover_frac[turnover] * np.pi
            dy = radius_t * (1 - np.cos(theta_t))
            dx = radius_t * np.sin(theta_t)
            x[turnover] = x[turnover] + dx
            y[turnover] = y[turnover] + dy

            # make continuous
            x = x * (-1) ** si

            m_rot = models.AffineTransformation2D(
                models.Rotation2D._compute_matrix(
                    angle=rot.to('rad').value) * self.frame_unit,
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
            r = (t_y / t_x).value
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
                    angle=rot.to('rad').value) * self.frame_unit,
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


class WyattRasterScanModel(SkyMapModel, metaclass=RasterScanModelMeta):
    frame = cf.Frame2D(
                name='wyatt', axes_names=("x", "y"),
                unit=(u.cm, u.cm))


class WyattLissajousModel(SkyMapModel, metaclass=LissajousModelMeta):
    frame = cf.Frame2D(
                name='wyatt', axes_names=("x", "y"),
                unit=(u.cm, u.cm))


class SkyRasterScanModel(SkyMapModel, metaclass=RasterScanModelMeta):
    frame = cf.Frame2D(
            name='skyoffset', axes_names=('lon', 'lat'),
            unit=(u.deg, u.deg))

    def evaluate_at(self, ref_coord, *args):
        """Returns the mapping pattern as evaluated at given coordinates.
        """
        frame = _get_skyoffset_frame(ref_coord)
        return coord.SkyCoord(*self(*args), frame=frame).transform_to(
                ref_coord.frame)


class SkyLissajousModel(SkyMapModel, metaclass=LissajousModelMeta):
    frame = cf.Frame2D(
            name='skyoffset', axes_names=('lon', 'lat'),
            unit=(u.deg, u.deg))

    def evaluate_at(self, ref_coord, *args):
        """Returns the mapping pattern as evaluated at given coordinates.
        """
        frame = _get_skyoffset_frame(ref_coord)
        return coord.SkyCoord(*self(*args), frame=frame).transform_to(
                ref_coord.frame)


class ProjModel(Model):
    """Base class for models that transform the detector locations.
    """

    def __init__(self, *args, **kwargs):
        inputs = kwargs.pop('inputs', self.input_frame.axes_names)
        outputs = kwargs.pop('outputs', self.output_frame.axes_names)
        kwargs.setdefault('name', self._name)
        super().__init__(*args, **kwargs)
        self.inputs = inputs
        self.outputs = outputs

    def mpl_axes_params(self):
        return dict(aspect='equal')


class ArrayProjModel(ProjModel):
    """A model that transforms the detector locations per array to
    a common instrument coordinate system.

    """
    toltec_instru_spec = {
            'a1100': {
                'rot_from_a1100': 0. * u.deg
                },
            'a1400': {
                'rot_from_a1100': 180. * u.deg
                },
            'a2000': {
                'rot_from_a1100': 0. * u.deg
                },
            'toltec': {
                'rot_from_a1100': 90. * u.deg,
                'array_names': ('a1100', 'a1400', 'a2000'),
                # 'plate_scale': ()
                'fov_diam': 4. * u.arcmin,
                'array_diam': 127049.101 * u.um  # from a1100 design spec.
                },
            }
    input_frame = cf.Frame2D(
                name='array',
                axes_names=("x", "y"),
                unit=(u.um, u.um))
    output_frame = cf.Frame2D(
                name='toltec',
                axes_names=("az_offset", "alt_offset"),
                unit=(u.deg, u.deg))
    _name = f'{output_frame.name}_proj'

    n_inputs = 3
    n_outputs = 2

    def __init__(self, **kwargs):
        spec = self.toltec_instru_spec

        # The mirror put array on the perspective of an observer.
        m_mirr = np.array([[1, 0], [0, -1]])

        plate_scale = spec['toltec']['fov_diam'] / spec['toltec']['array_diam']

        m_projs = dict()

        for array_name in self.array_names:

            # this rot and scale put the arrays on the on the sky in Altaz
            rot = (
                    spec['toltec']['rot_from_a1100'] -
                    spec[array_name]['rot_from_a1100']
                    )
            m_rot = models.Rotation2D._compute_matrix(
                    angle=rot.to('rad').value)

            m_projs[array_name] = models.AffineTransformation2D(
                (m_rot @ m_mirr) * u.cm,
                translation=(0., 0.) * u.cm) | (
                    models.Multiply(plate_scale) &
                    models.Multiply(plate_scale))
        self._m_projs = m_projs
        super().__init__(
                inputs=('axes_names', ) + self.input_frame.axes_names,
                **kwargs)

    @timeit(_name)
    def evaluate(self, array_name, x, y):
        out_unit = u.deg
        x_out = np.empty(x.shape) * out_unit
        y_out = np.empty(y.shape) * out_unit
        for n in self.array_names:
            m = array_name == n
            xx, yy = self._m_projs[n](x[m], y[m])
            x_out[m] = xx.to(out_unit)
            y_out[m] = yy.to(out_unit)
        return x_out, y_out

    @property
    def array_names(self):
        return self.toltec_instru_spec['toltec']['array_names']

    def prepare_inputs(self, array_name, *inputs, **kwargs):
        # this is necessary to handle the array_name inputs
        array_name_idx = np.arange(array_name.size).reshape(array_name.shape)
        inputs_new, broadcasts = super().prepare_inputs(
                array_name_idx, *inputs, **kwargs)
        inputs_new[0] = np.ravel(array_name)[inputs_new[0].astype(int)]
        return inputs_new, broadcasts


class WyattProjModel(ProjModel):
    """A projection model for the Wyatt robot arm.

    This model is an affine transformation that projects the designed positions
    of detectors on the toltec frame to a plane in front of the cryostat
    window, on which the Wyatt robot arm moves.

    Parameters
    ----------
    rot: `astropy.units.Quantity`
        Rotation angle between the Wyatt frame and the TolTEC frame.
    scale: 2-tuple of `astropy.units.Quantity`
        Scale between the Wyatt frame and the TolTEC frame
    ref_coord: 2-tuple of `astropy.units.Quantity`
        The coordinate of the TolTEC frame origin on the Wyatt frame.
    """

    input_frame = ArrayProjModel.output_frame
    output_frame = cf.Frame2D(
                name='wyatt',
                axes_names=("x", "y"),
                unit=(u.cm, u.cm))
    _name = f'{output_frame.name}_proj'

    n_inputs = 2
    n_outputs = 2

    crval0 = Parameter(default=0., unit=output_frame.unit[0])
    crval1 = Parameter(default=0., unit=output_frame.unit[1])

    def __init__(self, rot, scale, ref_coord=None, **kwargs):
        if not scale[0].unit.is_equivalent(u.m / u.deg):
            raise ValueError("invalid unit for scale.")
        if ref_coord is not None:
            if 'crval0' in kwargs or 'crval1' in kwargs:
                raise ValueError(
                        "ref_coord cannot be specified along with crvals")
            kwargs['crval0'] = ref_coord[0]
            kwargs['crval1'] = ref_coord[1]
            kwargs['n_models'] = np.asarray(ref_coord[0]).size

        m_rot = models.Rotation2D._compute_matrix(angle=rot.to('rad').value)

        self._t2w_0 = models.AffineTransformation2D(
                m_rot * u.deg,
                translation=(0., 0.) * u.deg) | (
                    models.Multiply(scale[0]) & models.Multiply(scale[1])
                    )
        super().__init__(**kwargs)

    @timeit(_name)
    def evaluate(self, x, y, crval0, crval1):
        c0, c1 = self._t2w_0(x, y)
        return c0 + crval0, c1 + crval1

    def get_map_wcs(self, pixscale, ref_coord=None):

        """Return a WCS object that describes a Wyatt map of given pixel scale.

        Parameters
        ----------
        pixscale: 2-tuple of `astropy.units.Quantity`
            Pixel scale of Wyatt map on the Wyatt frame, specified as
            value per pix.
        ref_coord: 2-tuple of `astropy.units.Quantity`, optional
            The coordinate of pixel (0, 0).
        """

        # transformation to go from Wyatt coords to pix coords
        w2m = (
            models.Multiply(1. / pixscale[0]) &
            models.Multiply(1. / pixscale[1])
            )

        # the coord frame used in the array design.
        af = cf.Frame2D(
                name=self.array_name, axes_names=("x", "y"),
                unit=(u.um, u.um))
        # the coord frame on the Wyatt plane
        wf = cf.Frame2D(
                name='wyatt', axes_names=("x", "y"),
                unit=(u.cm, u.cm))
        # the coord frame on the wyatt map. It is aligned with wyatt.
        mf = cf.Frame2D(
                name='wyattmap', axes_names=("x", "y"),
                unit=(u.pix, u.pix))

        if ref_coord is None:
            ref_coord = self.crval0, self.crval1
        a2w = self._a2w_0 | (
                models.Shift(ref_coord[0]) & models.Shift(ref_coord[1]))
        pipeline = [
                (af, a2w),
                (wf, w2m),
                (mf, None)
                ]
        return wcs.WCS(pipeline)


class SkyProjModel(ProjModel):
    """A sky projection model for TolTEC.

    Parameters
    ----------
    ref_coord: 2-tuple of `astropy.units.Quantity`
        The coordinate of the TolTEC frame origin on the sky.
    """

    site = {
        'name': 'LMT',
        'name_long': 'Large Millimeter Telescope',
        'location': coord.EarthLocation.from_geodetic(
                "-97d18m53s", '+18d59m06s', 4600 * u.m),
        'timezone': timezone('America/Mexico_City'),
        }

    observer = Observer(
            name=site['name_long'],
            location=site['location'],
            timezone=site['timezone'],
            )

    input_frame = ArrayProjModel.output_frame
    output_frame = cf.Frame2D(
                name='sky',
                axes_names=("lon", "lat"),
                unit=(u.deg, u.deg))
    _name = f'{output_frame.name}_proj'

    n_inputs = 2
    n_outputs = 2

    crval0 = Parameter(default=180., unit=output_frame.unit[0])
    crval1 = Parameter(default=30., unit=output_frame.unit[1])
    mjd_obs = Parameter(default=Time(2000.0, format='jyear').mjd, unit=u.day)

    def __init__(
            self, ref_coord=None, time_obs=None,
            evaluate_frame=None, **kwargs):
        if ref_coord is not None:
            if 'crval0' in kwargs or 'crval1' in kwargs:
                raise ValueError(
                        "ref_coord cannot be specified along with crvals")
            kwargs['crval0'] = ref_coord[0]
            kwargs['crval1'] = ref_coord[1]
            kwargs['n_models'] = np.asarray(ref_coord[0]).size
        if time_obs is not None:
            if 'mjd_obs' in kwargs:
                raise ValueError(
                        "time_obs cannot be specified along with mjd_obs")
            kwargs['mjd_obs'] = time_obs.mjd * u.day
        self.evaluate_frame = evaluate_frame
        super().__init__(**kwargs)

    @classmethod
    def _get_native_frame(cls, mjd_obs):
        return cls.observer.altaz(time=Time(mjd_obs, format='mjd'))

    def get_native_frame(self):
        return self._get_native_frame(self.mjd_obs)

    @classmethod
    def _get_projected_frame(
            cls, crval0, crval1, mjd_obs, also_return_native_frame=False):
        ref_frame = cls._get_native_frame(mjd_obs)
        ref_coord = coord.SkyCoord(
                crval0.value * u.deg, crval1.value * u.deg,
                frame='icrs').transform_to(ref_frame)
        ref_offset_frame = _get_skyoffset_frame(ref_coord)
        if also_return_native_frame:
            return ref_offset_frame, ref_frame
        return ref_offset_frame

    def get_projected_frame(self, **kwargs):
        return self._get_projected_frame(
                self.crval0, self.crval1, self.mjd_obs, **kwargs)

    @timeit(_name)
    def evaluate(self, x, y, crval0, crval1, mjd_obs):

        ref_offset_frame, ref_frame = self._get_projected_frame(
                crval0, crval1, mjd_obs, also_return_native_frame=True)
        det_coords_offset = coord.SkyCoord(x, y, frame=ref_offset_frame)
        with Timer(f"transform det coords to altaz"):
            det_coords = det_coords_offset.transform_to(ref_frame)

        frame = self.evaluate_frame
        if frame is None or frame == 'native':
            return det_coords.az, det_coords.alt
        with Timer(f"transform det coords to {frame}"):
            det_coords = det_coords.transform_to(frame)
            attrs = list(
                    det_coords.get_representation_component_names().keys())
            return (getattr(det_coords, attrs[0]),
                    getattr(det_coords, attrs[1]))

    def __call__(self, *args, frame=None, **kwargs):
        if frame is None:
            frame = self.evaluate_frame
        old_evaluate_frame = self.evaluate_frame
        self.evaluate_frame = frame
        result = super().__call__(*args, **kwargs)
        self.evaluate_frame = old_evaluate_frame
        return result

    def mpl_axes_params(self):
        w = celestial_frame_to_wcs(coord.ICRS())
        w.wcs.crval = [
                self.crval0.value,
                self.crval1.value,
                ]
        return dict(super().mpl_axes_params(), projection=w)


class ToltecObsSimulator(object):
    """A class that make simulated observations for TolTEC.

    The simulator makes use of a suite of models:

                    telescope pointing (lon, lat)
                                            |
                                            v
    detectors positions (x, y) -> [SkyProjectionModel]
                                            |
                                            v
                        projected detectors positions (lon, lat)
                                            |
                                            v
    source catalogs (lon, lat, flux) -> [SkyModel]
                                            |
                                            v
                            detector power loading  (power)
                                            |
                                            v
                                       [KidsProbeModel]
                                            |
                                            v
                            detector raw readout (I, Q)

    Parameters
    ----------
    source: astorpy.model.Model
        The optical power loading model, which computes the optical
        power applied to the detectors at a given time.

    model: astorpy.model.Model
        The KIDs detector probe model, which computes I and Q for
        given probe frequency and input optical power.

    **kwargs:
         Additional attributes that get stored as the meta data of this
         simulator.

     """

    def __init__(self, source=None, model=None, **kwargs):
        self._source = source
        self._model = model
        self._meta = kwargs

    @property
    def meta(self):
        return self._meta

    @property
    def tones(self):
        return self._tones


def _make_nw_cmap():
    # make a color map
    nws = np.arange(13)
    from_list = mcolors.LinearSegmentedColormap.from_list
    cmap = from_list(None, plt.get_cmap('tab20')(nws), len(nws))
    cmap_kwargs = dict(cmap=cmap, vmin=nws[0] - 0.5, vmax=nws[-1] + 0.5)
    cax_ticks = nws
    cax_label = 'Network'
    return cax_ticks, cax_label, cmap_kwargs


def plot_arrays(calobj):
    logger = get_logger()

    tbl = calobj.get_array_prop_table()
    array_names = tbl.meta['array_names']
    n_arrays = len(array_names)

    logger.debug(f"array prop table:\n{tbl}")

    x_a = tbl['x'].quantity.to(u.cm)
    y_a = tbl['y'].quantity.to(u.cm)

    m_proj = ArrayProjModel()
    logger.debug(f"proj model:\n{m_proj}")

    x_p, y_p = m_proj(tbl['array_name'], x_a, y_a)

    fig, axes = plt.subplots(2, n_arrays, subplot_kw=m_proj.mpl_axes_params())

    _, _, cmap_kwargs = _make_nw_cmap()
    for i, array_name in enumerate(array_names):
        m = tbl['array_name'] == array_name
        mtbl = tbl[m]
        mtbl.meta = tbl.meta[array_name]

        # per array pos
        mx_a = x_a[m]
        my_a = y_a[m]
        mx_p = x_p[m]
        my_p = y_p[m]

        n_detectors = mtbl.meta['n_detectors']
        if i == 0:
            n_detectors_0 = n_detectors
            wl_center_0 = mtbl.meta['wl_center']
        s = (
                (n_detectors_0 / n_detectors) ** 0.5 *
                mtbl.meta['wl_center'] /
                wl_center_0) * 3,
        c = mtbl['nw']

        axes[0, i].scatter(mx_a, my_a, c=c, s=s, **cmap_kwargs)
        axes[1, i].scatter(
                mx_p.to(u.arcmin).value,
                my_p.to(u.arcmin).value,
                c=c, s=s, **cmap_kwargs)

    axes[0, 0].set_ylabel(f"{m_proj.input_frame.name} frame ({y_a.unit})")
    axes[1, 0].set_ylabel(f"{m_proj.output_frame.name} frame (arcmin)")
    fig.tight_layout()
    plt.show()


def plot_projected(calobj, proj_model, **kwargs):

    logger = get_logger()

    tbl = calobj.get_array_prop_table()
    array_names = tbl.meta['array_names']
    n_arrays = len(array_names)
    x_a = tbl['x'].quantity.to(u.cm)
    y_a = tbl['y'].quantity.to(u.cm)
    x_t, y_t = ArrayProjModel()(tbl['array_name'], x_a, y_a)

    # combine the array projection with sky projection
    m_proj = proj_model(**kwargs)
    logger.debug(f"proj model:\n{m_proj}")

    crvals = (
            m_proj.crval0,
            m_proj.crval1,
            )
    proj_unit = crvals[0].unit

    x_p, y_p = m_proj(x_t, y_t)

    props = np.full((n_arrays, ), None, dtype=object)
    for i, array_name in enumerate(array_names):
        m = tbl['array_name'] == array_name

        mtbl = tbl[m]
        mtbl.meta = tbl.meta[array_name]

        # per array pos
        mx_a = x_a[m]
        my_a = y_a[m]
        mx_p = x_p[m]
        my_p = y_p[m]

        # edge detectors for outline, note the vert indices are
        # with respect to per array table
        iv = mtbl.meta['edge_indices']
        vx_a = mx_a[iv]
        vy_a = my_a[iv]
        vx_p = mx_p[iv]
        vy_p = my_p[iv]

        props[i] = (
                mtbl,
                mx_a, my_a, mx_p, my_p,
                vx_a, vy_a, vx_p, vy_p,
                )

    cax_ticks, cax_label, cmap_kwargs = _make_nw_cmap()

    fig = plt.figure(constrained_layout=True, figsize=(16, 8))
    axes = np.full((2, n_arrays), None, dtype=object)
    for i in range(n_arrays):
        axes[0, i] = fig.add_subplot(
                2, n_arrays, i + 1, **dict(
                    aspect='equal',
                    sharex=None if i == 0 else axes[0, 0],
                    sharey=None if i == 0 else axes[0, 0],
                    ))
        axes[1, i] = fig.add_subplot(
                2, n_arrays, i + 1 + n_arrays, **dict(
                    m_proj.mpl_axes_params(),
                    aspect='equal',
                    sharex=None if i == 0 else axes[1, 0],
                    sharey=None if i == 0 else axes[1, 0],
                    ))

    axes[0, 0].set_ylabel(
            f"{ArrayProjModel.input_frame.name} frame ({y_a.unit})")
    axes[1, 0].set_ylabel(f"{m_proj.output_frame.name} frame ({proj_unit})")

    for i, prop in enumerate(props):
        (
            mtbl,
            mx_a, my_a, mx_p, my_p,
            vx_a, vy_a, vx_p, vy_p,
            ) = prop
        n_detectors = mtbl.meta['n_detectors']
        if i == 0:
            n_detectors_0 = n_detectors
            wl_center_0 = mtbl.meta['wl_center']
        s = (
                (n_detectors_0 / n_detectors) ** 0.5 *
                mtbl.meta['wl_center'] /
                wl_center_0) * 3,
        c = mtbl['nw']

        axes[0, i].set_title(mtbl.meta['name_long'])
        im = axes[0, i].scatter(
                mx_a, my_a, s=s, c=c, **cmap_kwargs)
        axes[0, i].plot(
                0, 0,
                marker='+', color='red')

        # assuming 1cm = 1pix
        reg_a = PolygonPixelRegion(
                vertices=PixCoord(x=vx_a, y=vy_a))
        axes[0, i].add_patch(reg_a.as_artist(
            facecolor='none', edgecolor='red', lw=2))

        # projected plane
        if proj_model == WyattProjModel:
            plot_kw = dict()
        else:
            plot_kw = dict(transform=axes[1, i].get_transform('icrs'))

        im = axes[1, i].scatter(
                mx_p.to(proj_unit), my_p.to(proj_unit),
                s=s, c=c, **cmap_kwargs, **plot_kw)

        axes[1, i].plot(
                crvals[0].value,
                crvals[1].value,
                marker='+', color='red', **plot_kw)

        if proj_model == WyattProjModel:
            reg_p = PolygonPixelRegion(
                    vertices=PixCoord(
                        x=vx_p.to(proj_unit),
                        y=vy_p.to(proj_unit)))
            axes[1, i].add_patch(reg_p.as_artist(
                facecolor='none', edgecolor='red', lw=2))
        elif proj_model == SkyProjModel:
            reg_p = PolygonSkyRegion(
                    vertices=coord.SkyCoord(ra=vx_p, dec=vy_p, frame='icrs'))
            patch = reg_p.to_pixel(axes[1, i].wcs).as_artist(
                    facecolor='none', edgecolor='red', lw=2)
            axes[1, i].add_patch(patch)
            axes[1, i].coords.grid(
                    color='#cccc66', linestyle='-')
            overlay = axes[1, i].get_coords_overlay(
                    m_proj.get_native_frame())
            overlay.grid(color='#66aacc', ls='-')

    cax = fig.colorbar(
            im, ax=axes[:, -1], shrink=0.8, location='right', ticks=cax_ticks)
    cax.set_label(cax_label)
    plt.show()


def plot_obs_on_wyatt(calobj, m_obs, **wyatt_proj_kwargs):

    logger = get_logger()
    t_total = m_obs.get_total_time()
    t = np.arange(0, t_total.to(u.s).value, 0.5) * u.s
    n_pts = len(t.value)

    logger.debug(f"create {n_pts} pointings")
    x_t, y_t = m_obs(t)

    wyatt_proj_kwargs.update({
            'ref_coord': (x_t, y_t)
            })

    # porject the edge detectors
    tbl = calobj.get_array_prop_table()
    array_names = tbl.meta['array_names']
    n_arrays = len(array_names)
    array_unit = u.cm

    m_proj = ArrayProjModel(
            n_models=n_pts) | WyattProjModel(**wyatt_proj_kwargs)
    wyatt_unit = WyattProjModel.crval0.unit

    props = np.full((n_arrays, ), None, dtype=object)
    for i, array_name in enumerate(array_names):
        m = tbl['array_name'] == array_name

        mtbl = tbl[m]
        mtbl.meta = tbl.meta[array_name]

        # edge detectors for outline, note the vert indices are
        # with respect to per array table
        iv = mtbl.meta['edge_indices']

        # per array pos
        vn_a = mtbl['array_name'][iv]
        vx_a = mtbl['x'].quantity.to(array_unit)[iv]
        vy_a = mtbl['y'].quantity.to(array_unit)[iv]
        vx_p, vy_p = m_proj(
                np.tile(vn_a, (n_pts, 1)),
                np.tile(vx_a, (n_pts, 1)),
                np.tile(vy_a, (n_pts, 1))
                )

        props[i] = (
                mtbl,
                vx_a, vy_a, vx_p, vy_p,
                )

    fig, axes = plt.subplots(
            2, n_arrays, squeeze=False,
            sharex='row', sharey='row', subplot_kw={'aspect': 'equal'},
            constrained_layout=True, figsize=(16, 8))
    axes[0, 0].set_ylabel(f"array frame ({array_unit})")
    axes[1, 0].set_ylabel(f"wyatt frame ({wyatt_unit})")

    for i, prop in enumerate(props):
        mtbl, vx_a, vy_a, vx_p, vy_p = prop

        axes[0, i].set_title(mtbl.meta['name_long'])

        reg_a = PolygonPixelRegion(
                vertices=PixCoord(
                    x=vx_a.to(array_unit), y=vy_a.to(array_unit)))
        axes[0, i].add_patch(reg_a.as_artist(
            facecolor='none', edgecolor='#ff2222', lw=0.1))

        for t in range(n_pts):
            reg_p = PolygonPixelRegion(
                    vertices=PixCoord(
                        x=vx_p[t].to(wyatt_unit), y=vy_p[t].to(wyatt_unit)))
            axes[1, i].add_patch(reg_p.as_artist(
                facecolor='none', edgecolor='#ff2222', lw=0.1))

        axes[0, i].plot(
                0, 0,
                marker='+', color='red')
        axes[1, i].plot(
                m_proj['wyatt_proj'].crval0.value,
                m_proj['wyatt_proj'].crval1.value,
                marker='+', color='red')

    plt.show()


def plot_obs_on_sky(calobj, m_obs, ref_obj):

    from astroquery.utils import parse_coordinates

    ref_coord = parse_coordinates(ref_obj)

    logger = get_logger()

    t_total = m_obs.get_total_time()
    t = np.arange(0, t_total.to(u.s).value, 0.5) * u.s
    n_pts = len(t.value)

    logger.debug(f"create {n_pts} pointings")

    obs_coords = m_obs.evaluate_at(ref_coord, t)

    from astropy.wcs import WCS
    from astroquery.skyview import SkyView
    # from astropy.visualization import ZScaleInterval, ImageNormalize
    from astropy.visualization import make_lupton_rgb
    # make rgb 2mass
    hdulists = SkyView.get_images(
            ref_coord,
            survey=['2MASS-K', '2MASS-H', '2MASS-J'])
    scales = [1.5, 1.0, 1.0]

    def _bkg_subtracted_data(hdu, scale=1.):
        ni, nj = hdu.data.shape
        mask = np.ones_like(hdu.data, dtype=bool)
        frac = 5
        mask[
                ni // frac:(frac - 1) * ni // 4,
                nj // frac:(frac - 1) * nj // 4] = False
        data_bkg = hdu.data[mask]
        bkg = 3 * np.nanmedian(data_bkg) - 2 * np.nanmean(data_bkg)
        return (hdu.data - bkg) * scale

    image = make_lupton_rgb(
            *(_bkg_subtracted_data(
                hl[0], scale=scale)
                for hl, scale in zip(hdulists, scales)),
            Q=10, stretch=50)
    w = WCS(hdulists[0][0].header)
    fig, ax = plt.subplots(1, 1, subplot_kw={
        'projection': w
        })
    ax.imshow(image, origin='lower')
    # ax.coords.grid(color='white', alpha=0.5, linestyle='solid')
    ax.plot(
            obs_coords.ra, obs_coords.dec,
            transform=ax.get_transform('icrs'),
            color='red'
            )
    ax.set_title(f'{ref_obj}')
    plt.show()


if __name__ == "__main__":

    import sys
    from tolteca.cal import ToltecCalib

    maap = MultiActionArgumentParser(
            description="Make simulated observation."
            )

    maap.add_argument(
            '--calobj', '-c',
            help='Path to calibration object.',
            required=True
            )

    act_plot_arrays = maap.add_action_parser(
            'plot_arrays',
            help='Plot the projected detectors in toltec frame'
            )

    @act_plot_arrays.parser_action
    def plot_arrays_action(option):

        calobj = ToltecCalib.from_indexfile(option.calobj)
        plot_arrays(calobj)

    act_plot_projected = maap.add_action_parser(
            'plot_projected',
            help='Plot the projected detectors'
            )
    act_plot_projected.add_argument(
            'proj_model',
            choices=['wyatt', 'sky']
            )

    act_plot_projected.add_argument(
            '--time_utc', '-t',
            help='The time of the obs in UTC.',
            default='2020-04-14T00:00:00'
            )

    @act_plot_projected.parser_action
    def plot_projected_action(option):

        calobj = ToltecCalib.from_indexfile(option.calobj)
        if option.proj_model == 'wyatt':
            proj_model = WyattProjModel
            proj_kwargs = {
                    'rot': -2. * u.deg,
                    'scale': (30. / 4., 30. / 4.) * u.cm / u.arcmin,
                    'ref_coord': (15., 15.) * u.cm,
                    }
        elif option.proj_model == 'sky':
            proj_model = SkyProjModel
            proj_kwargs = {
                    'mjd_obs': Time(option.time_utc).mjd * u.day,
                    'ref_coord': (180., 30.) * u.deg,
                    'evaluate_frame': 'icrs'
                    }
        plot_projected(calobj, proj_model, **proj_kwargs)

    act_plot_obs_on_wyatt = maap.add_action_parser(
            'plot_obs_on_wyatt',
            help='Plot an obs pattern on Waytt'
            )
    act_plot_obs_on_wyatt.add_argument(
            "pattern", choices=['raster', 'lissajous'])

    @act_plot_obs_on_wyatt.parser_action
    def plot_obs_on_wyatt_action(option):

        calobj = ToltecCalib.from_indexfile(option.calobj)

        wyatt_proj_kwargs = {
            'rot': -2. * u.deg,
            'scale': (30. / 4., 30. / 4.) * u.cm / u.arcmin,
            'ref_coord': (15., 15.) * u.cm,
            }

        if option.pattern == 'raster':
            raster_scan_kwargs = {
                'rot': 30. * u.deg,
                'length': 50. * u.cm,
                'space': 5. * u.cm,
                'n_scans': 10 * u.dimensionless_unscaled,
                'speed': 1. * u.cm / u.s,
                't_turnover': 5 * u.s,
                }
            m_obs = WyattRasterScanModel(**raster_scan_kwargs)
        elif option.pattern == 'lissajous':
            lissajous_kwargs = {
                'rot': 30. * u.deg,
                'x_length': 50. * u.cm,
                'y_length': 50. * u.cm,
                'x_omega': 0.5 * np.pi * u.rad / u.s,
                'y_omega': 0.7 * np.pi * u.rad / u.s,
                'delta': 30 * u.deg
                # 'delta': 60. * u.deg
                }
            m_obs = WyattLissajousModel(**lissajous_kwargs)
        plot_obs_on_wyatt(calobj, m_obs, **wyatt_proj_kwargs)

    act_plot_obs_on_sky = maap.add_action_parser(
            'plot_obs_on_sky',
            help='Plot an obs pattern on the sky'
            )
    act_plot_obs_on_sky.add_argument(
            "pattern", choices=['raster', 'lissajous'])
    act_plot_obs_on_sky.add_argument(
            "--ref_obj", '-r', help='The reference object.')

    @act_plot_obs_on_sky.parser_action
    def plot_obs_on_sky_action(option):

        calobj = ToltecCalib.from_indexfile(option.calobj)

        if option.pattern == 'raster':
            raster_scan_kwargs = {
                'rot': 30. * u.deg,
                'length': 2. * u.arcmin,
                'space': 5. * u.arcsec,
                'n_scans': 24 * u.dimensionless_unscaled,
                'speed': 1. * u.arcsec / u.s,
                't_turnover': 5 * u.s,
                }
            m_obs = SkyRasterScanModel(**raster_scan_kwargs)
        elif option.pattern == 'lissajous':
            lissajous_kwargs = {
                'rot': 30. * u.deg,
                'x_length': 2. * u.arcmin,
                'y_length': 2. * u.arcmin,
                'x_omega': 0.07 * np.pi * u.rad / u.s,
                'y_omega': 0.05 * np.pi * u.rad / u.s,
                'delta': 30 * u.deg
                # 'delta': 60. * u.deg
                }
            m_obs = SkyLissajousModel(**lissajous_kwargs)
        ref_obj = option.ref_obj
        plot_obs_on_sky(calobj, m_obs, ref_obj)

    act_plot_simu_sky_point_source = maap.add_action_parser(
            'plot_simu_sky_point_source',
            help='Plot an end-to-end simulation.'
            )
    act_plot_simu_sky_point_source.add_argument(
            "pattern", choices=['raster', 'lissajous'])

    @act_plot_simu_sky_point_source.parser_action
    def plot_simu_sky_point_source_action(option):

        logger = get_logger()

        calobj = ToltecCalib.from_indexfile(option.calobj)

        if option.pattern == 'raster':
            raster_scan_kwargs = {
                'rot': 30. * u.deg,
                'length': 4. * u.arcmin,
                'space': 5. * u.arcsec,
                'n_scans': 24 * u.dimensionless_unscaled,
                'speed': 30. * u.arcsec / u.s,
                't_turnover': 5 * u.s,
                }
            m_obs = SkyRasterScanModel(**raster_scan_kwargs)
        elif option.pattern == 'lissajous':
            lissajous_kwargs = {
                'rot': 30. * u.deg,
                'x_length': 2. * u.arcmin,
                'y_length': 2. * u.arcmin,
                'x_omega': 0.07 * np.pi * u.rad / u.s,
                'y_omega': 0.05 * np.pi * u.rad / u.s,
                'delta': 30 * u.deg
                # 'delta': 60. * u.deg
                }
            m_obs = SkyLissajousModel(**lissajous_kwargs)

        # get object flux model
        beam_model = models.Gaussian2D
        beam_kwargs = {
                'x_fwhm': 5 * u.arcsec,
                'y_fwhm': 5 * u.arcsec,
                }

        x_stddev = beam_kwargs['x_fwhm'] / GAUSSIAN_SIGMA_TO_FWHM
        y_stddev = beam_kwargs['y_fwhm'] / GAUSSIAN_SIGMA_TO_FWHM
        m_beam = beam_model(
                amplitude=1. / (2 * np.pi * x_stddev * y_stddev) * u.mJy,
                x_mean=0. * u.arcsec,
                y_mean=0. * u.arcsec,
                x_stddev=x_stddev,
                y_stddev=y_stddev,
                )
        logger.debug(f"beam:\n{m_beam}")

        # make a source catalog
        sources = Table(
                [
                    Column(name='name', dtype='|S32'),
                    Column(name='ra', unit=u.deg),
                    Column(name='dec', unit=u.deg),
                    Column(name='flux', unit=u.mJy),
                    ])
        sources.add_row(['src0', 180., 0., 1.])
        sources.add_row(['src1', 180., 30. / 3600., 1.])

        logger.debug(f"sources:\n{sources}")

        # define a field center
        ref_coord = coord.SkyCoord(
                ra=sources['ra'].quantity[0],
                dec=sources['dec'].quantity[0],
                frame='icrs')

        # define the instru sampling rate
        fsmp = 122. * u.Hz
        # t_exp = 1 * u.min
        t_exp = 1 * u.s
        t = np.arange(0, t_exp.to(u.s).value, (1 / fsmp).to(u.s).value) * u.s
        n_pts = len(t.value)

        # these are the field center as a function of t
        obs_coords = m_obs.evaluate_at(ref_coord, t)

        # get detector positions, which requires absolute time
        # to get the altaz to equatorial transformation
        t0_obs = Time('2020-04-13 00:00:00')

        # here we only project in alt az, and we transform the source coord
        # to alt az for faster computation.

        tbl = calobj.get_array_prop_table()

        x_a = tbl['x'].quantity.to(u.cm)
        y_a = tbl['y'].quantity.to(u.cm)
        x_t, y_t = ArrayProjModel()(tbl['array_name'], x_a, y_a)

        # combine the array projection with sky projection
        m_proj = SkyProjModel(
                ref_coord=(
                    obs_coords.ra.degree, obs_coords.dec.degree) * u.deg,
                mjd_obs=(t0_obs + t).mjd * u.day,
                )
        logger.debug(f"proj model:\n{m_proj}")

        projected_frame, native_frame = m_proj.get_projected_frame(
                also_return_native_frame=True)

        # x_t and y_t are the coords in projected_frame
        # make the detector positions in the projected frame
        # az_p, alt_p = m_proj(
        #         np.tile(x_t, (n_pts, 1)),
        #         np.tile(y_t, (n_pts, 1)),
        #         frame='native',
        #         )

        # transform the sources on to the projected frame
        # this has to be done in two steps due to bug in
        # astropy.
        with Timer("make src coords"):
            src_coords = coord.SkyCoord(
                ra=sources['ra'][:, np.newaxis],
                dec=sources['dec'][:, np.newaxis],
                frame='icrs').transform_to(
                        native_frame).transform_to(
                            projected_frame)

        print(src_coords.shape)
        # now compute the distance for each detector to each source at each
        # time step
        dists = coord.angle_utilities.angular_separation(
                x_t[np.newaxis, :, np.newaxis],
                y_t[np.newaxis, :, np.newaxis],
                src_coords.lon[:, np.newaxis, :],
                src_coords.lat[:, np.newaxis, :]
                )
        print(dists.to(u.arcmin))
        print(dists.shape)

        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

        m = tbl['array_name'] == 'a1100'

        cax_ticks, cax_label, cmap_kwargs = _make_nw_cmap()
        ax.scatter(
                x_t[m].to(u.arcmin), y_t[m].to(u.arcmin),
                c=tbl['nw'][m], **cmap_kwargs)
        for i in range(len(sources)):
            ax.plot(
                    src_coords[i].lon.arcminute,
                    src_coords[i].lat.arcminute)
        plt.show()

    option = maap.parse_args(sys.argv[1:])
    maap.bootstrap_actions(option)
