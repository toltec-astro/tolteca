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

from tollan.utils.log import timeit
from tollan.utils.cli.multi_action_argparser import \
        MultiActionArgumentParser
from regions import PixCoord, PolygonPixelRegion
from astropy.visualization import quantity_support
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from kidsproc.kidsmodel.simulator import KidsSimulator
import numpy as np
from scipy import signal
import itertools
from astropy.visualization import quantity_support
import pickle
from tollan.utils.log import timeit
import concurrent
import psutil
from tolteca.recipes.simu_hwp_noise import save_or_show
from astropy import coordinates as coord
from astropy.modeling import Model, Parameter
from astropy.modeling import models
# from astropy import coordinates as coord
from astropy import units as u
from gwcs import wcs
from gwcs import coordinate_frames as cf
from astropy.table import Table
from tolteca.recipes import get_logger
from abc import ABCMeta


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

        @timeit
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

        @timeit
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
        frame = ref_coord.skyoffset_frame()
        return coord.SkyCoord(*self(*args), frame=frame).transform_to(
                ref_coord.frame)


class SkyLissajousModel(SkyMapModel, metaclass=LissajousModelMeta):
    frame = cf.Frame2D(
            name='skyoffset', axes_names=('lon', 'lat'),
            unit=(u.deg, u.deg))

    def evaluate_at(self, ref_coord, *args):
        """Returns the mapping pattern as evaluated at given coordinates.
        """
        frame = ref_coord.skyoffset_frame()
        return coord.SkyCoord(*self(*args), frame=frame).transform_to(
                ref_coord.frame)


class SkyProjModel(Model):
    """A model that transforms the detector locations to
    sky coordinates for a given pointing.
    """

    n_inputs = 2
    n_outputs = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, x, y):
        return NotImplemented

    @classmethod
    def from_fitswcs(cls, w):
        """Construct a sky projection model from fits WCS object."""
        return NotImplemented


class WyattProjModel(SkyProjModel):
    """A "Sky" model for the Wyatt robot arm.

    This model is an affine transformation that projects the
    designed positions of detectors on the array to a fiducial
    plane in front of the cryostat window, on which the Wyatt
    robot arm moves.

    Parameters
    ----------
    array_name: str
        The name of the array, choose from 'a1100', 'a1400' and 'a2000'.
    rot: `astropy.units.Quantity`
        Rotation angle between the Wyatt frame and the TolTEC frame.
    scale: 2-tuple of float
        Scale between the Wyatt frame and the TolTEC frame.
    ref_coord: 2-tuple of `astropy.units.Quantity`
        The coordinate of the TolTEC frame origin on the Wyatt frame.
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
                }
            }

    crval0 = Parameter(default=0., unit=u.cm)
    crval1 = Parameter(default=0., unit=u.cm)

    def __init__(
            self, array_name,
            rot,
            scale,
            ref_coord=(0, 0) * u.cm,
            **kwargs
            ):
        spec = self.toltec_instru_spec
        if array_name not in spec:
            raise ValueError(
                    f"array name shall be one of "
                    f"{spec['toltec']['array_names']}")
        self.array_name = array_name

        # The mirror put array on the perspective of an observer.
        m_mirr = np.array([[1, 0], [0, -1]])

        # the rotation consistent of the angle between wyatt and toltec,
        # and between toltec and the array.
        # the later is computed from subtracting angle between
        # toltec to a1100 and the array to a1100.
        rot = (
                rot +
                spec['toltec']['rot_from_a1100'] -
                spec[self.array_name]['rot_from_a1100']
                )
        m_rot = models.Rotation2D._compute_matrix(angle=rot.to('rad').value)
        m_scal = np.array([[scale[0], 0], [0, scale[1]]])

        # This transforms detector coordinates on the array frame to
        # the Wyatt frame, with respect to the ref_coord == (0, 0)
        # the supplied ref_coord is set as params of the model.
        self._a2w_0 = models.AffineTransformation2D(
            (m_scal @ m_rot @ m_mirr) * u.cm,
            translation=(0., 0.) * u.cm)
        super().__init__(
                crval0=ref_coord[0], crval1=ref_coord[1],
                n_models=np.asarray(ref_coord[0]).size, **kwargs)

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

    @timeit
    def evaluate(self, x, y, crval0, crval1):
        c0, c1 = self._a2w_0(x, y)
        return c0 + crval0, c1 + crval1


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


def plot_wyatt_plane(calobj, **kwargs):

    array_names = ['a1100', 'a1400', 'a2000']
    n_arrays = len(array_names)

    # make a color map
    nws = np.arange(13)
    from_list = mcolors.LinearSegmentedColormap.from_list
    cm = from_list(None, plt.get_cmap('tab20')(nws), len(nws))

    cm_kwargs = dict(cmap=cm, vmin=nws[0] - 0.5, vmax=nws[-1] + 0.5)

    fig, axes = plt.subplots(
            2, n_arrays, squeeze=False,
            sharex='row', sharey='row', subplot_kw={'aspect': 'equal'},
            constrained_layout=True, figsize=(16, 8))

    props = np.full((n_arrays, ), None, dtype=object)
    for i, array_name in enumerate(array_names):
        tbl = calobj.get_array_prop_table(array_name)
        m_proj = WyattProjModel(
                array_name=array_name,
                **kwargs)

        x_a = tbl['x'].quantity.to(u.cm)
        y_a = tbl['y'].quantity.to(u.cm)
        x_w, y_w = m_proj(x_a, y_a)

        n_kids = len(tbl)
        props[i] = (n_kids, tbl, m_proj, x_a, y_a, x_w, y_w)

    for i, prop in enumerate(props):
        n_kids, tbl, m_proj, x_a, y_a, x_w, y_w = prop
        s = (
                (props[0][0] / n_kids) ** 0.5 * tbl.meta['wl_center'] /
                props[0][1].meta['wl_center']) * 3,
        c = tbl['nw']
        im = axes[0, i].scatter(
                x_a, y_a, s=s, c=c, **cm_kwargs)
        axes[0, i].plot(
                0, 0,
                marker='+', color='red')

        im = axes[1, i].scatter(
                x_w, y_w, s=s, c=c, **cm_kwargs)
        axes[1, i].plot(
                m_proj.crval0.value, m_proj.crval1.value,
                marker='+', color='red')

        axes[0, i].set_title(tbl.meta['name_long'])

        verts = tbl[tbl.meta['edge_indices']]
        vx_a = verts['x'].quantity
        vy_a = verts['y'].quantity
        vx_w, vy_w = m_proj(vx_a, vy_a)

        reg_a = PolygonPixelRegion(
                vertices=PixCoord(x=vx_a.to(u.cm), y=vy_a.to(u.cm)))
        reg_w = PolygonPixelRegion(
                vertices=PixCoord(x=vx_w.to(u.cm), y=vy_w.to(u.cm)))

        axes[0, i].add_patch(reg_a.as_artist(
            facecolor='none', edgecolor='red', lw=2))
        axes[1, i].add_patch(reg_w.as_artist(
            facecolor='none', edgecolor='red', lw=2))

    cax = fig.colorbar(
            im, ax=axes[:, -1], shrink=0.8, location='right', ticks=nws)
    cax.set_label('Network')
    axes[0, 0].set_ylabel(f"Array Plane ({y_a.unit})")
    axes[1, 0].set_ylabel(f"Wyatt Plane ({y_w.unit})")
    plt.show()


def plot_obs_on_wyatt(calobj, m_obs, **wyatt_proj_kwargs):

    logger = get_logger()
    t_total = m_obs.get_total_time()
    t = np.arange(0, t_total.to(u.s).value, 0.5) * u.s

    logger.debug(f"create {len(t.value)} pointings")
    x_t, y_t = m_obs(t)

    wyatt_proj_kwargs.update({
            'ref_coord': (x_t, y_t)
            })

    array_names = ['a1100', 'a1400', 'a2000']
    n_arrays = len(array_names)

    fig, axes = plt.subplots(
            2, n_arrays, squeeze=False,
            sharex='row', sharey='row', subplot_kw={'aspect': 'equal'},
            constrained_layout=True, figsize=(16, 8))

    props = np.full((n_arrays, ), None, dtype=object)
    for i, array_name in enumerate(array_names):
        tbl = calobj.get_array_prop_table(array_name)
        m_proj = WyattProjModel(
                array_name=array_name,
                **wyatt_proj_kwargs)

        x_a = np.tile(tbl['x'].quantity.to(u.cm), (len(m_proj), 1))
        y_a = np.tile(tbl['y'].quantity.to(u.cm), (len(m_proj), 1))
        x_w, y_w = m_proj(x_a, y_a)

        verts = tbl[tbl.meta['edge_indices']]
        vx_a = np.tile(verts['x'].quantity.to(u.cm), (len(m_proj), 1))
        vy_a = np.tile(verts['y'].quantity.to(u.cm), (len(m_proj), 1))
        vx_w, vy_w = m_proj(vx_a, vy_a)

        n_kids = len(tbl)
        props[i] = (
                n_kids, tbl, m_proj,
                # x_a, y_a, x_w, y_w,
                vx_a, vy_a, vx_w, vy_w,
                )

    for i, prop in enumerate(props):
        (
            n_kids, tbl, m_proj,
            vx_a, vy_a, vx_w, vy_w, ) = prop

        axes[0, i].set_title(tbl.meta['name_long'])
        for t in range(vx_a.shape[0]):
            reg_a = PolygonPixelRegion(
                    vertices=PixCoord(
                        x=vx_a[t].to(u.cm), y=vy_a[t].to(u.cm)))
            reg_w = PolygonPixelRegion(
                    vertices=PixCoord(
                        x=vx_w[t].to(u.cm), y=vy_w[t].to(u.cm)))

            axes[0, i].add_patch(reg_a.as_artist(
                facecolor='none', edgecolor='#ff2222', lw=0.1))
            axes[1, i].add_patch(reg_w.as_artist(
                facecolor='none', edgecolor='#ff2222', lw=0.1))

        axes[0, i].plot(
                0, 0,
                marker='+', color='red')
        axes[1, i].plot(
                m_proj.crval0.value, m_proj.crval1.value,
                marker='+', color='red')

    axes[0, 0].set_ylabel(f"Array Plane ({vy_a.unit})")
    axes[1, 0].set_ylabel(f"Wyatt Plane ({vy_w.unit})")
    plt.show()


def plot_obs_on_sky(calobj, m_obs, ref_obj):

    from astroquery.utils import parse_coordinates

    ref_coord = parse_coordinates(ref_obj)

    logger = get_logger()

    t_total = m_obs.get_total_time()
    t = np.arange(0, t_total.to(u.s).value, 0.5) * u.s

    logger.debug(f"create {len(t.value)} pointings")

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
    # ax.plot(
    #         ref_coord.ra, ref_coord.dec,
    #         transform=ax.get_transform('icrs'),
    #         marker='+', color='red')
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

    act_plot_wyatt = maap.add_action_parser(
            'plot_wyatt',
            help='Plot the detectors on Wyatt plane.'
            )

    @act_plot_wyatt.parser_action
    def plot_wyatt_action(option):

        calobj = ToltecCalib.from_indexfile(option.calobj)
        wyatt_proj_kwargs = {
                'rot': -2. * u.deg,
                'scale': (3., 3.),
                'ref_coord': (12., 12.) * u.cm,
                }
        plot_wyatt_plane(calobj, **wyatt_proj_kwargs)

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
            'scale': (3., 3.),
            'ref_coord': (0., 0.) * u.cm,
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
                'x_omega': 0.13 * np.pi * u.rad / u.s,
                'y_omega': 0.19 * np.pi * u.rad / u.s,
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
                'x_omega': 0.13 * np.pi * u.rad / u.s,
                'y_omega': 0.19 * np.pi * u.rad / u.s,
                'delta': 30 * u.deg
                # 'delta': 60. * u.deg
                }
            m_obs = SkyLissajousModel(**lissajous_kwargs)
        ref_obj = option.ref_obj
        plot_obs_on_sky(calobj, m_obs, ref_obj)

    option = maap.parse_args(sys.argv[1:])
    maap.bootstrap_actions(option)
