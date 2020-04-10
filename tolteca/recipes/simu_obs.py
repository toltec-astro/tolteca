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


# class RasterScanModelMeta(ABCMeta):
#     """A meta class to create model that generate a raster scan pattern.
#     """
#     def __new__(mcs, name, bases, namespace, **kwargs):
#         return super().__new__(mcs, name, bases, namespace)

#     def __init__(cls, name, bases, namespace, frame=None):
#         super().__init__(name, bases, namespace)
#         # validate frame
#         if frame.naxes != 2:
#             raise ValueError("invalid frame")
#         if len(set(frame.unit)) != 1:
#             raise ValueError("invalid frame unit.")
#         cls.frame = frame
#         cls.frame_unit = frame.unit[0]
#         cls.length = Parameter(default=10., unit=cls.frame_unit)
#         cls.space = Parameter(default=1., unit=cls.frame_unit)
#         cls.n_scans = Parameter(default=10.)
#         cls.rot = Parameter(default=0., unit=cls.frame_unit)
#         cls.speed = Parameter(default=1., unit=cls.frame_unit / u.s)
#         # cls.accel = Parameter(default=1., unit=cls.frame_unit / u.s ** 2)
#         cls.t_turnover = Parameter(default=1., unit=u.s)

#         def evaluate(
#                 self, t, length, space, n_scans, rot, speed, t_turnover):
#             """This computes a raster patten around the origin.

#             This assumes a circular turn over trajectory.
#             """
#             t = np.asarray(t)
#             n_spaces = n_scans - 1

#             # bbox_width = length
#             # bbox_height = space * n_spaces
#             # # (x0, y0, w, h)
#             # bbox = (
#             #         -bbox_width / 2., -bbox_height / 2.,
#             #         bbox_width, bbox_height)
#             t_per_scan = length / speed
#             si_frac_turnover = t_per_scan / (t_turnover + t_per_scan)
#             # scan index
#             sif = t / (t_turnover + t_per_scan)
#             si = sif.astype(int)
#             si_frac = sif - si

#             if si_frac > si_frac_turnover:
#                 s_frac = 1.
#                 turnover_frac = si_frac - (
#                         1. - si_frac) * t_per_scan / t_turnover
#             else:
#                 s_frac = si_frac * (t_per_scan + t_turnover) / t_per_scan
#                 turnover_frac = None
#             x = (s_frac - 0.5) * length
#             y = (si / n_spaces - 0.5) * n_spaces * space
#             if turnover_frac is not None:
#                 # turn over point
#                 r_t = space / 2
#                 theta_t = turnover_frac * np.pi
#                 dy = r_t * (1 - np.cos(theta_t))
#                 dx = r_t * np.sin(theta_t)
#                 x = x + dx
#                 y = y + dy
#             else:
#                 pass
#             x = x * (-1) ** si

#             m_rot = models.Rotation2D(angle=rot.to('deg').value)

#             xx, yy = m_rot(x, y)
#             return xx, yy

#         cls.evaluate = evaluate


# class WyattRasterScanModelMixin(
#         Model,
#         metaclass=RasterScanModelMeta,
#         frame=cf.Frame2D(
#                 name='wyatt', axes_names=("x", "y"),
#                 unit=(u.cm, u.cm))
#         ):
#     pass


class WyattRasterScanModel(SkyMapModel):

    frame = cf.Frame2D(
                name='wyatt', axes_names=("x", "y"),
                unit=(u.cm, u.cm))
    frame = frame
    frame_unit = frame.unit[0]
    length = Parameter(default=10., unit=frame_unit)
    space = Parameter(default=1., unit=frame_unit)
    n_scans = Parameter(default=10.)
    rot = Parameter(default=0., unit=u.deg)
    speed = Parameter(default=1., unit=frame_unit / u.s)
    # accel = Parameter(default=1., unit=cls.frame_unit / u.s ** 2)
    t_turnover = Parameter(default=1., unit=u.s)

    def get_total_time(self):
        return self.n_scans * self.length / self.speed + (
                self.n_scans - 1.) * self.t_turnover

    @timeit
    def evaluate(
            self, t, length, space, n_scans, rot, speed, t_turnover):
        """This computes a raster patten around the origin.

        This assumes a circular turn over trajectory.
        """
        t = np.asarray(t)
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

    act_plot_raster_on_wyatt = maap.add_action_parser(
            'plot_raster_on_wyatt',
            help='Plot a raster pattern on Waytt'
            )

    @act_plot_raster_on_wyatt.parser_action
    def plot_raster_on_wyatt_action(option):

        logger = get_logger()

        calobj = ToltecCalib.from_indexfile(option.calobj)

        wyatt_proj_kwargs = {
            'rot': -2. * u.deg,
            'scale': (3., 3.),
            'ref_coord': (0., 0.) * u.cm,
            }

        raster_scan_kwargs = {
            'rot': 30. * u.deg,
            'length': 50. * u.cm,
            'space': 5. * u.cm,
            'n_scans': 10,
            'speed': 1. * u.cm / u.s,
            't_turnover': 5 * u.s,
            }

        m_obs = WyattRasterScanModel(**raster_scan_kwargs)

        t_total = m_obs.get_total_time()
        t = np.arange(0, t_total, 0.5) * u.s

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

    option = maap.parse_args(sys.argv[1:])
    maap.bootstrap_actions(option)
