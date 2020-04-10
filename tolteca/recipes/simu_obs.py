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
        super().__init__(crval0=ref_coord[0], crval1=ref_coord[1], **kwargs)

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
                name=array_name, axes_names=("x", "y"),
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


# def main(data_args, save_data=None):
#     if isinstance(data_args, str):
#         with open(data_args, 'rb') as fo:
#             data = pickle.load(fo)
#     else:
#         data = timeit(make_data)(*data_args)
#         if save_data is not None:
#             with open(save_data, 'wb') as fo:
#                 pickle.dump(data, fo)
#     make_plot(data)


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
    import argparse
    from tolteca.cal import ToltecCalib

    parser = argparse.ArgumentParser(
            description='Make simulated observation.')

    parser.add_argument(
            '--calobj', '-c',
            help='Path to calibration object.',
            required=True
            )

    option = parser.parse_args(sys.argv[1:])

    logger = get_logger()

    calobj = ToltecCalib.from_indexfile(option.calobj)

    wyatt_proj_kwargs = {
            'rot': -2. * u.deg,
            'scale': (3., 3.),
            'ref_coord': (12., 12.) * u.cm,
            }

    plot_wyatt_plane(calobj, **wyatt_proj_kwargs)
