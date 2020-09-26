#! /usr/bin/env python

# Author:
#   Zhiyuan Ma
#
# History:
#   2020/04/02 Zhiyuan Ma:
#       - Created.

"""
This recipe defines a TolTEC KIDs data simulator.

This recipe additionally requires to install a custom version of animatplot::

    pip install git+https://github.com/Jerry-Ma/animatplot.git

"""

from contextlib import contextmanager
from astroquery.utils import parse_coordinates
from kidsproc.kidsmodel.simulator import KidsSimulator
from astroplan import Observer
from astropy.wcs.utils import celestial_frame_to_wcs
from astropy.time import Time
from pytz import timezone
from tollan.utils.log import timeit
from tollan.utils.cli.multi_action_argparser import \
        MultiActionArgumentParser
from regions import PixCoord, PolygonPixelRegion, PolygonSkyRegion
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from tollan.utils.log import timeit
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
from astropy.coordinates.baseframe import frame_transform_graph


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
                mx_p.to_value(u.arcmin),
                my_p.to_value(u.arcmin),
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
    t = np.arange(0, t_total.to_value(u.s), 0.5) * u.s
    n_pts = t.size

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

    logger = get_logger()

    ref_coord = parse_coordinates(ref_obj)
    logger.debug(f"ref obj: {ref_obj} {ref_coord.to_string('hmsdms')}")

    t_total = m_obs.get_total_time()
    t = np.arange(0, t_total.to_value(u.s), 0.5) * u.s
    n_pts = t.size

    logger.debug(f"create {n_pts} pointings")

    obs_coords = m_obs.evaluate_at(ref_coord, t)

    from astropy.wcs import WCS
    from astroquery.skyview import SkyView
    # from astropy.visualization import ZScaleInterval, ImageNormalize
    from astropy.visualization import make_lupton_rgb
    # make rgb 2mass
    hdulists = SkyView.get_images(
            ref_coord,
            # survey=['WISE 12', 'WISE 4.6', 'WISE 3.4'],
            survey=['2MASS-K', '2MASS-H', '2MASS-J'],
            )
    # scales = [0.3, 0.8, 1.0]
    scales = [1.5, 1.0, 1.0]  # white balance

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
        projs = {
                'wyatt': {
                    'proj_model': WyattProjModel,
                    'rot': -2. * u.deg,
                    'scale': (30. / 4., 30. / 4.) * u.cm / u.arcmin,
                    'ref_coord': (15., 15.) * u.cm,
                    },
                'sky': {
                    'proj_model': SkyProjModel,
                    'mjd_obs': Time(option.time_utc).mjd * u.day,
                    'ref_coord': (180., 30.) * u.deg,
                    'evaluate_frame': 'icrs'
                    }
                }
        plot_projected(calobj, **projs[option.proj_model])

    act_plot_obs = maap.add_action_parser(
            'plot_obs',
            help='Plot an obs pattern.'
            )
    act_plot_obs.add_argument(
            "pattern", choices=['raster', 'lissajous'])

    act_plot_obs.add_argument(
            "--target_frame", '-t', choices=['wyatt', 'sky'], required=True)

    act_plot_obs.add_argument(
            "--ref_obj", '-r', default='M51',
            help='The reference object of target frame is sky.')

    @act_plot_obs.parser_action
    def plot_obs_action(option):

        calobj = ToltecCalib.from_indexfile(option.calobj)

        proj_kw = {
                'wyatt': {
                    'rot': -2. * u.deg,
                    'scale': (30. / 4., 30. / 4.) * u.cm / u.arcmin,
                    'ref_coord': (15., 15.) * u.cm,
                    },
                'sky': {
                    'ref_obj': option.ref_obj
                    }
                }
        if option.target_frame == 'wyatt':
            patterns = {
                    'raster': {
                        'model': WyattRasterScanModel,
                        'rot': 30. * u.deg,
                        'length': 50. * u.cm,
                        'space': 5. * u.cm,
                        'n_scans': 10 * u.dimensionless_unscaled,
                        'speed': 1. * u.cm / u.s,
                        't_turnover': 5 * u.s,
                        },
                    'lissajous': {
                        'model': WyattLissajousModel,
                        'rot': 30. * u.deg,
                        'x_length': 50. * u.cm,
                        'y_length': 50. * u.cm,
                        'x_omega': 0.5 * np.pi * u.rad / u.s,
                        'y_omega': 0.7 * np.pi * u.rad / u.s,
                        'delta': 30 * u.deg
                        # 'delta': 60. * u.deg
                        },
                    }
            m_obs = ProjModel.from_dict(patterns[option.pattern])
            plot_obs_on_wyatt(calobj, m_obs, **proj_kw[option.target_frame])
        elif option.target_frame == 'sky':
            patterns = {
                    'raster': {
                        'model': SkyRasterScanModel,
                        'rot': 30. * u.deg,
                        'length': 2. * u.arcmin,
                        'space': 5. * u.arcsec,
                        'n_scans': 24 * u.dimensionless_unscaled,
                        'speed': 1. * u.arcsec / u.s,
                        't_turnover': 5 * u.s,
                        },
                    'lissajous': {
                        'model': SkyLissajousModel,
                        'rot': 30. * u.deg,
                        'x_length': 2. * u.arcmin,
                        'y_length': 2. * u.arcmin,
                        'x_omega': 0.07 * np.pi * u.rad / u.s,
                        'y_omega': 0.05 * np.pi * u.rad / u.s,
                        'delta': 30 * u.deg
                        }
                    }
            m_obs = ProjModel.from_dict(patterns[option.pattern])
            plot_obs_on_sky(calobj, m_obs, **proj_kw[option.target_frame])

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
        tbl = calobj.get_array_prop_table()
        # tbl = tbl[tbl['nw'] == 3]

        simulator = ToltecObsSimulator(tbl)

        # the obs definition
        obs_params = {
            'patterns': {
                'raster': {
                    'model': SkyRasterScanModel,
                    'rot': 30. * u.deg,
                    'length': 4. * u.arcmin,
                    'space': 5. * u.arcsec,
                    'n_scans': 24 * u.dimensionless_unscaled,
                    'speed': 30. * u.arcsec / u.s,
                    't_turnover': 5 * u.s,
                    },
                'lissajous': {
                    'model': SkyLissajousModel,
                    'rot': 30. * u.deg,
                    'x_length': 2. * u.arcmin,
                    'y_length': 2. * u.arcmin,
                    'x_omega': 0.7 * np.pi * u.rad / u.s,
                    'y_omega': 0.5 * np.pi * u.rad / u.s,
                    'delta': 30 * u.deg
                    }
                },
            'fsmp': 12.2 * u.Hz,
            't_exp': 2 * u.min,
            't0': Time('2020-04-13 00:00:00')
            }
        m_obs = ProjModel.from_dict(obs_params['patterns'][option.pattern])

        # make a source catalog
        sources = Table(
                [
                    Column(name='name', dtype='|S32'),
                    Column(name='ra', unit=u.deg),
                    Column(name='dec', unit=u.deg),
                    Column(name='flux', unit=u.mJy),
                    ])
        sources.add_row(['src0', 180., 0., 50.])
        sources.add_row(['src1', 180., 30. / 3600., 0.25])

        logger.debug(f"sources:\n{sources}")

        with simulator.obs_context(obs_model=m_obs, sources=sources) as obs:
            # make t grid
            t = np.arange(
                    0, obs_params['t_exp'].to_value(u.s),
                    (1 / obs_params['fsmp']).to_value(u.s)) * u.s
            s, obs_info = obs(obs_params['t0'], t)

        with simulator.probe_context(fp=None) as probe:
            rs, xs, iqs = probe(s)

        # make some diagnostic plots
        tbl = simulator.table

        m = tbl['array_name'] == 'a1100'
        mtbl = tbl[m]
        mtbl.meta = tbl.meta['a1100']

        # unpack the obs_info
        native_frame = obs_info['native_frame']
        projected_frame = obs_info['projected_frame']
        src_coords = obs_info['src_coords']

        import animatplot as amp

        fps = 2 * u.Hz
        # fps = 12 * u.Hz
        t_slice = slice(
                None, None,
                int(np.ceil(
                    (obs_params['fsmp'] / fps).to_value(
                        u.dimensionless_unscaled))))
        fps = (obs_params['fsmp'] / t_slice.step).to_value(u.Hz)
        timeline = amp.Timeline(
                t[t_slice].to_value(u.s),
                fps=1 if fps < 1 else fps,
                units='s')
        # xx = x_t[m].to_value(u.arcmin)
        # yy = y_t[m].to_value(u.arcmin)
        xx = mtbl['x_t'].quantity.to_value(u.deg)
        yy = mtbl['y_t'].quantity.to_value(u.deg)

        ss = s[m, t_slice].T.to_value(u.MJy/u.sr)
        rrs = rs[m, t_slice].T
        xxs = xs[m, t_slice].T

        cmap = 'viridis'
        cmap_kwargs = dict(
                cmap=cmap,
                vmin=np.min(ss),
                vmax=np.max(ss),
                )

        import matplotlib.path as mpath
        import matplotlib.markers as mmarkers
        from matplotlib.transforms import Affine2D

        def make_fg_marker(fg):
            _transform = Affine2D().scale(0.5).rotate_deg(30)
            polypath = mpath.Path.unit_regular_polygon(6)
            verts = polypath.vertices
            top = mpath.Path(verts[(1, 0, 5, 4, 1), :])
            rot = [0, -60, -180, -240][fg]
            marker = mmarkers.MarkerStyle(top)
            marker._transform = _transform.rotate_deg(rot)
            return marker

        from astropy.visualization.wcsaxes import WCSAxesSubplot, conf
        from astropy.visualization.wcsaxes.transforms import (
                CoordinateTransform)

        # coord_meta = {
        #         'type': ('longitude', 'latitude'),
        #         'unit': (u.deg, u.deg),
        #         'wrap': (180, None),
        #         'name': ('Az Offset', 'Alt Offset')}
        from astropy.visualization.wcsaxes.utils import get_coord_meta
        coord_meta = get_coord_meta(native_frame[0])
        # coord_meta = get_coord_meta('icrs')
        conf.coordinate_range_samples = 5
        conf.frame_boundary_samples = 10
        conf.grid_samples = 5
        conf.contour_grid_samples = 5

        fig = plt.figure()
        ax = WCSAxesSubplot(
                fig, 3, 1, 1,
                aspect='equal',
                # transform=Affine2D(),
                transform=(
                    # CoordinateTransform(native_frame[0], 'icrs') +
                    CoordinateTransform(projected_frame[0], native_frame[0])
                    ),
                coord_meta=coord_meta,
                )
        fig.add_axes(ax)
        bx = fig.add_subplot(3, 1, 2)
        cx = fig.add_subplot(3, 1, 3)

        def amp_post_update(block, i):
            ax.reset_wcs(
                    transform=(
                        # CoordinateTransform(native_frame[i], 'icrs')
                        CoordinateTransform(
                            projected_frame[i], native_frame[i])
                        ),
                    coord_meta=coord_meta)
        # fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        # ax.set_facecolor(plt.get_cmap(cmap)(0.))
        ax.set_facecolor('#4488aa')
        nfg = 4
        pos_blocks = np.full((nfg, ), None, dtype=object)
        for i in range(nfg):
            mfg = mtbl['fg'] == i
            pos_blocks[i] = amp.blocks.Scatter(
                    xx[mfg],
                    yy[mfg],
                    s=np.abs(np.hypot(xx[0] - xx[2], yy[0] - yy[2])),
                    s_in_data_unit=True,
                    c=ss[:, mfg],
                    ax=ax,
                    # post_update=None,
                    post_update=amp_post_update if i == 0 else None,
                    marker=make_fg_marker(i),
                    # edgecolor='#cccccc',
                    **cmap_kwargs
                    )
        # add a block for the IQ values
        signal_blocks = np.full((2, ), None, dtype=object)
        for i, (vv, aa) in enumerate(zip((rrs, xxs), (bx, cx))):
            signal_blocks[i] = amp.blocks.Line(
                    np.tile(mtbl['f'], (vv.shape[0], 1)),
                    vv,
                    ax=aa,
                    marker='o',
                    linestyle='none',
                    )
        anim = amp.Animation(np.hstack([pos_blocks, signal_blocks]), timeline)

        anim.controls()

        # cax_ticks, cax_label, cmap_kwargs = _make_nw_cmap()
        # im = ax.scatter(
        #         x_t[m].to(u.arcmin), y_t[m].to(u.arcmin),
        #         # c=dists[0, m, 0].to(u.arcmin)
        #         c=s[0, m, 0].to_value(u.MJy / u.sr)
        #         # c=tbl['nw'][m], **cmap_kwargs
        #         )
        # fig.colorbar(
        #     im, ax=ax, shrink=0.8)
        # normalize lw with flux
        lws = 0.2 * sources['flux'] / np.max(sources['flux'])
        for i in range(len(sources)):
            ax.plot(
                    src_coords[i].lon,
                    src_coords[i].lat,
                    linewidth=lws[i], color='#aaaaaa')
        cax = fig.colorbar(
            pos_blocks[0].scat, ax=ax, shrink=0.8)
        cax.set_label("Surface Brightness (MJy/sr)")
        plt.show()

    option = maap.parse_args(sys.argv[1:])
    maap.bootstrap_actions(option)
