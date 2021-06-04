#!/usr/bin/env python

# Author:
#   Zhiyuan Ma

"""
This recipe matches (right to left) two array property tables using the
detector positions.

Inputs:

* left APT: The table acts as the base of the match. The result APT will be the
same shape as this table.

* right APT: The table to match. Each entry of this table will be either
assigned to the base table and included in the result APT, or end up in the
result "unmatched" APT.

Outputs:

* matched APT: the matched table containing info from both input tables.

* unmatched APT: The subset of right APT that does not find a match.

Procedure:

1. Compute the initial transformation that matches the outline of the
left table and right table

2. figure out frequency groups if not already present.
(has to be done interactively, not implemented yet)

3. figure out initial match on a per fg, per network basis.
(has to be done interactively, not implemented yet)

4. Solve the WCS transformation between the matched points

5. Apply the transformation on the entire array and do a 2-d Cartesian match.

"""

import sys
import argparse
from pathlib import Path

import numpy as np
from astropy.table import Table, join
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.modeling.models import Rotation2D

from tolteca.recipes import get_logger
from tollan.utils.mpl import save_or_show


# from astropy.io.misc.yaml import AstropyDumper
# import yaml


# def should_use_block(value):
#     return '\n' in value or len(value) > 100


# def my_represent_scalar(self, tag, value, style=None):
#     if style is None:
#         if should_use_block(value):
#             style = '>'
#         else:
#             style = self.default_style
#     if isinstance(value, str):
#         if should_use_block(value):
#             style = '>'

#     node = yaml.representer.ScalarNode(tag, value, style=style)
#     if self.alias_key is not None:
#         self.represented_objects[self.alias_key] = node
#     return node


# AstropyDumper.represent_scalar = my_represent_scalar


def load_apt(source, ensure_columns=None):
    logger = get_logger()
    try:
        apt = Table.read(source, format='ascii.ecsv')
    except Exception:
        from tolteca.cal.toltec import ToltecArrayProp

        apt = ToltecArrayProp.from_indexfile(source).get()
    # validate that in the apt we have the columns required
    if ensure_columns is not None:
        if set(apt.colnames).issuperset(ensure_columns):
            pass
        else:
            raise ValueError(
                f"APT missing required columns "
                f"{set(ensure_columns) - set(apt.colnames)}")
    # make availabel array_name
    if 'array_name' not in apt.colnames:
        logger.debug("add array_name column")
        array_names = [_nw_to_array(nw) for nw in apt['nw']]
        apt['array_name'] = array_names
    # clean up column meta, from topcat
    for c in apt.colnames:
        col = apt[c]
        if col.meta.pop('$ID', None) is not None:
            if col.meta.pop('Expression', None) is not None:
                col.description = None
    return apt


def _nw_to_array(nw):
    if nw < 7:
        return 'a1100'
    if nw < 11:
        return 'a1400'
    return 'a2000'


def get_xy(apt, frame='array'):
    # get the coordinates in specified frame
    if frame not in ['toltec', 'array']:
        raise ValueError("invalid frame.")
    if frame == 'toltec':
        # the toltec frame default column is x_t and y_t
        if 'x_t' not in apt.colnames:
            # make toltec frame
            from tolteca.simu.toltec import ArrayProjModel
            x_t, y_t = ArrayProjModel()(
                    apt['array_name'],
                    apt['x'].to(u.cm),
                    apt['y'].to(u.cm))
            apt['x_t'] = x_t
            apt['y_t'] = y_t
        result = Table()
        result['x'] = apt['x_t']
        result['y'] = apt['y_t']
        return result
    if frame == 'array':
        if apt['x'].unit is None:
            result = Table()
            result['x'] = apt['x'] << u.dimensionless_unscaled
            result['y'] = apt['y'] << u.dimensionless_unscaled
            return result
        return apt[['x', 'y']]


def get_f(apt):
    for colname in ['f', 'f_in']:
        if colname in apt.colnames:
            f = apt[colname]
            if f.unit is None:
                f = f << u.Hz
            result = Table()
            result['f'] = f.to(u.Hz)
            return result
    raise ValueError("unable to get f column")


def match_outline(xy_left, xy_right):

    logger = get_logger()

    def get_bbox(xy, q=0.05):
        x, y = xy['x'].quantity, xy['y'].quantity
        # left, right, bottom, top
        unit = x.unit
        x_value, y_value = x.to_value(unit), y.to_value(unit)
        ll = np.quantile(x_value, q)
        rr = np.quantile(x_value, 1 - q)
        bb = np.quantile(y_value, q)
        tt = np.quantile(y_value, 1 - q)
        return np.array([ll, rr, bb, tt]) << unit

    bb_left = get_bbox(xy_left)
    logger.debug("bbox left: {bb_left}")

    bb_right = get_bbox(xy_right)
    logger.debug("bbox right: {bb_right}")

    # scale to go from left to right
    x_scale = (bb_right[1] - bb_right[0]) / (bb_left[1] - bb_left[0])
    y_scale = (bb_right[3] - bb_right[2]) / (bb_left[3] - bb_left[2])
    logger.debug(f"scale: {x_scale=} {y_scale=}")

    # shift
    bb_left_s = np.hstack([
        bb_left[:2] * x_scale,
        bb_left[2:] * y_scale])

    x_shift = (bb_right[1] + bb_right[0] - bb_left_s[1] - bb_left_s[0]) * 0.5
    y_shift = (bb_right[3] + bb_right[2] - bb_left_s[3] - bb_left_s[2]) * 0.5
    logger.debug(f"shfit: {x_shift=} {y_shift=}")

    # no rotation is needed if we use toltec frame
    from astropy.modeling.models import Shift, Multiply

    def transform(xy):
        x, y = xy['x'], xy['y']

        transform_model = (
                (Multiply(x_scale) & Multiply(y_scale))
                | (Shift(x_shift) & Shift(y_shift))
                )

        result = Table()
        x1, y1 = transform_model(x, y)
        result['x'] = x1
        result['y'] = y1
        # result['x'] = x * x_scale + x_shift
        # result['y'] = y * y_scale + y_shift
        result.meta['x_scale'] = x_scale
        result.meta['y_scale'] = y_scale
        result.meta['x_shift'] = x_shift
        result.meta['y_shift'] = y_shift
        result.meta['expression'] = (
                'x = x * x_scale + x_shift;'
                ' y = y * y_scale + y_shift')
        # result.meta['model'] = transform_model
        return result

    return transform


def match_per_nw_fg(
        xy_left, apt_left, xy_right, apt_right,
        use_pickled=False,
        use_pickle_file=None,
        pickle_to_file='match_per_nw_fg.pickle'):
    # here we go through each nw and fg in right, plot left
    # and right, using matplotlib event to select pairs
    from matplotlib.patches import Polygon

    logger = get_logger()
    g = apt_right.group_by(['nw', 'fg'])

    idx_left = np.array(range(len(apt_left)))
    idx_right = np.array(range(len(apt_right)))

    import pickle

    if use_pickle_file is not None:
        # force skip
        logger.warning("input pickle file is used, skip match")
        use_pickled = True
        pfile = Path(use_pickle_file)
    else:
        pfile = Path(pickle_to_file)
    if pfile.exists():
        with open(pfile, 'rb') as fo:
            matched_tris = pickle.load(fo)
            if use_pickled:
                return matched_tris
    else:
        matched_tris = dict()
    logger.warning(f"saving tris to {pfile}")
    for key, group in zip(g.groups.keys, g.groups):
        logger.debug(
            f'****** nw={key["nw"]} fg={key["fg"]} *******\n'
            f'{group}\n'
            )

        if key['fg'] not in range(2):
            continue

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.set_title(f"nw={key['nw']} fg={key['fg']}")
        m_left = (apt_left['nw'] == key['nw']) & (
                (
                    apt_left['fg'] == key['fg'])
                ) & (
                        apt_left['flag'] == 0  # active only
                        )
        # two polarization groups, the co-exist on same spot
        m_right_pg0 = (apt_right['nw'] == key['nw']) & (
                (
                    apt_right['fg'] == key['fg'])
                )
        m_right_pg1 = (apt_right['nw'] == key['nw']) & (
                (
                    apt_right['fg'] == key['fg'] + 2)
                )
        scales = {
                'x': 1.,
                'y': 1.,
                }
        offsets = {
                'x': 0.,
                'y': 0.,
                }
        rotate = [0.0, ]

        def trans_left(x, y):
            m_rot = Rotation2D._compute_matrix(
                    angle=(rotate[0] << u.deg).to_value('rad'))

            cx = np.median(x)
            cy = np.median(y)
            xy = m_rot @ np.c_[
                    (x - cx) * scales['x'],
                    (y - cy) * scales['y'],
                    ].T
            return xy[0] + cx + offsets['x'], xy[1] + cy + offsets['y']
        sc_left = ax.scatter(
                *trans_left(xy_left[m_left]['x'], xy_left[m_left]['y']),
                s=60,
                marker='o', facecolor='none', color='black', picker=True)
        sc_right_pg0 = ax.scatter(
                xy_right[m_right_pg0]['x'], xy_right[m_right_pg0]['y'], s=40,
                marker='o', color='red', picker=True)
        sc_right_pg1 = ax.scatter(
                xy_right[m_right_pg1]['x'], xy_right[m_right_pg1]['y'], s=30,
                marker='o', color='green', picker=True)
        fig.text(
                0.98, 0.5,
                '''
Interactive matcher:
left-click: select
         e: save
         d: delete
      left: -x
      right: +x
      down: -y
        up: +y
         7: x in
         8: x out
         9: y in
         0: y out
''',
                fontsize=10, fontfamily='monospace',
                horizontalalignment='right',
                verticalalignment='center',
                )

        mode = 'append'
        mtk = (key['nw'], key['fg'])
        if mtk not in matched_tris:
            tris = matched_tris[mtk] = list()
        else:
            tris = matched_tris[mtk]
        a_tris = []

        def _plot_tris():
            for a in a_tris:
                try:
                    a.remove()
                except Exception:
                    pass

            def _xy(r):
                return [r['x'], r['y']]

            def _xy_l(r):
                return list(trans_left(r['x'], r['y']))
            for tri in tris:
                if not set(tri.keys()).issuperset({
                        'left', 'right_pg0', 'right_pg1'}):
                    continue
                p = Polygon(
                    np.array([
                        _xy_l(xy_left[tri['left']]),
                        _xy(xy_right[tri['right_pg0']]),
                        _xy(xy_right[tri['right_pg1']]),
                        ]
                    ))
                ax.add_patch(p)
                a_tris.append(p)
            ax.figure.canvas.draw_idle()

        def onpick(event):
            ind = event.ind[0]
            artist = event.artist
            if artist is sc_left:
                print(f"clicked left {ind}")
                t = {'left': idx_left[m_left][ind]}
            elif artist is sc_right_pg0:
                print(f"clicked right_pg0 {ind}")
                t = {'right_pg0': idx_right[m_right_pg0][ind]}
            elif artist is sc_right_pg1:
                print(f"clicked right_pg1 {ind}")
                t = {'right_pg1': idx_right[m_right_pg1][ind]}
            if mode == 'append':
                if len(tris) == 0 or set(tris[-1].keys()).issuperset({
                        'left', 'right_pg0', 'right_pg1'}):
                    tris.append(dict())
                    print("********* new entry added *********")
                else:
                    print("update last entry")
                d = tris[-1]
                d.update(t)
                d.update(key)
                _plot_tris()
                print(f"current d={d} tris size={len(tris)}")
            # update all tris
        fig.canvas.mpl_connect('pick_event', onpick)

        def _update_sc_left():
            x, y = trans_left(xy_left[m_left]['x'], xy_left[m_left]['y'])
            sc_left.set_offsets(np.vstack([x, y]).T)
            _plot_tris()
            ax.figure.canvas.draw_idle()

        def onkey(event):
            if event.key == 'd':
                print("remove last tris")
                if len(tris) == 0:
                    return
                tris.remove(tris[-1])
                _plot_tris()
                return
            if event.key == 'e':
                print("dump tris")
                with open(pfile, 'wb') as fo:
                    pickle.dump(matched_tris, fo)
                return
            if event.key.startswith('shift+'):
                d = 0.25
                ds = 1.005
                key = event.key.split('+')[-1]
            else:
                d = 1.25
                ds = 1.025
                key = event.key
            if key == 'left':
                offsets['x'] -= d
                _update_sc_left()
                return
            if key == 'right':
                offsets['x'] += d
                _update_sc_left()
                return
            if key == 'up':
                offsets['y'] += d
                _update_sc_left()
                return
            if key == 'down':
                offsets['y'] -= d
                _update_sc_left()
                return
            if key == '7':
                scales['x'] /= ds
                _update_sc_left()
                return
            if key == '8':
                scales['x'] *= ds
                _update_sc_left()
                return
            if key == '9':
                scales['y'] *= ds
                _update_sc_left()
                return
            if key == '0':
                scales['y'] /= ds
                _update_sc_left()
                return
            if key == 'o':
                rotate[0] -= 1.
                _update_sc_left()
                return
            if key == 'p':
                rotate[0] += 1.
                _update_sc_left()
                return
            if key == 'r':
                offsets['x'] = 0
                offsets['y'] = 0
                scales['x'] = 1
                scales['y'] = 1
                rotate[0] = 0.
                _update_sc_left()
                return
        fig.canvas.mpl_connect('key_press_event', onkey)
        _plot_tris()
        plt.show()

    return matched_tris


def _cal_wcs(xy0, xy1):

    from astropy.wcs.utils import fit_wcs_from_points
    from astropy import wcs

    w = wcs.WCS(naxis=2)

    center = np.mean(xy0, axis=1)
    print(f'wcs center: {center}')
    w.wcs.crpix = center
    scale = 1. / 3600.  # 1 arcsec pixel
    w.wcs.cd = np.array([[-1, 0.], [0., 1]]) * scale
    w.wcs.crval = [180., 0.]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    c = SkyCoord(ra=180. * u.degree, dec=0. * u.degree, frame='icrs')
    rd0 = SkyCoord(*w.all_pix2world(*xy0, 0), frame='icrs', unit='deg')

    w1 = fit_wcs_from_points(
            xy1, rd0, proj_point=c, projection='TAN', sip_degree=4)
    w.printwcs()
    w1.printwcs()
    return locals()


def cal_wcs(
        xy_left, apt_left, xy_right, apt_right, matched_tris,
        save_plot=None
        ):

    import matplotlib.colors as mcolors

    logger = get_logger()

    array_names = ['a1100', 'a1400', 'a2000']
    n_arrays = len(array_names)

    # make check plot
    fig, axes = plt.subplots(
            2, n_arrays, constrained_layout=True, figsize=(15, 10),
            sharex=True, sharey=True)

    # collate tris by array
    mtd = dict()
    for array_name in array_names:
        mtd[array_name] = dict(
                left=list(),
                rpg0=list(),
                rpg1=list(),
                )
    for (nw, fg), tris in matched_tris.items():
        logger.debug(f"unpack matched {nw=} {fg=} n={len(tris)}")
        array_name = _nw_to_array(nw)
        d = mtd[array_name]
        for tri in tris:
            if not set(tri.keys()).issuperset(
                    {'left', 'right_pg0', 'right_pg1'}):
                logger.debug(f"discard incomplete {tri}")
                continue
            # maybe there are duplicates, so we check the existence of
            # them in xy_left_idx. this is O(n^2) but I guess it won't be
            # too slow
            if tri['left'] not in d['left']:
                d['left'].append(tri['left'])
                d['rpg0'].append(tri['right_pg0'])
                d['rpg1'].append(tri['right_pg1'])
            else:
                logger.debug(f"discard duplicated {tri}")
    # for each array, calculate wcs
    result = dict()
    for i, array_name in enumerate(array_names):
        xy_left_in = xy_left[mtd[array_name]['left']]
        xy_rpg0_in = xy_right[mtd[array_name]['rpg0']]
        xy_rpg1_in = xy_right[mtd[array_name]['rpg1']]

        xy0 = (xy_left_in['x'], xy_left_in['y'])
        xy1 = (
            (xy_rpg0_in['x'] + xy_rpg1_in['x']) * 0.5,
            (xy_rpg0_in['y'] + xy_rpg1_in['y']) * 0.5,
            )
        wd = result[array_name] = _cal_wcs(xy0, xy1)
        # calculated corrected xy position
        rd = wd['w1'].all_pix2world(*xy1, 0)
        xy = wd['w'].all_world2pix(*rd, 0)

        # plot the points
        ax = axes[0, i]
        ax.set_title(f'{array_name} n={len(xy_left_in)}')
        ax.scatter(*xy, label='corrected pos', color='C1')
        ax.scatter(
            *xy0, label='designed pos',
            marker='o', edgecolor='C2', facecolors='none')
        ax.scatter(*xy1, marker='x', label='measured pos')
        ax.set_aspect('equal')
        if i == n_arrays - 1:
            ax.legend()

        # make plot of distortion. we generate a pixel grid in
        # the designed focal plane and evaluate the observed positions
        ax = axes[1, i]
        mm = (apt_left['array_name'] == array_name) & (
                apt_left['ori'] == 0
                )
        jj, ii = (
                xy_left['x'][mm],
                xy_left['y'][mm]
                )
        # jj, ii = np.meshgrid(
        #         # np.linspace(xy0[0].min(), xy0[0].max(), 50),
        #         # np.linspace(xy0[1].min(), xy0[1].max(), 50),
        #         )
        if isinstance(jj, u.Quantity):
            jj = jj.value
            ii = ii.value
        xy2 = (np.ravel(jj), np.ravel(ii))
        # fiducial rd coordinate, if forwarded with distortion
        rd2 = wd['w'].all_pix2world(*xy2, 0)
        # distorted xy , from measured wcs
        xy3 = wd['w1'].all_world2pix(*rd2, 0, quiet=True)
        # offsets
        uu, vv = xy3[0] - xy2[0], xy3[1] - xy2[1]
        # remove the zeroth order
        ax.set_title(f'0th offset: {uu.mean():.2g}, {vv.mean():.2g}')
        uu = uu - np.mean(uu)
        vv = vv - np.mean(vv)

        zz = (np.hypot(uu, vv)).reshape(ii.shape)
        # c = ax.pcolormesh(jj, ii, zz)
        vmin = np.quantile(zz, 0.1)
        vmax = np.quantile(zz, 0.9)
        c = ax.quiver(
                jj, ii, uu, vv, zz,
                cmap='rainbow',
                norm=mcolors.Normalize(vmin=vmin, vmax=vmax),)
        fig.colorbar(c)
        ax.set_aspect('equal')
    save_or_show(fig, filepath=save_plot, save=save_plot is not None)
    return result


def correct_with_wcs(xy_right, apt_right, wcs_info, save_plot=None):
    # we need to do this on a per array bases
    x_cor = np.full((len(xy_right),), np.nan, dtype='d')
    y_cor = np.full((len(xy_right),), np.nan, dtype='d')

    # make check plot
    fig, axes = plt.subplots(
            2, len(wcs_info), constrained_layout=True, figsize=(15, 10),
            sharex=True, sharey=True)

    for i, (array_name, wd) in enumerate(wcs_info.items()):
        w1 = wd['w1']
        w = wd['w']
        m = (apt_right['array_name'] == array_name)
        rd = w1.all_pix2world(xy_right[m]['x'], xy_right[m]['y'], 0)
        xy = w.all_world2pix(*rd, 0)
        x_cor[m] = xy[0]
        y_cor[m] = xy[1]
        axes[0, i].set_title(f'{array_name} measured')
        axes[1, i].set_title(f'{array_name} corrected')
        axes[0, i].scatter(
                xy_right[m]['x'], xy_right[m]['y'], s=10,
                c=apt_right['snr'][m]
                )
        axes[1, i].scatter(
                xy[0], xy[1], s=10,
                c=apt_right['snr'][m]
                )
        axes[0, i].set_xlim((0, 60))
        axes[0, i].set_ylim((0, 60))
        axes[0, i].set_aspect('equal')
    save_or_show(fig, filepath=save_plot, save=save_plot is not None)
    result = Table()
    assert np.isnan(x_cor).sum() == 0
    result['x'] = x_cor
    result['y'] = y_cor
    return result


def match_corrected_2d(
        xy_left,
        apt_left,
        xy_right,
        apt_right,
        pickle_to_file=None,
        save_plot_per_group=None,
        save_plot_summary=None,
        ):

    logger = get_logger()
    from tollan.utils.wraps.stilts import stilts_match2d, ensure_stilts
    stilts_cmd = ensure_stilts()

    idx_left = np.array(range(len(apt_left)))
    idx_right = np.array(range(len(apt_right)))

    xy0 = Table(xy_left)
    xy0['idx'] = idx_left
    xy1 = Table(xy_right)
    xy1['idx'] = idx_right

    # here we need to match on a per fg basis.
    # for those with no fg, we match to those that did not get a
    # match in the apt.
    g = apt_right.group_by(['nw', 'fg'])
    nws = np.unique(apt_right['nw'])
    fgs = range(4)

    # we plot the histogram of separations for each nw/fg
    fig, axes = plt.subplots(
            len(fgs), len(nws),
            constrained_layout=True,
            figsize=(16, 8)
            )

    matched_dict = dict()
    # load from pickle if needed to speed up
    import pickle
    if pickle_to_file is not None:
        pfile = Path(pickle_to_file)
        if pfile.exists():
            with open(pfile, 'rb') as fo:
                matched_dict = pickle.load(fo)
    else:
        pfile = None

    use_pickled = True
    if not use_pickled or pfile is None or not pfile.exists():
        for key, group in zip(g.groups.keys, g.groups):
            nw = key['nw']
            fg = key['fg']
            logger.debug(
                f'****** {nw=} {fg=} *******\n'
                f'{group}\n'
                )
            # for entries with fg=-1, we skip and will deal with later
            if fg < 0:
                continue
            # if nw > 2:
            #     continue
            m_left = (apt_left['nw'] == nw) & (
                    apt_left['fg'] == fg) & (
                        apt_left['flag'] == 0  # active only
                        )
            m_right = (apt_right['nw'] == nw) & (
                    apt_right['fg'] == fg)
            # make the match
            # xy0 and xy1 are left and right with idx left and right
            matched = matched_dict[(nw, fg)] = stilts_match2d(
                    xy0[m_left], xy1[m_right],
                    'x y',
                    1,
                    extra_args=['join=all1', 'find=best'],
                    stilts_cmd=stilts_cmd
                    )
            assert len(matched) == m_left.sum()
        if pfile is not None:
            with open(pfile, 'wb') as fo:
                pickle.dump(matched_dict, fo)
    else:
        logger.info(f"use pickled match from {pfile}")
    for (nw, fg), matched in matched_dict.items():
        # plot the separation
        m_left = (apt_left['nw'] == nw) & (
                apt_left['fg'] == fg) & (
                    apt_left['flag'] == 0  # active only
                    )
        m_right = (apt_right['nw'] == nw) & (
                apt_right['fg'] == fg)

        ifg = list(fgs).index(fg)
        inw = list(nws).index(nw)
        ax = axes[ifg, inw]
        ax.hist(matched['Separation'], histtype='step', align='mid')
        n_matched = (matched['Separation'] >= 0).sum()
        n_left = m_left.sum()
        n_right = m_right.sum()
        ax.text(
                0.5, 0.5,
                f'{n_matched}/{n_left}/{n_right}',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        if inw == 0:
            ax.set_ylabel(f"{fg=}")
        if ifg == 0:
            ax.set_title(f"{nw=}")
    save_or_show(
            fig,
            filepath=save_plot_per_group, save=save_plot_per_group is not None)
    # make a summary plot
    array_names = ['a1100', 'a1400', 'a2000']
    n_arrays = len(array_names)
    fig, axes = plt.subplots(
            2, n_arrays, figsize=(10, 5), constrained_layout=True)
    # dispatch matched table per array
    matched_array = dict(a1100=dict(), a1400=dict(), a2000=dict())
    for k, v in matched_dict.items():
        matched_array[_nw_to_array(k[0])][k] = v
    # dispatch matched table per network
    matched_nw = dict()
    for nw in nws:
        matched_nw[nw] = dict()
    for k, v in matched_dict.items():
        matched_nw[k[0]][k] = v

    # for each array, create the plot
    for i, (array_name, md) in enumerate(matched_array.items()):
        ax = axes[0, i]
        ax.hist([t['Separation'] for t in md.values()], label=[
            f'{nw=} {pg=}' for (nw, pg) in md.keys()
            ], histtype='barstacked', align='mid')
        ax.set_title(f"{array_name}")
        ax.set_xlabel("Pos. Offset (input pos. unit)")
        if i == 0:
            ax.set_ylabel("Count")
        # ax.legend()
    # plot in the second row the mean spearation
    sep_med = np.full((len(fgs), len(nws)), np.nan)
    frac_matched = np.full((len(fgs), len(nws)), np.nan)
    for (nw, fg), v in matched_dict.items():
        inw = list(nws).index(nw)
        ifg = list(fgs).index(fg)
        sep_med[ifg, inw] = np.ma.median(v['Separation'])
        n_matched = (v['Separation'] >= 0).sum()
        n_designed = len(v)
        frac_matched[ifg, inw] = n_matched / n_designed
    ax = axes[1, 0]
    c = ax.imshow(
            sep_med,
            extent=[-0.5, len(nws) - 0.5, -0.5, len(fgs) - 0.5],
            )
    ax.set_xlabel("nw")
    ax.set_ylabel("fg")
    fig.colorbar(c, ax=ax)
    ax = axes[1, 1]
    c = ax.imshow(
            frac_matched,
            extent=[-0.5, len(nws) - 0.5, -0.5, len(fgs) - 0.5],
            )
    ax.set_xlabel("nw")
    ax.set_ylabel("fg")
    fig.colorbar(c, ax=ax)
    axes[1, 2].set_axis_off()
    save_or_show(
            fig,
            filepath=save_plot_summary, save=save_plot_summary is not None)
    # now we compose the matched apt by joining the index
    result = Table(apt_left)
    # we use -1 for now because later we need to join
    # so this column cannot be masked column.

    # add xy_left to result
    result['x_t_m'] = xy_left['x']
    result['y_t_m'] = xy_left['y']

    # add matching info
    result['idx_left'] = idx_left
    result['idx_right'] = -1
    result['Separation'] = -1.0
    for v in matched_dict.values():
        m = ~v['Separation'].mask
        # update idx_2 for sucessful match to left
        result['idx_right'][v[m]['idx_1']] = v[m]['idx_2']
        result['Separation'][v[m]['idx_1']] = v[m]['Separation']
    m_matched_left = result['idx_right'] >= 0
    n_matched = m_matched_left.sum()
    print("**********************************************")
    print(f'Successful match: {n_matched}/{len(apt_left)}')

    def report_by(key, lst, md):
        t = Table()
        t[key] = lst
        t['n_designed'] = -1
        t['n_detected'] = -1
        t['n_matched'] = -1
        for i, (n, d) in enumerate(md.items()):
            m_left = apt_left[key] == n
            m_right = apt_right[key] == n
            n_left = m_left.sum()
            n_right = m_right.sum()
            n_matched = (m_left & m_matched_left).sum()
            ia = lst.index(n)
            t['n_designed'][ia] = n_left
            t['n_detected'][ia] = n_right
            t['n_matched'][ia] = n_matched

        t['frac_detected2designed'] = t['n_detected'] / t['n_designed']
        t['frac_detected2designed'].info.format = '.2f'
        t['frac_matched2detected'] = t['n_matched'] / t['n_detected']
        t['frac_matched2detected'].info.format = '.2f'
        t['frac_matched2designed'] = t['n_matched'] / t['n_designed']
        t['frac_matched2designed'].info.format = '.2f'
        t.pprint_all()

    print("By Array:\n==================================")
    report_by('array_name', array_names, matched_array)

    print("By network:\n==================================")
    report_by('nw', list(nws), matched_nw)
    print("************************************************")

    # try to figure out the unmatched portions
    # we need to exclude the nws that are not enabled
    m_check = apt_left['nw'] == nws[0]
    for nw in nws[1:]:
        m_check = m_check | (result['nw'] == nw)
    result_check = result[m_check]
    m_matched_check = result_check['idx_right'] >= 0
    result_unmatched = result_check[~m_matched_check]
    print(
        f"check unmatched left "
        f"{(~m_matched_check).sum()}/{len(result_check)}")
    # figure out the unmatched on right
    idx_right_matched = result_check[m_matched_check]['idx_right']
    assert len(set(idx_right_matched)) == len(idx_right_matched)
    idx_right_unmatched = sorted(list(set(idx_right) - set(idx_right_matched)))
    print("unmatched right: {len(idx_right_unmatched)}/len(apt_right)")
    apt_right_unmatched = apt_right[idx_right_unmatched]

    m_right_nofg = apt_right_unmatched['fg'] < 0
    print(f"no fg: {m_right_nofg.sum()}/{len(apt_right_unmatched)}")
    # plot the unmatched with unmatched result
    # fig, axes = plt.subplots(1, n_arrays, constrained_layout=True)
    # for i, (array_name) in enumerate(array_names):
    #     ax = axes[i]
    #     m_a_left = apt_left[m_check]['array_name'] == array_name
    #     m_a_right = apt_right_unmatched['array_name'] == array_name
    #     xy_left_unmatched = xy_left[m_check][(~m_matched) & m_a_left]
    #     xy_right_unmatched = xy_right[idx_right_unmatched][m_a_right]
    #     xy_right_nofg_unmatched = xy_right[idx_right_unmatched][
    #             m_a_right & m_right_nofg]
    #     ax.scatter(
    #             xy_left_unmatched['x'], xy_left_unmatched['y'], s=60,
    #             marker='o', facecolor='none', color='black', picker=True)
    #     ax.scatter(
    #             xy_right_unmatched['x'], xy_right_unmatched['y'], s=40,
    #             marker='o', color='green', picker=True)
    #     ax.scatter(
    #             xy_right_nofg_unmatched['x'],
    #             xy_right_nofg_unmatched['y'], s=40,
    #             marker='o', color='red', picker=True)
    #     ax.set_aspect('equal')
    # plt.show()
    # join the right to left with id_right
    t_right = Table(apt_right)
    t_right.add_column(idx_right, index=0, name='idx_right')
    t_right.add_column(xy_right['x'], index=0, name='x_m_c')
    t_right.add_column(xy_right['y'], index=0, name='y_m_c')
    t_right.rename_column('x', 'x_m')
    t_right.rename_column('y', 'y_m')
    t_right.rename_column('snr', 'snr_m')
    t_right.rename_column('fwhmx', 'fwhm_x_m')
    t_right.rename_column('fwhmy', 'fwhm_y_m')
    t_right.rename_column('flag', 'f_flag')
    t_right.add_column(t_right['tone_id'], index=0, name='ti')
    t_right.remove_column('tone_id')
    t_right.add_column(t_right['fg'], index=0, name='fg_hint')
    t_right.remove_column('fg')

    # remove duplicated column names
    rmcols_right = list(
            set(t_right.colnames).intersection(set(result.colnames))) + [
                    # these are columns from manual fg tagging
                    'fg0', 'fg1', 'fg2', 'fg3'
                    ]
    rmcols_right.remove("idx_right")
    print(f"remove cols right: {rmcols_right}")
    t_right.remove_columns(rmcols_right)
    result = join(result, t_right, keys='idx_right', join_type='left')
    # sort the result by idx_left
    result.sort(['idx_left'])
    result.remove_column('idx_left')
    result_unmatched.sort(['idx_left'])
    result_unmatched.remove_column('idx_left')

    # make the matching info columns masked
    result.add_column(
            np.ma.array(result['idx_right'], mask=result['idx_right'] < 0),
            index=len(apt_left.colnames) + 2, name='di')
    result.remove_column('idx_right')
    result.add_column(
            np.ma.array(result['Separation'], mask=result['Separation'] < 0),
            index=len(apt_left.colnames) + 2, name='offset_xy')
    result.remove_column('Separation')

    result.pprint(max_width=-1)

    xy_m_unit = result['x_m'].unit

    def update_col_meta(t):
        # update columns with description
        d = {
                'x_t': {
                    'description': 'The x position designed, in TolTEC frame.'
                    },
                'y_t': {
                    'description': 'The y position designed, in TolTEC frame.'
                    },
                'x_t_m': {
                    'description':
                    'The x position designed, aligned with measured frame.'
                    },
                'y_t_m': {
                    'description':
                    'The y position designed, aligned with measured frame.'
                    },
                'x_m': {
                    'description': 'The x position measured.'
                    },
                'snr_m': {
                    'description': 'The measured source SNR.'
                    },
                'fwhm_x_m': {
                    'description': 'The measured source x FWHM.'
                    },
                'fwhm_y_m': {
                    'description': 'The measured source y FWHM.'
                    },
                'y_m': {
                    'description': 'The y position measured.'
                    },
                'x_m_c': {
                    'description':
                    'The x position measured, corrected for distortion.',
                    'unit': xy_m_unit,
                    },
                'y_m_c': {
                    'description':
                    'The y position measured, corrected for distortion.',
                    'unit': xy_m_unit,
                    },
                'di': {
                    'description':
                    "Detector index from all network concatenated in order."
                    },
                'ti': {
                    'description':
                    "Detector tone index in its network."
                    },
                'offset_xy': {
                    'description': 'The positional offset when matching.',
                    'unit': xy_m_unit,
                    },
                'f_in': {
                    'description': 'The probing frequency.',
                    'unit': u.Hz,
                    },
                'f_out': {
                    'description':
                    'The measured resonance frequency.',
                    'unit': u.Hz,
                    },
                'f_flag': {
                    'description':
                    'The flag of the KIDs model fit.',
                    },
                'fg_hint': {
                    'description': "Externally tagged fg used for matching."
                    }
                }
        for c in t.colnames:
            if c in d:
                for a in ['description', 'meta', 'unit']:
                    if getattr(t[c], a) is None:
                        if a in d[c]:
                            setattr(t[c], a, d[c][a])
        return t

    update_col_meta(result)
    update_col_meta(result_unmatched)
    update_col_meta(apt_right_unmatched)
    return result, result_unmatched, apt_right_unmatched


def main(args):

    parser = argparse.ArgumentParser(
            description='Match two array property tables')
    parser.add_argument(
            '--left', '-l',
            metavar='APT',
            required=True,
            help='The left table which act as the base of match.'
            )
    parser.add_argument(
            '--right', '-r',
            metavar='APT',
            required=True,
            help='The matching table.'
            )
    parser.add_argument(
            "-f", "--overwrite",
            action='store_true',
            help="If set, overwrite the existing output file",
            )
    parser.add_argument(
            "--skip_init_match",
            action='store_true',
            help="If set, do not do match and use pickled init match",
            )
    parser.add_argument(
            "--save_plot",
            action='store_true',
            help="If set, save plots to output folder instead of showing",
            )
    parser.add_argument(
            "--init_match_dump",
            default=None,
            help="Use this specified init match dump file",
            )

    option = parser.parse_args(args)
    logger = get_logger()

    def check_outdir(p):
        if not option.overwrite and p.exists():
            raise RuntimeError(f"{p} exists, use -f to overwrite")
        if not p.exists():
            p.mkdir()
        return p

    def add_name_suffix(p, suffix):
        return p.parent / f'{p.stem}{suffix}{p.suffix}'

    jobkey = Path(option.right).stem
    outdir = check_outdir(Path(f'{jobkey}_matched'))
    logger.info(f"output to dir: {outdir}")
    matched_filepath = outdir / f'{jobkey}_matched.ecsv'
    unmatched_filepath_left = outdir / f'{jobkey}_unmatched_left.ecsv'
    unmatched_filepath_right = outdir / f'{jobkey}_unmatched_right.ecsv'
    manual_match_pickle_file = outdir / f'{jobkey}_match.pickle'
    fig_dist_cor_file = outdir / f'fig_{jobkey}_distortion_correction.png'
    fig_pos_cor_file = outdir / f'fig_{jobkey}_position_corrected.png'
    auto_match_pickle_file = outdir / f'{jobkey}_auto_match.pickle'
    fig_matched_hist_details = outdir / f'fig_{jobkey}_hist_matched.png'
    fig_matched_summary = outdir / f'fig_{jobkey}_matched_summary.png'

    # load input tables
    apt_left = load_apt(option.left, ['fg', ])
    logger.info(f"apt_left:\n{apt_left}")

    apt_right = load_apt(option.right, ['snr', 'f_in'])
    logger.info(f"apt_right:\n{apt_right}")

    # do the match
    # we need to define some heuristics
    match_outline_snr_cut = 1
    # compute initial offsets and make plots of data clouds

    xy_left = get_xy(apt_left, frame='toltec')
    xy_right = get_xy(apt_right, frame='array')

    # select good snr ones for matching outline
    m_snr_good = apt_right['snr'] >= match_outline_snr_cut
    xy_right_snr_good = xy_right[m_snr_good]
    outline_transform = match_outline(xy_left, xy_right_snr_good)

    # these are for the next frequency group setup
    xy_left_outline_matched = outline_transform(xy_left)
    xy_right_outline_matched = xy_right

    if 'fg' not in apt_right.colnames:
        raise NotImplementedError("fg needs to be created manually")
    else:
        pass

    # now we are ready to match each nw and each fg
    matched_tris = match_per_nw_fg(
            xy_left_outline_matched,
            apt_left,
            xy_right_outline_matched,
            apt_right,
            use_pickled=option.skip_init_match,
            use_pickle_file=option.init_match_dump,
            pickle_to_file=manual_match_pickle_file
            )
    wcs_info = cal_wcs(
            xy_left_outline_matched,
            apt_left,
            xy_right_outline_matched,
            apt_right,
            matched_tris,
            save_plot=fig_dist_cor_file if option.save_plot else None,
            )
    # using the computed wcs we can de-distort all the positions
    # in the right apt, and we can then match them in the outline matched
    # coords
    xy_right_outline_matched_corrected = correct_with_wcs(
            xy_right_outline_matched,
            apt_right,
            wcs_info,
            save_plot=fig_pos_cor_file if option.save_plot else None,
            )
    # now we can match the corrected xy with the left xy
    apt_matched, apt_unmatched_left, apt_unmatched_right = match_corrected_2d(
            xy_left_outline_matched,
            apt_left,
            xy_right_outline_matched_corrected,
            apt_right,
            pickle_to_file=auto_match_pickle_file,
            save_plot_per_group=fig_matched_hist_details
            if option.save_plot else None,
            save_plot_summary=fig_matched_summary
            if option.save_plot else None,
            )
    # attach some meta data
    import datetime
    from astropy.time import Time
    ut = Time(datetime.datetime.utcnow(), scale='utc')
    meta = {
            'generated_by': ' '.join([Path(sys.argv[0]).name] + sys.argv[1:]),
            'created_on': ut.isot,
            'jobkey': jobkey,
            'align_snr_cut': match_outline_snr_cut,
            'align_transform': xy_left_outline_matched.meta,
            'distortion_correction': {
                array_name: {
                    'wcs_fiducial': wd['w'].to_header(
                        relax=True).tostring(),
                    'wcs_fitted': wd['w1'].to_header(
                        relax=True).tostring(),
                    }
                for array_name, wd in wcs_info.items()
                }
            }
    apt_matched.meta['match_apt'] = meta

    logger.debug(f"save matched APT: {matched_filepath}")
    apt_matched.write(
            matched_filepath, format='ascii.ecsv',
            overwrite=option.overwrite)

    logger.debug(f"save unmatched APT: {unmatched_filepath_left}")
    apt_unmatched_left.write(
            unmatched_filepath_left, format='ascii.ecsv',
            overwrite=option.overwrite)
    logger.debug(f"save unmatched APT: {unmatched_filepath_right}")
    apt_unmatched_right.write(
            unmatched_filepath_right, format='ascii.ecsv',
            overwrite=option.overwrite)


if __name__ == "__main__":
    main(sys.argv[1:])
