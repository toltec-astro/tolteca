# type: ignore

from multiprocessing import Pool
from pathlib import Path

import astropy.units as u
import matplotlib
import numpy as np
from astropy.table import Column, Table, join, vstack
from scipy.signal import correlate
from tollan.utils.log import get_logger, init_log, timeit
from tollan.utils.wraps.stilts import stilts_match1d
from tolteca.common.toltec import toltec_info
from tolteca.datamodels.io.toltec.table import KidsModelParamsIO

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt


@timeit
def _match_d21(
        d21_0, d21_1, roi=None, max_shift=10 << u.MHz, return_locals=False):
    """Given two D21 spectrum, use correlation to find the relative
    frequency shift between them.

    Parameters
    ----------
    d21_0, d21_1: D21
        The D21s to match.
    roi: callable, optional
        If given, this will be used to filter the frequency as ``fs =
        roi(fs)``.
    return_locals: bool
        If set, the ``locals()`` is returned for further diagnostics.

    Returns
    -------
    float:
        The relative frequency shift.
    dict: optional
        The ``locals()`` dict, if `return_locals` is set.
    """
    fs0 = d21_0.frequency
    adiqs0 = d21_0.D21
    adiqscov0 = d21_0.d21_cov
    fs1 = d21_1.frequency
    adiqs1 = d21_1.D21
    adiqscov1 = d21_1.d21_cov
    if np.any((fs0 - fs1) != 0.):
        raise RuntimeError("find shift only works with same fs grid")
    fs = fs0
    if roi is not None:
        mask = roi(fs)
    else:
        mask = np.ones_like(fs, dtype=bool)
    fs = fs[mask]
    adiqs0 = adiqs0[mask]
    adiqscov0 = adiqscov0[mask]
    adiqs1 = adiqs1[mask]
    adiqscov1 = adiqscov1[mask]
    nfs = len(fs)
    cross_correlation = correlate(adiqs0, adiqs1, mode='same')
    dfs = np.arange(-nfs // 2, nfs // 2) * (fs[1] - fs[0])
    # print(dfs.shape)
    # the shift should always be less than max_shift
    m = np.abs(dfs) < max_shift
    cc = cross_correlation.copy()
    cc[~m] = -np.inf
    icc = cc.argmax()
    shift = -dfs[icc]
    shift_snr = (cc[icc] - np.median(cc[m])) / np.std(cc[m])
    if return_locals:
        return shift, locals()
    return shift


@timeit
def _match_tonelists(tl_0, tl_1, Qr_at_500MHz):
    """
    Create dummy D21 spectra for tone lists to derive the shift in freqs.
    """
    tl_0_Hz = tl_0.to_value(u.Hz)
    tl_1_Hz = tl_1.to_value(u.Hz)

    f_min = min(tl_0_Hz.min(), tl_1_Hz.min())
    f_max = min(tl_0_Hz.max(), tl_1_Hz.max())
    f_step = 500e6 / Qr_at_500MHz / 10
    f_pad = 3 * f_step

    f_bins = np.arange(f_min - f_pad, f_max + f_pad + f_step, f_step)

    fs0 = ((f_bins[:-1] + f_bins[1:]) * 0.5) << u.Hz

    class NS:
        pass

    d21_0 = NS()
    d21_1 = NS()

    d21_0.frequency = d21_1.frequency = fs0
    d21_0.d21_cov = d21_1.d21_cov = np.ones_like(fs0)
    d21_0.D21 = np.histogram(tl_0_Hz, bins=f_bins)[0].astype(float)
    d21_1.D21 = np.histogram(tl_1_Hz, bins=f_bins)[0].astype(float)

    return _match_d21(
        d21_0,
        d21_1,
        roi=None, return_locals=True)



def _get_apt_obsnum(apt):
    meta = apt.meta
    return meta.get('Header.Toltec.Obsnum', meta.get('obsnum', None))


def _get_kmp_files(data_rootpath, obsnum):
    files = []
    obsnum_str = f'{obsnum:06d}'
    for p in [
        # f'toltec/tcs/toltec*[0-9]/toltec*[0-9]_{obsnum_str}_*.nc',
        # f'toltec/ics/toltec*[0-9]/toltec*[0-9]_{obsnum_str}_*.nc',
        f'toltec/reduced/toltec*[0-9]_{obsnum_str}_*tune.txt',
    ]:
        files.extend(data_rootpath.glob(p))
    if not files:
        return None
    kmps = [KidsModelParamsIO(f).read() for f in files]
    # make an index table
    rows = []
    for kmp in kmps:
        rows.append({
            'interface': kmp.meta['interface'],
            'nw': kmp.meta['nwid'],
            'kmp': kmp,
        })
    index_table = Table(rows=rows)
    index_table.sort('nw')
    return index_table


def _kmp_to_apt(kmp):
    interface = kmp.meta['interface']
    nw = toltec_info[interface]['nw']
    array_name = toltec_info[interface]['array_name']
    array = toltec_info[toltec_info[interface]['array_name']]['index']
    tbl = kmp.table.copy()
    # prefix all columns with kids_model_
    for c in tbl.colnames:
        tbl.rename_column(c, f'kids_{c}')
    n_chans = len(tbl)
    tbl.add_column(Column(np.full((n_chans, ), nw), name='nw'), 0)
    tbl.add_column(Column(np.full((n_chans, ), array), name='array'), 0)
    tbl.add_column(Column(range(n_chans), name='kids_tone'), 0)
    return tbl


def _make_init_apt(kmp_index):
    apt = []
    for nw in sorted(kmp_index['nw']):
        kmp = kmp_index[kmp_index['nw'] == nw][0]['kmp']
        apt.append(_kmp_to_apt(kmp))
    apt = vstack(apt, metadata_conflicts='silent')
    apt.add_column(Column(range(len(apt)), name='det_id'), 0)
    for k in ['obsnum', 'subobsnum', 'scannum']:
        apt.meta[k] = kmp.meta[k]
    apt.meta['Header.Toltec.ObsNum'] = apt.meta['obsnum']
    apt.meta['Header.Toltec.SubObsNum'] = apt.meta['subobsnum']
    apt.meta['Header.Toltec.ScanNum'] = apt.meta['scannum']
    return apt


def _make_matched_apt_nw(apt_left, apt_right, Qr_at_500MHz=None, debug_plot_kw=None):
    dbk = debug_plot_kw or {}
    logger = get_logger()
    nw = apt_left['nw'][0]
    fr_left = apt_left['kids_fr'].quantity
    for c in ['kids_fr', 'tone_freq']:
        if c in apt_right.colnames:
            col_fr_right = c
            if apt_right[c].unit is None:
                apt_right[c].unit = u.Hz
            break
    else:
        raise ValueError("no fr found in apt_right.")
    fr_right = apt_right[col_fr_right].quantity
    if 'kids_Qr' in apt_right.colnames:
        Qr = apt_right['kids_Qr']
    elif 'kids_Qr' in apt_left.colnames:
        Qr = apt_left['kids_Qr']
    elif Qr_at_500MHz is not None:
        Qr = [Qr_at_500MHz]
    else:
        raise ValueError("no Qr info")
    Qr = np.median(Qr)
    logger.debug(f"match assume {Qr=} for {nw=}")
    shift, shift_data = _match_tonelists(fr_left, fr_right, Qr_at_500MHz=np.median(Qr))
    logger.debug(f"found {shift=} for {nw=}")
    # apply shift
    fr_left_shifted = fr_left + shift
    eps = 50 << u.kHz
    # here we match the good kids only first, then fill in the rest with bad
    good = apt_right['flag'] > 0
    id_matched = _match1d(
        x_left=fr_left_shifted.to_value(u.Hz),
        x_right=fr_right.to_value(u.Hz)[good],
        eps=eps.to_value(u.Hz),
        join_type='left',
        fill_id_value=-1
        )
    id_matched['is_good_match'] = False
    mgood = id_matched['idx_right'] >= 0
    id_matched['is_good_match'][mgood] = True
    id_matched['idx_right'][mgood] = np.where(good)[0][id_matched['idx_right'][mgood]]
    bad_matched = _match1d(
        x_left=fr_left_shifted.to_value(u.Hz)[~mgood],
        x_right=fr_right.to_value(u.Hz)[~good],
        eps=eps.to_value(u.Hz),
        join_type='inner',
    )
    mbad_idx = np.where(~mgood)[0]
    bad_idx = np.where(~good)[0]
    logger.debug(f"matched n_good={np.sum(mgood)} n_bad={len(bad_matched)}")
    for e in bad_matched:
        li = mbad_idx[e['idx_left']]
        ri = bad_idx[e['idx_right']]
        id_matched['idx_right'][id_matched['idx_left'] == li] = ri
        id_matched['x_right'][id_matched['idx_left'] == li] = fr_right.to_value(u.Hz)[ri]
    logger.debug(f"id_matched\n{id_matched}")
    # import pdb
    # pdb.set_trace()
    apt_matched = apt_left.copy()
    apt_matched['det_id_right'] = -1
    for e in id_matched:
        m = apt_matched['kids_tone'] == e['idx_left']
        assert np.sum(m) == 1
        if e['idx_right'] == -1:
            # no match
            apt_matched['det_id_right'][m] = -1
        else:
            mr = apt_right['kids_tone'] == e['idx_right']
            assert np.sum(mr) == 1
            apt_matched['det_id_right'][m] = apt_right['det_id'][mr]
    n_chans = len(apt_matched)
    n_matched = np.sum(apt_matched['det_id_right'] >= 0)
    n_missed = n_chans - n_matched
    logger.debug(f'matched {n_matched=} {n_missed=} total={n_chans}')

    obsnum_left = _get_apt_obsnum(apt_left)
    obsnum_right = _get_apt_obsnum(apt_right)
    nw = apt_left['nw'][0]
    if dbk.get('f_shift_enabled', False):
        fig, axes = plt.subplots(
                1, 2,
                figsize=(10, 4),
                sharex='col',
                constrained_layout=True,
                squeeze=False,
                )
        fig.suptitle(f'{obsnum_left=} {obsnum_right=} {nw=}')
        ax = axes.ravel()[0]
        fs = shift_data['fs']
        adiqs0 = shift_data['adiqs0']
        adiqs1 = shift_data['adiqs1']
        shift = shift_data['shift']
        # plot shift data
        ax.plot(
            fs, adiqs0, color='C0', linestyle='-',
            label='f_in'
            )
        ax.plot(
            fs, adiqs1, color='C1', linestyle='-',
            label='f_tone'
            )
        ax.plot(
                fs + shift,
                adiqs0,
                label=f'shift={shift.to_value(u.kHz):g} kHz',
                color='C3', linestyle='--',
                )
        ax.legend(loc='upper right')
        ax = axes.ravel()[1]
        ax.plot(
            shift_data['dfs'], shift_data['cross_correlation'],
            label=f'SNR={shift_data["shift_snr"]}'
            )
        max_shift = shift_data['max_shift']
        ax.axvline(
            -max_shift.to_value(u.Hz),
            color='gray', linestyle='--')
        ax.axvline(
            max_shift.to_value(u.Hz),
            color='gray', linestyle='--')
        ax.axvline(
            0,
            color='gray', linestyle=':')
        ax.axvline(
            -shift.to_value(u.Hz), 0.9, 1, color='cyan', linewidth=2)
        ax.set_xlim(
            max_shift.to_value(u.Hz) * -3, max_shift.to_value(u.Hz) * 3)
        ax.legend()
        plt.show()
    if dbk.get('match_enabled', False):
        n_segs = 10
        fig, axes = plt.subplots(
                n_segs, 1,
                figsize=(20, n_segs * 1),
                # sharex='col',
                constrained_layout=True,
                squeeze=False,
                )
        fig.suptitle(f'{obsnum_left=} {obsnum_right=} {nw=}')
        f_min = u.Quantity([fr_left.min(), fr_right.min()]).min()
        f_max = u.Quantity([fr_left.max(), fr_right.max()]).max()
        f_bins = np.linspace(f_min, f_max, n_segs + 1)
        f_pad = (200e3 << u.Hz) + shift
        # f_bin_centers = 0.5 * (f_bins[:-1] + f_bins[1:])
        for i, (f_bin_min, f_bin_max) in enumerate(zip(f_bins[:-1], f_bins[1:])):
            f_bin_min = f_bin_min - f_pad
            f_bin_max = f_bin_max + f_pad
            s_left = (fr_left >= f_bin_min) & (fr_left <= f_bin_max)
            s_right = (fr_right >= f_bin_min) & (fr_right <= f_bin_max)
            ax = axes[i, 0]
            ax.invert_yaxis()
            plot_sequence(ax, fr_left[s_left].to_value(u.MHz), y=0)
            plot_sequence(ax, fr_left_shifted[s_left].to_value(u.MHz), y=1)
            plot_sequence(ax, fr_right[s_right].to_value(u.MHz), y=2)
            mm = id_matched[s_left]
            for x0, x1 in zip(fr_left[s_left], fr_left_shifted[s_left]):
                ax.plot([x0.to_value(u.MHz), x1.to_value(u.MHz)], [0, 1], color='gray')
            for e in id_matched[s_left]:
                if e['is_good_match']:
                    c = 'green'
                else:
                    c = 'red'
                if e['idx_right'] >= 0:
                    ax.plot([e['x_left'] / 1e6, e['x_right'] / 1e6], [1, 2], color=c)
        plt.show()

    return apt_matched


def _match1d(x_left, x_right, eps, join_type='left', fill_id_value=None):
    id_col = 'idx'
    left = Table()
    left[id_col] = np.arange(len(x_left))
    left['x'] = x_left
    right = Table()
    right[id_col] = np.arange(len(x_right))
    right['x'] = x_right
    join = {
            'left': 'all1',
            'right': 'all2',
            'inner': '1and2',
            'outer': '1or2'
            }[join_type]
    result = stilts_match1d(left, right, 'x', eps, extra_args=[
        f'join={join}', 'find=best',
        'fixcols=all',
        f'suffix1=_left',
        f'suffix2=_right',
        ])
    if hasattr(result['idx_right'], 'mask') and fill_id_value is not None:
        result['idx_right'] = result['idx_right'].filled(fill_id_value)
    return result

def _match_apt_worker(arg):
    logger = get_logger()
    apt_left, apt_right, nw, debug_plot_kw = arg
    apt_left_nw = apt_left[apt_left['nw'] == nw]
    apt_right_nw = apt_right[apt_right['nw'] == nw]
    if len(apt_left_nw) == 0:
        logger.debug(f"no entry for apt_left {nw=}")
        return None
    if len(apt_right_nw) == 0:
        logger.debug(f'no entry for apt_right {nw=}')
        # make mock match
        mapt = apt_left_nw.copy()
        mapt['det_id_right'] = -1
        return mapt
    mapt = _make_matched_apt_nw(apt_left_nw, apt_right_nw, debug_plot_kw=debug_plot_kw)
    return mapt

def make_matched_apt(apt_left, apt_right, debug_plot_kw=None, n_procs=4):
    logger = get_logger()
    logger.debug(f"match apt: n_left={len(apt_left)} n_right={len(apt_right)}")

    nws = np.unique(apt_left['nw'])

    logger.debug(f"matching with {n_procs=}")
    with timeit("run matching"):
        with Pool(processes=n_procs) as pool:
            apt_matched = list(
                pool.imap(
                _match_apt_worker, [(apt_left, apt_right, nw, debug_plot_kw) for nw in nws]
                ))
        apt_matched = [a for a in apt_matched if a is not None]
    # apt_matched = []
    # for nw in nws:
    #     apt_left_nw = apt_left[apt_left['nw'] == nw]
    #     apt_right_nw = apt_right[apt_right['nw'] == nw]
    #     if len(apt_left_nw) == 0:
    #         logger.debug(f"no entry for apt_left {nw=}")
    #         continue
    #     if len(apt_right_nw) == 0:
    #         logger.debug(f'no entry for apt_right {nw=}')
    #         # make mock match
    #         mapt = apt_left_nw.copy()
    #         mapt['det_id_right'] = -1
    #         apt_matched.append(mapt)
    #         continue
    #     mapt = _make_matched_apt_nw(apt_left_nw, apt_right_nw, debug_plot_kw=debug_plot_kw)
    #     apt_matched.append(mapt)
    # import pdb
    # pdb.set_trace()
    apt_matched = vstack(apt_matched)
    logger.debug(f"matched apt:\n{apt_matched}")
    # do the join
    beammap_cols = [c for c in apt_right.colnames if not c.startswith('kids_')]
    apt_b = apt_right[beammap_cols]
    # resolve conflict metadata
    for k in list(apt_matched.meta.keys()):
        if k in apt_b.meta:
            apt_matched.meta[k + '_matched'] = apt_b.meta.pop(k)
    apt_matched = join(
        apt_matched, apt_b,
        join_type='left',
        keys_left="det_id_right",
        keys_right='det_id',
        uniq_col_name='{col_name}{table_name}',
        table_names=['', '_matched'],
        )
    apt_matched.sort('det_id')
    apt_matched.remove_column('det_id_matched')
    for c in beammap_cols:
        apt_matched[c].fill(0.)
    apt_matched['tone_freq'] = apt_matched['kids_f_out']
    logger.debug(f"joined apt:\n{apt_matched}")
    return apt_matched


def plot_sequence(ax, x, y=0, color_seq=None, size_seq=None, **kwargs):
    nx = len(x)
    if color_seq is None:
        color_seq = ['blue', 'red', 'green', 'black']
    if size_seq is None:
        size_seq = [5]
    nc = len(color_seq)
    colors = [color_seq[i % nc] for i in range(nx)]

    ns = len(size_seq)
    sizes = [size_seq[i % ns] for i in range(nx)]

    if np.isscalar(y):
        y = np.full((nx, ), y)
    ax.scatter(x, y, c=colors, s=sizes, **kwargs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_rootpath", '-r', default='data_lmt', type=Path, help='Data root path.')
    parser.add_argument("--apt_in_file", required=True, help='Input APT file to use')
    parser.add_argument("obsnum", type=int, help='obsnum to compose APT for.')
    parser.add_argument("--output_dir", default=Path(''), type=Path, help='output dir')
    parser.add_argument("--log_level", default='INFO', help='output dir')
    parser.add_argument(
        '--debug_plot', action='store_true'
        )
    option = parser.parse_args()

    if option.debug_plot:
        debug_plot_kw = {
            # 'f_shift_enabled': True,
            "match_enabled": True,
            }
    else:
        debug_plot_kw = None

    init_log(level=option.log_level)
    logger = get_logger()

    data_rootpath = option.data_rootpath

    apt_in = Table.read(option.apt_in_file, format='ascii')
    apt_in.add_column(Column(range(len(apt_in)), name='det_id'), 0)
    obsnum_in = _get_apt_obsnum(apt_in)
    obsnum = option.obsnum
    logger.debug(f"make apt for {obsnum=} from {obsnum_in=}\n{apt_in}")
    kmp_index = _get_kmp_files(data_rootpath, obsnum)
    if kmp_index is None:
        raise ValueError(f"unable to locate kmp files for {obsnum=}")
    logger.debug(f"kmps:\n{kmp_index}")
    apt_left = _make_init_apt(kmp_index)
    apt_right = apt_in
    apt_matched = make_matched_apt(apt_left, apt_right, debug_plot_kw=debug_plot_kw)

    # change the dtype of int columns for compatibility with citlali
    for c in apt_matched.colnames:
        if apt_matched[c].dtype == np.int64:
            apt_matched[c] = apt_matched[c].astype(float)

    logger.debug(f"apt_matched:\n{apt_matched}")
    apt_out_name = f'apt_{obsnum}_matched.ecsv'
    apt_out_filepath = option.output_dir.joinpath(apt_out_name)

    apt_matched.write(apt_out_filepath, format='ascii.ecsv', overwrite=True)
