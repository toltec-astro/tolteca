#!/usr/bin/env python
import sys
import numexpr as ne
from numpy.polynomial import Polynomial
from numba import njit
from astropy.table import QTable, Table, unique, Column, vstack
import dill
from astropy.stats import sigma_clipped_stats
from pathlib import Path
from tolteca.datamodels.io.toltec import NcFileIO
import numpy as np
from scipy.ndimage import median_filter
import scipy.signal
import astropy.units as u
import matplotlib.pyplot as plt
from tollan.utils.log import init_log, get_logger, timeit
from findpeaks import findpeaks
import lmfit
import tqdm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pyximport
pyximport.install(setup_args={"script_args" : ["--verbose"]})
from cpeakdetect import peakdetect


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
    from scipy.signal import correlate
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


def find_spike_S21(swp, spike_thresh=0.1):
    """Given a sweep data, find the spikes."""
    logger = get_logger()
    S21 = swp.S21.to_value(u.adu)
    # work in the log space
    y = np.log10(np.abs(S21))
    y_med = median_filter(y, size=(1, 5))
    y_range = np.max(y_med, axis=-1) - np.min(y_med, axis=-1)
    s_spike = (y - y_med) / y_range[:, np.newaxis]
    mask = np.abs(s_spike) > spike_thresh
    logger.debug(f'masked spike {mask.sum()}/{mask.size}')
    return locals()


@timeit
def despike(swp, debug_plot_kw=None, **kwargs):
    """Return sweep data that have spikes identified and interpolated away."""
    dbk = debug_plot_kw or {}
    kwargs.setdefault("spike_thresh", 0.05)
    ctx = find_spike_S21(swp, **kwargs)
    m = ~ctx['mask']
    fs_Hz = swp.frequency.to_value(u.Hz)
    S21_adu = swp.S21.to_value(u.adu).copy()
    # S21_interp = interp1d(fs_Hz[m], S21_adu[m])
    for di in range(fs_Hz.shape[0]):
        swp.S21[di] = np.interp(fs_Hz[di], fs_Hz[di][m[di]], S21_adu[di][m[di]]) << u.adu
    # attach the despike context for further use
    swp.despike = ctx

    if dbk.get("despike_enabled", False):
        nrows = dbk.get("nrows", 5)
        ncols = dbk.get("ncols", 5)
        di0 = dbk.get('di0', 0)
        fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(15, 15),
                constrained_layout=True,
                squeeze=False,
                )
        nw = swp.meta['nwid']
        obsnum = swp.meta['obsnum']
        fig.suptitle(f'{obsnum=} {nw=}')
        axes = axes.ravel()
        # create new axes on the right and on the top of the current axes.
        for i, ax in enumerate(axes):
            divider = make_axes_locatable(ax)
            ax_res = divider.append_axes("bottom", size="20%", pad="2%", sharex=ax)
            di = i + di0
            xx = swp.frequency[di].to_value(u.MHz)
            yy_orig = np.log10(np.abs(S21_adu[di]))
            yy = np.log10(np.abs(swp.S21.value[di]))
            ax.plot(xx, yy_orig, color='C0', label='orig')
            ax.plot(xx, ctx['y_med'][di], color='C1', label='S21_medfilt', linestyle='--')
            ax.plot(xx, yy, color='C2', label='despiked')
            # ax.plot(xx, yy - yy_orig, color='C3', label='diff')
            ax_res.plot(xx, ctx['s_spike'][di], color='C1')
            ax_res.axhline(ctx['spike_thresh'], color='gray', linestyle=':')
            tshade = np.full((len(xx), ), ctx['spike_thresh'])
            tshade[m[di]] = 0
            tshaden = np.full((len(xx), ), -ctx['spike_thresh'])
            tshaden[m[di]] = 0
            ax_res.fill_between(xx, tshade, color='black', alpha=0.8)
            ax_res.fill_between(xx, tshaden, color='black', alpha=0.8)
            ax_res.set_ylim(-ctx['spike_thresh'] * 1.1, ctx['spike_thresh'] * 1.1)
            ax_res.axhline(-ctx['spike_thresh'], color='gray', linestyle=':')
            ax.legend()
        plt.show()

    return swp


def _make_group_1d(x, dist):
    """return a list of list of indices that are groups of the x based on dist."""
    logger = get_logger()
    i_sorted = np.argsort(x)
    x_sorted = x[i_sorted]
    dx_sorted = np.diff(x_sorted)
    dist_sorted = dist[i_sorted]

    groups = [[0]]
    for j, xx in enumerate(dx_sorted):
        i0 = j
        i1  = j + 1
        cg = groups[-1]
        if np.abs(xx) < dist_sorted[i0]:
            cg.append(i1)
        else:
            groups.append([i1])
    igroups = [i_sorted[g] for g in groups]
    groupsizes = [len(g) for g in groups]
    unique, counts = np.unique(groupsizes, return_counts=True)
    logger.debug(f"mean group dist: {dist.mean()}")
    logger.debug(f"unique group sizes:\n{np.vstack([unique, counts])}")
    return locals()


@timeit
def find_peak_fp_fast(fs, y, Qrs, exclude_edge=5e3 << u.Hz, min_height=0.01):
    """Return a list of frequencies by looking for peaks in y.

    y has the same shape as fs.
    """
    logger = get_logger()
    # make a copy of y since we'll modify it
    # flip y to make it S21 peaks
    fstep = (fs[1] - fs[0])
    y = y.copy()
    # pad max value at side otherwise the function does not work for some
    # reason
    m = y.max()
    y[0] = m
    y[-1] = m
    fwhm = fs.mean() / np.mean(Qrs) / fstep
    exclude_edge_samples = exclude_edge / fstep
    if np.isnan(fwhm):
        import pdb
        pdb.set_trace()
    lookahead = int(fwhm * 0.5)
    min_lookahead, max_lookahead = 10, (len(y) // 5)
    if lookahead < min_lookahead:
        logger.debug(f"use {min_lookahead=}")
        lookahead = min_lookahead
    if lookahead > max_lookahead:
        logger.debug(f"use {max_lookahead=}")
        lookahead = max_lookahead
    logger.debug(f"Qrs={Qrs.tolist()} {lookahead=} {min_height=} {exclude_edge_samples=}")
    peaks, labx, peaks_good, peak_is_good = peakdetect(
        y,
        lookahead=lookahead,
        offset_for_height=lookahead,
        min_height=min_height,
        exclude_edge=exclude_edge_samples
        )
    if True:
    # if False:
        x = np.arange(len(y))
        for ll in np.unique(labx):
            plt.plot(x[labx==ll], y[labx==ll], marker='.')

        for i in range(len(peaks)):
            ip, yp, h, i0, i1, i2, i3 = peaks[i]
            if peak_is_good[i]:
                plt.plot(x[ip], y[ip], color='green', marker='o')
            else:
                plt.plot(x[ip], y[ip], color='red', marker='x')
            plt.plot([i0, i3], [y[i0], y[i3]], linestyle=':')
            plt.plot([i1, i2], [y[i1], y[i2]], linestyle='--', color='blue')
            ih = 0.5 * (i1 + i2)
            mh = 0.5 * (y[i1] + y[i2])
            plt.plot(ih, mh, marker='*')
            plt.text(x[ip], y[ip], f'{yp=:.2f}\n{h=:.2f}', ha='center', va='bottom')
            plt.text(ih, mh, f'{mh=:.2f}', ha='center', va='bottom')
        plt.show()

    if len(peaks_good) == 0:
        f_peaks = [] << u.Hz
        return locals()
    results_i = [p[0] for p in peaks_good]
    f_peaks = fs[results_i]
    return locals()


def _peakdetect_slow(y, lookahead, offset_for_height, min_height, exclude_edge, min_peak_pts=2):
    logger = get_logger()
    fp = findpeaks(method='peakdetect', lookahead=lookahead, interpolate=None, verbose=2)

    results = fp.fit(y)
    d = results['df']
    if np.all(np.isnan(d['labx'])):
        # no peak found
        f_peaks = [] << u.Hz
        return [], [], [], []
    lab = labx = d['labx'].astype(int)
    # get peak lables
    lab_values = np.unique(lab[d['peak']])
    heights = []
    peaks = []
    peak_is_good = []
    for ll in lab_values:
        # yy = y[lab == ll]
        # print(d['peak'][lab==ll])
        # print(d['valley'][lab==ll])
        ip = np.where(d['peak'] & (lab == ll))[0][0]
        ip0 = int(ip - offset_for_height)
        ip1 = int(ip + offset_for_height)
        il = np.where(lab == ll)[0]
        if ip0 < il[0]:
            ip0 = il[0]
        if ip1 > il[-1]:
            ip1 = il[-1]
        ymin = 0.5 * (y[ip0] + y[ip1])
        ymax = y[ip]
        yh = ymax - ymin
        heights.append(ymax - ymin)
        peaks.append((ip, y[ip], yh, il[0], ip0, ip1, il[-1]))
        # check minimum number of peak points above min_height + ymin
        yc = ymin + min_height
        n_peak_pts = (y[lab == ll] >= yc).sum()
        # exclude edge
        pts_good = n_peak_pts >= min_peak_pts
        pos_good = (ip > exclude_edge) or (ip < len(y) - exclude_edge)
        peak_is_good.append((yh >= min_height) and pos_good and pts_good)
    heights = np.array(heights)
    peak_is_good = np.array(peak_is_good)
    logger.debug(f"peak heights: {heights} {peak_is_good=}")
    # lab_good = heights >= min_height
    peaks_good = [p for i, p in enumerate(peaks) if peak_is_good[i]]

    return peaks, labx, peaks_good, peak_is_good
    # get mask to select the good peak
    # m = np.zeros_like(lab)
    # for i, ll in enumerate(lab_values):
    #     if lab_good[i]:
    #         m[lab == ll] = True
    # results_i = d['x'][d['peak'] & m]
    # f_peaks_all = fs[results_i]
    # # exclude edge
    # m = (f_peaks_all > fs.min() + exclude_edge) & (f_peaks_all < fs.max() - exclude_edge)
    # f_peaks = f_peaks_all[m]
    # import pdb
    # pdb.set_trace()


    # return peaks, labx, peaks_good, peak_is_good


@timeit
def find_peak_fp(fs, y, Qrs, exclude_edge=5e3 << u.Hz, min_height_value=None, min_height_n_sigma=None, plot=False):
    """Return a list of frequencies by looking for peaks in y.

    y has the same shape as fs.
    """
    peakdetect_func = _peakdetect_slow
    # peakdetect_func = peakdetect
    logger = get_logger()
    # make a copy of y since we'll modify it
    # flip y to make it S21 peaks
    fstep = (fs[1] - fs[0])
    fwhm = fs.mean() / np.median(Qrs) / fstep
    if np.isnan(fwhm):
        import pdb
        pdb.set_trace()
    lookahead = int(fwhm * 0.5)
    min_lookahead, max_lookahead = 10, (len(y) // 5)
    if lookahead < min_lookahead:
        logger.debug(f"use {min_lookahead=}")
        lookahead = min_lookahead
    if lookahead > max_lookahead:
        logger.debug(f"use {max_lookahead=}")
        lookahead = max_lookahead
    exclude_edge_samples = exclude_edge / fstep

    logger.debug(f"lookahead={lookahead} {exclude_edge_samples=}")
    y = y.copy()
    # calculate y stddev with a linear trend subtraced
    # ny =  len(y)
    # y_lin = np.interp(np.arange(ny), [0, ny-1], [y[0], y[-1]])
    # yy = y - y_lin
    # calculate y stddev with a slow baseline subtraced
    y_slow = median_filter(y, size=(lookahead * 2, ))
    yy = y - y_slow
    yy_mean0, yy_med0, yy_std0 = sigma_clipped_stats(yy, sigma=2)
    if min_height_n_sigma is not None:
        min_height = min_height_n_sigma * yy_std0
    if min_height_value is not None:
        min_height = max(min_height_value, min_height)
    # pad max value at side otherwise the function does not work for some
    # reason
    m = y.max()
    y[0] = m
    y[-1] = m

    # run a predetect to identify prominant peaks and mask out
    logger.debug(f"predetect with {min_height=} * 2 {min_height_n_sigma=}")
    peaks, labx, peaks_good, peak_is_good = peakdetect_func(
        y,
        lookahead=lookahead,
        offset_for_height=lookahead,
        min_height=min_height * 2,
        exclude_edge=exclude_edge_samples
        )

    pm = np.ones((len(y), ), dtype=bool)
    if len(peaks_good) > 0:
        for i in np.where(peak_is_good)[0]:
            p = peaks[i]
            ip, yp, h, i0, i1, i2, i3 = p
            pm[i1:i2 + 1] = False
        logger.debug(f"mask {len(peaks_good)} peaks. n_masked={(~pm).sum()}")
        yy_mean, yy_med, yy_std = sigma_clipped_stats(yy[pm], sigma=2)
        logger.debug(
            f"changed stats: yy_mean={yy_mean0} -> {yy_mean}\n"
            f"changed stats: yy_med={yy_med0} -> {yy_med}\n"
            f"changed stats: yy_std={yy_std0} -> {yy_std}"
            )
    else:
        logger.debug("no prominant peaks found")
        yy_mean = yy_mean0
        yy_med = yy_med0
        yy_std = yy_std0
    if min_height_n_sigma is not None:
        min_height = min_height_n_sigma * yy_std
    if min_height_value is not None:
        min_height = max(min_height_value, min_height)
    # now the actual detection with proper std
    # here we require the number of peak points to be half of the lookahead
    logger.debug(f"Qrs={Qrs.tolist()} {lookahead=} {min_height=} {min_height_n_sigma=} {exclude_edge_samples=}")
    peaks, labx, peaks_good, peak_is_good = peakdetect_func(
        y,
        lookahead=lookahead,
        offset_for_height=lookahead,
        min_height=min_height,
        min_peak_pts=lookahead // 2,
        exclude_edge=exclude_edge_samples
        )
    # print(f"select {lab_good.sum()} out of {lab_good.size} peaks")
    if plot:
    # if False:
    # if len(peaks_good) > 0 and any(p[2] < 2 * min_height for p in peaks_good):
        # fp.plot()
        fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
        x = np.arange(len(y))
        for ll in np.unique(labx):
            axes[0].plot(x[labx==ll], y[labx==ll], marker='.')
            axes[0].plot(x[labx==ll], y_slow[labx==ll], linestyle='--')
            axes[1].plot(x[labx==ll], yy[labx==ll], linestyle='-')
        axes[1].axhline(yy_mean, color='gray', linestyle='--')
        axes[1].axhline(yy_med, color='black', linestyle='-')
        axes[1].axhline(yy_med + yy_std, linestyle=':')
        axes[1].axhline(yy_med - yy_std, linestyle=':')
        axes[1].axhline(yy_med + yy_std * min_height_n_sigma, color='black', linestyle='--')

        for i in range(len(peaks)):
            ip, yp, h, i0, i1, i2, i3 = peaks[i]
            if peak_is_good[i]:
                axes[0].plot(x[ip], y[ip], color='green', marker='o')
            else:
                axes[0].plot(x[ip], y[ip], color='red', marker='x')
            axes[0].plot([i0, i3], [y[i0], y[i3]], linestyle=':')
            axes[0].plot([i1, i2], [y[i1], y[i2]], linestyle='--', color='blue')
            ih = 0.5 * (i1 + i2)
            mh = 0.5 * (y[i1] + y[i2])
            axes[0].plot(ih, mh, marker='*')
            axes[0].text(x[ip], y[ip], f'{yp=:.2f}\n{h=:.2f}', ha='center', va='bottom')
            axes[0].text(ih, mh, f'{mh=:.2f}', ha='center', va='bottom')

            mmh = 0.5 * (yy[i1] + yy[i2])
            yyc = mmh + min_height
            yyp = yy[ip]
            hh = yyp - mmh
            sig = hh / yy_std
            sig2 = (yyp - yyc) / yy_std
            nyyp = (yy[labx == labx[ip]] >= yyc).sum()
            axes[1].plot(ih, yyc, marker='x')
            axes[1].plot(ih, mmh, marker='*')
            axes[1].text(x[ip], yyp, f'{yyp=:.2f}\n{hh=:.2f}\n{sig=:.2f}\n{sig2=:.2f}\n{nyyp=}', ha='center', va='bottom')
            axes[1].text(ih, mmh, f'{mmh=:.2f}', ha='center', va='bottom')

        plt.show()

    # # get mask to select the good peak
    # m = np.zeros_like(lab)
    # for i, ll in enumerate(lab_values):
    #     if lab_good[i]:
    #         m[lab == ll] = True
    # results_i = d['x'][d['peak'] & m]
    # f_peaks_all = fs[results_i]
    # # exclude edge
    # m = (f_peaks_all > fs.min() + exclude_edge) & (f_peaks_all < fs.max() - exclude_edge)
    # f_peaks = f_peaks_all[m]
    # import pdb
    # pdb.set_trace()

    if len(peaks_good) == 0:
        f_peaks = [] << u.Hz
        return locals()
    results_i = [p[0] for p in peaks_good]
    f_peaks = fs[results_i]
    return locals()


@timeit
def find_peak_cwt(fs, y, Qrs):
    fs_Hz = fs.to_value(u.Hz)
    dfs = fs[1] - fs[0]
    cfs = 0.5 * (fs[-1] + fs[0])

    # bin Qrs to a grid of regular r
    Qrs = np.array(Qrs)
    r = 2 / Qrs
    r_bins = np.linspace(2 / 20000, 2 / 2000, 10)
    r_hist, _ = np.histogram(r, bins=r_bins)
    bin_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    Qrs_binned = []
    if np.sum(r < r_bins[0]) > 0:
        Qrs_binned.append(2 / r_bins[0])
    if np.sum(r > r_bins[-1]) > 0:
        Qrs_binned.append(2 / r_bins[-1])
    Qrs_binned.extend((2 / bin_centers[r_hist > 0]).tolist())
    Qrs_binned = np.array(Qrs_binned)
    # logger.info(f"bin {Qrs=} to {Qrs_binned}")
    widths = cfs / Qrs_binned / dfs
    peakind = scipy.signal.find_peaks_cwt(y, widths=widths, gap_thresh=widths.min())
    # print(peakind)
    if peakind.size == 0:
        f_peaks = [] << u.Hz
    else:
        f_peaks = fs[peakind] << u.Hz
    return locals()


@timeit
def _make_groups_by_channel(swp, fs_stats):
    """Return list of list of di that have the frequency overlap."""
    dist = fs_stats['range'].to_value(u.Hz)
    fc = fs_stats['center'].to_value(u.Hz)
    return _make_group_1d(fc, dist)


def _make_tone_list(swp, dis, Qrs, fp_method='fp', include_d21_peaks=True, d21_find_peak_kw=None, add_fs=None, min_dist_n_fwhm =0.5, find_peak_kw=None, min_group_size=1):
    """Make the list of tones to use for a set of tone indices

    Qrs : A list of Qrs to passes to the find peak algorithm
    add_fs : A list of positions to include to the detected peak list.
    min_dist_n_fwhm: number of fwhms to consider duplicates in final peak list.
    """
    d21_Qrs = Qrs * 1.5
    find_peak_kw = find_peak_kw or {}
    find_peak_kw.setdefault('Qrs', Qrs)
    d21_find_peak_kw = d21_find_peak_kw or {}
    d21_find_peak_kw.setdefault('Qrs', d21_Qrs)  # d21 peak are narrower
    logger = get_logger()
    # run find peak on sweep data
    if fp_method == 'cwt':
        def fp_func(fs, y, **kwargs):
            return find_peak_cwt(
                fs=fs,
                y=y,
                **kwargs
                )
    elif fp_method == 'fp':
        def fp_func(fs, y, **kwargs):
            return find_peak_fp(
                fs=fs,
                y=y,
                **kwargs
                )
    else:
        raise ValueError()
    check_dis = [
        # 466, 467
        # 642, 643
        # 618
        ]
    fp_ctxs = [fp_func(swp.frequency[di], -swp.despike['y_med'][di], plot=(di in check_dis), **find_peak_kw) for di in dis]
    if fp_method == 'cwt':
        fp_cwt_ctxs = fp_ctxs
    elif fp_method == 'fp':
        fp_fp_ctxs = fp_ctxs
    # d21 can find blended peaks. Here we interpolate
    subswp = swp[dis]
    subswp_fs_min = subswp.frequency.min()
    subswp_fs_max = subswp.frequency.max()
    subswp_fstep = subswp.frequency[0][1] - subswp.frequency[0][0]
    Qr_med = np.median(Qrs)
    d21_n_steps_total = (subswp_fs_max - subswp_fs_min) / subswp_fstep
    logger.debug(f"{min_dist_n_fwhm=} {subswp_fs_min=} {subswp_fs_max=} {Qr_med=} {subswp_fstep=} {d21_n_steps_total=}")
    d21_exclude_edge_samples = int(min_dist_n_fwhm * 0.5 * (subswp_fs_min + subswp_fs_max) / Qr_med / subswp_fstep)
    logger.debug(f"{d21_exclude_edge_samples=}")
    if d21_exclude_edge_samples > 0.1 * d21_n_steps_total:
        d21_exclude_edge_samples = int(0.1 * d21_n_steps_total)
        logger.debug(f"adjusted {d21_exclude_edge_samples=}")
    # d21_exclude_edge_samples = 5
    d21 = subswp.make_unified(
        flim=(subswp_fs_min, subswp_fs_max),
        resample=1,
        smooth=11,
        method='savgol',
        # some sensible guess of edge
        exclude_edge_samples=d21_exclude_edge_samples
        )
    # mask out no data region
    fp_d21_ctx = fp_func(
        d21.frequency[d21.d21_cov > 0], d21.D21.value[d21.d21_cov > 0], plot=(len(set(dis).intersection(check_dis)) > 0), **d21_find_peak_kw)

    # merge all found peaks

    f_tones = []
    if include_d21_peaks:
        f_tones = fp_d21_ctx['f_peaks'].to_value(u.Hz).tolist()
    if add_fs is not None:
        f_tones += add_fs.to_value(u.Hz).tolist()
    for ctx in fp_ctxs:
        f_tones += ctx['f_peaks'].to_value(u.Hz).tolist()
    f_tones = np.array(f_tones)
    if f_tones.size == 0:
        tone_list = [] << u.Hz
    else:
        # group all tones and unify them
        dist_min = f_tones / np.median(d21_Qrs) * min_dist_n_fwhm
        tone_list = _make_unique(f_tones, dist_min, min_group_size=min_group_size) << u.Hz
        logger.debug(f"found {len(tone_list)} {tone_list=} {dist_min=}")
    return locals()


def _make_unique(x, min_dist, func=np.mean, min_group_size=1):
    igroups = _make_group_1d(x, min_dist)['igroups']
    # take the mean of each group as the final f_tones
    result = [np.mean(x[ig]) for ig in igroups if len(ig) >= min_group_size]
    return result


def _get_plot_grid(dbk, i, n):
    nrows = dbk.get("nrows", 5)
    ncols = dbk.get("ncols", 5)
    if nrows * ncols >= n - i:
        npanels = n - i
        if npanels < 0:
            return 1, 1
        nrows, ncols = int(np.sqrt(npanels)) + 1, int(np.sqrt(npanels))
        if nrows * ncols - npanels >= ncols:
            nrows -= 1
    return nrows, ncols


@timeit
def make_tone_list(ref_data, sweep_data, debug_plot_kw=None):
    logger = get_logger()
    dbk = debug_plot_kw or {}

    swp_ref = ref_data['swp']
    # swp = sweep_data.read(tone=slice(0, 25))
    swp = sweep_data.read()
    # swp = sweep_data.read(block=0)
    swp = despike(swp, debug_plot_kw=dbk)
    # extract d21 spectrum
    # build per channel D21 data on orignal grid
    swp._D21 = swp._validate_D21(swp.diqs_df(
        swp.S21,
        swp.frequency,
        # smooth=ref_data['d21_kw'].get("smooth", None),
        smooth=15,
        method='savgol',
        ))
    # build d21 spectrum using tune data
    swp.make_unified(**ref_data['d21_kw'])
    # extract dark detector mask
    # 1 = dark detector; 0 = good
    d21_mask = ref_data['d21_mask']
    # set d21 in swp at dark_det to be good
    swp_ref.unified._D21.data[~d21_mask] = 0
    swp.unified._D21.data[~d21_mask] = 0
    # find global shift between ref and swp
    left = swp_ref
    right = swp
    shift, shift_data = _match_d21(
        left.unified,
        right.unified,
        roi=None, return_locals=True)
    nw = swp.meta['nwid']
    obsnum = swp.meta['obsnum']
    ref_obsnum = swp_ref.meta['obsnum']
    if dbk.get("d21shift_enabled", False):
        fig, axes = plt.subplots(
                1, 2,
                figsize=(10, 4),
                sharex='col',
                constrained_layout=True,
                squeeze=False,
                )
        fig.suptitle(f'{obsnum=} {ref_obsnum=} {nw=}')
        ax = axes.ravel()[0]
        fs = shift_data['fs']
        adiqs0 = shift_data['adiqs0']
        adiqs1 = shift_data['adiqs1']
        shift = shift_data['shift']
        # plot shift data
        ax.plot(
            fs, adiqs0, color='C0', linestyle='-',
            label=left.meta['obsnum']
            )
        ax.plot(
            fs, adiqs1, color='C1', linestyle='-',
            label=right.meta['obsnum']
            )
        ax.plot(
                fs + shift,
                adiqs0,
                label=f'shift={shift.to_value(u.kHz):g} kHz',
                color='C3', linestyle='--',
                )
        ax.legend(loc='upper right')
        ax = axes[1]
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

    # apply the shift to refdata output and generate f_ref
    # note that the ref table does not have the same order as the
    # the channels in the swp data.
    kmt = ref_data['kmp'].table
    kmt['di_ref'] = range(len(kmt))
    kmt['fr_ref'] = kmt['fr'].quantity + shift

    # generate some statistics of the frquency grid
    fs = swp.frequency
    n_tones, n_sweepsteps = fs.shape
    fs_min = fs.min(axis=1)
    fs_max = fs.max(axis=1)
    fs_center = fs.mean(axis=1)
    fs_range = fs_max - fs_min
    # import pdb
    # pdb.set_trace()
    fs_stats = {
        'n_tones': n_tones,
        'n_sweepsteps': n_sweepsteps,
        'min': fs_min,
        'max': fs_max,
        'center': fs_center,
        'tone_axis_data': swp.meta['tone_axis_data'],
        'sweep_axis_data': swp.meta['sweep_axis_data'],
        'range': fs_range,
        'df': fs[1] - fs[0],
        }

    # build connection between refdata kmt and sweep data
    n_tones_ref = len(kmt)
    fr_ref = kmt['fr_ref'].quantity
    m_inc= (fs_min[:, None] <= fr_ref[None, :]) & (fs_max[:, None] >= fr_ref[None, :])
    # this is a list of indices in kmt that fall into each swp tone.
    di_ref_per_di = [np.where(m_inc[di])[0] for di in range(n_tones)]
    # this is a list of indices in swp tone that contains given kmt
    di_per_di_ref = [np.where(m_inc[:, di_ref])[0] for di_ref in range(n_tones_ref)]

    # here we do the per-channel detection of peaks
    # the detection are done on a per-channelgroup basis, a group is defined
    # as the channels that share some frequency range
    channel_groups = _make_groups_by_channel(swp, fs_stats)
    di_groups = channel_groups['igroups']

    # for each channel group, build the tone list by inspecting the data
    tone_list_groups = []
    min_dist_n_fwhm = 0.5
    if swp.meta['data_kind'].name == 'VnaSweep':
        find_peak_kw = {
            'min_height_value': 0.1 / 20,  # 0.1 db half peak depth
            'min_height_n_sigma': 20
            }
        d21_find_peak_kw = {
            'min_height_n_sigma': 10,
            'min_height_value': 2.
            }
        min_group_size = 1
    else:
        find_peak_kw = {
            'min_height_value': 0.1 / 20,  # 0.2 db half peak depth
            'min_height_n_sigma': 20
            }
        d21_find_peak_kw = {
            'min_height_n_sigma': 10,
            'min_height_value': 2.
            }
        min_group_size = 1
    for dis in di_groups:
        # get the list of di_ref in kmt to extract or Qr
        di_refs = []
        for di in dis:
            di_refs.extend(di_ref_per_di[di])
        di_refs = list(set(di_refs))
        kmt_ref = kmt[di_refs]
        Qrs = kmt_ref['Qr']
        if len(kmt_ref) == 0:
            Qrs = np.array([np.median(kmt['Qr'])])
        # add_fs = kmt_ref['fr_ref'].quantity
        ctx = _make_tone_list(
            swp, dis, Qrs,
            fp_method='fp',
            add_fs=None,
            min_dist_n_fwhm=min_dist_n_fwhm,
            find_peak_kw=find_peak_kw,
            d21_find_peak_kw=d21_find_peak_kw,
            min_group_size=min_group_size
            )
        ctx['di_refs'] = di_refs
        ctx['kmt'] = kmt_ref
        tone_list_groups.append(ctx)
    if dbk.get("channelgroups_enabled", False):
        fig, ax = plt.subplots(
                1, 1,
                figsize=(10, 4),
                constrained_layout=True,
                # squeeze=False,
                )
        fig.suptitle(f'{obsnum=} {ref_obsnum=} {nw=}')
        fs = shift_data['fs']
        adiqs1 = shift_data['adiqs1']
        ax.plot(
            fs.to_value(u.MHz), adiqs1, color='C1', linestyle='-',
            label=right.meta['obsnum']
            )

        for tone_list_ctx in tone_list_groups:
            dis = tone_list_ctx['dis']
            color=f'C{len(dis) - 1}'
            for ctx in tone_list_ctx['fp_ctxs']:
                for f in ctx['f_peaks'].to_value(u.MHz):
                    ax.axvline(
                        f,
                        color='gray', linestyle=':',
                    )
            for f in tone_list_ctx['tone_list'].to_value(u.MHz):
                ax.axvline(
                    f,
                    color=color, linestyle='--',
                )
        ax.set_xlabel(f"{f} (MHz)")
        plt.show()

    # further process the tone list to merge them with duplicates rejected
    # this is the final tone freqs used for fitting.
    f_tones_fit_Hz = []
    for tone_list_ctx in tone_list_groups:
        f_tones_fit_Hz += tone_list_ctx['tone_list'].to_value(u.Hz).tolist()
    # just to make sure, we uniquefy this list by min_dist_n_fwhm
    min_dist_n_fwhm_uniquefy = min_dist_n_fwhm / 1.5  # to account for the closeby d21 tones
    f_tones_fit_Hz = np.array(f_tones_fit_Hz)
    min_dist = f_tones_fit_Hz / np.median(kmt['Qr']) * min_dist_n_fwhm_uniquefy
    f_tones_fit = _make_unique(f_tones_fit_Hz, min_dist=min_dist, min_group_size=1) << u.Hz
    n_tones_fit = len(f_tones_fit)
    logger.debug(f"final list of tones for fit: n_tones_fit={n_tones_fit}")

    # generate some properties of this list
    m_inc_fit = (fs_min[:, None] <= f_tones_fit[None, :]) & (fs_max[:, None] >= f_tones_fit[None, :])
    # this is a list of indices in fit tones that fall into each swp tone.
    di_fit_per_di = [np.where(m_inc_fit[di])[0] for di in range(n_tones)]
    # this is a list of indices in swp tone that contains given fit tone id
    di_per_di_fit = [np.where(m_inc_fit[:, di_fit])[0] for di_fit in range(n_tones_fit)]


    if dbk.get("tonelist_enabled", False):
        gi0 = dbk.get('gi0', 0)
        nrows, ncols = _get_plot_grid(dbk, gi0, len(tone_list_groups))
        fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(3 * ncols, 3 * nrows),
                constrained_layout=True,
                squeeze=False,
                )
        fig.suptitle(f'{obsnum=} {ref_obsnum=} {nw=}')
        axes = axes.ravel()
        for i, ax in enumerate(axes):
            divider = make_axes_locatable(ax)
            ax_d21 = divider.append_axes("bottom", size="30%", pad="2%", sharex=ax)
            fig.tight_layout()
            gi = i + gi0
            if gi >= len(tone_list_groups):
                continue
            tone_list_ctx = tone_list_groups[gi]
            tone_list = tone_list_ctx['tone_list']
            di_refs = tone_list_ctx['di_refs']
            dis = tone_list_ctx['dis'].tolist()
            if len(dis) > 10:
                info = f"{gi=}\nn_di_refs={len(di_refs)}\nn_dis={len(dis)}\nn_tones={len(tone_list)}"
            else:
                info = f"{gi=}\n{di_refs=}\n{dis=}\nn_tones={len(tone_list)}"
            ax.text(0, 0, info, transform=ax.transAxes, va='bottom', ha='left')
            legs = {}
            for j, di in enumerate(dis):
                xx = swp.frequency[di].to_value(u.MHz)
                yy = np.log10(np.abs(swp.S21.value[di]))
                ax.plot(xx, yy, color='C0', label='S21')
                for k, fp_k in enumerate(['fp_fp', 'fp_cwt']):
                    fp_ctx_k = f'{fp_k}_ctxs'
                    if fp_ctx_k not in tone_list_ctx:
                        continue
                    fp_ctx = tone_list_ctx[fp_ctx_k][j]
                    color = f'C{k+1}'
                    linestyle = ['-', '--', '-.'][k]
                    for f in fp_ctx['f_peaks']:
                        handle = ax.axvline(f.to_value(u.MHz), color=color, linestyle=linestyle)
                        if fp_k not in legs:
                            legs[fp_k] = handle
            for j, di_ref in enumerate(di_refs):
                handle = ax.axvline(kmt['fr_ref'].quantity[di_ref].to_value(u.MHz), color='black', linestyle='--')
                if 'fr_ref' not in legs:
                    legs['fr_ref'] = handle
            fp_d21 = tone_list_ctx['d21']
            ax_d21.plot(fp_d21.frequency.to_value(u.MHz), fp_d21.D21.value, color='C0', label='D21')
            for f in tone_list_ctx['fp_d21_ctx']['f_peaks']:
                handle = ax.axvline(f.to_value(u.MHz), color='magenta', linestyle='-.')
                ax_d21.axvline(f.to_value(u.MHz), color='magenta', linestyle='-.')
                if 'f_d21' not in legs:
                    legs['f_d21'] = handle
            for f in tone_list_ctx['tone_list']:
                handle = ax.axvline(f.to_value(u.MHz), color='cyan', linestyle='--')
                if 'f_guess' not in legs:
                    legs['f_guess'] = handle
            di_fits = []
            for di in dis:
                di_fits += di_fit_per_di[di].tolist()
            for f in f_tones_fit[di_fits]:
                handle = ax.axvline(f.to_value(u.MHz), color='red', linestyle=':')
                if 'f_fit' not in legs:
                    legs['f_fit'] = handle
            ax.set_xlabel("f (MHz)")
            if i == 0:
                ax.legend(labels=legs.keys(), handles=legs.values(), loc='upper right')
        if dbk.get("tonelist_save", None) is not None:
            fig.set_size_inches(3 * ncols, 3 * nrows)
            fig.savefig(dbk['tonelist_save'])
        else:
            fig.tight_layout()
            plt.show()

    # now re-group the f_tones_fit for model fits
    model_dist_n_fwhm = 3
    f_tones_fit_Hz = f_tones_fit.to_value(u.Hz)
    model_group_dist =  f_tones_fit_Hz / np.median(kmt['Qr']) * model_dist_n_fwhm
    di_fit_groups = _make_group_1d(f_tones_fit_Hz, model_group_dist)['igroups']
    model_group_ctxs = []
    for di_fits in di_fit_groups:
        # generate general infomation for model fitting
        # figure out the list of di that cover f_fit
        dis = []
        for di_fit in di_fits:
            dis += di_per_di_fit[di_fit].tolist()
        dis = list(set(dis))
        di_refs = []
        for di in dis:
            di_refs += di_ref_per_di[di].tolist()
        di_refs = list(set(di_refs))
        model_group_ctxs.append({
            'di_fits': di_fits.tolist(),
            'dis': dis,
            'di_refs': di_refs,
            'f_fit': f_tones_fit[di_fits],
            'kmt': kmt[di_refs],
            })
    if dbk.get("modelgroups_enabled", False):
        gi0 = dbk.get('gi0', 0)
        nrows, ncols = _get_plot_grid(dbk, gi0, len(di_fit_groups))
        fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(15, 15),
                constrained_layout=True,
                squeeze=False,
                )
        fig.suptitle(f'{obsnum=} {ref_obsnum=} {nw=}')
        axes = axes.ravel()
        for i, ax in enumerate(axes):
            # divider = make_axes_locatable(ax)
            # ax_res = divider.append_axes("bottom", size="20%", pad="2%", sharex=ax)
            gi = i + gi0
            model_group_ctx = model_group_ctxs[gi]
            di_refs = model_group_ctx['di_refs']
            di_fits = model_group_ctx['di_fits']
            dis = model_group_ctx['dis']
            if len(dis) > 10:
                info = f"{gi=}\nn_di_refs={len(di_refs)}\nn_dis={len(dis)}\nn_di_fits={len(di_fits)}"
            else:
                info = f"{gi=}\n{di_refs=}\n{dis=}\n{di_fits=}"
            ax.text(0, 1, info, transform=ax.transAxes, va='top', ha='left')
            legs = {}
            for j, di in enumerate(dis):
                xx = swp.frequency[di].to_value(u.MHz)
                yy = np.log10(np.abs(swp.S21.value[di]))
                ax.plot(xx, yy, color='C0', label='S21')
            for j, di_ref in enumerate(di_refs):
                handle = ax.axvline(kmt['fr_ref'].quantity[di_ref].to_value(u.MHz), color='black', linestyle='--')
                if 'fr_ref' not in legs:
                    legs['fr_ref'] = handle
            for f in model_group_ctx['f_fit']:
                handle = ax.axvline(f.to_value(u.MHz), color='cyan', linestyle='--')
                if 'f_fit' not in legs:
                    legs['f_fit'] = handle
            ax.set_xlabel("f (MHz)")
            ax.legend(labels=legs.keys(), handles=legs.values(), loc='upper right')
        plt.show()

    return locals()


def _ezclump(mask):
    """
    Finds the clumps (groups of data with the same values) for a 1D bool array.
    Returns a series of slices.
    """
    if mask.ndim > 1:
        mask = mask.ravel()
    idx = (mask[1:] ^ mask[:-1]).nonzero()
    idx = idx[0] + 1

    if mask[0]:
        if len(idx) == 0:
            return [slice(0, mask.size)]

        r = [slice(0, idx[0])]
        r.extend((slice(left, right)
                  for left, right in zip(idx[1:-1:2], idx[2::2])))
    else:
        if len(idx) == 0:
            return []

        r = [slice(left, right) for left, right in zip(idx[:-1:2], idx[1::2])]

    if mask[-1]:
        r.append(slice(idx[-1], mask.size))
    return r


@timeit
def _match_tonelists(tl_0, tl_1, Qr_at_500MHz):
    """
    Create dummy D21 spectra for tonelists to derive the shift in freqs.
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


def export_tone_list(tone_list_ctx, debug_plot_kw=None, vary_n_tones=True):
    """Dump table of all info for next step processing.

    """
    dbk = debug_plot_kw or {}
    logger = get_logger()
    model_group_ctxs = tone_list_ctx['model_group_ctxs']
    fs_stats = tone_list_ctx['fs_stats']

    tlt = []
    mi = 0
    for gi, model_group_ctx in enumerate(model_group_ctxs):
        fit_ctx = _prep_fit(tone_list_ctx['swp'], model_group_ctx)
        dis = fit_ctx['dis']
        di_fits = fit_ctx['di_fits']
        params = fit_ctx['params']
        for i, di in enumerate(dis):
            d = fit_ctx['dataset'][i]
            tone_indices = d['tone_indices']
            tone_ctxs = d['tone_ctxs']
            for j, tone_ctx in zip(tone_indices, tone_ctxs):
                ss = _ezclump(tone_ctx['datamask'])
                assert len(ss) == 1
                entry = {
                    "model_id": mi,
                    "group_id": gi,
                    'chan_id': di,
                    'tone_id': di_fits[j],
                    }
                for k in ['model_chan_id', 'model_tone_id',
                          'f_fit', 'f_fit_min', 'f_fit_max',
                          'fr_init', 'fr_init_min', 'fr_init_max',
                          'Qr_init', 'm0_init', 'm1_init',
                          'f_start', 'f_stop',
                          "index_start", "index_stop"]:
                    entry[k] = tone_ctx[k]
                # attach the fs_stats
                entry.update({
                    'chan_f_min': fs_stats['min'][di],
                    'chan_f_max': fs_stats['max'][di],
                    'chan_f_center': fs_stats['center'][di],
                    })
                tlt.append(entry)
                mi += 1
    tlt = QTable(rows=tlt)
    logger.info(f'output tlt:\n{tlt}')

    # Derived based on tlt.

    # generate subtable of unique tone id.
    # this is used for updating the targ_freqs list
    targ = unique(tlt, keys='tone_id', keep='first')
    targ_out = Table()

    # note this file is ideally to be generated by the
    # kmt.ecsv after the fitting. The file generated here
    # will have the same f_out and f_in.
    # this suppose to be the fitted fr. We use fr_init when fitting is not done.
    targ_out['f_out'] = targ['fr_init']
    # this suppose to be the input fit position
    targ_out['f_in'] = targ['f_fit']

    # add some more info
    targ_out['tone_id'] = targ['tone_id']
    targ_out['group_id'] = targ['group_id']
    # aggregates
    tlt_by_tone_id = Table(tlt).group_by("tone_id")
    targ_out['model_id_start'] = tlt_by_tone_id['model_id'].groups.aggregate(np.min)
    targ_out['model_id_stop'] = tlt_by_tone_id['model_id'].groups.aggregate(np.max) + 1
    targ_out['chan_id_start'] = tlt_by_tone_id['chan_id'].groups.aggregate(np.min)
    targ_out['chan_id_stop'] = tlt_by_tone_id['chan_id'].groups.aggregate(np.max) + 1
    targ_out['chan_f_min'] = tlt_by_tone_id['chan_f_min'].groups.aggregate(np.min).quantity
    targ_out['chan_f_max'] = tlt_by_tone_id['chan_f_max'].groups.aggregate(np.max).quantity
    targ_out['main_chan_id'] = -1
    targ_out['main_chan_f_center'] = np.nan
    # For each tone, find the chan that has the tone closest to the center
    for tg in tlt_by_tone_id.groups:
        ic = np.argmin(np.abs(tg['chan_f_center'].quantity - tg['fr_init'].quantity))
        mg = (targ_out['tone_id'] == tg['tone_id'][ic])
        targ_out['main_chan_id'][mg] = tg['chan_id'][ic]
        targ_out['main_chan_f_center'][mg] = tg['chan_f_center'].quantity[ic]
    # import pdb
    # pdb.set_trace()

    # propagate the ampcor values from the header
    sweep_data = tone_list_ctx['sweep_data']
    swp = tone_list_ctx['swp']
    chan_tone_amps = sweep_data.nc_node.variables['Header.Toltec.ToneAmps'][:]
    # logger.info(f"{chan_tone_amps=}")
    # print(chan_tone_amps)
    # TODO check interpolate on main_chan_id_f_center works better
    targ_out['ampcor'] = chan_tone_amps[targ_out['main_chan_id']]

    # import pdb
    # pdb.set_trace()
    # plt.plot(tone_amps)
    # plt.show()

    # generate per chan_id table for checking shift in fr
    # note that there might be channels that don't enter the tlt.
    # this is only applicable to TUNEs

    # to do this we need to run proper matching between the sweep center (f_in)
    # and the found tone list (f_out)

    # for TUNE, there might be a global shift that needs to be considered
    # when doing the match.
    if swp.meta['data_kind'].name =='VnaSweep':
        chk_out = QTable()
        chk_out['f_out'] = targ['fr_init']
        chk_out['f_in'] = targ['f_fit']
        chk_out['n_tones'] = 1

        # sort in fft order as required by the ROACH system
        swp = tone_list_ctx['swp']
        lofreq = swp.meta['flo_center']
        logger.info(f"subtract lo_freq = {lofreq}")
        dfs = targ_out['f_out'] - lofreq

        targ_out.add_column(Column(dfs, name='f_centered'), 0)
        targ_out.meta['Header.Toltec.LoCenterFreq'] = lofreq
        tones = targ_out['f_centered']
        max_ = 3 * np.max(tones)  # this is to shift the native tones to positive side
        isort = sorted(
            range(len(tones)),
            key=lambda i: tones[i] + max_ if tones[i] < 0 else tones[i]
            )
        targ_out = targ_out[isort]
        logger.info(f'output targ:\n{targ_out}')

        # plt.plot(targ_out['f_centered'])
        # plt.show()

        # expand this to a dummy tune.txt file for compatibility
        # TODO this will be phased out when we update tlaloc.
        compat_targ_freqs_dat = targ_out[['f_centered', 'f_out', 'f_in']]
        for p in ['flag', 'fp', 'Qr', 'Qc', 'fr', 'A', 'normI', 'normQ', 'slopeI', 'slopeQ', 'interceptI', 'interceptQ']:
            compat_targ_freqs_dat[p] = 0.
        logger.debug(f'output compat_targ_freqs.dat:\n{compat_targ_freqs_dat}')

        # also for the ampcor file:
        compat_targ_amps_dat = targ_out[['ampcor']]
        logger.debug(f'output compat_targ_amps.dat:\n{compat_targ_amps_dat}')
        return locals()
    # TUNE or targsweep
    f_in = fs_stats['center']
    f_out = targ_out['f_out'].quantity
    Qr_at_500MHz = np.median(targ['Qr_init'])
    shift, shift_data = _match_tonelists(f_in, f_out, Qr_at_500MHz=Qr_at_500MHz)

    if dbk.get("f_shift_enabled", False):
        nw = swp.meta['nwid']
        obsnum = swp.meta['obsnum']
        fig, axes = plt.subplots(
                1, 2,
                figsize=(10, 4),
                sharex='col',
                constrained_layout=True,
                squeeze=False,
                )
        fig.suptitle(f'{obsnum=} {nw=}')
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

    # apply the shift to the f_center before doing the match
    logger.info(f"found global fr shift {shift}")
    f_in_shifted = f_in + shift

    chk = []
    assert swp.frequency.shape[0] == fs_stats['n_tones']
    # we use a cursor to maintain the relative ordering of matched tones
    # tone_cursor = 0
    for chan_id in range(swp.frequency.shape[0]):
        mg = (tlt['chan_id'] == chan_id)
        # this is the sub table contains all tones in this channel
        tg = tlt[mg]

        entry = {
            'f_center': fs_stats['center'][chan_id],
            'chan_id': chan_id,
            'n_tones': len(tg)
            }

        if len(tg) == 0:
            # nothing to do, no data
            entry['group_id'] = -1
            entry['main_model_id'] = -1
            entry['main_tone_id'] = -1
            entry['fr_init'] = np.nan << u.Hz
            logger.debug(f"skip {chan_id=}: no tones")
            chk.append(entry)
            continue

        entry['group_id'] = tg['group_id'][0]
        # identify the closest tone in the sub table
        # we sort it by distance, and check for the main channel tag
        tg['_f_dist'] = np.abs(tg['chan_f_center'] - tg['fr_init'])
        tg.sort('_f_dist')
        tg['_is_main'] = False
        # check if any of those has this channel as main channel
        for i, e in enumerate(tg):
            s = targ_out[targ_out['tone_id'] == e['tone_id']]
            assert len(s) == 1
            tg['_is_main'][i] = (s[0]['main_chan_id'] == chan_id)
        # accept the closest main entry, that have tone_id larger than cursor
        main_tg = tg[tg['_is_main']]
        entry['n_tones_main'] = len(main_tg)
        # ctg = tg[tg['tone_id'] >= tone_cursor]
        # entry['n_tones_forward'] = len(ctg)
        # entry['n_tones_backward'] = len(tg) - len(ctg)
        if len(main_tg) == 0:
            entry['main_model_id'] = -1
            entry['main_tone_id'] = -1
            entry['fr_init'] = np.nan << u.Hz
            # no cursor update
            # new_cursor = tone_cursor
        else:
            # get the closest entry that have tone_id no less than the cursor
            # cmain_tg = main_tg[main_tg['tone_id'] >= tone_cursor]
            # entry['n_tones_main_forward'] = len(cmain_tg)
            # entry['n_tones_main_backward'] = len(main_tg) - len(cmain_tg)
            # if len(cmain_tg) > 0:
            #     # take the first entry
            #     entry['main_model_id'] = cmain_tg['model_id'][0]
            #     entry['main_tone_id'] = cmain_tg['tone_id'][0]
            #     entry['fr_init'] = cmain_tg['fr_init'][0]
            #     new_cursor = cmain_tg['tone_id'][0] + 1
            # else:
            if True:
                # take the first entry regardless of the cursor
                entry['main_model_id'] = main_tg['model_id'][0]
                entry['main_tone_id'] = main_tg['tone_id'][0]
                entry['fr_init'] = main_tg['fr_init'][0]
                # new_curosr = main_tg['tone_id'][0] + 1
        # tone_cursor = new_cursor
        chk.append(entry)
        # logger.debug(
        #     f"{chan_id=} n_tones={entry['n_tones']} n_tones_main={entry['n_tones_main']} cursor={tone_cursor}->{new_cursor}")
        logger.debug(
            f"{chan_id=} n_tones={entry['n_tones']} n_tones_main={entry['n_tones_main']}")

    chk = QTable(rows=chk)
    logger.debug(f"chk:\n{chk}")
    # build chk out
    chk_out = Table()
    chk_out['f_out'] = chk['fr_init']
    chk_out['f_in'] = chk['f_center']
    chk_out['x'] = (chk_out['f_out'] / chk_out['f_in'] - 1.)
    for c in chk.colnames:
        chk_out[c] = chk[c]

    # cols = id_cols
    # rows = []
    # unique_tone = set(tlt['tone_id'])
    # for chan_id in range(swp.frequency.shape[0]):
    #     subtbl = tlt[tlt['chan_id'] == chan_id]
    #     m = np.ones((len(subtbl), ), dtype=bool)
    #     for i, e in enumerate(subtbl):
    #         if e['tone_id'] not in unique_tone:
    #             m[i] = False
    #     subtbl = subtbl[m]
    #     s_tone_id = None
    #     fc = fs_stats['tone_axis_data']['f_center']
    #     if len(subtbl) == 0:
    #         d = {c: -1 for c in cols}
    #         d['fr_init'] = np.nan << u.Hz
    #         d['chan_id'] = chan_id
    #     elif len(subtbl) == 1:
    #         d = {c: subtbl[0][c] for c in cols}
    #         s_tone_id = subtbl[0]['tone_id']
    #         d['fr_init'] = subtbl[0]['fr_init']
    #     else:
    #         # select the closest one to the center
    #         imin = np.argmin(np.abs(fs_stats['center'][chan_id] - subtbl['fr_init']))
    #         logger.debug(f"select closest (imin={imin}) from {len(subtbl)} tones")
    #         d = {c: subtbl[imin][c] for c in cols}
    #         s_tone_id = subtbl[imin]['tone_id']
    #         d['fr_init'] = subtbl[imin]['fr_init']
    #     d['n_tones'] = len(subtbl)
    #     d['f_center'] = fc[chan_id]
    #     if s_tone_id is not None:
    #         unique_tone.remove(s_tone_id)
    #     rows.append(d)
    # chk = Table(rows=rows, names=['n_tones', 'f_center', 'fr_init'] + cols)
    # chk_out = Table()
    # chk_out['f_out'] = chk['fr_init']
    # chk_out['f_in'] = chk['f_center']
    # for p in chk.colnames:
    #     chk_out[p] = chk[p]
    logger.info(f"output tone check:\n{chk_out}")

    # here we need to further process the targ_out
    if not vary_n_tones:
        # check if we need to trim or pad the tone list
        # to maintain the same number of tones
        n_tones = len(targ_out)
        n_chans = fs_stats['n_tones']
        if n_tones != n_chans:
            logger.info(f"adjust tone list to match the channel list {n_tones=} {n_chans=}")
            if n_tones > n_chans:
                targ_out_mask = np.ones(n_tones, dtype=bool)
                n_trim = n_tones - n_chans
                for mtid in chk_out['group_id']:
                    if mtid < 0:
                        targ_out_mask[mtid] = False
                        n_trim -= 1
                        if n_trim == 0:
                            break
                logger.info(f"trimmed tones: {np.where(~targ_out_mask)[0]}")
                targ_out_trimmed = targ_out[targ_out_mask]
                if len(targ_out_trimmed) > n_chans:
                    targ_out_trimmed = targ_out_trimmed[:n_chans]
                targ_out_orig = targ_out
                targ_out = targ_out_trimmed
            else:
                # padd targ list
                npad = n_chans - n_tones
                pad_tbl = chk_out[chk_out['main_tone_id'] < 0]
                logger.info(f"pad {npad} tones from\n{pad_tbl}")
                pad_tbl = pad_tbl[-npad:]
                targ_out_padded = vstack([targ_out, targ_out[-npad:]])
                targ_out_padded['f_out'][-npad:] = pad_tbl['f_in']
                targ_out_padded['f_in'][-npad:] = pad_tbl['f_in']
                targ_out_padded['ampcor'][-npad:] = 1.
                targ_out_orig = targ_out
                targ_out = targ_out_padded

    # sort in fft order as required by the ROACH system
    swp = tone_list_ctx['swp']
    lofreq = swp.meta['flo_center']
    logger.info(f"subtract lo_freq = {lofreq}")
    dfs = targ_out['f_out'] - lofreq

    targ_out.add_column(Column(dfs, name='f_centered'), 0)
    targ_out.meta['Header.Toltec.LoCenterFreq'] = lofreq
    tones = targ_out['f_centered']
    max_ = 3 * np.max(tones)  # this is to shift the native tones to positive side
    isort = sorted(
        range(len(tones)),
        key=lambda i: tones[i] + max_ if tones[i] < 0 else tones[i]
        )
    targ_out = targ_out[isort]
    logger.info(f'output targ:\n{targ_out}')

    # plt.plot(targ_out['f_centered'])
    # plt.show()

    # expand this to a dummy tune.txt file for compatibility
    # TODO this will be phased out when we update tlaloc.
    compat_targ_freqs_dat = targ_out[['f_centered', 'f_out', 'f_in']]
    for p in ['flag', 'fp', 'Qr', 'Qc', 'fr', 'A', 'normI', 'normQ', 'slopeI', 'slopeQ', 'interceptI', 'interceptQ']:
        compat_targ_freqs_dat[p] = 0.
    logger.debug(f'output compat_targ_freqs.dat:\n{compat_targ_freqs_dat}')

    # also for the ampcor file:
    compat_targ_amps_dat = targ_out[['ampcor']]
    logger.debug(f'output compat_targ_amps.dat:\n{compat_targ_amps_dat}')

    return locals()


kids_model_name = 'lintrend'
# kids_model_name = 'lintrend2'
# kids_model_name = 'proper'
# kids_model_name = 'geometry'


# this is for y = x + 2 * a * r ** 3/ (r ** 2 + y ** 2)
# there are three roots
_ne_duffing_fmt = (
"""
(54*{a}*{r}**3+sqrt((54*{a}*{r}**3+18*{r}**2*{x}+2*{x}**3)**2+4*(3*{r}**2-{x}**2)**3)+18*{r}**2*{x}+2*{x}**3)**(1/3)/(3*2**(1/3))-(2**(1/3)*(3*{r}**2-{x}**2))/(3*(54*{a}*{r}**3+sqrt((54*{a}*{r}**3+18*{r}**2*{x}+2*{x}**3)**2+4*(3*{r}**2-{x}**2)**3)+18*{r}**2*{x}+2*{x}**3)**(1/3))+{x}/3
""".replace("\n",'').strip(),
"""
-((1-1.j*sqrt(3))*(54*{a}*{r}**3+sqrt((54*{a}*{r}**3+18*{r}**2*{x}+2*{x}**3)**2+4*(3*{r}**2-{x}**2)**3)+18*{r}**2*{x}+2*{x}**3)**(1/3))/(6*2**(1/3))+((1+1.j*sqrt(3))*(3*{r}**2-{x}**2))/(3*2**(2/3)*(54*{a}*{r}**3+sqrt((54*{a}*{r}**3+18*{r}**2*{x}+2*{x}**3)**2+4*(3*{r}**2-{x}**2)**3)+18*{r}**2*{x}+2*{x}**3)**(1/3))+{x}/3
""".replace("\n",'').strip(),
"""
-((1+1.j*sqrt(3))*(54*{a}*{r}**3+sqrt((54*{a}*{r}**3+18*{r}**2*{x}+2*{x}**3)**2+4*(3*{r}**2-{x}**2)**3)+18*{r}**2*{x}+2*{x}**3)**(1/3))/(6*2**(1/3))+((1-1.j*sqrt(3))*(3*{r}**2-{x}**2))/(3*2**(2/3)*(54*{a}*{r}**3+sqrt((54*{a}*{r}**3+18*{r}**2*{x}+2*{x}**3)**2+4*(3*{r}**2-{x}**2)**3)+18*{r}**2*{x}+2*{x}**3)**(1/3))+{x}/3
""".replace("\n",'').strip(),
)
# the equation that gives the value at bifurcation x = -sqrt(3) r
_ne_duffing_bf_fmt = """
1/3*(2**(1/3)*(4*{x}**3-3*sqrt(3)*{a}*{x}**3)**(1/3)+{x})
""".replace("\n",'').strip()



if kids_model_name == 'lintrend':

    def _kids_model_func_0(f, fr, Qr, g0, g1, fp, k0, k1, m0, m1):
        x = (f - fr) / fr
        r = 0.5 / Qr
        X = r + 1.j * x
        Sp = 0.5 / X
        g = g0 + 1.j * g1
        k = k0 + 1.j * k1
        m = m0 + 1.j * m1
        # kdf = (fp / Qr) * 1.5
        # slope = k * (f - fp)
        slope = k * (f - fr)
        # slope[f > fp + kdf] = k * kdf
        # slope[f < fp - kdf] * -k * kdf
        return g * Sp + slope + m

    def _kids_model_func(f, fr, Qr, g0, g1, fp, k0, k1, m0, m1):
        return ne.evaluate("""
            (g0 + 1.j * g1)
            * Qr / (1. + 2.j * Qr * (
                     (f - fr) / fr
                    )
                  )
            + (k0 + 1.j * k1) * (f-fr)
            + m0 + 1.j * m1
            """.replace("\n", '').strip()
            )


    _kids_param_names = ['fr', 'Qr', 'g0', 'g1', 'fp', 'k0', 'k1', 'm0', 'm1']


elif kids_model_name == 'lintrend2':

    def _kids_model_func(f, fr, Qr, g0, g1, a, fp, k0, k1, k2, k3, k4, k5, m0, m1):
        x = (f - fr) / fr
        r = 0.5 / Qr

        # solve for duffing equation numerically. This is too slow
        # ysp = []
        # for xx in x:
        #     duffing = Polynomial([
        #         xx * r ** 2 + 2 * a * r**3,
        #         -r ** 2,
        #         xx,
        #         -1.
        #         ])
        #     ysp.append(duffing.roots())
        # ysp = np.array(ysp).T

        # Use cubic root equation, this is fast but is not accurate around
        # the determinant when a is small.

        if a == 0:
            y = x
        else:
            # note the bifurcation is at x = -(sqrt3)r, where the equation becomes undefined.
            # where the two imag roots becomes real.
            # We make the x and r complex to avoid nan values in evaluation.
            ys = []
            for ne_fmt in _ne_duffing_fmt:
                ys.append(ne.evaluate(ne_fmt.format(a='a', x='(x + 0.j)', r=f'(r + 0.j)')))
            # bfcond = ne.evaluate('x + sqrt(3) * r')
            # because of the floating point issues, we rely on the 0 value of
            # first root imag part to set the bfcond.
            bfmask = ys[0].imag != 0
            # if a > 0.3:
            #     plt.plot(x[~bfmask], ys[0].real[~bfmask])
            #     plt.plot(x[bfmask], ys[1].real[bfmask])
            #     plt.plot(x[bfmask], ys[2].real[bfmask])
            #     plt.show()
            # here we take the larger value after bifurcation as the model
            y = ys[0].real
            y[bfmask] = ys[1][bfmask].real

            # sometimes the bfcond is zero and evaluate gives nan
            # we use the exact value in this case
            # mask = np.isclose(bfcond, 0.)
            nanmask = np.isnan(y)
            if np.any(nanmask):
            # if not np.all(np.isfinite(y[mask])):
                # replace with value at bifurcation.
                xm = x[nanmask]
                y[nanmask] = ne.evaluate(_ne_duffing_bf_fmt.format(a='a', x='xm + 0.j'))
            # if not np.all(np.isfinite(y)):
            #     import pdb
            #     pdb.set_trace()
            # if np.any(np.abs(y - x) > 1e-6):
            #     import pdb
            #     pdb.set_trace()
            # print(f"{a=}")
            # print(np.abs(y - x).mean())
            # print(np.abs(y - x).min())
            # print(np.abs(y - x).max())
        X = "(r + 1.j * y)"
        Sp = f"(0.5 / {X})"
        g = "(g0 + 1.j * g1)"
        m = "(m0 + 1.j * m1)"
        ff = "(f - fp)"
        # ff = f - fp
        k = "(k0 + 1.j * k1)"
        kk = "(k2 + 1.j * k3)"
        kkk = "(k4 + 1.j * k5)"
        s1 = f'({k} * {ff})'
        s2 = f'({kk} * 0.5 * (3 * {ff} ** 2 - 1))'
        s3 = f'({kkk} * 0.5 * (5 * {ff} ** 3 - 3 * {ff}))'
        # s1 = f'({k} * ff)'
        # s2 = f'({kk} * 0.5 * (3 * ff ** 2 - 1))'
        # s3 = f'({kkk} * 0.5 * (5 * ff ** 3 - 3 * ff))'

        expr = f"{g} * {Sp} + {m} + {s1} + {s2} + {s3}"
        return ne.evaluate(expr)

    _kids_param_names = ['fr', 'Qr', 'g0', 'g1', 'a', 'fp', 'k0', 'k1', 'k2', 'k3', 'k4', 'k5', 'm0', 'm1']



elif kids_model_name == 'proper':

    def _kids_model_func_0(f, fr, Qr, g0, g1, tau, Qc, A, a, k0, k1, k2, k3, k4, k5):
        x = (f - fr) / fr
        r = 0.5 / Qr
        X = r + 1.j * x + 1.j * a * r ** 2 / (r ** 2 + x ** 2)
        Sp = 0.5 / X
        g = g0 + 1.j * g1
        aa = g * np.exp(-2.j * np.pi * f * tau)
        bb = (1 + 1.j * A) / Qc
        k = k0 + 1.j * k1
        kk = k2 + 1.j * k3
        kkk = k4 + 1.j * k5
        ff = (f - fr)
        slope = k * ff + kk * 0.5 * (3 * ff ** 2 - 1) + kkk * 0.5 * (5 * ff ** 3 - 3 * ff)
        return aa * (1 - bb * Sp) + slope



    def _kids_model_func(f, fr, Qr, g0, g1, tau, Qc, A, a, k0, k1, k2, k3, k4, k5):
        x = (f - fr) / fr
        r = 0.5 / Qr
        # solve for duffing equation
        # note the bifurcation is at x = -(sqrt3)r, where the equation becomes undefined.
        # where the two imag roots becomes real.
        # We make the x and r complex to avoid nan values in evaluation.
        ys = []
        for ne_fmt in _ne_duffing_fmt:
            ys.append(ne.evaluate(ne_fmt.format(a='a', x='(x + 0.j)', r=f'(r + 0.j)')))
        # bfcond = ne.evaluate('x + sqrt(3) * r')
        # because of the floating point issues, we rely on the 0 value of
        # first root imag part to set the bfcond.
        bfmask = ys[0].imag != 0
        # if a > 0.3:
        #     plt.plot(x[~bfmask], ys[0].real[~bfmask])
        #     plt.plot(x[bfmask], ys[1].real[bfmask])
        #     plt.plot(x[bfmask], ys[2].real[bfmask])
        #     plt.show()
        # here we take the larger value after bifurcation as the model
        y = ys[0].real
        y[bfmask] = ys[1][bfmask].real

        # sometimes the bfcond is zero and evaluate gives nan
        # we use the exact value in this case
        # mask = np.isclose(bfcond, 0.)
        nanmask = np.isnan(y)
        if np.any(nanmask):
        # if not np.all(np.isfinite(y[mask])):
            # replace with exact value
            xm = x[nanmask]
            y[nanmask] = ne.evaluate(_ne_duffing_bf_fmt.format(a='a', x='xm + 0.j'))
        if not np.all(np.isfinite(y)):
            import pdb
            pdb.set_trace()
        X = "(r + 1.j * y)"
        Sp = f"(0.5 / {X})"
        g = "(g0 + 1.j * g1)"
        pi = np.pi
        aa = f"({g} * exp(-2.j * pi * f * tau))"
        bb = "((1. + 1.j * A) / Qc)"
        # ff = "(f - fr)"
        ff = f - f[f.size // 2]
        k = "(k0 + 1.j * k1)"
        kk = "(k2 + 1.j * k3)"
        kkk = "(k4 + 1.j * k5)"
        # s1 = f'({k} * {ff})'
        # s2 = f'({kk} * 0.5 * (3 * {ff} ** 2 - 1))'
        # s3 = f'({kkk} * 0.5 * (5 * {ff} ** 3 - 3 * {ff}))'
        s1 = f'({k} * ff)'
        s2 = f'({kk} * 0.5 * (3 * ff ** 2 - 1))'
        s3 = f'({kkk} * 0.5 * (5 * ff ** 3 - 3 * ff))'

        expr = f"{aa} * (1 - {bb} * {Sp}) + {s1} + {s2} + {s3}"
        return ne.evaluate(expr)
        # if np.any(np.isnan(S21)):
        #     import pdb
        #     pdb.set_trace()
        # return S21
        # return ne.evaluate("""
        #     (g0 + 1.j * g1) * exp(-2.j * pi * f * tau)
        #     * (1. - (1. + 1.j * A) / Qc
        #        * Qr / (1. + 2.j * Qr * (
        #              (f - fr) / fr
        #             + a / (1 + 4 * Qr ** 2 * (f-fr) ** 2 / fr ** 2)
        #             )
        #           ))
        #     + (k0 + 1.j * k1) * (f-fr)
        #     + (k2 + 1.j * k3) * 0.5 * (3. * (f-fr) ** 2 - 1.)
        #     + (k4 + 1.j * k5) * 0.5 * (5. * (f-fr) ** 3 - 3. * (f-fr))""".replace("\n", '').strip()
        #     )



        # return ne.evaluate(
        #     "(g0 + 1.j * g1) * exp(-2.j * pi * f * tau) "
        #     "* (1. - (1. + 1.j * A) / Qc * Qr / "
        #     "(1. + 2.j * Qr * ((f-fr)/fr + 1. / (1. + 4 * Qr ** 2 * ((f-fr)/fr) ** 2)))) "
        #     "+ (k0 + 1.j * k1) * (f-fr) "
        #     "+ (k2 + 1.j * k3) * 0.5 * (3. * (f-fr) ** 2 - 1.) "
        #     "+ (k4 + 1.j * k5) * 0.5 * (5. * (f-fr) ** 3 - 3. * (f-fr))"
        #     )

    _kids_param_names = ['fr', 'Qr', 'g0', 'g1', 'tau', 'Qc', 'A', 'a', 'k0', 'k1', 'k2', 'k3', 'k4', 'k5']



elif kids_model_name == 'geometry':

    # @njit(parallel=False)
    def _kids_model_func(f, fr, Qr, g0, g1, tau, Qc, phi_c):
        x = (f - fr) / fr
        r = 0.5 / Qr
        X = r + 1.j * x
        Sp = 0.5 / X
        gg = g0 + 1.j * g1
        cc = 1. / (Qc * np.exp(1.j * phi_c))
        aa = gg * np.exp(-2.j * np.pi * f * tau)
        return aa * (1 - Sp * cc)

    _kids_param_names = ['fr', 'Qr', 'g0', 'g1', 'tau', 'Qc', 'phi_c']
else:
    raise ValueError()


def _get_kids_pname(name, i, j):
    return f'm{i}{j}_{name}'

def _add_kids_param(params, name, i, j, **kwargs):
    params.add(_get_kids_pname(name, i, j), **kwargs)

def _get_kids_param(params, name, i, j):
    return params[_get_kids_pname(name, i, j)]



def _kids_model_func_eval(params, i, j, f):
    # eval multiple kids model at f for model index i
    return _kids_model_func(
        f,
        *(_get_kids_param(params, n, i, j) for n in _kids_param_names),
        )

def _kids_model_eval(params, di, tone_indices, f):
    # this evel kids model for a set of data.
    # each data item has (x, y, tone_indices)
    S21_mdl = np.zeros(f.shape, dtype=np.complex)
    for j in tone_indices:
        S21_mdl += _kids_model_func_eval(params, di, j, f)
    return S21_mdl


# @njit
# def _kids_model_func_eval_njit(params_tuple, npars, i, j, f):
#     # eval multiple kids model at f for model index i
#     i0 = npars * j
#     # i1 = i0 + npars
#     return _kids_model_func(
#         f,
#         params_tuple[i0],
#         params_tuple[i0 + 1],
#         params_tuple[i0 + 2],
#         params_tuple[i0 + 3],
#         params_tuple[i0 + 4],
#         params_tuple[i0 + 5],
#         params_tuple[i0 + 6],
#         params_tuple[i0 + 7],
#         params_tuple[i0 + 8],
#         )

# @njit
# def _kids_model_eval_njit(params_tuple, npars, di, nt, f):
#     S21_mdl = np.zeros(f.shape, dtype=np.complex)
#     for j in range(nt):
#         S21_mdl += _kids_model_func_eval_njit(params_tuple, npars, di, j, f)
#     return S21_mdl


# def _params_to_tuple(params, i, tone_indices):
#     p = params.valuesdict()
#     result = []
#     for j in tone_indices:
#         for pn in _kids_param_names:
#             result.append(_get_kids_param(params, pn, i, j).value)
#     return tuple(result)


def _kids_fit_objective_func(params, dataset):
    # this evel kids model for a set of data.
    # each data item has (x, y, tone_indices)
    resid = []
    # npars = len(_kids_param_names)
    # ndata = len(dataset)
    for i, d in enumerate(dataset):
        f = d['f']
        S21 = d['S21']
        tone_indices = d['tone_indices']
        # tone_masks = d['tone_masks']
        S21_mdl = _kids_model_eval(
            # params.valuesdict(),
            params,
            i, tone_indices, f)
        # S21_mdl = _kids_model_eval_njit(
        #     _params_to_tuple(params, i, tone_indices), npars, i, len(tone_indices), f)
        resid.append(S21 - S21_mdl)
    return np.hstack(resid).view(float)


def _prep_fit(swp, model_group_ctx, fit_n_fwhm=2):
    """Generate model for fitting a group"""
    logger = get_logger()
    di_fits = model_group_ctx['di_fits']
    di_refs = model_group_ctx['di_refs']
    dis = model_group_ctx['dis']
    kmt = model_group_ctx['kmt']
    Qr0 = np.median(kmt['Qr'])
    logger.debug(f"prepare model to fit n_tones={len(di_fits)}, n_chans={len(dis)} {Qr0=}")

    # for each tone and chan, build the model params
    params = lmfit.Parameters()
    dataset = []
    f_fit = model_group_ctx['f_fit']
    f_fit_Hz = f_fit.to_value(u.Hz)
    for i, di in enumerate(dis):
        xx = swp.frequency[di].to_value(u.Hz)
        yy = swp.S21[di].value
        # fp = xx.mean()
        m_init = yy.mean()
        mm = np.zeros((len(xx), ), dtype=bool)
        tone_indices = []
        tone_ctxs = []
        for j, f in enumerate(f_fit_Hz):
            df = f / Qr0 * fit_n_fwhm
            ff_min = f - df
            ff_max = f + df
            has_data = (xx >= ff_min) & (xx <= ff_max)
            mm[has_data] = True
            if has_data.sum() > 0:
                # add params for this model
                # _add_kids_param(params, 'fr', i, j, value=f) # min=f - df / 2, max=f + df / 2)
                fi_min = f - df / 2
                fi_max = f + df / 2
                fd = xx[has_data]
                sd = np.where(has_data)[0]
                _add_kids_param(params, 'fr', i, j, value=f, min=fi_min, max=fi_max)
                _add_kids_param(params, 'Qr', i, j, value=Qr0, min=100)
                _add_kids_param(params, 'g0', i, j, value=1.)
                _add_kids_param(params, 'g1', i, j, value=0.)
                if kids_model_name == 'lintrend':
                    _add_kids_param(params, 'fp', i, j, value=f, vary=False)
                    _add_kids_param(params, 'k0', i, j, value=0., vary=True)
                    _add_kids_param(params, 'k1', i, j, value=0., vary=True)
                    _add_kids_param(params, 'm0', i, j, value=m_init.real)
                    _add_kids_param(params, 'm1', i, j, value=m_init.imag)
                elif kids_model_name == 'lintrend2':
                    _add_kids_param(
                        params, 'a', i, j,
                        # value=1e-7,
                        value=0,
                        min=0.0,
                        # max=4*np.sqrt(3)/9,
                        max=0.5,
                        # vary=True
                        vary=False
                        )
                    _add_kids_param(params, 'fp', i, j, value=f, vary=False)
                    _add_kids_param(params, 'k0', i, j, value=0., vary=True)
                    _add_kids_param(params, 'k1', i, j, value=0., vary=True)
                    _add_kids_param(params, 'k2', i, j, value=0., vary=True)
                    _add_kids_param(params, 'k3', i, j, value=0., vary=True)
                    _add_kids_param(params, 'k4', i, j, value=0., vary=False)
                    _add_kids_param(params, 'k5', i, j, value=0., vary=False)
                    _add_kids_param(params, 'm0', i, j, value=m_init.real)
                    _add_kids_param(params, 'm1', i, j, value=m_init.imag)
                elif kids_model_name == 'proper':
                    _get_kids_param(params, 'g0', i, j).value = m_init.real
                    _get_kids_param(params, 'g1', i, j).value = m_init.imag
                    _add_kids_param(params, 'tau', i, j, value=0., vary=False)
                    _add_kids_param(params, 'Qc', i, j, value=1e5, min=100)
                    _add_kids_param(params, 'A', i, j, value=0.)
                    _add_kids_param(
                        params, 'a', i, j, value=1e-7,
                        min=0.0,
                        # max=4*np.sqrt(3)/9,
                        max=0.5,
                        vary=True)
                    _add_kids_param(params, 'k0', i, j, value=0., vary=True)
                    _add_kids_param(params, 'k1', i, j, value=0., vary=True)
                    _add_kids_param(params, 'k2', i, j, value=0., vary=True)
                    _add_kids_param(params, 'k3', i, j, value=0., vary=True)
                    _add_kids_param(params, 'k4', i, j, value=0., vary=True)
                    _add_kids_param(params, 'k5', i, j, value=0., vary=True)
                elif kids_model_name == 'geometry':
                    _add_kids_param(params, 'tau', i, j, value=0.)
                    _add_kids_param(params, 'Qc', i, j, value=4e4)
                    _add_kids_param(params, 'phi_c', i, j, value=0.)
                tone_indices.append(j)
                tone_ctxs.append({
                    "model_chan_id": i,
                    "model_tone_id": j,
                    "di_fit": di_fits[j],
                    "f_fit": f << u.Hz,
                    "f_fit_min": ff_min << u.Hz,
                    "f_fit_max": ff_max << u.Hz,
                    "fr_init": f << u.Hz,
                    "fr_init_min": fi_min << u.Hz,
                    "fr_init_max": fi_max << u.Hz,
                    'Qr_init': Qr0,
                    "m0_init": m_init.real,
                    "m1_init": m_init.imag,
                    "f_start": fd[0] << u.Hz,
                    "f_stop": fd[-1] << u.Hz,
                    "index_start": sd[0],
                    "index_stop": sd[-1],
                    "datamask": has_data,
                    })
        dataset.append({
            'di': di,
            'f': xx[mm],
            'S21': yy[mm],
            'tone_indices': tone_indices,
            'tone_masks': [tc['datamask'][mm] for tc in tone_ctxs],
            'tone_ctxs': tone_ctxs,
            'f_fit': f_fit[tone_indices],
            })
    fit_input_f = np.hstack([d['f'] for d in dataset])
    fit_input_S21 = np.hstack([d['S21'] for d in dataset])

    # connect model params
    # for the same tone, link the fr and Qr
    # for tones on the same swp, link the k
    for j in range(len(f_fit_Hz)):
        i_first = None
        for i in range(len(dis)):
            tone_indices = dataset[i]['tone_indices']
            if j not in tone_indices:
                continue
            if i_first is None:
                i_first = i
            if i != i_first:
                _get_kids_param(params, 'fr', i, j).expr = _get_kids_pname('fr', i_first, j)
                _get_kids_param(params, 'Qr', i, j).expr = _get_kids_pname('Qr', i_first, j)
    for i in range(len(dis)):
        j_first = None
        for j in range(len(f_fit_Hz)):
            tone_indices = dataset[i]['tone_indices']
            if j not in tone_indices:
                continue
            if j_first is None:
                j_first = j
            if j != j_first:
                pass
                # _get_kids_param(params, 'k0', i, j).expr = _get_kids_pname('k0', i, j_first)
                # _get_kids_param(params, 'k1', i, j).expr = _get_kids_pname('k1', i, j_first)
                # _get_kids_param(params, 'k2', i, j).expr = _get_kids_pname('k2', i, j_first)
                # _get_kids_param(params, 'k3', i, j).expr = _get_kids_pname('k3', i, j_first)
                # _get_kids_param(params, 'k4', i, j).expr = _get_kids_pname('k4', i, j_first)
                # _get_kids_param(params, 'k5', i, j).expr = _get_kids_pname('k5', i, j_first)

    # logger.debug("fit init params:")
    # params.pretty_print()
    return locals()

def _fit_worker(arg):
    params, dataset = arg
    return lmfit.minimize(
        _kids_fit_objective_func, params, args=(dataset, ),
        method='leastsq',
        # method='powell',
        )# , max_nfev=1000)


def do_fit(swp, model_group_ctxs, debug_plot_kw=None, n_procs=4):
    """Generate models for fitting all groups."""
    dbk = debug_plot_kw or {}
    nw = swp.meta['nwid']
    obsnum = swp.meta['obsnum']

    fit_ctxs = []
    for model_group_ctx in model_group_ctxs:
        fit_ctxs.append(_prep_fit(swp, model_group_ctx))
    if dbk.get("prepfit_enabled", False):
        gi0 = dbk.get('gi0', 0)
        nrows, ncols = _get_plot_grid(dbk, gi0, len(fit_ctxs))
        fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(15, 15),
                constrained_layout=True,
                squeeze=False,
                )
        fig.suptitle(f'{obsnum=} {nw=}')
        axes = axes.ravel()
        for i, ax in enumerate(axes):
            # divider = make_axes_locatable(ax)
            # ax_res = divider.append_axes("bottom", size="20%", pad="2%", sharex=ax)
            gi = i + gi0
            if gi >= len(fit_ctxs):
                continue
            fit_ctx = fit_ctxs[gi]
            di_refs = fit_ctx['di_refs']
            di_fits = fit_ctx['di_fits']
            dis = fit_ctx['dis']
            if len(dis) > 10:
                info = f"{gi=}\nn_di_refs={len(di_refs)}\nn_dis={len(dis)}\nn_di_fits={len(di_fits)}"
            else:
                info = f"{gi=}\n{di_refs=}\n{dis=}\n{di_fits=}"
            ax.text(0, 1, info, transform=ax.transAxes, va='top', ha='left')
            legs = {}
            for j, di in enumerate(dis):
                xx = swp.frequency[di].to_value(u.MHz)
                yy = np.log10(np.abs(swp.S21.value[di]))
                ax.plot(xx, yy, color='C0', label='S21')
            # plot fit input x and y
            handle = ax.plot(
                fit_ctx['fit_input_f'] / 1e6,
                np.log10(np.abs(fit_ctx['fit_input_S21'])),
                color='red', linestyle='none', marker='.')
            if 'fit_input' not in legs:
                legs['fit_input'] = handle
            # plot tone list for this group
            for f in fit_ctx['f_fit']:
                handle = ax.axvline(f.to_value(u.MHz), color='red', linestyle='--')
                if 'f_fit' not in legs:
                    legs['f_fit'] = handle
            # render the init model
            for j, di in enumerate(dis):
                xx = swp.frequency[di].to_value(u.Hz)
                yy_mdl = _kids_model_eval(
                    fit_ctx['params'].valuesdict(), j,
                    fit_ctx['dataset'][j]['tone_indices'], xx)
                ax.plot(xx / 1e6, np.log10(np.abs(yy_mdl)), color='green', linestyle=':')
            ax.set_xlabel("f (MHz)")
            ax.legend(labels=legs.keys(), handles=legs.values())
        plt.show()
    # now ready to do fit
    from multiprocessing import Pool

    logger.info(f"fitting with {n_procs=}")
    with timeit("run fitting"):
        with Pool(processes=n_procs) as pool:
            fit_outs = list(tqdm.tqdm(
                pool.imap(
                _fit_worker, [(fit_ctx['params'], fit_ctx['dataset']) for fit_ctx in fit_ctxs]
                ), total=len(fit_ctxs)))
        # fit_outs = list(tqdm.tqdm(
        #     map(
        #     _fit_worker, [(fit_ctx['params'], fit_ctx['dataset']) for fit_ctx in fit_ctxs]
        #     ), total=len(fit_ctxs)))

    for gi, fit_ctx in enumerate(fit_ctxs):
        # tone_list = group_tone_list[gi]
        # fit_out = timeit(lmfit.minimize)(
        #     _kids_fit_objective_func,
        #     fit_ctx['params'],
        #     args=(fit_ctx['dataset'],)
        #     )
        fit_out = fit_ctx['fit_out'] = fit_outs[gi]
        # lmfit.report_fit(fit_out.params)
    if dbk.get("checkfit_enabled", False):
        gi0 = dbk.get('gi0', 0)
        nrows, ncols = _get_plot_grid(dbk, gi0, len(fit_ctxs))
        fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(15, 15),
                constrained_layout=True,
                squeeze=False,
                )
        fig.suptitle(f'{obsnum=} {nw=}')
        axes = axes.ravel()
        for i, ax in enumerate(axes):
            divider = make_axes_locatable(ax)
            ax_base = divider.append_axes("bottom", size="20%", pad="2%", sharex=ax)
            ax_res = divider.append_axes("bottom", size="20%", pad="2%", sharex=ax)
            gi = i + gi0
            if gi >= len(fit_ctxs):
                continue
            fit_ctx = fit_ctxs[gi]
            di_refs = fit_ctx['di_refs']
            di_fits = fit_ctx['di_fits']
            dis = fit_ctx['dis']
            if len(dis) > 10:
                info = f"{gi=}\nn_di_refs={len(di_refs)}\nn_dis={len(dis)}\nn_di_fits={len(di_fits)}"
            else:
                info = f"{gi=}\n{di_refs=}\n{dis=}\n{di_fits=}"
            ax.text(0, 1, info, transform=ax.transAxes, va='top', ha='left')
            legs = {}
            for j, di in enumerate(dis):
                xx = swp.frequency[di].to_value(u.MHz)
                yy = np.log10(np.abs(swp.S21.value[di]))
                ax.plot(xx, yy, color='C0', label='S21')
            # plot fit input x and y
            handle = ax.plot(
                fit_ctx['fit_input_f'] / 1e6,
                np.log10(np.abs(fit_ctx['fit_input_S21'])),
                color='red', linestyle='none', marker='.')
            if 'fit_input' not in legs:
                legs['fit_input'] = handle
            # plot tone list for this group
            for f in fit_ctx['f_fit']:
                handle = ax.axvline(f.to_value(u.MHz), color='red', linestyle='--')
                if 'f_fit' not in legs:
                    legs['f_fit'] = handle
            # render the fitted model and residual
            for j, di in enumerate(dis):
                xx = swp.frequency[di].to_value(u.Hz)
                yy_mdl = _kids_model_eval(
                    fit_ctx['fit_out'].params.valuesdict(), j,
                    fit_ctx['dataset'][j]['tone_indices'], xx)
                ax.plot(xx / 1e6, np.log10(np.abs(yy_mdl)), color='green', linestyle=':')
                yy = swp.S21.value[di]
                rr = yy - yy_mdl
                # ax_res.plot(xx / 1e6, rr.real / yy.real, color='blue', linestyle=':')
                # ax_res.plot(xx / 1e6, rr.imag / yy.imag, color='red', linestyle=':')
                ax_res.plot(xx / 1e6, rr.real, color='blue', linestyle=':')
                ax_res.plot(xx / 1e6, rr.imag, color='red', linestyle=':')
                ax_res.plot(xx / 1e6, np.abs(rr), color='green', linestyle=':')
                # render baseline
                baseline_params = fit_ctx['fit_out'].params.valuesdict().copy()
                for k in baseline_params:
                    if k.endswith("g0") or k.endswith("g1"):
                        baseline_params[k] = 0.
                yy_base = _kids_model_eval(
                    baseline_params, j,
                    fit_ctx['dataset'][j]['tone_indices'], xx)
                bb = yy_base
                ax_base.plot(xx / 1e6, bb.real, color='blue', linestyle=':')
                ax_base.plot(xx / 1e6, bb.imag, color='red', linestyle=':')
                ax_base.plot(xx / 1e6, np.abs(bb), color='green', linestyle=':')

                # debug check overdriven
                od_params = fit_ctx['fit_out'].params.valuesdict().copy()
                # for v in [0.0, 0.3, 0.5, 0.7, 0.9]:
                for v in [0.0,]:
                    for k in od_params:
                        if k.endswith("_a"):
                            od_params[k] = v
                    yy_od = _kids_model_eval(
                        od_params, j,
                        fit_ctx['dataset'][j]['tone_indices'], xx)
                    ax.plot(xx / 1e6, np.log10(np.abs(yy_od)), color='cyan', linestyle='-')

            # plot best fit frequencies
            for k, p in fit_ctx['fit_out'].params.items():
                if k.endswith("_fr"):
                    unc = p.stderr
                    if unc is not None:
                        ax.axvspan((p.value - unc) / 1e6, (p.value + unc) / 1e6, color='green', alpha=0.5)
                    ax.axvline(p.value / 1e6, color='green', linestyle='-')
            ax.set_xlabel("f (MHz)")
            ax.legend(labels=legs.keys(), handles=legs.values())
        plt.show()
    # collect fit result
    # this is a flat list of all models, with indices back to input di
    # and di_ref
    kmt_out = []
    for fit_ctx in fit_ctxs:
        dis = fit_ctx['dis']
        di_fits = fit_ctx['di_fits']
        params = fit_ctx['params']
        params_out = fit_ctx['fit_out'].params
        # print(params)
        for i, di in enumerate(dis):
            d = fit_ctx['dataset'][i]
            tone_indices = d['tone_indices']
            for j in tone_indices:
                fr0 = _get_kids_param(params, 'fr', i, j).value << u.Hz
                Qr0 = _get_kids_param(params, 'Qr', i, j).value
                assert fit_ctx['f_fit'][j] == fr0
                entry = {
                    "chan_id": di,
                    'tone_id': di_fits[j],
                    "fr0": fr0,
                    "Qr0": Qr0,
                    }
                for pn in _kids_param_names:
                    if pn in ['fr', 'fp']:
                        unit = u.Hz
                    elif pn in ['k0', 'k1']:
                        unit = u.s
                    else:
                        unit = None
                    entry[pn] = u.Quantity(
                        _get_kids_param(params_out, pn, i, j).value,
                        unit
                        )
                    entry[f'{pn}_unc'] = u.Quantity(
                        _get_kids_param(params_out, pn, i, j).stderr or 0.,
                        unit,
                        )
                for k in ['nfev', 'success', 'chisqr', 'redchi', 'ndata', 'nvarys', 'nfree']:
                    entry[k] = getattr(fit_ctx['fit_out'], k)
                kmt_out.append(entry)
    kmt_out = QTable(rows=kmt_out)
    logger.info(f'output kmt:\n{kmt_out}')
    return locals()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ref_file', '-r', help='reference data for guessing tone list.')
    parser.add_argument(
        '--output_dir'
        )
    parser.add_argument(
        '--debug_plot', action='store_true'
        )
    parser.add_argument(
        '--log_level', default='INFO'
        )
    parser.add_argument(
        '--no_fit', action='store_true',
        )
    parser.add_argument(
        '--save_fit_only', action='store_true',
        )
    parser.add_argument(
        '--n_procs', default=4, type=int,
        )
    parser.add_argument(
        '--tune_mode', action='store_true',
        )

    parser.add_argument("sweep_file", help='sweep data for fitting.')
    # parser.add_argument("--config", "-c", help='YAML config file.')
    option = parser.parse_args()

    init_log(level=option.log_level)
    logger = get_logger()

    ref_file = Path(option.ref_file)
    with open(ref_file, 'rb') as fo:
        ref_data = dill.load(fo)

    sweep_file = Path(option.sweep_file)

    logger.info(f"Reduce {sweep_file} with {ref_file}")

    sweep_data = NcFileIO(sweep_file).open()

    # check if tune mode is applicable
    if sweep_data.meta['data_kind'].name == 'Tune' and option.tune_mode:
        logger.info(f"Running in tune mode, number of tones are fixed")
        # check block index
        if sweep_data.meta['n_blocks'] == 1:
            logger.info(f"Skip processing in the middle of TUNE.")
            sys.exit(0)
        vary_n_tones = False
    else:
        vary_n_tones = True

    output_dir = option.output_dir or ref_file.parent
    output_dir = Path(output_dir)
    if option.debug_plot:
        debug_plot_kw = {
            # 'despike_enabled': True,
            # 'd21shift_enabled': True,
            # 'channelgroups_enabled': True,
            # 'tonelist_enabled': True,
            # 'tonelist_save': output_dir.joinpath(sweep_file.stem + '_tonelist.png'),
            # 'modelgroups_enabled': True,
            # "prepfit_enabled": True,
            # 'f_shift_enabled': True,
            "checkfit_enabled": True,
            'gi0': 0,
            'di0': 0,
            'nrows': 5,
            'ncols': 10
            }
    else:
        debug_plot_kw = None
    ctx = timeit(make_tone_list)(
        ref_data=ref_data,
        sweep_data=sweep_data,
        debug_plot_kw=debug_plot_kw)

    stem = sweep_file.resolve().stem
    if not option.save_fit_only:
        # dump the fit ctx for external usage
        ctx = export_tone_list(ctx, debug_plot_kw=debug_plot_kw, vary_n_tones=vary_n_tones)

        def _post_proc_and_save(tbl, suffix, format='ascii.ecsv'):
            tbl = Table(tbl)
            output_file = output_dir.joinpath(stem + suffix)
            tbl.meta['Header.Toltec.ObsNum'] = sweep_data.meta['obsnum']
            tbl.meta['Header.Toltec.SubObsNum'] = sweep_data.meta['subobsnum']
            tbl.meta['Header.Toltec.ScanNum'] = sweep_data.meta['scannum']
            tbl.meta['Header.Toltec.RoachIndex'] = sweep_data.meta['roachid']
            tbl.write(output_file, overwrite=True, format=format)
            logger.info(f"saved file {output_file}")
        _post_proc_and_save(ctx['tlt'], '_tonelist.ecsv')
        _post_proc_and_save(ctx['targ_out'], '_targfreqs.ecsv')
        _post_proc_and_save(ctx['compat_targ_freqs_dat'], '_targfreqs.dat')
        _post_proc_and_save(ctx['compat_targ_amps_dat'], '_ampcor.dat', format='ascii.no_header')
        _post_proc_and_save(ctx['chk_out'], '_tonecheck.ecsv')

    if option.no_fit:
        return

    # model data
    ctx = timeit(do_fit)(swp=ctx['swp'], model_group_ctxs=ctx['model_group_ctxs'], debug_plot_kw=debug_plot_kw, n_procs=option.n_procs)
    kmt = ctx['kmt_out']
    kmt = Table(kmt)
    output_file = output_dir.joinpath(stem + '_kmt.txt')
    kmt.meta['Header.Toltec.ObsNum'] = sweep_data.meta['obsnum']
    kmt.meta['Header.Toltec.SubObsNum'] = sweep_data.meta['subobsnum']
    kmt.meta['Header.Toltec.ScanNum'] = sweep_data.meta['scannum']
    kmt.meta['Header.Toltec.RoachIndex'] = sweep_data.meta['roachid']
    kmt.write(output_file, overwrite=True, format='ascii.ecsv')
    return


if __name__ == "__main__":
    # import cProfile
    # with cProfile.Profile() as pr:
    if True:
        main()
        # pr.print_stats(sort='cumtime')
