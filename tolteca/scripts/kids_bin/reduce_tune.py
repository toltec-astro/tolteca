#!/usr/bin/env python
from numba import njit
import dill
from pathlib import Path
from tolteca.datamodels.io.toltec import NcFileIO
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from tollan.utils.log import init_log, get_logger
from findpeaks import findpeaks
from astropy.table import vstack
import lmfit


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


def find_peak_S21_fp(swp, exclude_edge=5e3):
    fs = swp.frequency.to_value(u.Hz)
    S21 = swp.S21.to_value(u.adu)
    # this does the peak finding in inverted logS21, normalized to lowest point
    S21p = -np.log10(np.abs(S21))
    S21p = S21p - np.min(S21p)

    # pad max value at side otherwise the function does not work for some
    # reason
    m = S21p.max()
    S21p[0] = m
    S21p[-1] = m
    lookahead = 10
    fp = findpeaks(method='peakdetect', lookahead=lookahead, interpolate=None)
    results = fp.fit(S21p)
    d = results['df']
    results_i = d['x'][d['peak']]
    # fp.plot()
    results = fs[results_i]
    m = (results > fs.min() + exclude_edge) & (results < fs.max() - exclude_edge)
    results = results[m] << u.Hz
    return results


def make_tone_list(ref_data, sweep_data, debug_plot=False):
    logger = get_logger()
    swp_ref = ref_data['swp']
    swp = sweep_data.read()
    # extract d21 spectrum
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
    #if debug_plot:
    if False:
        fig, axes = plt.subplots(
                1, 2,
                figsize=(10, 4),
                sharex='col',
                constrained_layout=True,
                )
        fig.suptitle(f'{obsnum=} {ref_obsnum=} {nw=}')
        ax = axes[0]
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
    # apply the shift to vna output for making tone list
    kmt = ref_data['kmp'].table
    fr_ref = kmt['fr'].quantity
    # print(fr_ref)
    f_init = fr_ref + shift
    kmt['f_init'] = f_init
    fs = swp.frequency
    fs_min = fs.min(axis=1)[:, None]
    fs_max = fs.max(axis=1)[:, None]
    m_inc= (fs_min <= f_init[None, :]) & (fs_max >= f_init[None, :])
    n_tones = fs.shape[0]
    idx_inc = [np.where(m_inc[i])[0] for i in range(n_tones)]
    fr_ref_inc = [fr_ref[i] for i in idx_inc]
    fs_inc = [f_init[i] for i in idx_inc]

    # run find peak on sweep data
    fp_inc = [find_peak_S21_fp(swp.get_sweep(i)) for i in range(n_tones)]

    # for each channel, make the final list of tones based on
    # fs_inc and fp_inc
    # this build the kmp table for each channel.
    kmt_init = list()
    peak_search_lim_nfwhm = 0.75
    used_as_main = set()
    for i in range(n_tones):
        di = idx_inc[i]
        kk = kmt[di].copy()
        kk['is_main_det'] = False
        # print(kk)
        # need to figure out the main detector in this channel
        # we record all detector idx that have been used before
        for j, dd in enumerate(di):
            if dd in used_as_main:
                kk['is_main_det'][j] = False
                # print(f'set {j} {dd} false')
            else:
                kk['is_main_det'][j] = True
                # print(f'set {j} {dd} true')
                used_as_main.add(dd)
                break
            # print(kk)
            # import pdb
            # pdb.set_trace()
        # kk['is_main_det'] = (i == idx_inc[i])
        # sort by fr for checking merging state
        if len(kk) > 1:
            kk.sort('f_init')
            # allow no merging by limiting the shift to 1/3 of the gap
            df_init = np.diff(kk['f_init'].quantity) * 0.3
        ff_init = kk['f_init'].quantity
        fr_init = list()
        for j, e in enumerate(kk):
            ff0 = ff_init[j]  # the shifted f_ref
            ff_peaks = fp_inc[i]  # the found peaks
            # get the closest peak
            if len(ff_peaks) > 0:
                ff1 = ff_peaks[np.argmin(np.abs(ff_peaks - ff0))]
                # check if it is too far away
                fwhm = kk['fr'].quantity[j] / e['Qr']
                dfmax = fwhm * peak_search_lim_nfwhm
                # also this should not change the ordering,
                # import pdb
                # pdb.set_trace()
                if len(kk) > 1:
                    # need to check spacing to make sure the peak
                    # does not mess the ordering
                    dfmax0 = dfmax
                    if j == 0:
                        dfmax = min(df_init[0], dfmax)
                    if j == len(kk) - 1:
                        dfmax = min(df_init[-1], dfmax)
                    else:
                        dfmax = min(df_init[j-1], df_init[j], dfmax)
                    if dfmax != dfmax0:
                        logger.debug("update dfmax to preserve ordering.")
                if np.abs(ff1 - ff0) < dfmax:
                    # accept this peak
                    _fr_init = ff1
                else:
                    # accept the shifted value
                    _fr_init = ff0
            else:
                _fr_init = ff0
            fr_init.append(_fr_init)
        # update the table
        kk['fr_init'] = fr_init
        kmt_init.append(kk)
    if debug_plot:
        nrows = 5
        ncols = 5
        fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(15, 15),
                constrained_layout=True,
                )
        fig.suptitle(f'{obsnum=} {ref_obsnum=} {nw=}')
        axes = axes.ravel()
        for i, ax in enumerate(axes):
            i = i + 100
            xx = fs[i].to_value(u.MHz)
            yy = np.log(np.abs(swp.S21[i].value))
            ax.plot(xx, yy)
            for ff_ref in fr_ref_inc[i]:
                ax.axvline(ff_ref.to_value(u.MHz), color='C1', label=f'{ref_obsnum}', linestyle=':')
            for ff in fs_inc[i]:
                ax.axvline(ff.to_value(u.MHz), color='C0', label=f'{ref_obsnum} shifted', linestyle=':')
            for ff_peak in fp_inc[i]:
                ax.axvline(ff_peak.to_value(u.MHz), color='black', label=f'{obsnum} find peak', linestyle='--')
            for ff_fr_init in kmt_init[i]['fr_init'].quantity:
                ax.axvline(ff_fr_init.to_value(u.MHz), color='red', label='fr_init', linestyle='-')
            if ax is axes[0]:
                ax.legend()
        plt.show()
    return locals()


def fit_sweep(swp, kmt_init, debug_plot=False):
    # run model fitting to extract parameters for each tone
    n_tones = len(kmt_init)

    kmt = list()
    fit_S21 = list()
    for i in range(n_tones):
        x = swp.frequency[i].to_value(u.Hz)
        y = swp.S21[i].to_value(u.adu)
        s = swp.uncertainty[i].quantity.to_value(u.adu)
        ctx = _fit_sweep(fs=x, S21=y, S21_unc=s, kmt_init=kmt_init[i])
        fit_S21.append(ctx['fit_S21'])
        kmt.append(ctx['kmt_main'])
    kmt = vstack(kmt)
    if debug_plot:
        nrows = 5
        ncols = 5
        fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(15, 15),
                constrained_layout=True,
                )
        axes = axes.ravel()
        for i, ax in enumerate(axes):
            i = i + 100
            fs = swp.frequency[i].to_value(u.MHz)
            S21 = swp.S21[i].value
            S21m = fit_S21[i]
            ax.plot(S21.real, S21.imag, marker='x', linestyle='none', color='C0')
            ax.plot(S21m.real, S21m.imag, linestyle='-', color='black')
        plt.show()

    # check distance. use f_init if offset is large
    m_large_offset = np.abs(kmt['fr'] - kmt['f_init']) > (kmt['f_init'] / kmt['Qr_init'])
    print(f"number of detector with m_large_offset: {m_large_offset.sum()}")
    kmt['f_out'] = kmt['fr']
    kmt[m_large_offset]['f_out'] = kmt['f_init'][m_large_offset]
    # TODO this might break when we vary the number of tones
    kmt['f_in'] = kmt['fp']
    # strip out the units
    for c in kmt.colnames:
        kmt[c].unit = None
    print(kmt)
    # usecols
    cols = 'f_out f_in flag fp Qr Qc fr A normI normQ slopeI slopeQ interceptI interceptQ'.split()
    kmt = kmt[cols]
    return locals()


@njit
def _kids_model(f, fr, Qr, g0, g1, fp, k0, k1, m0, m1):
    x = (f - fr) / fr
    r = 0.5 / Qr
    X = r + 1.j * x
    Sp = 0.5 / X
    g = g0 + 1.j * g1
    k = k0 + 1.j * k1
    m = m0 + 1.j * m1
    return g * Sp + k * (f - fp) + m


def _fit_sweep(fs, S21, S21_unc, kmt_init):
    fp = kmt_init['fp'].quantity.to_value(u.Hz)[0]

    def kids_model(f, fr, Qr, g0, g1, k0, k1, m0, m1):
        return _kids_model(f, fr, Qr, g0, g1, fp, k0, k1, m0, m1)

    class ResonatorModel(lmfit.model.Model):
        __doc__ = "resonator model" + lmfit.models.COMMON_INIT_DOC

        def __init__(self, *args, **kwargs):
            super().__init__(kids_model, *args, **kwargs)
            self.set_param_hint('Qr', min=0)  # enforce Q is positive

    mdl = None
    # build the composite model
    f_init_kw = dict()
    fr_init = kmt_init['fr_init'].quantity.to_value(u.Hz)
    # print(kmt_init)
    window_data = list()
    for i, e in enumerate(kmt_init):
        prefix = f'r{i}_'
        if mdl is None:
            mdl = ResonatorModel(prefix=prefix)
        else:
            mdl = mdl + ResonatorModel(prefix=prefix) 
        ff = f_init_kw[prefix + 'fr'] = fr_init[i]
        qq = f_init_kw[prefix + 'Qr'] = kmt_init['Qr'][i]
        f_init_kw[prefix + 'g0'] = kmt_init['normI'][i]
        f_init_kw[prefix + 'g1'] = kmt_init['normQ'][i]
        f_init_kw[prefix + 'k0'] = 0
        f_init_kw[prefix + 'k1'] = 0
        # guess m0 and m1 values from data
        f_init_kw[prefix + 'm0'] = S21.real[0]
        f_init_kw[prefix + 'm1'] = S21.imag[0]
        # create weight data for this model with lorentz profile
        xx = 2 * qq * (fs - ff) / ff
        ss = 1 + xx ** 2
        window_data.append(ss)
    # add in quadrature the window data
    window_data = np.sqrt(np.sum(window_data, axis=0))
    # scale it to minimum of 1
    window_data = window_data / window_data.min()
    # plt.plot(window_data)
    # plt.show()
    # print(mdl)
    resonator = mdl
    params = resonator.make_params(**f_init_kw)
    weights = (1 / S21_unc) *  (1 / window_data)
    # weights = (1 / S21_unc)
    # print(weights)
    result = resonator.fit(
        S21, params=params,
        weights=weights,
        f=fs,
        # verbose=True
        )
    # print(result.fit_report() + '\n')
    # result.params.pretty_print()
    fit_S21 = resonator.eval(params=result.params, f=fs)
    # result.plot()
    # plt.show()
    # import pdb
    # pdb.set_trace()
    # update kmt
    kmt = kmt_init.copy()
    for i, e in enumerate(kmt):
        prefix = f'r{i}_'
        kmt['fr'][i] = result.params[prefix + 'fr'].value
        kmt['Qr'][i] = result.params[prefix + 'Qr'].value
        kmt['normI'][i] = result.params[prefix + 'g0'].value
        kmt['normQ'][i] = result.params[prefix + 'g1'].value
        kmt['slopeI'][i] = result.params[prefix + 'k0'].value
        kmt['slopeQ'][i] = result.params[prefix + 'k1'].value
        kmt['interceptI'][i] = result.params[prefix + 'm0'].value
        kmt['interceptQ'][i] = result.params[prefix + 'm1'].value
    kmt['Qr_init'] = kmt_init['Qr']
    kmt_main = kmt[kmt['is_main_det']]
    assert len(kmt_main) == 1
    return locals()


if __name__ == "__main__":
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
    # this generates the per channel list of model init params
    ctx = make_tone_list(
        ref_data=ref_data, sweep_data=sweep_data, debug_plot=option.debug_plot)

    ctx = fit_sweep(swp=ctx['swp'], kmt_init=ctx['kmt_init'], debug_plot=option.debug_plot)

    kmt = ctx['kmt']
    output_dir = option.output_dir or ref_file.parent
    output_file = output_dir.joinpath(sweep_file.stem + '.txt')
    kmt.meta['Header.Toltec.ObsNum'] = sweep_data.meta['obsnum']
    kmt.meta['Header.Toltec.SubObsNum'] = sweep_data.meta['subobsnum']
    kmt.meta['Header.Toltec.ScanNum'] = sweep_data.meta['scannum']
    kmt.write(output_file, overwrite=True, format='ascii.ecsv')
