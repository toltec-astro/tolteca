#! /usr/bin/env python

# Author:
#   Zhiyuan Ma
#
# History:
#   2020/02/15 Zhiyuan Ma:
#       - First staged.

"""
This recipe makes use of the `KidsSimulator` to investigate
the total PSD noise as function of Qr and the readout noise.

"""


from astropy import units as u
from kidsproc.kidsmodel.simulator import KidsSimulator
import numpy as np
from scipy import signal
import itertools
from astropy.visualization import quantity_support
# import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from tollan.utils.log import timeit
import concurrent
import psutil
from tolteca.recipes.simu_hwp_noise import save_or_show


def simulate_qr(Qr, readout_noise, band='T1'):
    """Make simulated timestream and calculate some useful quantities
    of the HWP under some assumed parameters.

    Parameters
    ----------
    Qr: float
        Qr of the detector.

    readout_noise: float
        Readout noise specified in the same unit as Qr.

    hwp_var_temp: float
        The HWP temperature variation, peak-to-peak.

    band: str
        The TolTEC band to usage.

    Returns
    -------
    dict
        Dict that contains all stated local variables.
    """

    # Some globals generated from Grant's Mapping-Speed-Calculator code
    # The below is for elev 60 deg, atm 50 quantiles
    # if band == 'T1':
    #     background = 11.86 * u.pW
    #     bkg_temp = 11.40 * u.K
    #     responsivity = 5.794e-5 / u.pW
    #     psd_phot = ((102.61 * u.aW * responsivity) ** 2 * 2.).decompose()
    # elif band == 'T2':
    #     background = 8.23 * u.pW
    #     bkg_temp = 10.84 * u.K
    #     responsivity = 1.1e-4 / u.pW
    #     psd_phot = ((79.38 * u.aW * responsivity) ** 2 * 2.).decompose()
    # elif band == 'T3':
    #     background = 5.69 * u.pW
    #     bkg_temp = 8.92 * u.K
    #     responsivity = 1.1e-4 / u.pW
    #     psd_phot = ((56.97 * u.aW * responsivity) ** 2 * 2.).decompose()
    # The below is for elev 45 deg, atm 25 quantiles
    if band == 'T1':
        background = 10.01 * u.pW
        bkg_temp = 9.64 * u.K
        responsivity = 5.794e-5 / u.pW
        psd_phot = ((90.86 * u.aW * responsivity) ** 2 * 2.).decompose()
    elif band == 'T2':
        background = 7.15 * u.pW
        bkg_temp = 9.43 * u.K
        responsivity = 1.1e-4 / u.pW
        psd_phot = ((71.58 * u.aW * responsivity) ** 2 * 2.).decompose()
    elif band == 'T3':
        background = 5.29 * u.pW
        bkg_temp = 8.34 * u.K
        responsivity = 1.1e-4 / u.pW
        psd_phot = ((53.96 * u.aW * responsivity) ** 2 * 2.).decompose()
    else:
        raise ValueError(f'unknown band {band}. Select among T1, T2, and T3.')
    fsmp = 488. * u.Hz
    tlen = 60. * u.s

    # reduction params
    psd_flim = (6, 10.)
    psd_size = 1024 * 8

    sim = KidsSimulator(
            fr=None, Qr=Qr,
            background=background, responsivity=responsivity)

    # create a resonance circle model
    swp_xs, swp_iqs = sim.sweep_x(n_steps=176, n_fwhms=5)

    # create a time grid
    time = np.arange(0., tlen.to(u.s).value, (1. / fsmp).to(u.s).value) * u.s

    # phot_snr = 100.
    phot_snr = np.inf
    # create simulated timestream
    delta_ps = np.random.normal(
            0, background.to('pW').value / phot_snr, time.shape) * u.pW
    # tsm_ps = np.full(time.shape, background.to('pW').value) * u.pW
    tsm_ps = background + delta_ps
    # tsm_rs, tsm_xs, tsm_iqs = sim.probe_p(tsm_ps)
    tsm_rs, tsm_xs, _ = sim.probe_p(tsm_ps)

    delta_rs = np.random.normal(
            0, 0.005 * np.mean(tsm_rs), time.shape)

    tsm_rs = tsm_rs + delta_rs
    tsm_iqs = sim._rx2iqcomplex(tsm_rs + 1.j * tsm_xs)
    # add some noise in I and Q
    delta_is = np.random.normal(0, readout_noise, tsm_iqs.shape)
    delta_qs = np.random.normal(0, readout_noise, tsm_iqs.shape)

    tsm_aiqs = np.abs(tsm_iqs)
    tsm_iqs_out = tsm_iqs.real + delta_is * tsm_aiqs + 1.j * (
            tsm_iqs.imag + delta_qs * tsm_aiqs)
    tsm_rxs_out = sim.solve_x(tsm_iqs_out)

    # psd of the time stream
    psd_rs = signal.welch(tsm_rs, fsmp.to('Hz').value, nperseg=psd_size)
    psd_xs = signal.welch(tsm_xs, fsmp.to('Hz').value, nperseg=psd_size)
    psd_rs_out = signal.welch(
            tsm_rxs_out.real, fsmp.to('Hz').value,
            nperseg=psd_size)
    psd_xs_out = signal.welch(
                tsm_rxs_out.imag, fsmp.to('Hz').value,
                nperseg=psd_size)

    _psd_f_mask = (psd_rs_out[0] >= psd_flim[0]) & (
                   psd_rs_out[0] < psd_flim[1])

    psd_r_level = np.mean(psd_rs_out[1][_psd_f_mask])
    psd_x_level = np.mean(psd_xs_out[1][_psd_f_mask])

    exclude = [
           "time", "swp_iqs", "swp_xs",
           "sim", "tsm_iqs", "tsm_rxs_out", "tsm_iqs_out",
           "tsm_ps", "tsm_rs", "tsm_xs", "delta_is", "delta_qs",
           "psd_rs", "psd_xs", "psd_rs_out", "psd_xs_out",
           'exclude',
            ]
    locals_ = locals()
    result = {
            k: v for k, v in locals_.items() if k not in exclude}
    return result


def worker(args):
    i, j, args, kwargs = args
    return i, j, simulate_qr(*args, **kwargs)


class DetectorProps(object):
    pass


def make_data(band, Qrs, readout_noises):
    detector_stats = np.empty(
            (len(Qrs), len(readout_noises)),
            dtype=object)
    psd_r_levels = np.full(detector_stats.shape, np.nan)
    psd_x_levels = np.full(detector_stats.shape, np.nan)

    max_workers = psutil.cpu_count(logical=False)
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers) as executor:

        for i, j, s in executor.map(worker, (
                (i, j, (Qr, readout_noise), dict(band=band))
                for (i, Qr), (j, readout_noise) in
                itertools.product(
                        enumerate(Qrs),
                        enumerate(readout_noises)
                ))):
            detector_stats[i, j] = s
            psd_x_levels[i, j] = s['psd_x_level']
            psd_r_levels[i, j] = s['psd_r_level']
    del executor

    psd_phot = detector_stats[0, 0]['psd_phot']
    kappas = np.sqrt(psd_phot / psd_x_levels)
    kappa_threshs = np.arange(10)

    exclude = [
           "detector_stats",
           'exclude',
            ]
    locals_ = locals()
    result = {
            k: v for k, v in locals_.items() if k not in exclude}
    return result


def make_plot(data):

    Qrs = data['Qrs']
    readout_noises = data['readout_noises']
    band = data['band']
    psd_phot = data['psd_phot']
    kappa_threshs = np.asanyarray(data['kappa_threshs'])
    psd_r_levels = data['psd_r_levels']
    psd_x_levels = data['psd_x_levels']
    kappas = data['kappas']

    print(psd_r_levels)
    print(psd_x_levels)
    # make some plots

    with quantity_support():

        # PSD vs dT_HWP for all combination of Qrs and readout_noises
        fig = plt.figure(constrained_layout=True, figsize=(16, 12))
        # fig.set_tight_layout(True)
        gs = gridspec.GridSpec(
                ncols=3, nrows=3,
                figure=fig, width_ratios=[50, 1, 50])
        ax = fig.add_subplot(gs[0, 0])
        bx = fig.add_subplot(gs[1, 0])
        cx = fig.add_subplot(gs[2, 0])

        cax = fig.add_subplot(gs[0, 1])
        cbx = fig.add_subplot(gs[1, 1])
        ccx = fig.add_subplot(gs[2, 1])

        dx = fig.add_subplot(gs[0, 2])
        ex = fig.add_subplot(gs[1, 2])
        fx = fig.add_subplot(gs[2, 2])

        for x in (ax, bx, cx):
            # ax.set_aspect('equal')
            x.set_xlabel(r"$\sigma_{readout}$")
            x.set_ylabel("$Q_r$")
        for x in (dx, ex, fx):
            x.set_xlabel("$Q_r$")

        def _make_edges(x):
            x = np.asanyarray(x)
            middle = list(0.5 * (x[:-1] + x[1:]))
            edges = np.array(
                [x[0] * 2 - middle[0]] + middle + [
                 x[-1] * 2 - middle[-1]
                ])
            return edges

        def pcolor_with_contour(
                ax, x, y, z, l,
                cax, cax_label, contour_kwargs):
            im = ax.pcolormesh(
                    _make_edges(x),
                    _make_edges(y),
                    np.log10(z),
                    # vmin=psd_phot / kappa_threshs[-1] ** 2,
                    # vmax=psd_phot / kappa_threshs[0] ** 2,
                    )
            cp = ax.contour(
                    x, y, z, levels=l
                    )
            ax.clabel(cp, inline=True, colors='black', **contour_kwargs)
            ax.figure.colorbar(im, cax=cax)
            cax.set_ylabel(cax_label)

        for xx, z, l, cax, cax_label, ckw in (
                (
                    ax,
                    psd_r_levels,
                    sorted(psd_phot / kappa_threshs ** 2),
                    cax,
                    r"$Log\ PSD_{r}$",
                    dict(fmt='%.2e'),
                    ),
                (
                    bx,
                    psd_x_levels,
                    sorted(psd_phot / kappa_threshs ** 2),
                    cbx,
                    r"$Log\ PSD_{x}$",
                    dict(fmt='%.2e'),
                    ),
                ):
            pcolor_with_contour(
                    xx, readout_noises, Qrs, z, l,
                    cax, cax_label, ckw)

        from scipy.ndimage.filters import gaussian_filter
        kappas_s = gaussian_filter(kappas, sigma=3)
        ccp = cx.contourf(
                readout_noises,
                Qrs,
                kappas_s,
                levels=kappa_threshs,
                )
        cx.clabel(ccp, inline=True, colors='black')
        fig.colorbar(ccp, cax=ccx)

        def reset_ax(ax):
            anno_kwargs = dict(
                    xycoords=(ax.transAxes, ax.transData),
                    xytext=(2, 2),
                    textcoords='offset points',
                    horizontalalignment='left',
                    verticalalignment='bottom'
                    )
            hline_kwargs = dict(color='#cccccc', linestyle='--')
            if ax is dx or ax is ex:
                ax.set_yscale('log')
                ax.set_ylim(1e-18, 1e-15)
                for k in kappa_threshs:
                    ax.axhline(psd_phot / k ** 2, **hline_kwargs)
                    ax.annotate(
                            f"1 / {k ** 2:g} BLIP",
                            xy=(0, psd_phot / k ** 2),
                            **anno_kwargs
                            )
                if ax is dx:
                    ax.set_ylabel(r'$PSD\ r$')
                else:
                    ax.set_ylabel(r'$PSD\ x$')
            elif ax is fx:
                ax.set_ylabel(r'$\kappa$')
                for k in kappa_threshs:
                    ax.axhline(k, **hline_kwargs)
                    # ax.annotate(
                    #         f"$\\kappa={k:g}$",
                    #         xy=(0, k),
                    #         **anno_kwargs
                    #         )
            return

        for x in (dx, ex, fx):
            reset_ax(x)

        cache = dict()

        from astropy.modeling.models import custom_model  # , fix_inputs
        from astropy.modeling import fitting

        curve_fit = fitting.LevMarLSQFitter()

        # @custom_model
        # def PsdModel(
        #         Qr, readout_noise,
        #         norm=1e-5,
        #         idx_Qr=-4.,
        #         idx_readout_noise=2.):
        #     return norm * Qr ** idx_Qr * readout_noise ** idx_readout_noise

        def onclick(event, **kwargs):
            px = kwargs.pop('inaxes', event.inaxes)
            if px is None:
                return
            if px is ax:
                px = dx
                data = psd_r_levels
                data1 = psd_x_levels
            elif px is bx:
                px = ex
                data = psd_x_levels
                data1 = psd_r_levels
            elif px is cx:
                px = fx
                data = kappas
                data1 = None
            else:
                return
            from matplotlib.backend_bases import MouseButton
            if event.button is MouseButton.RIGHT:
                px.clear()
                reset_ax(px)
            else:
                def _idx(a, x):
                    return np.argmin(np.abs(a - x))

                j = _idx(readout_noises, event.xdata)
                y = data[:, j]

                if (j, px) in cache:
                    return
                lines, = p = px.plot(
                            Qrs, y,
                            label=f'$\\sigma_{{readout}}={event.xdata:g}$')
                if data1 is not None:
                    px.plot(
                            Qrs, data1[:, j],
                            color=lines.get_color(), linestyle='--')
                # fit Qr model
                if px in (dx, ex):
                    @custom_model
                    def PsdQrModel(
                            Qr,
                            norm=1e-5,
                            idx_Qr=-2.,
                            ):
                        return norm * Qr ** idx_Qr * readout_noises[j] ** 2

                    model_init = PsdQrModel()
                    model = curve_fit(model_init, Qrs, data[:, j])
                    px.plot(
                            Qrs, model(Qrs),
                            color=lines.get_color(), linestyle=':',
                            label=f"$PSD = {model.norm.value:.2g}"
                                  f"Qr^{{{model.idx_Qr.value:.2f}}}$")
                cache[(j, px)] = p
                px.legend()
            px.figure.canvas.draw_idle()

        def _onclick_locked(event):
            for xx in (ax, bx, cx):
                onclick(event, inaxes=xx)
        # fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('button_press_event', _onclick_locked)
        save_or_show(fig, f"simu_qr_noise_{band}.png")


def main(data_args, save_data=None):
    if isinstance(data_args, str):
        with open(data_args, 'rb') as fo:
            data = pickle.load(fo)
    else:
        data = timeit(make_data)(*data_args)
        if save_data is not None:
            with open(save_data, 'wb') as fo:
                pickle.dump(data, fo)
    make_plot(data)


if __name__ == "__main__":
    import sys
    data_args = sys.argv[1]
    if data_args in ['T1', 'T2', 'T3']:
        band = data_args
        # Qrs = np.array([1e4, 1.5e4, 2.0e4])
        Qrs = np.arange(1e4, 8e4, 1e3)
        # readout_noises = np.array([10, 20, 40])
        # readout_noises = np.arange(5, 501, 5)
        readout_noises = np.arange(1e-3, 1e-2, 1e-3)
        data_args = (band, Qrs, readout_noises)
    if len(sys.argv) >= 3:
        save_data = sys.argv[2]
    else:
        save_data = None
    main(data_args, save_data=save_data)
