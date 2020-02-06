#! /usr/bin/env python

# Author:
#   Zhiyuan Ma
#
# Contributor(s):
#   Zhiyuan Ma, Giles Novak, Grant Wilson
#
# History:
#   2020/02/05 Zhiyuan Ma:
#       - Allow pickling the generated data.
#       - Correct the quoted \kappa values.
#         Previously they were (mistakenly) set to show \kappa^2.
#       - Implement calculation of \kappa for given noise degradation.
#       - Make the heatmaps of threshold temperatures.
#   2020/02/02 Zhiyuan Ma:
#       - Handle all three TolTEC bands.
#       - Include kappa values in legend.
#   2020/01/28 Zhiyuan Ma:
#       - First staged.

"""This recipe makes use of the `KidsSimulator` to investigate
the impact of the half wave plate to the observed noise.

"""


from astropy import units as u
from astropy import log
from tolteca.kidsutils.kidsmodel.simulator import KidsSimulator
import numpy as np
from scipy import signal
import itertools
from astropy.visualization import quantity_support
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import fsolve
from tollan.utils.log import timeit, init_log
import concurrent
import psutil


init_log(level='DEBUG')


def simulate_hwp(Qr, readout_noise, hwp_var_temp, band='T1'):
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

    # hwp defs
    f_hwp = 2. * u.Hz
    two_omega = 2. * 2 * np.pi * u.rad * f_hwp
    four_omega = 2. * two_omega
    var_factor = hwp_var_temp / bkg_temp

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

    # create simulated timestream
    hwp_signal = 0.5 * var_factor * background * np.sin(two_omega * time)
    tsm_ps = background + hwp_signal
    tsm_rs, tsm_xs, tsm_iqs = sim.probe_p(tsm_ps)

    # add some noise in I and Q
    delta_is = np.random.normal(0, readout_noise, tsm_iqs.shape)
    delta_qs = np.random.normal(0, readout_noise, tsm_iqs.shape)

    tsm_iqs_out = tsm_iqs.real + delta_is + 1.j * (tsm_iqs.imag + delta_qs)
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

    psd_x_level = np.mean(psd_xs_out[1][_psd_f_mask])

    exclude = [
           "time", "swp_iqs", "swp_xs", "hwp_signal",
           "sim", "tsm_iqs", "tsm_rxs_out", "tsm_iqs_out",
           "tsm_ps", "tsm_rs", "tsm_xs", "delta_is", "delta_qs",
           "psd_rs", "psd_xs", "psd_rs_out", "psd_xs_out",
           'exclude',
            ]
    locals_ = locals()
    result = {
            k: v for k, v in locals_.items() if k not in exclude}
    return result


def make_tabular_legend(
        ax, handles, *cols,
        colnames=None, colwidths=5, **kwargs):

    print(handles)
    print(cols)
    if set([len(handles)]) != set(map(len, cols)):
        raise ValueError("all columns need to have the same size")

    if colnames is not None and len(colnames) != len(cols):
        raise ValueError("size of colnames need to match that of the cols")
    if isinstance(colwidths, int):
        colwidths = [colwidths] * len(handles)

    dummy_patch = matplotlib.patches.Rectangle(
            (1, 1), 1, 1,
            fill=False, edgecolor='none', visible=False)
    handles = [dummy_patch, ] + list(handles)
    labels = []
    if colnames is not None:
        label_row = []
        for colwidth, colname in zip(colwidths, colnames):
            if isinstance(colname, tuple):
                name, width = colname
            else:
                name = colname
                width = len(colname)
            print(name, width, colwidth)
            label_row.append(' ' * int(width) + name)
        labels.append(' '.join(label_row))
    for i, row in enumerate(zip(*cols)):
        label_row = []
        for j, value in enumerate(row):
            fmt = f'{{:>{colwidths[j]}s}}'
            label_row.append(fmt.format(str(value)))
        labels.append(' '.join(label_row))

    return ax.legend(handles, labels, **kwargs)


def save_or_show(fig, filepath,
                 bbox_inches='tight',
                 **kwargs):
    '''Save figure or show  plot, depending on
    the last sys.argv'''
    argv = sys.argv[1:]
    if not argv:
        save = False
    else:
        try:
            s = int(argv[-1])
            save = True if s == 1 else False
        except ValueError:
            if argv[-1].lower() in ['true', 'save']:
                save = True
            elif argv[-1].lower() in ['plot', ]:
                save = False
            else:
                save = False
    if save:
        fig.savefig(
            filepath,
            dpi='figure',
            format=Path(filepath).suffix.lstrip('.'), **kwargs)
        log.info('figure saved: {0}'.format(filepath))
    else:
        plt.show()


HwpTrend = type("HwpTrend", (object, ), dict())
HwpConstraint = type("HwpConstraint", (object, ), dict())


def worker(args):
    i, j, k, args, kwargs = args
    return i, j, k, simulate_hwp(*args, **kwargs)


def make_data(band, Qrs, readout_noises, var_temps):
    hwp_stats = np.empty(
            (len(Qrs), len(readout_noises), len(var_temps)),
            dtype=object)

    max_workers = psutil.cpu_count(logical=False)
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers) as executor:

        for i, j, k, s in executor.map(worker, (
                (i, j, k, (Qr, readout_noise, var_temp), dict(band=band))
                for (i, Qr), (j, readout_noise), (k, var_temp) in
                itertools.product(
                        enumerate(Qrs),
                        enumerate(readout_noises),
                        enumerate(var_temps)
                ))):
            hwp_stats[i, j, k] = s
    del executor

    psd_phot = hwp_stats[0, 0, 0]['psd_phot']
    noise_impact_threshs = [1.1, 2 ** 0.5]

    # reduction on var_temp
    hwp_trends = np.empty(
            (len(Qrs), len(readout_noises)), dtype=object)
    for i, j in itertools.product(*map(range, hwp_stats.shape[:2])):
        h = hwp_trends[i, j] = HwpTrend()
        h.Qr = hwp_stats[i, j][0]['Qr']
        h.readout_noise = hwp_stats[i, j][0]['readout_noise']
        h.var_temps = [
                s['hwp_var_temp'].to('K').value for s in hwp_stats[i, j]
                ] * u.K
        h.psd_x_levels_raw = np.asanyarray(
                [s['psd_x_level'] for s in hwp_stats[i, j]])
        h.psd_x_levels = gaussian_filter1d(h.psd_x_levels_raw, 5)
        h.kappas = np.sqrt(psd_phot / h.psd_x_levels)
        h.noise_impacts = (
                (1. + 1. / h.kappas ** 2) / (1. + 1. / h.kappas[0] ** 2)
                ) ** 0.5
        assert h.noise_impacts[0] == 1.
        h.noise_impact_threshs = noise_impact_threshs
        h.noise_thresh_temps = []
        for thresh in h.noise_impact_threshs:
            def func(x):
                return np.interp(
                        x, h.var_temps.to('K').value,
                        h.noise_impacts) - thresh
            h.noise_thresh_temps.append(fsolve(func, 1.) * u.K)
    # reduction on noise thresh
    hwp_constraints = np.empty(
            (len(noise_impact_threshs), ), dtype=object)
    for k in range(len(hwp_constraints)):
        hc = hwp_constraints[k] = HwpConstraint()
        hc.Qrs = Qrs
        hc.readout_noises = readout_noises
        hc.noise_impact_thresh = noise_impact_threshs[k]
        hc.noise_thresh_temps = np.full(
                hwp_trends.shape, np.nan) * u.K
        for i, j in itertools.product(*map(range, hwp_trends.shape)):
            hc.noise_thresh_temps[i, j] = \
                    hwp_trends[i, j].noise_thresh_temps[k]
    # only keep the reduced data
    try:
        del func
        del hwp_stats
    except UnboundLocalError:
        pass
    return locals()


def make_plot(data):

    Qrs = data['Qrs']
    readout_noises = data['readout_noises']
    var_temps = data['var_temps']
    band = data['band']
    psd_phot = data['psd_phot']
    hwp_trends = data['hwp_trends']
    noise_impact_threshs = data['noise_impact_threshs']
    hwp_constraints = data['hwp_constraints']

    # make some plots

    with quantity_support():

        # PSD vs dT_HWP for all combination of Qrs and readout_noises
        fig = plt.figure(constrained_layout=True, figsize=(20, 12))
        # fig.set_tight_layout(True)
        gs = gridspec.GridSpec(
                ncols=3, nrows=2, figure=fig,
                width_ratios=[3, 2, 0.1])
        ax = fig.add_subplot(gs[0, 0])
        bx = fig.add_subplot(gs[1, 0])
        cx = fig.add_subplot(gs[0, 1])
        dx = fig.add_subplot(gs[1, 1])
        ex = fig.add_subplot(gs[:, 2])

        for x in (ax, bx):
            x.set_xlim(-1, 3.5)
            x.set_xlabel(r"$\Delta T_{HWP}$ (K)")
        for x in (cx, dx):
            # ax.set_aspect('equal')
            x.set_xlabel(r"$\sigma_{readout}$")
            x.set_ylabel("$Q_r$")
        ax.set_yscale("log")
        ax.set_ylabel("PSD x ($Hz^{-1}$)")
        bx.set_ylabel("$NEP / NEP_0$")
        bx.set_ylim(1, 1.5)
        cxes = (cx, dx)

        leg_handles = []
        leg_Qrs = []
        leg_readout_noises = []
        leg_factors = []  # the temps
        leg_temps = []
        leg_show_temps_mask = slice(None, None, len(var_temps) // 5)
        plot_Qrs = np.array([1e4, 1.5e4, 2e4])
        plot_readout_noises = np.array([10, 20, 40])
        i_plot = []
        j_plot = []
        for v in zip(plot_Qrs, plot_readout_noises):
            for vv, vs, ii, in zip(v, (Qrs, readout_noises), (i_plot, j_plot)):
                i = np.where(np.abs(vv - vs) < 0.001 * vv)[0]
                if len(i) == 1:
                    ii.append(i[0])
        print(i_plot)
        print(j_plot)

        for i, j in itertools.product(*map(range, hwp_trends.shape)):
            h = hwp_trends[i, j]
            if i in i_plot and j in j_plot:
                leg_Qrs.append(f'{h.Qr / 1e3:.0f}k')
                leg_readout_noises.append(f'{h.readout_noise:.0f}')
                color = f"C{i_plot.index(i)}"
                marker = ['o', 'x', 'D', '^', '.', 'v'][
                        j_plot.index(j) % len(j_plot)]
                leg_handles.append(ax.plot(
                    h.var_temps, h.psd_x_levels_raw,
                    color=color,
                    marker=marker,
                    markersize=3, linestyle='none')[0])
                ax.plot(
                        h.var_temps, h.psd_x_levels,
                        color=color, linewidth=2)
                bx.plot(
                        h.var_temps[::5], h.noise_impacts[::5],
                        color=color, marker=marker, markersize=5)
                bx.plot(
                        h.noise_thresh_temps, h.noise_impact_threshs,
                        color=color, marker=marker, markersize=10)
                leg_factors.append([
                    f'{f:.2f}' for f in h.kappas[leg_show_temps_mask]])
                leg_temps.append(list(map(lambda x: f"{x:.1f}K", (
                        hwp_constraints[k].noise_thresh_temps.to(
                            'K').value[i, j]
                        for k in range(len(noise_impact_threshs))
                        ))))
        leg_labels = [
                ('$Q_r$', 0),
                (r'$\sigma_{readout}$', 0),
                ] + [
                (f"$\\kappa_{{{t:g}}}$", 0.3 * i)
                for i, t in enumerate(hwp_trends[0, 0].var_temps[
                    leg_show_temps_mask])] + [
                (f"${{\\delta}}T_{{+{f - 1:.1%}}}$".replace('%', r'\%'), 0)
                for f in hwp_trends[0, 0].noise_impact_threshs
                        ]
        colwidths = [3, 3] + [4, ] * len(leg_factors[0]) + [5, ] * len(
                    leg_temps[0])
        make_tabular_legend(
                ax, leg_handles, leg_Qrs, leg_readout_noises,
                *zip(*leg_factors), *zip(*leg_temps),
                colnames=leg_labels,
                colwidths=colwidths,
                prop={'family': 'monospace', 'size': 'small'},
                )

        for i, blip_factor in enumerate((1, 4, 9, 16)):
            ax.axhline(
                    psd_phot / blip_factor,
                    color='#444444',
                    # label=f'BLIP={psd_phot:.2e}',
                    linewidth=(2 - i) if 2 - i > 0 else 1,
                    )
            if blip_factor == 1:
                text = "BLIP"
            else:
                text = f'1/{blip_factor:d} BLIP'
            ax.annotate(
                    text,
                    xy=(0, psd_phot / blip_factor),
                    xycoords=(ax.transAxes, ax.transData),
                    xytext=(2, 2),
                    textcoords='offset points',
                    horizontalalignment='left',
                    verticalalignment='bottom'
                    )
        from matplotlib.offsetbox import AnchoredText
        ax.add_artist(AnchoredText(
                "TolTEC {}".format(
                    dict(T1='1.1mm', T2='1.4mm', T3='2.0mm')[band]
                    ),
                prop=dict(size=15), frameon=False,
                loc='lower left',
                ))

        for k, (noise_impact_span, color) in enumerate(zip(zip(
                [1.] + noise_impact_threshs[:-1],
                noise_impact_threshs,
                ),
                ['#66ffcc', '#ffcc66']
                )):
            bx.axhspan(*noise_impact_span, color=color, alpha=0.5)
            bx.annotate(
                    f"{noise_impact_span[-1] - 1.:.2%} noise penalty",
                    xy=(0, noise_impact_span[-1]),
                    xycoords=(bx.transAxes, bx.transData),
                    xytext=(2, 2),
                    textcoords='offset points',
                    horizontalalignment='left',
                    verticalalignment='bottom'
                    )
            hc = hwp_constraints[k]
            leg_temps.append(list(map(lambda x: f"{x:.1f}K", np.ravel(
                hc.noise_thresh_temps.to('K').value).tolist())))

            def _make_edges(x):
                x = np.asanyarray(x)
                middle = list(0.5 * (x[:-1] + x[1:]))
                edges = np.array(
                    [x[0] * 2 - middle[0]] + middle + [
                     x[-1] * 2 - middle[-1]
                    ])
                return edges
            im = cxes[k].pcolormesh(
                    _make_edges(hc.readout_noises),
                    _make_edges(hc.Qrs),
                    hc.noise_thresh_temps.to('K').value,
                    vmin=0.5, vmax=2.5)
            if k == 0:
                fig.colorbar(im, cax=ex)
                ex.set_ylabel(r"Threshold of ${\delta}T$ (K)")
            cxes[k].add_artist(AnchoredText(
                f"{noise_impact_span[-1] - 1.:.2%} noise penalty",
                prop=dict(size=15), frameon=False,
                loc='upper left',
                ))

        save_or_show(fig, f"simu_hwp_noise_{band}.png")


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
        Qrs = np.arange(8e3, 2.21e4, 1e3)
        # readout_noises = np.array([10, 20, 40])
        readout_noises = np.arange(5, 51, 5)
        var_temps = np.arange(0., 4.0, 0.05) * u.K
        data_args = (band, Qrs, readout_noises, var_temps)
    if len(sys.argv) >= 3:
        save_data = sys.argv[2]
    else:
        save_data = None
    main(data_args, save_data=save_data)
