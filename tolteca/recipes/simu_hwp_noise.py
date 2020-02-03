#! /usr/bin/env python

# Author:
#   Zhiyuan Ma
#
# Contributor(s):
#   Zhiyuan Ma, Giles Novak, Grant Wilson
#
# History:
#   2020/02/02 Zhiyuan Ma:
#      - Handl all three TolTEC bands.
#      - Include kappa values in legend.
#   2020/01/28 Zhiyuan Ma:
#      - First staged.

"""This recipe makes use of the `KidsSimulator` to investigate
the impact of the half wave plate to the observed noise.

"""


from astropy import units as u
from tolteca.kidsutils.kidsmodel.simulator import KidsSimulator
import numpy as np
from scipy import signal
import itertools
import matplotlib.patches


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

    # some globals
    if band == 'T1':
        background = 11.86 * u.pW
        bkg_temp = 11.40 * u.K
        responsivity = 5.794e-5 / u.pW
        psd_phot = ((102.61 * u.aW * responsivity) ** 2 * 2.).decompose()
    elif band == 'T2':
        background = 8.23 * u.pW
        bkg_temp = 10.84 * u.K
        responsivity = 1.1e-4 / u.pW
        psd_phot = ((79.38 * u.aW * responsivity) ** 2 * 2.).decompose()
    elif band == 'T3':
        background = 5.69 * u.pW
        bkg_temp = 8.92 * u.K
        responsivity = 1.1e-4 / u.pW
        psd_phot = ((56.97 * u.aW * responsivity) ** 2 * 2.).decompose()
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

    return locals()


def make_tabular_legend(
        ax, handles, *cols,
        colnames=None, colwidths=5, **kwargs):

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
            label_row.append(name + ' ' * (colwidth - width))
        labels.append(' '.join(label_row))
    for i, row in enumerate(zip(*cols)):
        label_row = []
        for j, value in enumerate(row):
            fmt = f'{{:{colwidths[j]}s}}'
            label_row.append(fmt.format(str(value)))
        labels.append(' '.join(label_row))

    return ax.legend(handles, labels, **kwargs)


def main():
    import sys

    band = sys.argv[1]

    # make a grid of hwp_simulations
    Qrs = [1e4, 1.5e4, 2.0e4]
    readout_noises = [10, 20, 40]
    var_temps = np.array([0., 1.0, 1.5, 2.0]) * u.K

    hwp_stats = np.empty(
            (len(Qrs), len(readout_noises), len(var_temps)),
            dtype=object)

    for (i, Qr), (j, readout_noise), (k, var_temp) in itertools.product(
            enumerate(Qrs), enumerate(readout_noises), enumerate(var_temps)):
        hwp_stats[i, j, k] = simulate_hwp(
                Qr, readout_noise, var_temp, band=band)
    psd_phot = hwp_stats[0, 0, 0]['psd_phot']

    # make some plots
    from astropy.visualization import quantity_support
    import matplotlib.pyplot as plt

    with quantity_support():

        # PSD vs dT_HWP for all combination of Qrs and readout_noises
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.set_tight_layout(True)

        leg_handles = []
        leg_Qrs = []
        leg_readout_noises = []
        leg_show_temps_mask = slice(0, -1)
        leg_factors = []  # the temps
        leg_labels = [
                ('$Q_r$', 4),
                (r'$\sigma_{readout}$', 4),
                ] + [
                (f"$\\kappa_{{{t:g}}}$", len(f'{t:g}'))
                for t in var_temps[leg_show_temps_mask]]
        for i, j in itertools.product(*map(range, hwp_stats.shape[:2])):
            Qr = hwp_stats[i, j][0]['Qr']
            readout_noise = hwp_stats[i, j][0]['readout_noise']
            leg_Qrs.append(f'{Qr / 1e3:.0f}k')
            leg_readout_noises.append(f'{readout_noise:.0f}')

            var_temps = [
                s['hwp_var_temp'].to('K').value for s in hwp_stats[i, j]] * u.K
            psd_x_levels = np.asanyarray(
                    [s['psd_x_level'] for s in hwp_stats[i, j]])
            color = f"C{i}"
            marker = ['o', 'x', 'D', '^', '.', 'v'][j]
            leg_handles.append(ax.plot(
                    var_temps, psd_x_levels,
                    color=color, marker=marker)[0])
            # psd_x_factors = (psd_x_levels / psd_x_levels[0])[
            #         leg_show_temps_mask]
            # phot_hwp_noise_facotor = (
            #         psd_phot + psd_x_levels[leg_show_temps_mask]) / (
            #         psd_phot + psd_x_levels[0])
            phot_noise_facotor = (
                    psd_phot / psd_x_levels[leg_show_temps_mask])
            leg_factors.append([f'{f:.2f}' for f in phot_noise_facotor])
            # for fi, f in enumerate(psd_x_factors):
            #     ax.annotate(
            #         f'{f:.1f}',
            #         xy=(var_temps[fi], psd_x_levels[fi]),
            #         xytext=(0, i - j),
            #         textcoords='offset points',
            #         )
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
        ax.set_yscale("log")
        ax.set_xlabel(r"$\Delta T_{HWP}$ (K)")
        ax.set_ylabel("PSD x ($Hz^{-1}$)")
        ax.set_xlim(-1, 3.5)
        make_tabular_legend(
                ax, leg_handles, leg_Qrs, leg_readout_noises,
                *zip(*leg_factors),
                colnames=leg_labels, colwidths=[4, 3, 6, 6, 6],
                prop={'family': 'monospace', 'size': 'small'},
                )
        from matplotlib.offsetbox import AnchoredText
        ax.add_artist(AnchoredText(
                "TolTEC {}".format(
                    dict(T1='1.1mm', T2='1.4mm', T3='2.0mm')[band]
                    ),
                prop=dict(size=15), frameon=False,
                loc='upper left',
                ))
    plt.show()


if __name__ == "__main__":
    main()
