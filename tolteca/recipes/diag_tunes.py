#! /usr/bin/env python

# Author:
#   Zhiyuan Ma

"""This recipe makes use of the `tolteca` and `kidsproc` package to make
diagnostic plot for a collection of tune files.

The code requires an external tool `stilts` to match the tones in
different files. This code will try download it automatically if not
already installed. Please refer to http://www.star.bris.ac.uk/~mbt/stilts/
for more information.
"""

from tolteca.recipes import get_logger
from tolteca.fs.toltec import ToltecDataset
from tollan.utils.cli.multi_action_argparser import \
        MultiActionArgumentParser
from tollan.utils.wraps.stilts import ensure_stilts
import numpy as np
import functools
import matplotlib.pyplot as plt
from tollan.utils.mpl import save_or_show


def main():
    logger = get_logger()
    stilts_cmd = ensure_stilts()
    logger.debug(f'use stilts: "{stilts_cmd}"')


def _match_d21(d21_0, d21_1, roi=None, ax=None):
    """Given two D21 spectrum, use correlation to find the relative
    frequency shift between them.

    Parameters
    ----------
    d21_0, d21_1: D21
        The D21s to match.
    roi: callable, optional
        If given, this will be used to filter the frequency as ``fs =
        roi(fs)``.
    ax: matplotlib.Axes, optional
        If given, diagnostic plots are added to this axes.

    Returns
    -------
    float:
        The relative frequency shift.
    """
    (fs0, adiqs0, adiqscov0), (fs1, adiqs1, adiqscov1) = d21_0, d21_1
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
    shift = -dfs[cross_correlation.argmax()]
    if ax is not None:
        ax.plot(fs, adiqs0, color='C0', linestyle='-')
        ax.plot(fs, adiqs1, color='C1', linestyle='-')
        ax.plot(
                fs + shift,
                adiqs0,
                label=f'shift={shift/1e3:.0f}kHz',
                color='C3', linestyle='--',
                )
        ax.legend(loc='upper right')
    return shift


def d21_worker(args, **kwargs):
    i, swp = args
    return i, swp.d21(**kwargs)


def match_tones(tunes, pairing='diff', roi=None, fig_title=None):
    """Given a set of tune files, try find the relative shift of the
    resonance frequencies of all detectors among them.

    Parameters
    ----------
    tunes: ToltecDataset
        The tune files.

    pairing: choose from ['diff', 'first']
        The pairing method. `diff` will match neighbouring tunes, and
        `first` will match to the first one.

    roi: dict or object, optional
        The range of frequencies to match. If dict, the keys shall be query
        strings to the `tunes` dataset, and the value shall be a callable that
        takes the frequency array and compute a mask for it. If not dict, it is
        equivalent to applying this to all entries. If None, the entire
        frequency range is used.
    """

    logger = get_logger()
    logger.debug(f"match tunes: {tunes}")
    logger.debug(f"roi: {roi}")

    swps = tunes.load_data(
            lambda fo: fo.sweeploc(index=-1)[:].read()).data_objs

    d21_kwargs = dict(fstep=1000, flim=(4.0e8, 1.0e9), smooth=0)
    import psutil
    max_workers = psutil.cpu_count(logical=False)
    import concurrent
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers) as executor:
        for i, d21 in executor.map(functools.partial(
                d21_worker, **d21_kwargs), enumerate(swps)):
            swps[i]._d21 = d21
    tunes.index_table['data_objs'] = swps
    del executor

    _roi = dict()
    # resolve roi dict
    if roi is not None:
        if not isinstance(roi, dict):
            roi = dict('obsid>0', roi)
        for q, r in roi.items():
            e = tunes.select(q)
            if len(e) > 0:
                for ee in e:
                    if ee in _roi:
                        raise ValueError(
                                "each entry can only have one roi")
                    _roi[e.data_objs[e.index(ee)]] = roi
    else:
        roi = dict()

    # Read the sweep object from the file IO object. A TolTEC tune file
    # contains multipe sweep blocks, and here we read the last one using the
    # `sweeploc` method.

    panel_size = (6, 2)
    n_panels = len(swps) - 1
    fig, axes = plt.subplots(
            n_panels, 1,
            figsize=(panel_size[0], panel_size[1] * n_panels),
            sharex=True,
            constrained_layout=True,
            )
    if fig_title is not None:
        fig.subtitle(fig_title)

    if pairing == 'diff':
        pairs = zip(swps[:-1], swps[1:])
    elif pairing == 'first':
        pairs = [(swps[0], swps[i]) for i in range(1, len(swps))]
    else:
        raise ValueError('pairing has to be one of "diff" or "first".')

    shifts = dict()
    for i, (left, right) in enumerate(pairs):
        ax = np.ravel(axes)[i]
        shift = _match_d21(
                left.d21(**d21_kwargs),
                right.d21(**d21_kwargs),
                roi=roi.get(right), ax=ax)
        shifts[right] = (left, shift)
    # update the table
    tunes['tone_match_id_self'] = range(len(tunes))
    tunes['tone_match_fshift'] = [0., ] * len(tunes)
    tunes['tone_match_id_other'] = [0, ] * len(tunes)
    for i, e in enumerate(tunes):
        if i == 0:
            continue
        other_swp, shift = shifts[tunes.data_objs[i]]
        e['tone_match_fshift'] = shift
        e['tone_match_id_other'] = tunes[
                np.where(tunes.data_objs == other_swp)[0]][
                        'tone_match_id_self']
    save_or_show(
            fig, 'fig_tone_match.png',
            window_type='scrollable',
            size=(panel_size[0], panel_size[1])
            )
    logger.debug(f"tunes: {tunes}")
    return tunes


def plot_trend(swps):
    logger = get_logger()
    import matplotlib.pyplot as plt
    tones = list(range(10))
    fig, axes = plt.subplots(2, len(tones))
    atten_in = []
    atten_out = []
    atten_total = []
    d21max = []
    for swp in swps:
        atten_in.append(swp.meta['atten_in'])
        atten_out.append(swp.meta['atten_out'])
        atten_total.append(swp.meta['atten_in'] + swp.meta['atten_out'])
        jd21max = np.argmax(swp.adiqs_df, axis=-1)
        d21max.append([
            swp.adiqs[i, jd21max[i]]
            for i in range(swp.fs.shape[0])
            ])
        for i in range(axes.shape[-1]):
            ax = axes[0, i]
            ax.plot(
                swp.iqs[i, :].real,
                swp.iqs[i, :].imag,
                marker='.')
            ax = axes[1, i]
            ax.plot(swp.fs[i, :], swp.adiqs[i, :])
    fig, axes = plt.subplots(len(tones))
    for i in range(axes.shape[-1]):
        ax = axes[i]
        ax.plot(
            atten_out,
            [
                d21max[j][i]
                for j in range(len(swps))
                ])
    plt.show()

    logger.debug(f"swps: {swps}")
    return swps


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]

    maap = MultiActionArgumentParser(
            description="Diagnostics for a set of TolTEC KIDs tune files."
            )
    # Set up the `index` action.
    # The purpose of this is to collect a set of tune files and
    # try match the tones among them. This is necessary because
    # consecutive tune files does not necessarily result in
    # the same set of tune positions.
    # The end result of this action group is to dump an index
    # file that contains a list of tune files, such that neighbouring
    # ones have at least 10 tones found to be referring to the same
    # detector.
    act_index = maap.add_action_parser(
            'index',
            help="Build an index file that have tones"
            " matched for a set of tune files"
            )
    act_index.add_argument(
            "files",
            metavar="FILE",
            nargs='+',
            help="The files to use",
            )
    act_index.add_argument(
            "-s", "--select",
            metavar="COND",
            help='A selection predicate, e.g.,:'
            '"(obsid>8900) & (nwid==3) & (fileext=="nc")"',
            )
    act_index.add_argument(
            "-o", "--output",
            metavar="OUTPUT_FILE",
            required=True,
            help="The output index file",
            )
    act_index.add_argument(
            "-f", "--overwrite",
            action='store_true',
            help="If set, overwrite the existing index file",
            )

    @act_index.parser_action
    def index_action(option):
        # This function is called when `index` is specified in the cmd
        # Collect the dataset from the command line arguments
        dataset = ToltecDataset.from_files(*option.files).select(
                '(kindstr=="tune")'
                )
        # Apply any selection filtering
        if option.select:
            dataset = dataset.select(option.select).open_files()
        else:
            dataset = dataset.open_files()
        # Run the tone matching algo.
        dataset = match_tones(dataset, pairing='first')
        # Dump the result file.
        dataset.write_index_table(
                option.output, overwrite=option.overwrite,
                format='ascii.commented_header')

    act_plot = maap.add_action_parser(
            'plot',
            help="Make diagnostic plots"
            )
    act_plot.add_argument(
            'something', nargs='*'
            )

    @act_plot.parser_action
    def plot_action(option):
        print(option)

    option = maap.parse_args(args)
    maap.bootstrap_actions(option)
