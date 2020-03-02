#! /usr/bin/env python

# Author:
#   Zhiyuan Ma

"""This recipe implements the "autodrive" procedure to determine the best
drive attenuations for the KIDs.

"""

from tolteca.recipes import get_logger
from tolteca.fs.toltec import ToltecDataset
from tollan.utils.cli.multi_action_argparser import \
        MultiActionArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from tollan.utils.mpl import save_or_show
from tollan.utils.slice import parse_slice
from pathlib import Path
import itertools
# from kneed import KneeLocator


def load_autodrive_data(dataset):
    """Given a dataset, try load all the necessary data to run the autodrive.

    Parameters
    ----------
    dataset: ToltecDataset
        The input dataset.

    """

    logger = get_logger()
    logger.debug(f"load autodrive data from: {dataset}")

    # split the dataset for tunes and reduced files
    targs = dataset.select('kindstr=="targsweep"')
    calibs = dataset.select('fileext=="txt"')
    # Read the sweep object from the file IO object. A TolTEC tune file
    # contains multipe sweep blocks, and here we read the last one using the
    # `sweeploc` method.
    targs.load_data(
            lambda fo: fo.sweeploc(index=-1)[:].read())
    join_keys = ['nwid', 'obsid', 'subobsid', 'scanid']
    targs = targs.left_join(
            calibs.load_data(lambda fo: fo),
            join_keys, [('data_obj', 'mdl_obj'), ], )
    # join the mdls to swps
    logger.debug(f"targs: {targs}")
    return targs


def find_best_a_naive(a, y, n_flat, thresh):
    """A simple brutal force finder of the best atten value.

    The algorithm goes through the values in decreasing
    order and looks for the first point hat has y higher by
    some factor.
    """
    logger = get_logger()
    # sort in descending order so that the flat
    # section is at start. the a
    a = a[~np.isnan(a)]
    if len(a) < n_flat:
        logger.debug("unable to find best a: not enough data")
        return np.nan
    i = np.argsort(a)[::-1]
    aa = a[i]
    yy = y[i]
    ii = n_flat // 2  # the running index
    y0 = np.nanmean(yy[:ii])
    while ii < len(aa):
        if yy[ii] > y0 * thresh:
            result = aa[ii]
            break
        ii += 1
    else:
        logger.debug("unable to find best a: no more value to traverse.")
        result = np.nan
    return result


def autodrive(
        targs, toneloc=None, plot=True, output_ref_atten=None, output=None):
    logger = get_logger()
    swps = targs.data_objs
    mdls = targs['mdl_obj']
    for _, (swp, mdl) in enumerate(zip(swps, mdls)):
        swp.mdl = mdl
        swp.iqs_mdl = mdl.model(swp.fs)
        swp.iqs_derot = swp.mdl.model.derotate(swp.iqs, swp.fs)
        swp.adiqs_derot = np.abs(swp.diqs_df(swp.iqs_derot, swp.fs, smooth=0))

    if toneloc is None or output is not None:
        toneloc = slice(None)

    if isinstance(toneloc, slice):
        toneloc = slice(*toneloc.indices(swps[0].iqs.shape[0]))
        tis = list(range(toneloc.start, toneloc.stop, toneloc.step))
    else:
        tis = list(toneloc)
    logger.debug(f"tones: {toneloc}")

    n_tis = len(tis)
    n_swps = len(swps)

    logger.debug(f"n_tis={n_tis} n_swps={n_swps}")

    a_drvs = np.empty((n_tis, n_swps), dtype=np.double)
    a_tots = np.empty((n_tis, n_swps), dtype=np.double)
    Qrs = np.empty((n_tis, n_swps), dtype=np.double)
    frs = np.empty((n_tis, n_swps), dtype=np.double)
    adiqs_derot_max = np.empty((n_tis, n_swps), dtype=np.double)
    data_extra = np.empty((n_tis, n_swps), dtype=object)

    for (i, ti), (j, swp) in itertools.product(
            enumerate(tis), enumerate(swps)):
        a_drvs[i, j] = a_drv = swp.meta['atten_out']
        a_tots[i, j] = a_tot = swp.meta['atten_in'] + swp.meta['atten_out']
        # compute abs of derotated d21 and find the maximum
        fs = swp.fs[ti, :].to('Hz').value
        iqs = swp.iqs[ti, :]
        iqs_mdl = swp.iqs_mdl[ti, :]
        iqs_derot = swp.iqs_derot[ti, :]
        # adiqs_derot = swp.adiqs_derot[ti, :]
        adiqs_derot = np.abs(np.gradient(iqs_derot, fs))
        # we only find the max within one fwhm of the resonance
        Qrs[i, j] = Qr = swp.mdl.model.Qr[ti]
        frs[i, j] = fr = swp.mdl.model.fr[ti]
        fwhm = fr / swp.mdl.model.Qr[ti]
        flim = (fr - fwhm), (fr + fwhm)
        fm = (fs >= flim[0]) & (fs < flim[1])
        if np.any(fm):
            adiqs_derot_max[i, j] = np.max(adiqs_derot[fm])
        else:
            adiqs_derot_max[i, j] = np.nan
        data_extra[i, j] = (
                fs, iqs, iqs_mdl, iqs_derot, adiqs_derot,
                Qr, fr, fwhm, flim, fm
                )

    # reduce the swps to get the best a_drv
    # this is done by looking from a "flat" section
    # in the adiqs_derot_max vs a_drvs, and find the
    # minimum a_drv that have adiqs_derot_max higher by
    # some factor of the "flat" section.
    a_drv_bests = np.empty((n_tis, ), dtype=np.double)
    for i, ti in enumerate(tis):
        a_drv_bests[i] = find_best_a_naive(
                a_drvs[i], adiqs_derot_max[i],
                n_flat=5,
                thresh=1.2,
                )
        # kneedle = KneeLocator(
        #         a_drvs, adiqs_derot_max,
        #         S=1.0,
        #         curve='convex',
        #         direction='decreasing',
        #         interp_method='polynomial')
        # a_drv_bests[i] = a_drv_best = knee
    # compute autodrive cor table
    if output_ref_atten is None:
        # use the ref atten at 90 percentile
        p = 0.1
        a_drv_ref = np.quantile(a_drv_bests[~np.isnan(a_drv_bests)], p)
        logger.debug(
                f"a_drv_ref={a_drv_ref} (p={p})")
    else:
        a_drv_ref = output_ref_atten
    ampcors = np.ones((n_tis, ), dtype=np.double)
    for i, ti in enumerate(tis):
        a_drv = a_drv_bests[ti]
        ampcor = 10 ** ((a_drv_ref - a_drv) / 20.)
        if ampcor > 1.0 or np.isnan(ampcor):
            ampcor = 1.0
        ampcors[i] = ampcor
    if output is not None:
        with open(output, 'w') as fo:
            # fo.write(f"# a_drv_ref={a_drv_ref} dB\n")
            for ampcor in ampcors:
                fo.write(f"{ampcor}\n")
    ampcor_extra = np.mean(ampcors[i])
    # ampcor_extra = np.sqrt(np.sum(ampcors[i] ** 2)) / len(ampcors)
    a_drv_extra = -20. * np.log10(ampcor_extra)
    a_drv_common = a_drv_ref + a_drv_extra
    logger.debug(
            f"a_drv_common={a_drv_common} (extra={a_drv_extra})")

    if not plot:
        return
    # make plots of the failed cases
    # maximum number of panels is 10
    i_failed = np.where(np.isnan(a_drv_bests))[0]
    i_good = np.where(~np.isnan(a_drv_bests))[0]
    n_failed = len(i_failed)
    logger.debug(f"n_tones with failed autodrive: {n_failed}")

    i_plot = list(i_failed) + list(i_good[:n_failed])

    panel_size = (20, 6)
    n_rows = len(i_plot)
    if n_rows > 5:
        n_rows = 5
    fig, axes = plt.subplots(
            n_rows, 6,
            figsize=(panel_size[0], panel_size[1] * n_rows),
            dpi=50,
            constrained_layout=True,
            )
    for ii in range(n_rows):
        i = i_plot[ii]
        ax = axes[ii, 0]  # (I, Q)
        bx = axes[ii, 1]  # (I, Q) derotated
        cx = axes[ii, 2]  # adiqs
        dx = axes[ii, 3]  # max(adiqs) vs a_drv
        ex = axes[ii, 4]  # Qr vs a_drv
        fx = axes[ii, 5]  # fr vs a_drv
        for j in range(n_swps):
            id_ = swps[j].meta['obsid']
            a_drv = a_drvs[i, j]
            a_tot = a_tots[i, j]
            fs, iqs, iqs_mdl, iqs_derot, adiqs_derot, \
                Qr, fr, fwhm, flim, fm = data_extra[i, j]
            ax.plot(
                iqs.real, iqs.imag,
                label=f'${id_}\\ A_{{drive}}={a_drv},'
                      f'\\ A_{{tot}}={a_tot}$',
                marker='o',
                )
            ax.plot(
                    iqs_mdl.real, iqs_mdl.imag,
                    )
            bx.plot(
                    iqs_derot.real, iqs_derot.imag,
                    )
            cx.plot(fs, adiqs_derot)
            cx.axvline(flim[0], color='#cccccc')
            cx.axvline(flim[1], color='#cccccc')
        # trend
        assert tis[i] == i
        dx.plot(
                a_drvs[i],
                adiqs_derot_max[i],
                label=f'tone id={tis[i]}')
        a_drv_best = a_drv_bests[i]
        dx.axvline(
                a_drv_best,
                color='#000000',
                linewidth=3,
                label=f'$A_{{drv, best}}={a_drv_best}$')
        dx.legend(loc='upper right')
        ex.plot(
                a_drvs[i], Qrs[i])
        fx.plot(
                a_drvs[i], frs[i])

    fig2, axes = plt.subplots(2, 1)
    ax, bx = axes
    ax.hist(a_drv_bests)
    bx.plot([frs[ti, 0] for ti in tis],  ampcors)
    fig2.show()
    save_or_show(
        fig, 'fig_autodrive.png', window_type='scrollable',
        size=(panel_size[0], panel_size[1])
        )


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]

    maap = MultiActionArgumentParser(
            description="Run autodrive."
            )
    act_index = maap.add_action_parser(
            'index',
            help="Build an index table that have tones"
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
            help="The output filename",
            )
    act_index.add_argument(
            "-f", "--overwrite",
            action='store_true',
            help="If set, overwrite the existing index file",
            )

    @act_index.parser_action
    def index_action(option):
        output = Path(option.output)
        if output.exists() and not option.overwrite:
            raise RuntimeError(
                    f"output file {output} exists, use -f to overwrite")
        # This function is called when `index` is specified in the cmd
        # Collect the dataset from the command line arguments
        dataset = ToltecDataset.from_files(*option.files)
        # Apply any selection filtering
        if option.select:
            dataset = dataset.select(option.select).open_files()
        else:
            dataset = dataset.open_files()
        # Run the tone matching algo.
        dataset = load_autodrive_data(dataset)
        # Dump the results.
        # dataset.write_index_table(
        #         option.output, overwrite=option.overwrite,
        #         format='ascii.commented_header')
        dataset.dump(option.output)

    act_run = maap.add_action_parser(
            'run',
            help="Run autodrive"
            )
    act_run.add_argument(
            "-s", "--select",
            metavar="COND",
            help='A selection predicate, e.g.,:'
            '"(obsid>8900) & (nwid==3) & (fileext=="nc")"',
            )
    act_run.add_argument(
            '-i', '--input',
            help='The input filename, created by the "index" action.'
            )
    act_run.add_argument(
            '-t', '--tones',
            default=':10',
            help='The tones to examine. Default is the first 10.'
            )
    act_run.add_argument(
            '-r', '--output_ref_atten',
            default=None,
            help='The output reference driving atten.'
            )
    act_run.add_argument(
            '-p', '--plot',
            action='store_true',
            help='Make plots.'
            )
    act_run.add_argument(
            '-o', '--output',
            help='The output amplitude correction filename.'
            )

    @act_run.parser_action
    def run_action(option):
        input_ = Path(option.input)
        dataset = ToltecDataset.load(input_)
        if option.select is not None:
            dataset = dataset.select(option.select)
        autodrive(
                dataset,
                toneloc=parse_slice(option.tones),
                output_ref_atten=option.output_ref_atten,
                output=option.output,
                plot=option.plot)

    option = maap.parse_args(args)
    maap.bootstrap_actions(option)
