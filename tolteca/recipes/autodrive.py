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
from kneed import KneeLocator


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


def autodrive(targs, toneloc=None, plot=True):
    logger = get_logger()
    swps = targs.data_objs
    mdls = targs['mdl_obj']
    for _, (swp, mdl) in enumerate(zip(swps, mdls)):
        swp.mdl = mdl
        # import pdb
        # pdb.set_trace()
        swp.iqs_mdl = mdl.model(swp.fs)
        swp.iqs_derot = swp.mdl.model.derotate(swp.iqs, swp.fs)

    logger.debug(f"swps: {swps}")

    tis = list(range(10))
    if toneloc is None:
        toneloc = slice(None)

    if isinstance(toneloc, slice):
        toneloc = slice(*toneloc.indices(swps[0].iqs.shape[0]))
        tis = list(range(toneloc.start, toneloc.stop, toneloc.step))
    else:
        tis = list(toneloc)
    logger.debug(f"tones: {toneloc}")

    panel_size = (24, 6)
    n_panels = len(tis)
    if n_panels > 10:
        n_panels = 10
    fig, axes = plt.subplots(
            n_panels, 4,
            figsize=(panel_size[0], panel_size[1] * n_panels),
            dpi=40,
            constrained_layout=True,
            )
    a_drv_bests = np.empty((len(tis), ), dtype=float)
    for i, ti in enumerate(tis):
        if i < n_panels:
            ax = axes[ti, 0]  # (I, Q)
            bx = axes[ti, 1]  # (I, Q) derotated
            cx = axes[ti, 2]  # adiqs
            dx = axes[ti, 3]  # max(adiqs) vs a_drv
        a_drvs = np.empty((len(swps), ), dtype=float)
        a_tots = np.empty((len(swps), ), dtype=float)
        adiqs_derot_max = np.empty((len(swps), ), dtype=float)
        for k, swp in enumerate(swps):
            a_drvs[k] = a_drv = swp.meta['atten_out']
            a_tots[k] = a_tot = swp.meta['atten_in'] + swp.meta['atten_out']
            id_ = f"{swp.meta['obsid']}"
            fs = swp.fs[ti, :].to('Hz').value
            iqs = swp.iqs[ti, :]
            iqs_mdl = swp.iqs_mdl[ti, :]
            iqs_derot = swp.iqs_derot[ti, :]
            adiqs_derot = np.abs(np.gradient(iqs_derot, fs))
            fr = swp.mdl.model.fr[ti]
            fwhm = fr / swp.mdl.model.Qr[ti]
            flim = (fr - fwhm), (fr + fwhm)
            fm = (fs >= flim[0]) & (fs < flim[1])
            if np.any(fm):
                adiqs_derot_max[k] = np.max(adiqs_derot[fm])
            else:
                adiqs_derot_max[k] = np.nan
            # import pdb
            # pdb.set_trace()
            if i < n_panels:
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
        try:
            kneedle = KneeLocator(
                    a_drvs, adiqs_derot_max,
                    S=1.0,
                    curve='convex',
                    direction='decreasing',
                    interp_method='polynomial')
            a_drv_bests[i] = a_drv_best = kneedle.knee
        except Exception:
            a_drv_bests[i] = a_drv_best = np.nan
        # trend
        if i < n_panels:
            dx.plot(a_drvs, adiqs_derot_max)
            dx.axvline(
                    a_drv_best,
                    color='#000000',
                    linewidth=3,
                    label=f'$A_{{drv, best}}={a_drv_best}$')
            ax.legend(loc='upper right')
            dx.legend(loc='upper right')
    fig2, ax = plt.subplots(1, 1)
    ax.hist(a_drv_bests)
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
            '-p', '--plot',
            action='store_true',
            help='Make plots.'
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
                plot=option.plot)

    option = maap.parse_args(args)
    maap.bootstrap_actions(option)
