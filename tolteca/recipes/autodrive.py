#! /usr/bin/env python

# Author:
#   Zhiyuan Ma

"""This recipe implements the "autodrive" procedure to determine the best
drive attenuations for the KIDs.

"""

from tolteca.recipes import get_logger
from tolteca.datamodels.toltec import BasicObsDataset
from tollan.utils.cli.multi_action_argparser import \
        MultiActionArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from tollan.utils.mpl import save_or_show
from tollan.utils.slice import parse_slice
from pathlib import Path
import itertools
# from kneed import KneeLocator
from astropy.table import Table, join
import astropy.units as u


def load_autodrive_data(dataset):
    """Given a dataset, try load all the necessary data to run the autodrive.

    Parameters
    ----------
    dataset: BasicObsDataset
        The input dataset.

    """

    logger = get_logger()
    logger.debug(f"load autodrive data from: {dataset}")

    # split the dataset for tunes and reduced files
    targs = dataset.select('fileext=="nc" & filesuffix=="targsweep"')
    calibs = dataset.select('fileext=="txt" & filesuffix=="targsweep"')
    # Read the sweep object from the file IO object. A TolTEC tune file
    # contains multipe sweep blocks, and here we read the last one using the
    # `sweeploc` method.
    targs['data_obj'] = targs.read_all()
    calibs['mdl_obj'] = calibs.read_all()
    join_keys = ['roachid', 'obsnum', 'subobsnum', 'scannum']

    tbl = join(
            targs.index_table, calibs.index_table,
            keys=join_keys,
            join_type='right')
    targs = BasicObsDataset(index_table=tbl, bod_list=tbl['data_obj'])
    # targs = targs.right_join(
    #         calibs.load_data(lambda fo: fo),
    #         join_keys, [('data_obj', 'mdl_obj'), ], )
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


def find_best_a_naive2(a, y, n_init, n_sigma):
    """A simple brutal force finder of the best atten value.

    The algorithm goes through the values in decreasing
    order and looks for the first point hat has y higher by
    some factor of the standard deviation.
    """
    logger = get_logger()
    # sort in descending order so that the flat
    # section is at start. the a
    a = a[~np.isnan(y)]
    if len(a) < n_init:
        logger.debug("unable to find best a: not enough data")
        return np.nan
    i = np.argsort(a)[::-1]
    aa = a[i]
    yy = y[i]
    ii = n_init  # the running index
    y0 = np.mean(yy[:ii])
    sig = np.std(yy[:ii])

    while ii < len(aa):
        if yy[ii] > y0 + n_sigma * sig:
            result = aa[ii]
            break
        ii += 1
    else:
        logger.debug("unable to find best a: no more value to traverse.")
        result = np.nan
    return result


def find_best_a_naive3(a, y, n_flat, thresh, n_accept):
    """A simple brutal force finder of the best atten value.

    The algorithm goes through the values in decreasing
    order and looks for the first group of points that have y higher by
    some factor.
    """
    logger = get_logger()
    # sort in descending order so that the flat
    # section is at start. the a
    a = a[~np.isnan(y)]
    if len(a) < n_flat:
        logger.debug("unable to find best a: not enough data")
        return np.nan
    i = np.argsort(a)[::-1]
    aa = a[i]
    yy = y[i]
    y0 = np.mean(yy[:n_flat])

    cands = np.where(yy > y0 * thresh)[0]

    for idx in cands:
        if all(i in cands for i in range(idx, idx + n_accept)):
            return aa[idx]
    else:
        logger.debug("unable to find best a: no more value to traverse.")
        return np.nan


def autodrive(
        targs, toneloc=None, plot=True, output_ref_atten=None, output=None):
    logger = get_logger()
    swps = targs['data_obj']
    mdls = targs['mdl_obj']
    for _, (swp, mdl) in enumerate(zip(swps, mdls)):
        swp.mdl = mdl
        swp.S21_mdl = mdl.model(swp.frequency)
        # TODO sort out the unit stuff here
        swp.S21_derot = swp.mdl.model.derotate(
                swp.S21.to_value(u.adu), swp.frequency).to_value(u.dimensionless_unscaled) << u.adu
        swp.adiqs_derot = np.abs(swp.diqs_df(
            swp.S21_derot, swp.frequency, smooth=0))

    if toneloc is None or output is not None:
        toneloc = slice(None)

    if isinstance(toneloc, slice):
        toneloc = slice(*toneloc.indices(swps[0].S21.shape[0]))
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
        a_drvs[i, j] = a_drv = swp.meta['atten_drive']
        a_tots[i, j] = a_tot = swp.meta['atten_sense'] + swp.meta['atten_drive']
        # compute abs of derotated d21 and find the maximum
        fs = swp.frequency[ti, :].to('Hz').value
        iqs = swp.S21[ti, :]
        iqs_mdl = swp.S21_mdl[ti, :]
        iqs_derot = swp.S21_derot[ti, :].to_value(u.adu)
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
    # finder = find_best_a_naive2
    # finder_kwargs = dict(
    #         n_init=min(10, len(a_drvs) / 5),
    #         n_sigma=3,
    #         )
    finder = find_best_a_naive3
    finder_kwargs = dict(
            n_flat=min(10, len(a_drvs) // 5),
            thresh=1.2,
            n_accept=3,
            )
    logger.debug(f"finder={finder} kwargs={finder_kwargs}")
    for i, ti in enumerate(tis):
        # a_drv_bests[i] = find_best_a_naive(
        #         a_drvs[i], adiqs_derot_max[i],
        #         n_flat=5,
        #         thresh=1.2,
        #         )
        a_drv_bests[i] = finder(
                a_drvs[i], adiqs_derot_max[i],
                **finder_kwargs
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
        a_drv = a_drv_bests[i]
        ampcor = 10 ** ((a_drv_ref - a_drv) / 20.)
        if ampcor > 1.0 or np.isnan(ampcor):
            ampcor = 1.0
        ampcors[i] = ampcor
    if output is not None:
        output = Path(output)
        with open(output, 'w') as fo:
            # fo.write(f"# a_drv_ref={a_drv_ref} dB\n")
            for ampcor in ampcors:
                fo.write(f"{ampcor}\n")
        with open(output.with_suffix('.a_drv'), 'w') as fo:
            for a in a_drv_bests:
                fo.write(f"{a}\n")

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
    # i_good = np.where(~np.isnan(a_drv_bests))[0]
    n_failed = len(i_failed)
    logger.debug(f"n_tones with failed autodrive: {n_failed}")

    # i_plot = list(i_failed) + list(i_good[:n_failed])
    i_plot = list(range(10))

    panel_size = (20, 6)
    n_rows = len(i_plot)
    if n_rows > 10:
        n_rows = 10
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
            id_ = swps[j].meta['obsnum']
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
        # assert tis[i] == i
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
    bx.plot([frs[i, 0] for i, ti in enumerate(tis)],  ampcors)
    fig2.show()
    save_or_show(
        fig, 'fig_autodrive.png', window_type='scrollable',
        size=(panel_size[0], panel_size[1])
        )


def main(args):
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
            '"(obsnum>8900) & (roachid==3) & (fileext=="nc")"',
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
    def index_action(option, **kwargs):
        output = Path(option.output)
        if output.exists() and not option.overwrite:
            raise RuntimeError(
                    f"output file {output} exists, use -f to overwrite")
        # This function is called when `index` is specified in the cmd
        # Collect the dataset from the command line arguments
        dataset = BasicObsDataset.from_files(option.files)
        # Apply any selection filtering
        if option.select:
            dataset = dataset.select(option.select)
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
            '"(obsnum>8900) & (roachid==3) & (fileext=="nc")"',
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
    def run_action(option, **kwargs):
        input_ = Path(option.input)
        dataset = BasicObsDataset.load(input_)
        if option.select is not None:
            dataset = dataset.select(option.select)
        autodrive(
                dataset,
                toneloc=parse_slice(option.tones),
                output_ref_atten=option.output_ref_atten,
                output=option.output,
                plot=option.plot)

    act_collect = maap.add_action_parser(
            'collect',
            help="Collect autodrive result"
            )
    act_collect.add_argument(
            "files",
            metavar="FILE",
            nargs='+',
            help="The files to use",
            )
    act_collect.add_argument(
            '-o', '--output',
            help='The output drive attenution file.'
            )
    act_collect.add_argument(
            "-f", "--overwrite",
            action='store_true',
            help="If set, overwrite the existing index file",
            )
    act_collect.add_argument(
            '-p', '--plot',
            action='store_true',
            help='Make plots.'
            )
    act_collect.add_argument(
            '--save_plot',
            action='store_true',
            help='Save plot.'
            )

    @act_collect.parser_action
    def collect_action(option, **kwargs):
        output = Path(option.output)
        if output.exists() and not option.overwrite:
            raise RuntimeError(
                    f"output file {output} exists, use -f to overwrite")

        ps = [0, 0.05, 0.1, 0.5, 0.9, 0.95, 1]
        result = []
        colnames = ['nw', 'filename', ] + [f'p{p * 100:.0f}' for p in ps]
        # sort the files
        files = []
        for filepath in option.files:
            nw = int(Path(filepath).stem.split('_')[0].replace('toltec', ''))
            files.append((nw, filepath))
        files = sorted(files, key=lambda a: a[0])
        data = []
        for nw, filepath in files:
            tbl = Table.read(filepath, format='ascii.no_header')
            a = tbl['col1']
            a = a[~np.isnan(a)]
            result.append(
                    [nw, Path(filepath).stem, ] + [np.quantile(a, p) for p in ps])
            data.append(a)
        result = Table(rows=result, names=colnames)
        result.write(option.output, format='ascii.commented_header')
        cmd_file = Path(option.output).with_suffix('.cmd').as_posix()
        with open(cmd_file, 'w') as fo:
            args = []
            for r in result:
                args.append(f"-AttenDriveCmd[{r['nw']}]")
                args.append(f"{r['p95']}")
            fo.write('set ToltecBackend {}\n'.format(' '.join(args)))
        if option.plot:
            fig, axes = plt.subplots(
                    len(data), 1, sharex=True, squeeze=False)
            axes = np.ravel(axes)
            for i, a in enumerate(data):
                ax = axes[i]
                ax.hist(a)
                for j, p in enumerate(ps):
                    ax.axvline(result[i][-len(ps) + j], color=f"C{j}")
                ax.set_ylabel(f'NW {result["nw"][i]}')
            axes[-1].set_xlabel(f'Best driving atten. (dB)')
            axes[0].set_title(f"{result[0]['filename']}")
            save_or_show(
                    fig, Path(option.output).with_suffix('.png').as_posix(),
                    save=option.save_plot)

    option = maap.parse_args(args)
    maap.bootstrap_actions(option)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
