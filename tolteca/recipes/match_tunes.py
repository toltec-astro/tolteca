#! /usr/bin/env python

# Author:
#   Zhiyuan Ma

"""This recipe helps match the tones in two tune files.

The code requires an external tool `stilts` to match the tones in
different files. This code will try download it automatically if not
already installed. Please refer to http://www.star.bris.ac.uk/~mbt/stilts/
for more information.
"""

from tolteca.recipes import get_logger
from tolteca.datamodels.toltec import BasicObsDataset
from tollan.utils.wraps.stilts import stilts_match1d
from tollan.utils.cli.multi_action_argparser import \
        MultiActionArgumentParser
import numpy as np
import functools
import matplotlib.pyplot as plt
from tollan.utils.mpl import save_or_show
from pathlib import Path
from astropy.table import vstack
import astropy.units as u


def _calc_d21_worker(args, **kwargs):
    i, swp = args
    return i, swp.d21(**kwargs)


def prepare_d21_data(dataset):
    """Prepare the dataset for tone match."""

    logger = get_logger()
    logger.debug(f"load tones data from: {dataset}")

    # split the dataset for tunes and reduced files
    targs = dataset.select('((filesuffix=="targsweep") | (filesuffix=="tune")) & (fileext=="nc")')
    calibs = dataset.select('fileext=="txt"')
    join_keys = ['nwid', 'obsnum', 'subobsnum', 'scannum']
    targs = targs.join(
            calibs,
            keys=join_keys, join_type='left',
            uniq_col_name='{col_name}{table_name}',
            table_names=['', '_reduced']
            )
    # join the mdls to swps
    logger.debug(f"targs: {targs}")

    # Compute the D21
    d21_kwargs = dict(fstep=500 << u.Hz, flim=(4.0e8 << u.Hz, 1.0e9 << u.Hz), smooth=2)
    swps = targs.read_all()
    for swp in swps:
        swp.make_unified(**d21_kwargs)
    targs.index_table['swp'] = swps
    return targs


def _match_d21(d21_0, d21_1, roi=None, return_locals=False):
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
    if return_locals:
        return shift, locals()
    return shift


def find_global_shift(
        dataset, pairing='diff', roi=None, fig_title=None, plot=False):
    """Given a set of tune files, try find the relative shift of the
    resonance frequencies of all detectors among them.

    Parameters
    ----------
    dataset: ToltecDataset
        The dataset to use.

    pairing: choose from ['diff', 'first']
        The pairing method. `diff` will match neighbouring tunes, and
        `first` will match to the first one.

    roi: dict or object, optional
        The range of frequencies to match. If dict, the keys shall be query
        strings to the dataset, and the value shall be a callable that takes
        the frequency array and compute a mask for it. If not dict, it is
        equivalent to applying this to all entries. If None, the entire
        frequency range is used.
    """

    logger = get_logger()
    logger.debug(f"find tone shift for dataset: {dataset}")
    logger.debug(f"roi: {roi}")

    if len(dataset) < 2:
        raise RuntimeError("dataset has to have at lease 2 entries.")

    # check that all entries are from the same network
    if len(np.unique(dataset['nwid'])) != 1:
        raise RuntimeError("dataset shall be from the same network.")

    swps = dataset.data_objs

    _roi = dict()
    # resolve roi dict
    if roi is not None:
        if not isinstance(roi, dict):
            roi = {'obsid>0': roi}
        for q, r in roi.items():
            e = dataset.select(q)
            if len(e) > 0:
                for ee in e:
                    key = ee['data_obj']
                    if key in _roi:
                        raise ValueError(
                                "each entry can only have one roi")
                    _roi[key] = r
    else:
        pass
    roi = _roi

    # Do the actuall matching
    if pairing == 'diff':
        pairs = list(zip(swps[:-1], swps[1:]))
    elif pairing == 'first':
        pairs = [(swps[0], swps[i]) for i in range(1, len(swps))]
    else:
        raise ValueError('pairing has to be one of "diff" or "first".')

    shifts = dict()
    shifts_data = dict()
    for i, (left, right) in enumerate(pairs):
        shift, shift_data = _match_d21(
                left._d21,
                right._d21,
                roi=roi.get(right), return_locals=True)
        shifts[right] = (left, shift)
        shifts_data[right] = (left, shift_data)
    # Update the table with found shift
    # The shift is defined as self = other + shift
    dataset['tone_match_idx_self'] = range(len(dataset))
    dataset['tone_match_global_shift'] = [0., ] * len(dataset)
    dataset['tone_match_idx_other'] = [0, ] * len(dataset)
    for i, e in enumerate(dataset):
        if i == 0:
            continue
        other_swp, shift = shifts[swps[i]]
        e['tone_match_global_shift'] = shift.to_value('Hz')
        e['tone_match_idx_other'] = dataset[
                np.where(dataset.data_objs == other_swp)[0]][
                        'tone_match_idx_self']

    # add a unique label for self and other
    def make_uid(e):
        return f"{e['nwid']}_{e['obsid']}_{e['subobsid']}_{e['scanid']}"

    dataset['tone_match_uid_self'] = [
            make_uid(
                dataset[e['tone_match_idx_self']]) for e in dataset]
    dataset['tone_match_uid_other'] = [
            make_uid(
                dataset[e['tone_match_idx_other']]) for e in dataset]

    if plot:
        n_panels = len(shifts_data)
        if n_panels > 2:
            panel_size = (6, 2)
            window_type = 'scrollable'
            fig_size = panel_size
        else:
            panel_size = (12, 4)
            window_type = 'default'
            fig_size = (panel_size[0], panel_size[1] * n_panels)
        fig, axes = plt.subplots(
                n_panels, 1,
                figsize=(panel_size[0], panel_size[1] * n_panels),
                sharex=True,
                constrained_layout=True,
                squeeze=False,
                )
        if fig_title is not None:
            fig.subtitle(fig_title)

        for i, (left, right) in enumerate(pairs):
            shift_data = shifts_data[right][1]
            ax = np.ravel(axes)[i]
            fs = shift_data['fs']
            adiqs0 = shift_data['adiqs0']
            adiqs1 = shift_data['adiqs1']
            shift = shift_data['shift']
            assert shift == shifts[right][1]
            # plot shift data
            ax.plot(fs, adiqs0, color='C0', linestyle='-')
            ax.plot(fs, adiqs1, color='C1', linestyle='-')
            ax.plot(
                    fs + shift,
                    adiqs0,
                    label=f'shift={shift/1e3:g}kHz',
                    color='C3', linestyle='--',
                    )
            ax.legend(loc='upper right')

        save_or_show(
                fig, plot,
                window_type=window_type,
                size=fig_size
                )
    logger.debug("found global shifts: {}".format(
        dataset[[
            'nwid', 'obsid', 'subobsid', 'scanid',
            'tone_match_idx_self',
            'tone_match_idx_other',
            'tone_match_global_shift']]
        ))
    return dataset


def match_tones(
        left, right, eps=2000., shift_from_right=0.,
        match_col='fr',
        join_type='inner'):
    """Return a table with tones matched.

    This function makes use the ``stilts`` utility.

    Parameters
    ----------
    left: astropy.Table
        The left model params table.
    right: astropy.Table
        The right model params table.
    eps: float
        The error to tolerate in Hz.
    shift_from_right: float
        The frequency shift to apply to the right.
    match_col: str
        The column to use for the match.
        Default is the resonance frequency ``fr``.
    join_type: str
        Join type to use for the output table.
    """
    # make match column
    col = 'match_tones_col'
    idx_col = 'idx'
    _left = left.copy()
    _right = right.copy()
    _left[col] = _left[match_col]
    _left[idx_col] = list(range(len(_left)))
    _right[col] = _right[match_col] + shift_from_right
    _right[idx_col] = list(range(len(_right)))
    join = {
            'left': 'all1',
            'right': 'all2',
            'inner': '1and2',
            'outer': '1or2'
            }[join_type]
    return stilts_match1d(_left, _right, col, eps, extra_args=[
        f'join={join}', f'find=best',
        ])


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]

    maap = MultiActionArgumentParser(
            description="Match tones in tune files."
            )
    act_index = maap.add_action_parser(
            'index',
            help="Build an index table that have precomputed D21."
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
        dataset = prepare_d21_data(dataset)
        # Dump the results.
        # dataset.write_index_table(
        #         option.output, overwrite=option.overwrite,
        #         format='ascii.commented_header')
        dataset.dump(option.output)

    act_run = maap.add_action_parser(
            'run',
            help="Run match"
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
            "-j", "--join_type",
            choices=['left', 'right', 'outer', 'inner'],
            default='inner',
            help="Join type to use for the matched result.",
            )
    act_run.add_argument(
            "-p", "--pairing",
            choices=['first', 'diff'],
            default='diff',
            help="Pairing method for making the match.",
            )
    act_run.add_argument(
            "-o", "--output",
            metavar="OUTPUT_FILE",
            required=True,
            help="The output filename with matched tones",
            )
    act_run.add_argument(
            "-f", "--overwrite",
            action='store_true',
            help="If set, overwrite the existing index file",
            )
    act_run.add_argument(
            "--plot",
            action='store_true',
            help="generate plot",
            )

    @act_run.parser_action
    def run_action(option):
        logger = get_logger()
        input_ = Path(option.input)
        dataset = ToltecDataset.load(input_)
        if option.select is not None:
            dataset = dataset.select(option.select)

        def _match(d):
            logger.debug(f"match tones for {d.meta['select_query']}")

            def roi(fs):
                return (fs > (469.5 << u.MHz)) & (fs < (472.5e6 << u.MHz))
            d = find_global_shift(d, roi=roi, plot=False)
            # match tones
            d['tone_match_matched_obj'] = [None, ] * len(d)
            d['tone_match_n_matched'] = [-1, ] * len(d)
            d['tone_match_sep_median'] = [np.nan, ] * len(d)
            d['matched_tones'] = [None, ] * len(d)
            for i, entry in enumerate(d):
                j0 = entry['tone_match_idx_self']
                j1 = entry['tone_match_idx_other']
                # self = other + shift
                shift = entry['tone_match_global_shift']
                if j0 == j1:
                    continue
                # get the left and right entry
                left = d[j0]
                right = d[j1]
                cal_left = left['mdl_obj'].table
                cal_right = right['mdl_obj'].table
                print(f'apply global shift {shift} Hz')
                matched = match_tones(
                        cal_left, cal_right, shift_from_right=shift,
                        join_type=option.join_type, eps=30000.)
                d['tone_match_matched_obj'][i] = matched
                d['tone_match_n_matched'][i] = len(matched)
                d['tone_match_sep_median'][i] = np.nanmedian(
                        matched['Separation'])
            return d

        join_keys = ['nwid', 'obsid', 'subobsid', 'scanid']
        dataset = dataset.left_join(ToltecDataset.vstack(
            map(_match, dataset.split('nwid'))),
            join_keys, cols=r'tone_match_.*')
        logger.debug(f"dataset: {dataset}")

        ds = list(filter(
            lambda d: all(d['tone_match_matched_obj']),
            dataset.split("obsid", 'subobsid', 'scanid')))

        # build final matched tone catalogs
        for d in ds:
            def _prep(i, t):
                t = t.copy()
                t['nwid'] = [d[i]['nwid'], ] * len(t)
                return t
            tbl = vstack([
                _prep(i, t) for i, t in enumerate(d['tone_match_matched_obj'])
                    ])
            for c in [
                    'tone_match_uid_self',
                    'tone_match_uid_other',
                    'tone_match_n_matched',
                    'tone_match_sep_median',
                    'tone_match_global_shift',
                    ]:
                def escape_0(v):
                    if 'uid' in c:
                        return f'toltec{v}'
                    return v
                tbl.meta[c] = [escape_0(e[c]) for e in d]
            e = d[0]
            uid = f"{e['tone_match_uid_self']}-{e['tone_match_uid_other']}"
            output = Path(option.output).with_suffix(".asc").as_posix(
                    ).replace('.asc', f'_matched_{uid}.asc')
            tbl.write(
                    output, format='ascii.ecsv',
                    overwrite=option.overwrite,
                    delimiter=',',
                    )

        if option.plot:
            # plot per obsnum match hist
            n_panels = len(ds)  # the match base is excluded
            if n_panels > 2:
                panel_size = (6, 2)
                window_type = 'scrollable'
                fig_size = panel_size
            else:
                panel_size = (12, 4)
                window_type = 'default'
                fig_size = (panel_size[0], panel_size[1] * n_panels)
            fig, axes = plt.subplots(
                    n_panels, 1,
                    figsize=(panel_size[0], panel_size[1] * n_panels),
                    sharex=True,
                    constrained_layout=True,
                    squeeze=False,
                    )
            for i, d in enumerate(ds):
                ax = np.ravel(axes)[i]
                for j, entry in enumerate(d):
                    t = entry['tone_match_matched_obj']
                    ax.hist(
                        t['Separation'], label=f'nw={entry["nwid"]}',
                        histtype='step',
                        bins=np.arange(0, 50000., 500),
                        )
                ax.set_title(f"obsid={entry['obsid']}")
                ax.legend(loc='upper right')
            save_or_show(
                    fig, option.plot,
                    window_type=window_type,
                    size=fig_size
                    )
        logger.debug(f"write output {option.output}")
        dataset.write_index_table(
                option.output, overwrite=option.overwrite,
                exclude_cols='.+_obj',
                format='ascii.commented_header')

    option = maap.parse_args(args)
    maap.bootstrap_actions(option)
