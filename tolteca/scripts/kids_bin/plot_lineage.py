
from tolteca.datamodels.toltec import BasicObsDataset
from tollan.utils.log import init_log, get_logger, timeit
from astropy.table import Table, QTable, unique
import astropy.units as u
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
from pathlib import Path


def collect_data(data_rootpath, nw, obsnums):
    logger = get_logger()
    data_files = []
    for obsnum in obsnums:
        try:
            obsnum = int(obsnum)
        except ValueError:
            pass
        if isinstance(obsnum, int):
            obsnum = f'{obsnum:06d}'
        for p in [
            f'toltec/?cs/toltec{nw}/toltec{nw}_{obsnum}_*sweep.nc',
            f'toltec/?cs/toltec{nw}/toltec{nw}_{obsnum}_*tune.nc',
            ]:
            logger.debug(f"collect file from pattern {p} in {data_rootpath}")
            data_files.extend(data_rootpath.glob(p))

    logger.debug(f"collected files:\n{data_files}")
    logger.info(f"collected {len(data_files)} files")
    return BasicObsDataset.from_files(data_files)


def make_lineage_data(bods):
    logger = get_logger()
    t = bods.index_table.copy()
    logger.info(f"bod index:\n{t}")

    t.sort('ut')
    # get first all vna sweep index, those are the roots
    root_indices = np.where(t['filesuffix'] == 'vnasweep')[0]
    root_indices = np.append(root_indices, len(t))
    cal_groups = []
    for ri0, ri1 in zip(root_indices[:-1], root_indices[1:]):
        st = t[ri0:ri1]
        st = st[
            (st['filesuffix'] == 'vnasweep')
            | (st['filesuffix'] == 'targsweep')
            | (st['filesuffix'] == 'tune')
            ]
        cal_groups.append(st)

    logger.info(f"n_calgroups={len(cal_groups)}")
    lineage_ctxs = []
    for st in cal_groups:
        lineage_ctxs.append(_make_lineage_data(st))
    return locals()


def _make_lineage_data(index_table):
    logger = get_logger()

    t = index_table
    logger.info(f"make lineage data for bod index:\n{t}")

    lineage_ctxs = []
    tone_slice = slice(None, None)
    for e in t:
        if e['filesuffix'] == 'tune':
            swps = [
                e['_bod'].read(block=0, tone=tone_slice),
                e['_bod'].read(block=1, tone=tone_slice),
            ]
        else:
            swps = [e['_bod'].read(tone=tone_slice)]
        for iblock, swp in enumerate(swps):
            ctx = _make_swp_lineage_data(swp, iblock)
            lineage_ctxs.append(ctx)
    return locals()


def _make_swp_lineage_data(swp, iblock):
    logger = get_logger()
    kind_str = swp.meta['data_kind'].name.lower()
    name = "{kind_str}-nw{nwid}-{obsnum}[{iblock}]".format(kind_str=kind_str, iblock=iblock, **swp.meta)
    logger.debug(f"collect lineage data from swp {name}")
    n_chan, n_sweepsteps = swp.frequency.shape
    chan_id = np.arange(n_chan)
    tone_axis_data = swp.meta['tone_axis_data']
    # logger.debug(f"tone_axis_data\n{tone_axis_data}")
    f_tones = tone_axis_data['f_tone']
    f_chans = tone_axis_data['f_center']
    # check tone sorting

    def _check_tone_sorting(f_tones):
        df = np.diff(f_tones)
        if not np.all(df >= 0):
            id_not_sorted = np.where(df < 0)[0]
            logger.warning(f"corruped tone: not sorted at\n{id_not_sorted}\n{df[id_not_sorted]}\n{f_tones[id_not_sorted]} -> {f_tones[id_not_sorted + 1]}")
        if np.any(df == 0):
            id_dups = np.where(df == 0)[0]
            logger.info(f"found duplicated tone:\n{id_dups}\n{f_tones[id_dups]}")
        return df

    f_tone_spaces = np.hstack([
        _check_tone_sorting(f_tones[f_tones > 0]),
        _check_tone_sorting(f_tones[f_tones < 0]),
    ])
    _log_arr_stats(logger.debug, 'f_tone_spaces', f_tone_spaces.to(u.kHz))

    # now locate the model data and check the match
    kct_file = _get_reduced_filepath(swp.meta['file_loc'].path, '.txt')
    if kct_file is not None:
        kct = QTable.read(kct_file, format='ascii')
        logger.debug(f"kidscpp model table:\n{kct}")
        f_in_kct = kct['f_in'] << u.Hz
        f_out_kct = kct['f_out'] << u.Hz

        if kind_str != 'vnasweep':
            # check f_in is consistent with f_chan for non-vnasweep
            df_in_chan_kct = f_in_kct - f_chans
            if np.any(df_in_chan_kct != 0):
                unique_values = np.unique(df_in_chan_kct)
                if len(unique_values) < 10:
                    logger.warning(f'global offset between f_chan and f_in_kct: {unique_values}')
                else:
                    raise ValueError(f"inconsistent f_chan and f_in_kct: {df_in_chan_kct}")
        # check f_in same with f_out for all kinds
        adf_in_out_kct = np.abs(f_out_kct - f_in_kct)
        large_offset_thresh = (1e5 << u.Hz)
        if np.any(adf_in_out_kct> large_offset_thresh):
            id_large_offset_in_out_kct = np.where(adf_in_out_kct > large_offset_thresh)[0]
            logger.warning(f"large offset in f_in and f_out:\n{id_large_offset_in_out_kct}\n{adf_in_out_kct[id_large_offset_in_out_kct]}")
        _log_arr_stats(logger.info, 'adf_in_out_kct', adf_in_out_kct.to(u.kHz))
        # check ordering of f_out with respect to f_in
        kct['chan_id'] = range(len(kct))
        kct.sort('f_in')
        kct['tone_id'] = range(len(kct))
        kct.sort('f_out')
        kct['tone_out_id'] = range(len(kct))
        d_tone_id_in_out_kct = kct['tone_out_id'] - kct['tone_id']
        if np.any(d_tone_id_in_out_kct != 0):
            tone_id_mismatch_idx_kct = np.where(d_tone_id_in_out_kct != 0)[0]
            logger.warning(f"mismatch order in f_tone in and out:\n{tone_id_mismatch_idx_kct}\n{d_tone_id_in_out_kct[tone_id_mismatch_idx_kct]}")
    tlt_file = _get_reduced_filepath(swp.meta['file_loc'].path, '_tonelist.ecsv')
    if tlt_file is not None:
        tlt = QTable.read(tlt_file, format='ascii')
        print(tlt)
    targfreqs_dat_file = _get_reduced_filepath(swp.meta['file_loc'].path, '_targfreqs.dat')
    if targfreqs_dat_file is not None:
        ampcor_file = _get_reduced_filepath(swp.meta['file_loc'].path, '_ampcor.dat')
        if ampcor_file is None:
            raise ValueError("missing ampcor.dat file along with targfreqs.dat")
        tfd = QTable.read(targfreqs_dat_file, format='ascii')
        ampcor = QTable.read(ampcor_file, format='ascii.no_header')
        tfd['ampcor'] = ampcor
        f_in_tfd = tfd['f_in']
        f_out_tfd = tfd['f_out']
        if kind_str == 'tune':
            # check f_in is consistent with f_chan for non-vnasweep
            if len(f_in_tfd) != len(f_chans):
                logger.warning(f"mismatch number of tones in tune: {len(f_in_tfd)} != {len(f_chans)}")
            else:
                df_in_chan_tfd = f_in_tfd - f_chans
                if np.any(df_in_chan_tfd != 0):
                    unique_values = np.unique(df_in_chan_tfd)
                    if len(unique_values) < 10:
                        logger.warning(f'global offset between f_chan and f_in_tfd: {unique_values}')
                    else:
                        logger.warning(f"inconsistent f_chan and f_in_tfd: {df_in_chan_tfd}")
        # check f_in same with f_out for all kinds
        adf_in_out_tfd = np.abs(f_out_tfd - f_in_tfd)
        large_offset_thresh = (1e5 << u.Hz)
        if np.any(adf_in_out_tfd> large_offset_thresh):
            id_large_offset_in_out_tfd = np.where(adf_in_out_tfd > large_offset_thresh)[0]
            logger.warning(f"large offset in f_in and f_out:\n{id_large_offset_in_out_tfd}\n{adf_in_out_tfd[id_large_offset_in_out_tfd]}")
        _log_arr_stats(logger.info, 'adf_in_out_tfd', adf_in_out_tfd.to(u.kHz))
        # check ordering of f_out with respect to f_in
        tfd['chan_id'] = range(len(tfd))
        tfd.sort('f_in')
        tfd['tone_id'] = range(len(tfd))
        tfd.sort('f_out')
        tfd['tone_out_id'] = range(len(tfd))
        d_tone_id_in_out_tfd = tfd['tone_out_id'] - tfd['tone_id']
        if np.any(d_tone_id_in_out_tfd != 0):
            tone_id_mismatch_idx_tfd = np.where(d_tone_id_in_out_tfd != 0)[0]
            logger.warning(f"mismatch order in f_tone in and out:\n{tone_id_mismatch_idx_tfd}\n{d_tone_id_in_out_tfd[tone_id_mismatch_idx_tfd]}")

    return locals()


def _get_reduced_filepath(bod_filepath, suffix):
    logger = get_logger()
    data_rootpath = bod_filepath.parents[2]
    reduced_dir = data_rootpath.joinpath('reduced')
    stem = bod_filepath.stem
    file = reduced_dir.joinpath(f"{stem}{suffix}")
    logger.debug(f"get {suffix} file for {bod_filepath}: {file}")
    if file.exists():
        return file
    return None
    # raise ValueError(f"duplicated reduced file {suffix}")


def _log_arr_stats(log_func, name, arr):
    mean = np.mean(arr)
    median = np.median(arr)
    min = np.min(arr)
    max = np.max(arr)
    q25 = np.quantile(arr, 0.25)
    q75 = np.quantile(arr, 0.75)
    log_func(f'{name}: {mean=:.3f} {median=:.3f} {min=:.3f} {max=:.3f} {q25=:.3f} {q75=:.3f}')


def plot_sequence(ax, x, y=0, color_seq=None, size_seq=None, **kwargs):
    nx = len(x)
    if color_seq is None:
        color_seq = ['blue', 'red', 'green', 'black']
    if size_seq is None:
        size_seq = [5]
    nc = len(color_seq)
    colors = [color_seq[i % nc] for i in range(nx)]

    ns = len(size_seq)
    sizes = [size_seq[i % ns] for i in range(nx)]

    if np.isscalar(y):
        y = np.full((nx, ), y)
    ax.scatter(x, y, c=colors, s=sizes, **kwargs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--network", '-n', type=int)
    parser.add_argument("obsnums", nargs='+')
    parser.add_argument("--data_rootpath", '-d', type=Path)
    parser.add_argument("--log_level", default='INFO')

    option = parser.parse_args()

    logger = get_logger()
    init_log(level=option.log_level)
    import logging
    for n in ['NcFileIO', '_KidsDataAxisSlicer', 'ncopen', 'open']:
        logging.getLogger(n).disabled = True


    data_rootpath = option.data_rootpath or Path('data_lmt')

    nw = option.network or 0
    obsnums = option.obsnums

    logger.info(f"run for {nw=} {obsnums=}")

    bods = collect_data(data_rootpath, nw, obsnums)
    ctx = make_lineage_data(bods)

    lineage_ctxs = ctx['lineage_ctxs']
    nc = len(lineage_ctxs)
    panel_height = 4 if nc < 4 else 16 / nc
    panel_width = 20

    fig, axes = plt.subplots(nc, 1, figsize=(panel_width, panel_height), squeeze=False)

    def _iter_layer():
        i = 0
        while True:
            yield i
            i += 1

    for i, (ax, r) in enumerate(zip(axes.ravel(), lineage_ctxs)):
        ll = _iter_layer()
        ax.invert_yaxis()
        for j, lc in enumerate(r['lineage_ctxs']):
            name = lc['name']
            # chan_f
            y0 = next(ll)
            trans = transforms.blended_transform_factory(
                ax.transAxes, ax.transData)
            ax.text(0.5, y0, name, transform=trans, ha='left', va='center')
            x = lc['f_chans'].to_value(u.MHz)
            plot_sequence(ax, x, y=next(ll))
            if 'f_out_kct' in lc:
                x = lc['f_out_kct'].to_value(u.MHz)
                plot_sequence(ax, x, y=next(ll))
            if 'f_out_tfd' in lc:
                x = lc['f_out_tfd'].to_value(u.MHz)
                plot_sequence(ax, x, y=next(ll))
            if 'tlt' in lc:
                tlt = lc['tlt']
                targ = unique(tlt, keys='tone_id', keep='first')
                x = targ['fr_init'].to_value(u.MHz)               
                plot_sequence(ax, x, y=next(ll))
            y1 = next(ll)
            ax.axhline(y1, color='gray', linestyle=':')
            # plot tlt groups
            if 'tlt' in lc:
                tlt_by_group = lc['tlt'].group_by('group_id')
                color_seq = ['#aaccff', '#ccaaff']
                for k, (group_id, group) in enumerate(zip(tlt_by_group.groups.keys, tlt_by_group.groups)):
                    f_min = np.min(group['f_fit_min']).to_value(u.MHz)
                    f_max = np.max(group['f_fit_max']).to_value(u.MHz)
                    rect = patches.Rectangle(
                        (f_min,y0), f_max - f_min, y1 - y0,
                        linewidth=1,
                        edgecolor='none',
                        facecolor=color_seq[k % len(color_seq)],
                        zorder=0
                    )
                    ax.add_patch(rect)
    plt.show()
