#!/usr/bin/env python


from astropy.table import QTable
import sys
import astropy.units as u
from tolteca.datamodels.toltec import BasicObsDataset
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from tollan.utils.log import init_log, get_logger, timeit
from pathlib import Path
from tolteca.common.toltec import toltec_info
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib
from tolteca.utils import get_pkg_data_path


def _find_file(patterns, search_paths, unique=True):
    files = []
    for path in search_paths:
        for pattern in patterns:
            files += list(Path(path).glob(pattern))
    if unique:
        if len(files) == 1:
            return files[0]
        return None
    if files:
        return files
    return None


def _collect_kids_info(entry, search_paths):
    logger = get_logger()
    interface = entry['interface']
    nw = entry['nwid']
    obsnum = entry['obsnum']
    subobsnum =entry['subobsnum']
    scannum = entry['scannum']

    logger.debug(f"collect kids info for {interface}_{obsnum}_{subobsnum}_{scannum}")
    logger.debug(f"search paths: {search_paths}")
    prefix = f'{interface}_{obsnum:06d}_{subobsnum:03d}_{scannum:04d}_'

    def _load_table(t):
        if t is not None:
            return QTable.read(t, format='ascii')
        return None
    tonelist_table = _load_table(_find_file([
        f"{prefix}*_tonelist.ecsv"
        ], search_paths))
    checktone_table = _load_table(_find_file([
        f"{prefix}*_tonecheck.ecsv"
        ], search_paths))
    kidscpp_table = _load_table(_find_file([
        f"{prefix}*_vnasweep.txt",
        f"{prefix}*_targsweep.txt",
        f"{prefix}*_tune.txt",
        ], search_paths))
    targfreqs_table = _load_table(_find_file([
        f"{prefix}*_targ_freqs.ecsv"
        ], search_paths))
    kidsmodel_table = _load_table(_find_file([
        f"{prefix}*_kmt.ecsv",
        ], search_paths))
    tone_table = _load_table(_find_file([
        f"{prefix}*_kmt.ecsv",
        ], search_paths))
    context_data = _find_file([
        f"{prefix}*_ctx.pickle"
        ], search_paths)
    return locals()


def _make_kids_figure(layouts):

    nrows = len(layouts)
    if nrows == 1:
        figsize = (15, 5.5)
        gi_array = gi_nw = 0
    elif nrows == 2:
        figsize = (16, 11)
        gi_array = 0
        gi_nw = 1
    else:
        raise
        # figsize = (16, 10)

    fig = plt.figure(figsize=figsize)
    gs0 = gridspec.GridSpec(nrows, 1, figure=fig)

    array_names = toltec_info['array_names']
    kids_interfaces = [i for i in toltec_info['interfaces'] if 'nw' in toltec_info[i]]

    cmap_name = 'RdYlGn'
    cmap = matplotlib.colormaps[cmap_name]
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=1000)

    gss = dict()
    axes = dict()
    if 'array' in layouts:
        gs = gss['array'] = gs0[gi_array, 0].subgridspec(1, len(array_names), wspace=0.01, hspace=0.01)

        axes_array = dict()
        kw = {}
        for i, array_name in enumerate(array_names):
            dd = {
                # 'is_lim_ax': True,
                'is_label_ax': True,
                }
            if axes_array:
                kw['sharex'] = next(iter(axes_array.values()))['ax']
                kw['sharey'] = next(iter(axes_array.values()))['ax']
            else:
                pass
                # dd['is_lim_ax'] = False
            # print(kw)
            ax = fig.add_subplot(gs[0, i], **kw)
            dd.update({'ax': ax, 'cmap': cmap})
            dd.update(toltec_info[array_name])
            if i == 0:
                ax.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
                ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False)
            else:
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
                dd['is_label_ax'] = False
            axes_array[array_name] = dd
        axes.update(axes_array)
    if 'nw' in layouts:
        gs = gss['nw'] = gs0[gi_nw, 0].subgridspec(2, 7, wspace=0.01, hspace=0.01)
        axes_nw = dict()
        kw = {}
        for i, interface in enumerate(kids_interfaces):
            dd = {
                # 'is_lim_ax': True,
                'is_label_ax': True,
                }
            if axes_nw:
                kw['sharex'] = next(iter(axes_nw.values()))['ax']
                kw['sharey'] = next(iter(axes_nw.values()))['ax']
            else:
                pass
                # dd['is_lim_ax'] = False
            if toltec_info[interface]['array_name'] == 'a1100':
                ii = 0
            else:
                ii = 1
                i = i - 7
            ax = fig.add_subplot(gs[ii, i], **kw)
            dd.update({'ax': ax, 'cmap': cmap})
            dd.update(toltec_info[interface])
            if ii == 1 and i == 0:
                pass
            elif i == 0:
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
                dd['is_label_ax'] = False
            else:
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
                dd['is_label_ax'] = False
            axes_nw[interface] = dd
        axes.update(axes_nw)
    return {
        'fig': fig,
        'gss': gss,
        'axes': axes,
        'array_names': array_names,
        'interfaces': kids_interfaces,
        'cmap': cmap
        }



def _plot_finding_ratio_nw(ax_nw, ax_array, data, phi_lim=5 << u.deg):
    logger = get_logger()
    # make a path outline the 
    edge_indices = data['edge_indices']

    # tlt = d.get('tonelist_table', None)
    tft = data.get('targfreqs_table', None)
    apt = data.get('apt', None)
    tct = data.get("checktone_table", None)
    kct = data.get("kidscpp_table", None)
    if any([
            tft is None,
            apt is None,
            tct is None,
            kct is None,
            ]):
        return
    n_found = len(tft)
    # import pdb
    # pdb.set_trace()
    # assert np.allclose(tct['f_in'], kct['f_in'])

    r_med = np.median(0.5 / kct['Qr'])

    x_off = (tct['f_out'] - tct['f_in']) / tct['f_in']
    phi_off = (np.arctan2(x_off, r_med) << u.rad).to(u.deg)
    # print(phi_off)
    m_good = np.abs(phi_off) < phi_lim
    n_good = np.sum(m_good)
    m_single = tct['n_tones'] == 1
    n_single = np.sum(m_single)
    m_dup = tct['n_tones'] > 1
    n_dup = np.sum(m_dup)
    m_miss = tct['n_tones'] < 1
    n_miss = np.sum(m_miss)

    n_design = len(apt)
    frac_found = n_found / n_design
    frac_in_tune = n_good / n_design

    phi_lim_v = phi_lim.to_value(u.deg)
    bins = np.arange(-90 - phi_lim_v / 2, 90 + phi_lim_v / 2 + 0.1, phi_lim_v)
    phi_off_v = phi_off.to_value(u.deg)

    # print(phi_off_v[m_single])
    # print(phi_off_v[m_dup])
    # print(phi_off_v[m_miss])
    ax = ax_nw['ax']
    cmap = ax_nw['cmap']
    ax.axvspan(-90, -phi_lim.to_value(u.deg), color='gray', alpha=0.2)
    ax.axvspan(phi_lim.to_value(u.deg), 90, color='gray', alpha=0.2)
    ax.axvline(0., linestyle='-', color='black', linewidth=1)
    ax.hist(
        [
            phi_off_v[m_single & m_good],
            phi_off_v[m_dup & m_good],
            phi_off_v[m_single & ~m_good],
            phi_off_v[m_dup & ~m_good],

            # phi_off_v[m_miss],
        ],
        bins=bins, color=[cmap(1.), cmap(0), cmap(0.75), cmap(0.25)],
        stacked=True
        )
    ax.text(
        0.0, 1.0,
f"""
  Design: {n_design:3d}
   Found: {n_found:3d}
 In-tune: {n_good:3d}
  Single: {n_single:3d}
Multiple: {n_dup:3d}
  Missed: {n_miss:3d}
""".strip('\n'), ha='left', va='top',
        fontsize=8,
        fontfamily='monospace',
        transform=ax.transAxes
        )

    # array outline
    ax = ax_array['ax']
    cmap = ax_array['cmap']
    fillcolor = cmap(frac_found)
    ax.fill(
        apt['x_t'][edge_indices].to_value(u.arcsec),
        apt['y_t'][edge_indices].to_value(u.arcsec),
        color=cmap(frac_found))
    return locals()


def _plot_finding_ratio_array(ax_array, nw_ctxs):
    ax = ax_array['ax']
    cmap = ax_array['cmap']
    if not nw_ctxs:
        ax = ax_array['ax']
        ax.text(
            0.5, 0.5,
            "NO DATA", ha='center', va='center',
            transform=ax.transAxes,
            # fontsize=20,
            color=cmap(0),
            )
        return
    n_design = 0
    n_found = 0
    n_good = 0
    for ctx in nw_ctxs:
        n_design += ctx['n_design']
        n_found += ctx['n_found']
        n_good += ctx['n_good']
    frac_found = n_found / n_design
    frac_in_tune = n_good / n_design

    ax.text(
        0.5, 0.5,
        f"""
Found: {n_found:3d} / {n_design:3d} {frac_found:4.1%}
In-tune: {n_good:3d}/ {n_design:3d} {frac_in_tune:4.1%}
""".strip('\n'), ha='center', va='center',
        # fontsize=20,
        fontfamily='monospace',
        transform=ax.transAxes
        )


def _make_per_nw_apt_info(nw, apt):
    sel = apt['nw'] == nw
    utbl = apt[sel]
    # polygon region
    # build a polygon to describe the outline
    rows = np.sort(np.unique(utbl['i']))
    verts = []
    ui = np.where(utbl['ori'] == 0)[0]
    etbl = utbl[ui]
    for row in rows:
        # import pdb
        # pdb.set_trace()
        rv = sorted(
            np.where(etbl['i'] == row)[0], key=lambda i: etbl[i]['j'])
        if len(verts) > 0:
            verts = rv[:1] + verts + rv[-1:]
        else:
            if len(rv) >= 2:
                verts = [rv[0], rv[-1]]
            else:
                verts = rv
    verts = ui[verts]
    return {
        'select': sel,
        'apt': utbl,
        'edge_indices': verts,
        }


def _make_kids_plot(bods, apt_design=None, show_plot=True, output_dir=None, search_paths=None):
    logger = get_logger()
    bidx = bods.index_table

    # index by nw
    nws = np.unique(bidx['nwid'])

    nws_all = toltec_info['nws']

    logger.debug(f"make plot for  {len(nws)} / {len(nws_all)} kids data.")

    kids_info = {nw: {} for nw in nws}

    search_paths = search_paths or []
    if output_dir is not None:
        search_paths.append(Path(output_dir))
    for entry in bidx:
        nw = entry['nwid']
        search_paths = search_paths + [Path(entry['source']).parent]
        kids_info[nw].update(_collect_kids_info(entry, search_paths))
        if apt_design is not None:
            kids_info[nw].update(_make_per_nw_apt_info(nw, apt_design))

    fctx = _make_kids_figure(layouts=['nw', 'array'])

    nw_ctxs_per_array = {array_name: [] for array_name in toltec_info['array_names']}
    for interface in fctx['interfaces']:
        array_name = toltec_info[interface]['array_name']
        nw = toltec_info[interface]['nw']
        ax_nw = fctx['axes'][interface]
        ax_array = fctx['axes'][array_name]
        d = kids_info.get(nw, None)
        phi_lim = 5 << u.deg
        if d is None:
            ax = ax_nw['ax']
            cmap = ax_nw['cmap']
            ax.text(
                0.5, 0.5,
                "NO DATA", ha='center', va='center',
                transform=ax.transAxes,
                # fontsize=20,
                color=cmap(0.))
        else:
            ctx = _plot_finding_ratio_nw(
                ax_array=ax_array, ax_nw=ax_nw, data=d, phi_lim=phi_lim
                )
            if ctx is not None:
                nw_ctxs_per_array[array_name].append(ctx)
        ax = ax_nw['ax']
        ax.text(1, 1, f'{interface}', ha='right', va='top', transform=ax.transAxes)
        if ax_nw['is_label_ax']:
            ax.set_xlabel("Phase Offset $atan^{-1}(x / r)$ (deg)")
            ax.set_ylabel('Num. Dets')
            ax.set_xlim(-90, 90)
            ax.set_ylim(0, 350)


    for array_name in toltec_info['array_names']:
        ax_array = fctx['axes'][array_name]
        ax = ax_array['ax']
        ax.text(1, 1, f'{array_name}', ha='right', va='top', transform=ax.transAxes)
        ax_array['ax'].set_aspect("equal")
        if ax_array['is_label_ax']:
            ax.set_xlabel("$\Delta lon$ (arcsec)")
            ax.set_ylabel("$\Delta lat$ (arcsec)")
            fov = toltec_info['fov_diameter']
            ax.set_xlim(
                -(fov / 2).to_value(u.arcsec),
                (fov / 2).to_value(u.arcsec),
                )
            ax.set_ylim(
                -(fov / 2).to_value(u.arcsec),
                (fov / 2).to_value(u.arcsec),
                )

    # plot aggregated stats for array
    for array_name, nw_ctxs in nw_ctxs_per_array.items():
        ax_array = fctx['axes'][array_name]
        _plot_finding_ratio_array(ax_array=ax_array, nw_ctxs=nw_ctxs)

    fig = fctx['fig']
    obsnum = bidx['obsnum'][0]
    ut = bidx['ut'][0]
    fig.suptitle(f"KIDs Summary ObsNum={obsnum} ({ut})")
    fig.tight_layout()
    if show_plot:
        plt.show()
    else:
        # save the plot
        outname = f'ql_toltec_{obsnum}_kidsinfo.png'
        outfile = output_dir.joinpath(outname)
        fig.savefig(outfile)
        logger.info(f"figure saved: {outfile}")
    return locals()


def make_quicklook_prod(bods, **kwargs):
    logger = get_logger()

    dp = {}
    dp['kids'] = _make_kids_plot(bods, **kwargs)



def _get_or_create_default_apt(apt_filepath):
    logger = get_logger()
    apt_filepath = Path(apt_filepath)
    if apt_filepath.exists():
        logger.debug(f"use default apt {apt_filepath}")
        return QTable.read(apt_filepath, format='ascii.ecsv')

    from tolteca.simu.toltec import ToltecObsSimulatorConfig
    simulator = ToltecObsSimulatorConfig.from_dict({}).simulator
    apt = simulator.array_prop_table
    # cache apt
    apt.write(apt_filepath, format='ascii.ecsv', overwrite=True)
    logger.debug(f"create and save default apt to {apt_filepath}")
    return apt



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filepaths", nargs='+')
    # parser.add_argument(
    #     '--search_paths', nargs='*',
    #     )
    parser.add_argument(
        '--apt_design',
        )
    parser.add_argument(
        '--output_dir', required=True
        )
    parser.add_argument(
        '--log_level', default='INFO'
        )
    parser.add_argument(
        '--show_plot', action='store_true',
        )

    option = parser.parse_args()

    init_log(level=option.log_level)
    logger = get_logger()


    filepaths = [Path(fp) for fp in option.filepaths if Path(fp).exists()]
    if not filepaths:
        logger.info("no valid files, exit.")
        sys.exit(1)

    bods = timeit(BasicObsDataset.from_files)(filepaths)

    logger.debug(f"loaded {bods=} index\n{bods.index_table}")

    tbl = bods.index_table

    if len(np.unique(tbl['obsnum'])) > 1:
        logger.error("files are not from one obsnum, exist")
        sys.exit(1)


    # create output dir
    obsnum = tbl['obsnum'][0]

    output_dir = Path(option.output_dir)

    if option.apt_design is not None:
        apt = QTable.read(option.apt_design, format='ascii.ecsv')
    else:
        apt = _get_or_create_default_apt(output_dir.joinpath("apt_design.ecsv"))

    ql_output_dir = output_dir.joinpath(str(obsnum))
    if not ql_output_dir.exists():
        ql_output_dir.mkdir(parents=True)

    ql_prods = make_quicklook_prod(
        bods,
        show_plot=option.show_plot,
        output_dir=ql_output_dir, search_paths=[output_dir], apt_design=apt)
