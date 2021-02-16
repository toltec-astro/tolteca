#! /usr/bin/env python


from tollan.utils import getobj
from tollan.utils.registry import Registry, register_to
from tollan.utils.fmt import pformat_yaml
from tollan.utils.schema import create_relpath_validator
from tollan.utils.log import get_logger, logit, timeit
from tollan.utils.nc import ncopen, ncinfo
from tollan.utils.namespace import Namespace

import netCDF4

import re
import numpy as np
from astropy.table import Table
from astropy.time import Time
from astropy.io import fits
import astropy.units as u
from astroquery.utils import parse_coordinates
import yaml

import matplotlib.pyplot as plt

from schema import Optional, Or, Use, Schema

from ..utils import RuntimeContext, RuntimeContextError

# import these models as toplevel
from .base import (
        SkyRasterScanModel,
        SkyLissajousModel, resolve_sky_map_ref_frame)  # noqa: F401


__all__ = ['SimulatorRuntimeError', 'SimulatorRuntime']


_instru_simu_factory = Registry.create()
"""This holds the handler of the instrument simulator config."""


@register_to(_instru_simu_factory, 'toltec')
def _isf_toltec(cfg, cfg_rt):
    """Create and return `ToltecObsSimulator` from the config."""

    logger = get_logger()

    from ..cal import ToltecCalib
    from .toltec import ToltecObsSimulator

    path_validator = create_relpath_validator(cfg_rt['rootpath'])

    def get_calobj(p):
        return ToltecCalib.from_indexfile(path_validator(p))

    cfg = Schema({
        'name': 'toltec',
        'calobj': Use(get_calobj),
        Optional('select', default=None): str
        }).validate(cfg)

    logger.debug(f"simulator config: {cfg}")
    apt = cfg['calobj'].get_array_prop_table()
    if cfg['select'] is not None:
        n = len(apt)
        apt = apt[apt.to_pandas().eval(cfg['select']).to_numpy()]
        logger.info(f"select {len(apt)} of {n} detectors: {cfg['select']}")
    return ToltecObsSimulator(apt)


_simu_source_factory = Registry.create()
"""This holds the handler of the source config for the simulator."""


@register_to(_simu_source_factory, 'image')
def _ssf_image(cfg, cfg_rt):
    """Handle simulator source specified as a FITS file."""

    logger = get_logger()

    path_validator = create_relpath_validator(cfg_rt['rootpath'])

    cfg = Schema({
        'type': 'image',
        'filepath': Use(path_validator),
        }).validate(cfg)

    logger.debug(f"source config: {cfg}")

    # TODO: finish define the fits file format for simulator input
    # finish implement this function so a source_model is returned
    fits.open(cfg['filepath'])

    return NotImplemented


@register_to(_simu_source_factory, 'atmosphere_psd')
def _ssf_atm_psd(cfg, cfg_rt):
    """Handle creation of atmosphere signal timestreams via PSD."""

    # TODO: finish the implementation.
    # a base model is needed so that the returned model
    # is a subclass of that. This will allow the driver to
    # evaluate the model at appropriate stage.

    logger = get_logger()

    cfg = Schema({
        'type': 'atmosphere_psd',
        'k': str,
        'm': int,
        }).validate(cfg)

    logger.debug(f"source config: {cfg}")

    # k = cfg['k']
    # m = cfg['m']

    # from .atm_model import kgenerator
    # psd = kgenerator(k, m)

    def get_timestream(ra, dec, time):
        # time is a vector
        surface_brightness = NotImplemented
        return surface_brightness  # vector of the same size as time

    return get_timestream


@register_to(_simu_source_factory, 'point_source_catalog')
def _ssf_point_source_catalog(cfg, cfg_rt):
    """Handle simulator source specified as a point source catalog."""

    # from .base import SkyOffsetModel

    logger = get_logger()

    path_validator = create_relpath_validator(cfg_rt['rootpath'])

    cfg = Schema({
        'type': 'point_source_catalog',
        'filepath': Use(path_validator),
        }).validate(cfg)

    logger.debug(f"source config: {cfg}")

    tbl = Table.read(cfg['filepath'], format='ascii')

    # normalize tbl
    if 'name' not in tbl.colnames:
        tbl['name'] = [f'src_{i}' for i in range(len(tbl))]
    for c in tbl.colnames:
        if c == 'ra' and tbl[c].unit is None:
            tbl[c].unit = u.deg
            logger.debug(f"assume unit {u.deg} for column {c}")
        elif c == 'dec' and tbl[c].unit is None:
            tbl[c].unit = u.deg
            logger.debug(f"assume unit {u.deg} for column {c}")
        elif re.match(r'flux(_.+)?', c) and tbl[c].unit is None:
            tbl[c].unit = u.mJy
            logger.debug(f"assume unit {u.mJy} for column {c}")
    logger.debug(f"source catalog:\n{tbl}")
    return tbl


_mapping_model_factory = Registry.create()
"""This holds the handler of the mapping model for the simulator."""


@register_to(_mapping_model_factory, 'lmt_tcs')
def _mmf_lmt_tcs(cfg, cfg_rt):
    """Handle mapping model specified as LMT/TCS pointing file."""

    logger = get_logger()

    path_validator = create_relpath_validator(cfg_rt['rootpath'])

    cfg = Schema({
        'type': 'lmt_tcs',
        'filepath': Use(path_validator),
        }).validate(cfg)

    logger.debug(f"mapping config: {cfg}")

    # TODO: finish define the fits file format for simulator input
    with ncopen(cfg['filepath']) as fo:
        logger.debug(ncinfo(fo))

    from .toltec.tel import LmtTelFileIO
    m = LmtTelFileIO(source=cfg['filepath']).read()
    logger.debug(f"resolved mapping model: {m}")
    return m


def _register_mapping_model_factory(clspath):
    """This can be used to export `clspath` as mapping model factory."""

    # this is not public API so be careful for future changes.
    from astropy.coordinates.sky_coordinate_parsers import _get_frame_class

    @register_to(_mapping_model_factory, clspath)
    def _mmf_map_model(cfg, cfg_rt):
        """Handle mapping model specified as model defined in `~tolteca.simu`.
        """

        logger = get_logger(f'_mmf_{clspath}')

        # TODO add inspect to the cls constructor to find out the necessary
        # conversion of values
        cfg = Schema({
            'type': Use(getobj),
            'target': Use(parse_coordinates),
            'ref_frame': Use(_get_frame_class),
            't0': Use(Time),
            object: object,
            }).validate(cfg)

        logger.debug(f"mapping model config: {cfg}")

        cls = cfg.pop('type')
        target = cfg.pop('target')
        t0 = cfg.pop('t0')
        ref_frame = cfg.pop('ref_frame')
        kwargs = {
                k: u.Quantity(v)
                for k, v in cfg.items()
                }
        m = cls(t0=t0, target=target, ref_frame=ref_frame, **kwargs)
        logger.debug(f"resolved mapping model: {m}")
        return m


_register_mapping_model_factory('tolteca.simu:SkyRasterScanModel')
_register_mapping_model_factory('tolteca.simu:SkyLissajousModel')


class SimulatorRuntimeError(RuntimeContextError):
    """Raise when errors occur in `SimulatorRuntime`."""
    pass


class SimulatorRuntime(RuntimeContext):
    """A class that manages the runtime of the simulator."""

    @classmethod
    def extend_config_schema(cls):
        # this defines the subschema relevant to the simulator.
        return {
            'simu': {
                'instrument': {
                    'name': Or(*_instru_simu_factory.keys()),
                    object: object
                    },
                'obs_params': {
                    'f_smp_mapping': Use(u.Quantity),
                    'f_smp_data': Use(u.Quantity),
                    't_exp': Use(u.Quantity)
                    },
                'sources': [{
                    'type': Or(*_simu_source_factory.keys()),
                    object: object
                    }],
                'mapping': {
                    'type': Or(*_mapping_model_factory.keys()),
                    object: object
                    },
                Optional('mapping_only', default=False): bool,
                object: object
                },
            }

    def get_or_create_output_dir(self):
        cfg = self.config['simu']
        outdir = self.rootpath.joinpath(cfg['jobkey'])
        if not outdir.exists():
            with logit(self.logger.debug, 'create output dir'):
                outdir.mkdir(parents=True, exist_ok=True)
        return outdir

    def get_mapping_model(self):
        """Return the mapping model specified in the runtime config."""
        cfg = self.config['simu']
        cfg_rt = self.config['runtime']
        mapping = _mapping_model_factory[cfg['mapping']['type']](
                cfg['mapping'], cfg_rt)
        return mapping

    def get_source_model(self):
        """Return the source model specified in the runtime config."""
        cfg = self.config['simu']
        cfg_rt = self.config['runtime']

        # resolve sources
        sources = []
        for src in cfg['sources']:
            try:
                s = _simu_source_factory[src['type']](
                        src, cfg_rt
                        )
            except Exception as e:
                raise SimulatorRuntimeError(
                        f"invalid simulation source:\n{pformat_yaml(src)}\n"
                        f"{e}")
            sources.append(s)

        if not sources:
            raise SimulatorRuntimeError("no valid simulation sources found.")
        return sources

    def get_instrument_simulator(self):
        """Return the instrument simulator specified in the runtime config."""

        cfg = self.config['simu']
        cfg_rt = self.config['runtime']

        simobj = _instru_simu_factory[cfg['instrument']['name']](
                cfg['instrument'], cfg_rt)
        return simobj

    def get_obs_params(self):
        """Return the observation parameters specified in the runtime config.
        """
        cfg = self.config['simu']
        obs_params = cfg['obs_params']
        return obs_params

    def run(self):
        """Run the simulator.

        Returns
        -------
        `SimulatorResult` : The result context containing the simulated data.
        """

        simobj = self.get_instrument_simulator()

        # resolve mapping
        mapping = self.get_mapping_model()

        self.logger.debug(f"mapping: {mapping}")

        obs_params = self.get_obs_params()

        sources = self.get_source_model()

        self.logger.debug(
                f"simobj: {simobj}\nobs_params: {obs_params}\n"
                f"sources: {sources}")

        with simobj.obs_context(
                obs_model=mapping, sources=sources,
                ref_coord=mapping.target,
                ref_frame=mapping.ref_frame,
                ) as obs:
            # make t grid
            t = np.arange(
                    0, obs_params['t_exp'].to_value(u.s),
                    (1 / obs_params['f_smp']).to_value(u.s)) * u.s
            s, obs_info = obs(mapping.t0, t)

        with simobj.probe_context(fp=None) as probe:
            rs, xs, iqs = probe(s)

        return SimulatorResult(
                simctx=self,
                config=self.config,
                simobj=simobj,
                obs_params=obs_params,
                obs_info=obs_info,
                sources=sources,
                mapping=mapping,
                data={
                    # TODO make these names in-line with some standard notion.
                    'time': t,
                    'flux': s,
                    'rs': rs,
                    'xs': xs,
                    'iqs': iqs
                    }
                )

    def run_mapping_only(self):
        """Run the simulator to generate mapping file only."""
        simobj = self.get_instrument_simulator()

        mapping = self.get_mapping_model()
        self.logger.debug(f"mapping: {mapping}")

        obs_params = self.get_obs_params()

        t0 = mapping.t0
        ref_frame = mapping.ref_frame
        ref_coord = mapping.target
        # make t grid
        t = np.arange(
                0, obs_params['t_exp'].to_value(u.s),
                (1 / obs_params['f_smp_mapping']).to_value(u.s)) << u.s
        time_obs = t0 + t

        _ref_frame = resolve_sky_map_ref_frame(
                ref_frame, observer=simobj.observer, time_obs=time_obs)
        _ref_coord = ref_coord.transform_to(_ref_frame)
        obs_coords = mapping.evaluate_at(_ref_coord, t)
        # transform all obs_coords to equitorial
        obs_coords_icrs = obs_coords.transform_to('icrs')

        self.logger.debug(f"time_obs size: {time_obs.shape}")
        return SimulatorResult(
                simctx=self,
                config=self.config,
                simobj=simobj,
                obs_params=obs_params,
                obs_info=locals(),
                mapping=mapping,
                )

    @timeit
    def cli_run(self, args=None):
        """Run the simulator and save the result.
        """
        cfg = self.config['simu']

        mapping_only = cfg['mapping_only']
        if mapping_only:
            result = self.run_mapping_only()
        else:
            result = self.run()
        result.save(self.get_or_create_output_dir(), mapping_only=mapping_only)

        if not cfg.get('plot', False):
            return

        result.plot_animation()


class SimulatorResult(Namespace):
    """A class to hold simulator results."""

    def _save_lmt_tcs_tel(self, outdir):

        simctx = self.simctx
        cfg = self.config['simu']

        output_tel = outdir.joinpath('tel.nc')

        nc_tel = netCDF4.Dataset(output_tel, 'w', format='NETCDF4')

        def add_str(ds, name, s, dim=128):
            if not isinstance(dim, str) or dim is None:
                if dim is None:
                    dim = len(s)
                dim_name = f'{name}_slen'
                ds.createDimension(dim_name, dim)
            else:
                dim_name = dim
            v = ds.createVariable(name, 'S1', (dim_name, ))
            v[:] = netCDF4.stringtochar(np.array([s], dtype=f'S{dim}'))

        add_str(
                nc_tel,
                'Header.File.Name',
                output_tel.relative_to(simctx.rootpath).as_posix())
        add_str(
                nc_tel,
                'Header.Source.SourceName',
                cfg['jobkey'])

        d_time = 'time'

        nc_tel.createDimension(d_time, None)

        v_time = nc_tel.createVariable(
                'Data.TelescopeBackend.TelTime', 'f8', (d_time, ))
        time_obs = self.obs_info['time_obs']
        v_time[:] = time_obs.unix

        def add_coords_data(ds, name, arr, dims):
            v = ds.createVariable(name, 'f8', dims)
            v.units = 'deg'
            v[:] = arr.to_value(u.deg)

        obs_coords_icrs = self.obs_info['obs_coords_icrs']
        add_coords_data(
            nc_tel,
            'Data.TelescopeBackend.TelSourceRaAct',
            obs_coords_icrs.ra,
            (d_time, ))
        add_coords_data(
            nc_tel,
            'Data.TelescopeBackend.TelSourceDecAct',
            obs_coords_icrs.dec,
            (d_time, ))

        nc_tel.close()

    def _save_toltec_nc(self, outdir):
        simobj = self.simobj
        # output data_files
        nws = np.unique(simobj.table['nw'])

        iqs = self.data['iqs']

        for nw in nws:
            tbl = simobj.table
            m = tbl['nw'] == nw
            tbl = tbl[m]
            output_toltec = outdir.joinpath(f'toltec{nw}.nc')

            nc_toltec = netCDF4.Dataset(output_toltec, 'w', format='NETCDF4')
            nc_toltec.createDimension('nkids', len(tbl))
            nc_toltec.createDimension('time', None)
            v_I = nc_toltec.createVariable('I', 'f8', ('nkids', 'time'))
            v_Q = nc_toltec.createVariable('Q', 'f8', ('nkids', 'time'))
            v_I[:, :] = iqs.real[m, :]
            v_Q[:, :] = iqs.imag[m, :]
            nc_toltec.close()

    def _save_config(self, outdir):
        with open(outdir.joinpath('tolteca.yaml'), 'w') as fo:
            yaml.dump(self.config, fo, Dumper=self.simctx.yaml_dumper)

    @timeit
    def save(self, outdir, mapping_only=False):

        self._save_config(outdir)
        self._save_lmt_tcs_tel(outdir)
        if not mapping_only:
            self._save_toltec_nc(outdir)

    @timeit
    def plot_animation(self):

        try:
            import animatplot as amp
        except Exception:
            raise RuntimeContextError(
                    "Package `animatplot` is required to plot animation. "
                    "To install, run "
                    "`pip install "
                    "git+https://github.com/Jerry-Ma/animatplot.git`")

        simobj = self.simobj
        obs_params = self.obs_params
        obs_info = self.obs_info
        sources = self.sources

        # make some diagnostic plots
        tbl = simobj.table

        m = tbl['array_name'] == 'a1100'
        mtbl = tbl[m]
        mtbl.meta = tbl.meta['a1100']

        # unpack the obs_info
        native_frame = obs_info['native_frame']
        projected_frame = obs_info['projected_frame']
        src_coords = obs_info['src_coords']

        fps = 2 * u.Hz
        # fps = 12 * u.Hz
        t_slice = slice(
                None, None,
                int(np.ceil(
                    (obs_params['f_smp'] / fps).to_value(
                        u.dimensionless_unscaled))))
        fps = (obs_params['f_smp'] / t_slice.step).to_value(u.Hz)

        t = self.data['time']
        s = self.data['flux']
        rs = self.data['rs']
        xs = self.data['xs']

        timeline = amp.Timeline(
                t[t_slice].to_value(u.s),
                fps=1 if fps < 1 else fps,
                units='s')
        # xx = x_t[m].to_value(u.arcmin)
        # yy = y_t[m].to_value(u.arcmin)
        xx = mtbl['x_t'].quantity.to_value(u.deg)
        yy = mtbl['y_t'].quantity.to_value(u.deg)

        ss = s[m, t_slice].T.to_value(u.MJy/u.sr)
        rrs = rs[m, t_slice].T
        xxs = xs[m, t_slice].T

        cmap = 'viridis'
        cmap_kwargs = dict(
                cmap=cmap,
                vmin=np.min(ss),
                vmax=np.max(ss),
                )

        import matplotlib.path as mpath
        import matplotlib.markers as mmarkers
        from matplotlib.transforms import Affine2D

        def make_fg_marker(fg):
            _transform = Affine2D().scale(0.5).rotate_deg(30)
            polypath = mpath.Path.unit_regular_polygon(6)
            verts = polypath.vertices
            top = mpath.Path(verts[(1, 0, 5, 4, 1), :])
            rot = [0, -60, -180, -240][fg]
            marker = mmarkers.MarkerStyle(top)
            marker._transform = _transform.rotate_deg(rot)
            return marker

        from astropy.visualization.wcsaxes import WCSAxesSubplot, conf
        from astropy.visualization.wcsaxes.transforms import (
                CoordinateTransform)

        # coord_meta = {
        #         'type': ('longitude', 'latitude'),
        #         'unit': (u.deg, u.deg),
        #         'wrap': (180, None),
        #         'name': ('Az Offset', 'Alt Offset')}
        from astropy.visualization.wcsaxes.utils import get_coord_meta
        coord_meta = get_coord_meta(native_frame[0])
        # coord_meta = get_coord_meta('icrs')
        conf.coordinate_range_samples = 5
        conf.frame_boundary_samples = 10
        conf.grid_samples = 5
        conf.contour_grid_samples = 5

        fig = plt.figure()
        ax = WCSAxesSubplot(
                fig, 3, 1, 1,
                aspect='equal',
                # transform=Affine2D(),
                transform=(
                    # CoordinateTransform(native_frame[0], 'icrs') +
                    CoordinateTransform(projected_frame[0], native_frame[0])
                    ),
                coord_meta=coord_meta,
                )
        fig.add_axes(ax)
        bx = fig.add_subplot(3, 1, 2)
        cx = fig.add_subplot(3, 1, 3)

        def amp_post_update(block, i):
            ax.reset_wcs(
                    transform=(
                        # CoordinateTransform(native_frame[i], 'icrs')
                        CoordinateTransform(
                            projected_frame[i], native_frame[i])
                        ),
                    coord_meta=coord_meta)
        # fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        # ax.set_facecolor(plt.get_cmap(cmap)(0.))
        ax.set_facecolor('#4488aa')
        nfg = 4
        pos_blocks = np.full((nfg, ), None, dtype=object)
        for i in range(nfg):
            mfg = mtbl['fg'] == i
            pos_blocks[i] = amp.blocks.Scatter(
                    xx[mfg],
                    yy[mfg],
                    s=np.abs(np.hypot(xx[0] - xx[2], yy[0] - yy[2])),
                    s_in_data_unit=True,
                    c=ss[:, mfg],
                    ax=ax,
                    # post_update=None,
                    post_update=amp_post_update if i == 0 else None,
                    marker=make_fg_marker(i),
                    # edgecolor='#cccccc',
                    **cmap_kwargs
                    )
        # add a block for the IQ values
        signal_blocks = np.full((2, ), None, dtype=object)
        for i, (vv, aa) in enumerate(zip((rrs, xxs), (bx, cx))):
            signal_blocks[i] = amp.blocks.Line(
                    np.tile(mtbl['f'], (vv.shape[0], 1)),
                    vv,
                    ax=aa,
                    marker='o',
                    linestyle='none',
                    )
        anim = amp.Animation(np.hstack([pos_blocks, signal_blocks]), timeline)

        anim.controls()

        # cax_ticks, cax_label, cmap_kwargs = _make_nw_cmap()
        # im = ax.scatter(
        #         x_t[m].to(u.arcmin), y_t[m].to(u.arcmin),
        #         # c=dists[0, m, 0].to(u.arcmin)
        #         c=s[0, m, 0].to_value(u.MJy / u.sr)
        #         # c=tbl['nw'][m], **cmap_kwargs
        #         )
        # fig.colorbar(
        #     im, ax=ax, shrink=0.8)
        # normalize lw with flux
        lws = 0.2 * sources[0]['flux_a1100'] / np.max(sources[0]['flux_a1100'])
        for i in range(len(sources)):
            ax.plot(
                    src_coords[i].lon,
                    src_coords[i].lat,
                    linewidth=lws[i], color='#aaaaaa')
        cax = fig.colorbar(
            pos_blocks[0].scat, ax=ax, shrink=0.8)
        cax.set_label("Surface Brightness (MJy/sr)")
        plt.show()
