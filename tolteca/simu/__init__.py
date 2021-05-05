#!/usr/bin/env python

from tollan.utils import getobj
from tollan.utils import rupdate
from tollan.utils.registry import Registry, register_to
from tollan.utils.fmt import pformat_yaml
from tollan.utils.schema import create_relpath_validator
from tollan.utils.log import get_logger, logit, timeit
from tollan.utils.nc import ncopen, ncinfo
from tollan.utils.namespace import Namespace

import yaml
import netCDF4
from tollan.utils.nc import NcNodeMapper
from datetime import datetime
from collections import UserDict
from pathlib import Path
from contextlib import contextmanager
import numpy as np
from astropy.time import Time
import astropy.units as u
from astroquery.utils import parse_coordinates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from schema import Optional, Or, Use, Schema

from ..utils import RuntimeContext, RuntimeContextError, get_pkg_data_path

# import these models as toplevel
from .base import (
        SkyRasterScanModel,
        SkyLissajousModel,
        resolve_sky_map_ref_frame)  # noqa: F401


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
        try:
            return ToltecCalib.from_indexfile(path_validator(p))
        except Exception:
            logger.debug(
                    'invalid calibration object index file path,'
                    ' fallback to default')
            default_cal_indexfile = get_pkg_data_path().joinpath(
                    'cal/toltec_default/index.yaml')
            return ToltecCalib.from_indexfile(default_cal_indexfile)

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
        Optional('grouping', default=None): str,
        Optional('extname_map', default=None): dict
        }).validate(cfg)

    logger.debug(f"source config: {cfg}")

    from .base import SourceImageModel
    m = SourceImageModel.from_fits(
            cfg['filepath'],
            extname_map=cfg['extname_map'],
            grouping=cfg['grouping'])
    return m


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
        Optional('grouping', default=None): str,
        Optional('colname_map', default=None): dict
        }).validate(cfg)

    logger.debug(f"source config: {cfg}")

    from .base import SourceCatalogModel
    m = SourceCatalogModel.from_table(
            cfg['filepath'],
            colname_map=cfg['colname_map'],
            grouping=cfg['grouping'])
    return m


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
                'jobkey': str,
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
                Optional('plot', default=False): bool,
                Optional('save', default=False): bool,
                Optional('mapping_only', default=False): bool,
                Optional('perf_params', default={
                        # TODO refactor here to not repeat
                        'chunk_size': 10 << u.s,
                        'mapping_interp_len': 1 << u.s,
                        'erfa_interp_len': 300 << u.s,
                        'anim_frame_rate': 12 << u.Hz,
                    }): {
                    Optional('chunk_size', default=10 << u.s): Use(u.Quantity),
                    Optional('mapping_interp_len', default=1 << u.s): Use(
                        u.Quantity),
                    Optional('erfa_interp_len', default=300 << u.s): Use(
                        u.Quantity),
                    Optional('anim_frame_rate', default=12 << u.Hz): Use(
                        u.Quantity),
                    },
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
        cfg = self.config['simu']
        simobj = self.get_instrument_simulator()
        obs_params = self.get_obs_params()

        self.logger.debug(
                pformat_yaml({
                    'simobj': simobj,
                    'obsparams': obs_params,
                    }))

        # resolve mapping
        mapping = self.get_mapping_model()

        self.logger.debug(f"mapping:\n{mapping}")

        # resolve sources
        sources = self.get_source_model()

        self.logger.debug("sources: n_sources={}\n{}".format(
            len(sources), '\n'.join(
                f'-----\n{s}\n-----' for s in sources
                )))

        # create the time grid and run the simulation
        # here we resolve the special `ct` unit according to the number
        # of repeats of the mapping pattern
        t_exp = obs_params['t_exp']

        t_pattern = mapping.get_total_time()
        self.logger.debug(f"mapping pattern time: {t_pattern}")
        if t_exp.unit.is_equivalent(u.ct):
            ct_exp = t_exp.to_value(u.ct)
            t_exp = mapping.get_total_time() * ct_exp
            self.logger.debug(f"resolve t_exp={t_exp} from count={ct_exp}")
        t = np.arange(
                0, t_exp.to_value(u.s),
                (1 / obs_params['f_smp_data']).to_value(u.s)) * u.s

        # make chunks
        chunk_size = cfg['perf_params']['chunk_size']
        if chunk_size is None:
            t_chunks = [t]
        else:
            n_times_per_chunk = int((
                    chunk_size * obs_params['f_smp_data']).to_value(
                            u.dimensionless_unscaled))
            n_times = len(t)
            n_chunks = n_times // n_times_per_chunk + bool(
                    n_times % n_times_per_chunk)
            self.logger.debug(
                    f"n_times_per_chunk={n_times_per_chunk} n_times={len(t)} "
                    f"n_chunks={n_chunks}")

            t_chunks = []
            for i in range(n_chunks):
                t_chunks.append(
                        t[i * n_times_per_chunk:(i + 1) * n_times_per_chunk])

        # construct the simulator payload
        def data_generator():
            with simobj.mapping_context(
                    mapping=mapping, sources=sources
                    ) as obs, simobj.probe_context(
                            fp=None) as probe:
                for i, t in enumerate(t_chunks):
                    s, obs_info = obs(t)
                    self.logger.debug(
                            f'chunk #{i}: t=[{t.min()}, {t.max()}] '
                            f's=[{s.min()} {s.max()}]')
                    rs, xs, iqs = probe(s)
                    data = {
                        'time': t,
                        'flux': s,
                        'rs': rs,
                        'xs': xs,
                        'iqs': iqs,
                        'obs_info': obs_info,
                        }
                    yield data

        return SimulatorResult(
                simctx=self,
                config=self.config,
                simobj=simobj,
                obs_params=obs_params,
                sources=sources,
                mapping=mapping,
                data_generator=data_generator
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
        if cfg['plot']:
            result.plot_animation()
        if cfg['save']:
            result.save(
                    self.get_or_create_output_dir(), mapping_only=mapping_only)


class SimulatorResult(Namespace):
    """A class to hold simulator results."""

    logger = get_logger()

    outdir_lockfile = 'simresult.lock'
    outdir_statefile = 'simresult.state'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'data_generator') and hasattr(self, 'data'):
            raise ValueError("invalid result. can only have data"
                             "or data_generator")
        self._lazy = hasattr(self, 'data_generator')
        # wrap data in an iterator so we have a uniform implementation
        if not self._lazy:
            def _data_gen():
                yield self.data
            self.data_generator = _data_gen
        self.reset_iterdata()

    def reset_iterdata(self):
        """Reset the data iterator."""
        self._iterdata = self.data_generator()
        return self._iterdata

    def iterdata(self, reset=False):
        """Return data from the data iterator."""
        if reset:
            self.reset_iterdata()
        return next(self._iterdata)

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

        data = self.iterdata(reset=True)

        time_obs = data['time_obs']
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

    class _PersistentSimulatorState(UserDict):
        def __init__(self, filepath, init=None, update=None):
            if filepath.exists():
                with open(filepath, 'r') as fo:
                    state = yaml.load(fo)
                if update is not None:
                    rupdate(state, update)
            elif init is not None:
                state = init
            else:
                raise ValueError("cannot initialize state")
            self._filepath = filepath
            super().__init__(state)

        def sync(self):
            with open(self._filepath, 'w') as fo:
                yaml.dump(self.data, fo)
            return self

        def reload(self):
            with open(self._filepath, 'r') as fo:
                state = yaml.load(fo)
            self.data = state

        def __str__(self):
            return pformat_yaml({
                'state': self.data,
                'filepath': self._filepath})

    @contextmanager
    def writelock(self, outdir):
        outdir = Path(outdir)
        lockfile = outdir.joinpath(self.outdir_lockfile)
        if lockfile.exists():
            raise RuntimeError(f"cannot acquire write lock for {outdir}")
        state = self._PersistentSimulatorState(
                outdir.joinpath(self.outdir_statefile),
                init={
                    'obsnum': 0,
                    'subobsnum': 0,
                    'scannum': 0,
                    'cal_obsnum': 0,
                    'cal_subobsnum': 0,
                    'cal_scannum': 0,
                    })
        try:
            with open(lockfile, 'w'):
                pass
            yield state.sync()
        finally:
            try:
                lockfile.unlink()
            except Exception:
                self.logger.debug("failed release write lock", exc_info=True)

    @timeit
    def save(self, outdir, mapping_only=False):

        def make_output_filename(interface, state, suffix):
            filename = (
                    f'{interface}_{state["obsnum"]:06d}_'
                    f'{state["subobsnum"]:03d}_'
                    f'{state["scannum"]:04d}_'
                    f'{state["ut"].strftime("%Y_%m_%d_%H_%M_%S")}'
                    f'{suffix}'
                    )
            return outdir.joinpath(filename)

        with self.writelock(outdir) as state:
            state['obsnum'] += 1
            state['cal_obsnum'] += 1
            state['ut'] = datetime.utcnow()
            state.sync()
            self.logger.debug(f"outdir state:\n{state}")
            self._save_config(outdir)
            # save the data
            simctx = self.simctx
            obs_params = simctx.get_obs_params()
            simobj = self.simobj
            mapping = simctx.get_mapping_model()
            cfg = self.config['simu']

            output_tel = make_output_filename('tel', state, '.nc')
            nm_tel = NcNodeMapper(source=output_tel, mode='w')

            nm_tel.setstr(
                    'Header.File.Name',
                    output_tel.relative_to(simctx.rootpath).as_posix())

            nm_tel.setstr(
                    'Header.Source.SourceName',
                    cfg['jobkey'])

            # create data variables
            nc_tel = nm_tel.nc_node
            d_time = 'time'
            nc_tel.createDimension(d_time, None)
            v_time = nc_tel.createVariable(
                    'Data.TelescopeBackend.TelTime', 'f8', (d_time, ))
            v_ra = nc_tel.createVariable(
                    'Data.TelescopeBackend.TelRaAct', 'f8', (d_time, ))
            v_ra.unit = 'rad'
            v_dec = nc_tel.createVariable(
                    'Data.TelescopeBackend.TelDecAct', 'f8', (d_time, ))
            v_dec.unit = 'rad'
            v_alt = nc_tel.createVariable(
                    'Data.TelescopeBackend.TelElAct', 'f8', (d_time, ))
            v_alt.unit = 'rad'
            v_az = nc_tel.createVariable(
                    'Data.TelescopeBackend.TelAzAct', 'f8', (d_time, ))
            v_az.unit = 'rad'
            v_pa = nc_tel.createVariable(
                    'Data.TelescopeBackend.ActParAng', 'f8', (d_time, ))
            v_pa.unit = 'rad'
            v_hold = nc_tel.createVariable(
                    'Data.TelescopeBackend.Hold', 'f8', (d_time, )
                    )
            # not sure why d_coord is all 2 for the coords
            d_coord = 'Header.Source.Ra_xlen'
            nc_tel.createDimension(d_coord, 2)
            v_source_ra = nc_tel.createVariable(
                    'Header.Source.Ra', 'f8', (d_coord, ))
            v_source_ra.unit = 'rad'
            v_source_dec = nc_tel.createVariable(
                    'Header.Source.Dec', 'f8', (d_coord, ))
            v_source_dec.unit = 'rad'
            ref_coord = mapping.target.transform_to('icrs')
            v_source_ra[:] = ref_coord.ra.radian
            v_source_dec[:] = ref_coord.dec.radian

            # kids data
            tbl = simobj.table
            # dump the apt
            tbl.write(
                    make_output_filename('apt', state, '.ecsv'),
                    format='ascii.ecsv')
            nws = np.unique(tbl['nw'])

            def make_kidsdata_nc(nw):
                m = tbl['nw'] == nw
                mtbl = tbl[m]
                output_toltec = make_output_filename(
                        f'toltec{nw}', state, '_timestream.nc')
                nm_toltec = NcNodeMapper(source=output_toltec, mode='w')
                nc_toltec = nm_toltec.nc_node
                # add meta data
                nm_toltec.setstr(
                        'Header.Toltec.Filename',
                        output_toltec.relative_to(simctx.rootpath).as_posix())
                nm_toltec.setscalar(
                        'Header.Toltec.ObsType', 1, dtype='i4')  # Timestream
                nm_toltec.setscalar(
                        'Header.Toltec.Master', 0, dtype='i4')
                nm_toltec.setscalar(
                        'Header.Toltec.RepeatLevel', 0, dtype='i4')
                nm_toltec.setscalar(
                        'Header.Toltec.RoachIndex', nw, dtype='i4')
                nm_toltec.setscalar(
                        'Header.Toltec.ObsNum', state['obsnum'], dtype='i4')
                nm_toltec.setscalar(
                        'Header.Toltec.SubObsNum',
                        state['subobsnum'], dtype='i4')
                nm_toltec.setscalar(
                        'Header.Toltec.ScanNum',
                        state['scannum'], dtype='i4')
                nm_toltec.setscalar(
                        'Header.Toltec.TargSweepObsNum',
                        state['cal_obsnum'], dtype='i4')
                nm_toltec.setscalar(
                        'Header.Toltec.TargSweepSubObsNum',
                        state['cal_subobsnum'], dtype='i4')
                nm_toltec.setscalar(
                        'Header.Toltec.TargSweepScanNum',
                        state['cal_scannum'], dtype='i4')
                nm_toltec.setscalar(
                        'Header.Toltec.SampleFreq',
                        obs_params['f_smp_data'].to_value(u.Hz))
                nm_toltec.setscalar(
                        'Header.Toltec.LoCenterFreq', 0.)
                nm_toltec.setscalar(
                        'Header.Toltec.InputAtten', 0.)
                nm_toltec.setscalar(
                        'Header.Toltec.OutputAtten', 0.)
                nm_toltec.setscalar(
                        'Header.Toltec.AccumLen', 524288, dtype='i4')
                nm_toltec.setscalar(
                        'Header.Toltec.MaxNumTones', 1000, dtype='i4')

                nc_toltec.createDimension('numSweeps', 1)
                nc_toltec.createDimension('toneFreqLen', len(mtbl))
                v_tones = nc_toltec.createVariable(
                        'Header.Toltec.ToneFreq',
                        'f8', ('numSweeps', 'toneFreqLen')
                        )
                v_tones[0, :] = mtbl['fp']
                nc_toltec.createDimension('modelParamsNum', 15)
                nc_toltec.createDimension('modelParamsHeaderItemSize', 32)
                v_mph = nc_toltec.createVariable(
                        'Header.Toltec.ModelParamsHeader',
                        '|S1', ('modelParamsNum', 'modelParamsHeaderItemSize')
                        )
                v_mp = nc_toltec.createVariable(
                        'Header.Toltec.ModelParams',
                        'f8', ('numSweeps', 'modelParamsNum', 'toneFreqLen')
                        )
                mp_map = {
                        'f_centered': 'fp',
                        'f_out': 'fr',
                        'f_in': 'fp',
                        'flag': None,
                        'fp': 'fp',
                        'Qr': 'Qr',
                        'Qc': 1.,
                        'fr': 'fr',
                        'A': 0.,
                        'normI': 'g0',
                        'normQ': 'g1',
                        'slopeI': 'k0',
                        'slopeQ': 'k1',
                        'interceptI': 'm0',
                        'interceptQ': 'm1',
                        }
                for i, (k, v) in enumerate(mp_map.items()):
                    v_mph[i, :] = netCDF4.stringtochar(np.array(k, '|S32'))
                    if v is not None:
                        if isinstance(v, str):
                            v = mtbl[v]
                        v_mp[0, i, :] = v

                nc_toltec.createDimension('loclen', len(mtbl))
                nc_toltec.createDimension('iqlen', len(mtbl))
                nc_toltec.createDimension('tlen', 6)
                nc_toltec.createDimension('time', None)
                v_flo = nc_toltec.createVariable(
                        'Data.Toltec.LoFreq', 'i4', ('time', ))
                v_time = nc_toltec.createVariable(
                        'Data.Toltec.Ts', 'i4', ('time', 'tlen'))
                v_I = nc_toltec.createVariable(
                        'Data.Toltec.Is', 'i4', ('time', 'iqlen'))
                v_Q = nc_toltec.createVariable(
                        'Data.Toltec.Qs', 'i4', ('time', 'iqlen'))
                return locals()

            if not mapping_only:
                kds = {
                        nw: make_kidsdata_nc(nw)
                        for nw in nws
                        }

            for data in self.reset_iterdata():
                obs_coords_icrs = data['obs_info']['obs_coords_icrs']
                obs_coords_altaz = data['obs_info']['obs_coords_altaz']
                obs_parallactic_angle = data['obs_info'][
                        'obs_parallactic_angle']
                time_obs = data['obs_info']['time_obs']
                idx = nc_tel.dimensions[d_time].size
                v_time[idx:] = time_obs.unix
                v_ra[idx:] = obs_coords_icrs.ra.radian
                v_dec[idx:] = obs_coords_icrs.dec.radian
                v_az[idx:] = obs_coords_altaz.az.radian
                v_alt[idx:] = obs_coords_altaz.alt.radian
                v_pa[idx:] = obs_parallactic_angle.radian
                v_hold[idx:] = data['obs_info']['hold_flags']
                self.logger.info(
                        f'write [{idx}:{idx + len(time_obs)}] to'
                        f' {nc_tel.filepath()}')

                if mapping_only:
                    continue

                iqs = data['iqs']
                for nw, kd in kds.items():
                    nc_toltec = kd['nc_toltec']
                    idx = nc_toltec.dimensions['time'].size
                    self.logger.info(
                        f'write [{nc_toltec.dimensions["iqlen"].size}]'
                        f'[{idx}:{idx + len(time_obs)}] to'
                        f' {nc_toltec.filepath()}')
                    m = kd['m']
                    kd['v_flo'][idx:] = 0
                    kd['v_time'][idx:, 0] = data['time']
                    kd['v_I'][idx:, :] = iqs.real[m, :].T
                    kd['v_Q'][idx:, :] = iqs.imag[m, :].T
            # close the files
            nc_tel.close()

            if not mapping_only:
                for nw, kd in kds.items():
                    kd['nc_toltec'].close()

    def _save_config(self, outdir):
        with open(outdir.joinpath('tolteca.yaml'), 'w') as fo:
            yaml.dump(self.config, fo, Dumper=self.simctx.yaml_dumper)

    @timeit
    def save_simple(self, outdir, mapping_only=False):

        self._save_config(outdir)
        self._save_lmt_tcs_tel(outdir)
        if not mapping_only:
            self._save_toltec_nc(outdir)

    @timeit
    def plot_animation(self, reset=False):

        try:
            import animatplot as amp
        except Exception:
            raise RuntimeContextError(
                    "Package `animatplot` is required to plot animation. "
                    "To install, run "
                    "`pip install "
                    "git+https://github.com/Jerry-Ma/animatplot.git`")

        data = self.iterdata(reset=reset)

        cfg = self.config['simu']
        simobj = self.simobj
        obs_params = self.obs_params
        obs_info = data['obs_info']
        tbl = simobj.table

        array_names = np.unique(tbl['array_name'])
        n_arrays = len(array_names)

        # m = tbl['array_name'] == 'a1100'
        # mtbl = tbl[m]
        # mtbl.meta = tbl.meta['a1100']

        # unpack the obs_info
        projected_frame = obs_info['projected_frame']
        native_frame = obs_info['native_frame']

        fps = cfg['perf_params']['anim_frame_rate']
        # fps = 12 * u.Hz
        t_slice = slice(
                None, None,
                int(np.ceil(
                    (obs_params['f_smp_data'] / fps).to_value(
                        u.dimensionless_unscaled))))
        fps = (obs_params['f_smp_data'] / t_slice.step).to_value(u.Hz)

        t = data['time']
        s = data['flux']
        rs = data['rs']
        xs = data['xs']

        timeline = amp.Timeline(
                t[t_slice].to_value(u.s),
                fps=1 if fps < 1 else fps,
                units='s')
        # xx = x_t[m].to_value(u.arcmin)
        # yy = y_t[m].to_value(u.arcmin)
        xx = tbl['x_t'].to_value(u.deg)
        yy = tbl['y_t'].to_value(u.deg)

        ss = s[:, t_slice].T.to_value(u.MJy/u.sr)
        rrs = rs[:, t_slice].T
        xxs = xs[:, t_slice].T

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
        from astropy.visualization.wcsaxes.utils import get_coord_meta
        coord_meta = get_coord_meta(native_frame[0])
        # coord_meta = get_coord_meta('icrs')
        conf.coordinate_range_samples = 5
        conf.frame_boundary_samples = 10
        conf.grid_samples = 5
        conf.contour_grid_samples = 5

        fig = plt.figure(constrained_layout=True)
        gs_parent = gridspec.GridSpec(nrows=1, ncols=2, figure=fig)
        gs_array_view = gs_parent[0].subgridspec(nrows=n_arrays, ncols=1)
        gs_detector_view = gs_parent[1].subgridspec(nrows=2, ncols=1)
        array_axes = []
        for i, array_name in enumerate(array_names):
            ax = WCSAxesSubplot(
                fig, gs_array_view[i],
                aspect='equal',
                transform=(
                    # CoordinateTransform(native_frame[0], 'icrs') +
                    CoordinateTransform(projected_frame[0], native_frame[0])
                    ),
                coord_meta=coord_meta,
                )
            fig.add_axes(ax)
            ax.set_facecolor('#4488aa')
            array_axes.append(ax)
        bx = fig.add_subplot(gs_detector_view[0])
        cx = fig.add_subplot(gs_detector_view[1])

        def amp_post_update(block, i):
            for ax in array_axes:
                ax.reset_wcs(
                    transform=(
                        # CoordinateTransform(native_frame[i], 'icrs')
                        CoordinateTransform(
                            projected_frame[i], native_frame[i])
                        ),
                    coord_meta=coord_meta)
        cmap = 'viridis'
        nfg = 4
        pos_blocks = np.full((nfg, n_arrays, ), None, dtype=object)
        for i in range(nfg):
            for j, array_name in enumerate(array_names):
                m = (tbl['fg'] == i) & (tbl['array_name'] == array_name)
                s_m = ss[:, m]
                cmap_kwargs = dict(
                        cmap=cmap,
                        vmin=np.min(s_m),
                        vmax=np.max(s_m),
                        )
                pos_blocks[i, j] = amp.blocks.Scatter(
                    xx[m],
                    yy[m],
                    # 0 and 2 are to account the two pg of each
                    # detector position
                    s=np.hypot(
                        xx[0] - xx[2],
                        yy[0] - yy[2]),
                    s_in_data_unit=True,
                    c=s_m,
                    ax=array_axes[j],
                    # post_update=None,
                    # only update the wcs once
                    post_update=amp_post_update if i + j == 0 else None,
                    marker=make_fg_marker(i),
                    # edgecolor='#cccccc',
                    **cmap_kwargs
                    )
        # add a block for the IQ values
        signal_blocks = np.full((2, ), None, dtype=object)
        for i, (vv, aa) in enumerate(zip((rrs, xxs), (bx, cx))):
            signal_blocks[i] = amp.blocks.Line(
                    np.tile(tbl['f'], (vv.shape[0], 1)),
                    vv,
                    ax=aa,
                    marker='.',
                    linestyle='none',
                    )
        anim = amp.Animation(
                np.hstack([pos_blocks.ravel(), signal_blocks]), timeline)

        anim.controls()

        for i, ax in enumerate(array_axes):
            cax = fig.colorbar(
                pos_blocks[0, i].scat, ax=ax, shrink=0.8)
            cax.set_label("Surface Brightness (MJy/sr)")
        plt.show()
        self.plot_animation(reset=False)


def load_example_configs():

    example_dir = get_pkg_data_path().joinpath('examples')
    files = ['toltec_point_source.yaml']

    def load_yaml(f):
        with open(f, 'r') as fo:
            return yaml.safe_load(fo)

    configs = {
            f'{f.stem}': load_yaml(f)
            for f in map(example_dir.joinpath, files)}
    return configs


example_configs = load_example_configs()
