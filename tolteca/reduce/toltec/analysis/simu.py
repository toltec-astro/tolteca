from ... import steps_registry
from pathlib import Path
import numpy as np
import netCDF4
from astropy.table import Table, QTable
from astropy.time import Time
import astropy.units as u
from astropy.modeling.functional_models import GAUSSIAN_SIGMA_TO_FWHM
from ....datamodels.toltec import BasicObsDataset, BasicObsData
from ....datamodels.io.toltec.table import KidsModelParamsIO
from tollan.utils.log import get_logger, logit, timeit
from tollan.utils.fmt import pformat_yaml
import os
import functools
from tollan.utils.dataclass_schema import add_schema
from dataclasses import dataclass, field
from ....simu import PerfParamsConfig, sources_registry, RuntimeBase
from ....simu.mapping.lmt_tcs import LmtTcsTrajMappingModel
from ...engines.citlali import CitlaliConfig, CitlaliProc
from ....simu.sources.base import (SurfaceBrightnessModel, )
from ....simu.toltec.models import (
    ToltecPowerLoadingModel)
import shutil
from ....simu.toltec.toltec_info import toltec_info
from ....simu.toltec.simulator import ToltecObsSimulator
from ....simu.utils import make_time_grid, SkyBoundingBox

@steps_registry.register('simu')
@add_schema
@dataclass
class SimuStepConfig():
    '''
    The config class for injecting simulated signals to data.
    '''

    enabled: bool = field(
        default=True,
        metadata={
            'description': 'Enable/disable this pipeline step.'
            }
        )
    jobkey: str = field(
        default='simu',
        metadata={
            'description': 'The unique identifier the job.'
            }
        )
    citlali_config: CitlaliConfig = field(
        default_factory=CitlaliConfig,
        metadata={
            "description": "The dict for related citlali settings",
        }
    )
    sources: list = field(
        default_factory=list,
        metadata={
            'description': 'The list contains input sources for simulation.',
            'schema': list(sources_registry.item_schemas),
            'pformat_schema_type': f"[<{sources_registry.name}>, ...]"
            })
    perf_params: PerfParamsConfig = field(
        default_factory=PerfParamsConfig,
        metadata={
            'description': 'The dict contains the performance related'
                           ' parameters.',
            })

    def __post_init__(self):
        self.logger = get_logger()

    def __call__(self, cfg):
        return SimuExecutor(
            jobkey=self.jobkey,
            citlali_config=self.citlali_config,
            sources=self.sources,
            perf_params=self.perf_params
        )

    def run(self, cfg, inputs=None):
        """Run this reduction step."""
        if inputs is None:
            inputs = cfg.load_input_data()
        # get bods
        bods = [
            input for input in inputs
            if isinstance(input, BasicObsDataset)
            ]
        if len(bods) == 0:
            self.logger.debug("no valid input for this step, skip")
            return None
        assert len(bods) == 1
        bods = bods[0]
        output_dir = cfg.get_or_create_output_dir()
        simu_executor = self(cfg)
        return simu_executor(
                dataset=bods,
                output_dir=output_dir,
                )


class SimuExecutor(object):
    logger = get_logger()
    def __init__(self, jobkey, citlali_config, sources, perf_params):
        self._jobkey = jobkey
        self._citlali_config = citlali_config
        self._sources = sources
        self._perf_params=perf_params

    def __call__(self, dataset, output_dir):
        citlali_proc = CitlaliProc(citlali=None, config=self._citlali_config)
        citlali_cfg = citlali_proc._prepare_citlali_config(dataset, output_dir)
        input_items = citlali_cfg['inputs']
        for item in input_items:
            self._run_simu(item, output_dir)

    def get_or_create_simu_output_dir(self, rootpath):
        logger = get_logger()
        jobkey = self._jobkey
        simu_output_dirs = list(rootpath.glob(jobkey + '[0-9][0-9][0-9]'))
        if len(simu_output_dirs) > 0:
            # get index
            index = max([int(p.name.replace(jobkey, '')) for p in simu_output_dirs])
            index += 1
        else:
            index = 1
        output_dir = rootpath.joinpath(f'{jobkey}{index:03d}')
        if not output_dir.exists():
            with logit(logger.debug, 'create simu output dir'):
                output_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("unable to create simu output dir")
        return output_dir

    def _get_source_models(self):
        source_models = [s(None) for s in self._sources]
        # split the sources based on their base class
        # we need to make sure the TolTEC power loading source is only
        # specified once
        sources_sb = list()
        power_loading_model = None
        sources_unknown = list()
        for s in source_models:
            if isinstance(s, SurfaceBrightnessModel):
                sources_sb.append(s)
            elif isinstance(s, ToltecPowerLoadingModel):
                if power_loading_model is not None:
                    raise ValueError(
                        "multiple TolTEC power loading model found.")
                power_loading_model = s
            else:
                sources_unknown.append(s)
        self.logger.debug(f"surface brightness sources:\n{sources_sb}")
        if power_loading_model is not None:
            raise ValueError("source injection mode does not allow power loading model.")
        if sources_unknown:
            self.logger.warning(f"ignored sources:\n{sources_unknown}")
        return sources_sb

    def _run_simu(self, input, output_dir):
        name = input['meta']['name']
        self.logger.debug(
            f"run simulated source injection for {name}, output_dir={output_dir}")
        tel_filepath = Path([d['filepath'] for d in input['data_items'] if d['meta']['interface'] == "lmt"][0])
        perf_params = self._perf_params
        source_models = self._get_source_models()        
        mapping_model = LmtTcsTrajMappingModel(tel_filepath)
        t_simu = mapping_model.t_pattern
        simu_output_dir = self.get_or_create_simu_output_dir(rootpath=output_dir)
        
        # copy over input files and setup the nc node mapper
        output = self._create_output_files(simu_output_dir, input)
        self.logger.debug(f"simu output:\n{pformat_yaml(output)}")
    
        apt_filepath = Path([d['filepath'] for d in input['cal_items'] if d['type'] == "array_prop_table"][0])

        apt_in = Table.read(apt_filepath, format='ascii.ecsv')
        apt = Table()
        # fix cols for apt:
        for c in ['uid', 'array', 'nw', 'fg', 'pg', 'ori', 'loc', 'flag']:
            apt[c] = apt_in[c].astype(int)
        apt_dispatch = [
                ('tone_freq', 'tone_freq', u.Hz),
                ('responsivity', 'responsivity', None),
                ('flxscale', 'flxscale', None),
                ('x_t', 'x_t', u.arcsec),
                ('x_t_err', 'x_t_err', u.arcsec),
                ('y_t', 'y_t', u.arcsec),
                ('y_t_err', 'y_t_err', u.arcsec),
                ('a_fwhm', 'a_fwhm', u.arcsec),
                ('a_fwhm_err', 'a_fwhm_err', u.arcsec),
                ('b_fwhm', 'b_fwhm', u.arcsec),
                ('b_fwhm_err', 'b_fwhm_err', u.arcsec),
                ('x_t_raw', 'x_t_raw', u.arcsec),
                ('y_t_raw', 'y_t_raw', u.arcsec),
                ('x_t_derot', 'x_t_derot', u.arcsec),
                ('y_t_derot', 'y_t_derot', u.arcsec),
                ]
        for k, kk, ku in apt_dispatch:
            if ku is not None:
                apt[k] = apt_in[kk] << ku              
            else:
                apt[k] = apt_in[kk]

        kids_dispatch = [
                    ('fr', 'fr', u.Hz),
                    ('Qr', 'Qr', None),
                    ('g0', 'normI', None),
                    ('g1', 'normQ', None),
                    ('g', 'normI', None),
                    ('phi_g', 'normQ', None),
                    ('f0', 'fp', u.Hz),
                    ('k0', 'slopeI', u.s),
                    ('k1', 'slopeQ', u.s),
                    ('m0', 'interceptI', None),
                    ('m1', 'interceptQ', None),
                    ]
        for k, kk, ku in kids_dispatch:
            if ku is not None:
                apt[k] = np.full((len(apt), ), np.nan) << ku
            else:
                apt[k] = np.full((len(apt), ), np.nan)
        apt['kids_tone'] = np.full((len(apt), ), -1)
        # kids props. This needs to come from the kids model loaded from files
        nws = np.unique(apt['nw'])
        output_by_nw = {
            int(i['interface'].replace("toltec", "")) : i
            for i in output
            if i['interface'].startswith("toltec")
        }
        for nw in nws:
            if nw not in output_by_nw:
                raise ValueError(f"missing network {nw} data")
            d = output_by_nw[nw]
            m = apt['nw'] == nw
            kmt = d['kids_model'].table
            apt['kids_tone'][m] = range(len(kmt))
            for k, kk, ku in kids_dispatch:
                if ku is not None:
                    apt[k][m] = kmt[kk] << ku
                else:
                    apt[k][m] = kmt[kk]
        apt['f'] = apt['fr']
        apt['fp'] = apt['f0']

        # handle flxscale
        apt['mJybeam_per_MJysr'] = np.nan
        apt['array_name'] = [toltec_info['array_names'][a] for a in apt['array']]
        for array_name in toltec_info['array_names']:
            fwhm = toltec_info[array_name]['a_fwhm']
            beam_area = 2 * np.pi * (fwhm / GAUSSIAN_SIGMA_TO_FWHM) ** 2
            conv = (1 << u.mJy/u.beam).to_value(u.MJy / u.sr, equivalencies=u.beam_angular_area(beam_area))
            apt['mJybeam_per_MJysr'][apt['array_name'] == array_name] = 1 / conv
        simu = ToltecObsSimulator(array_prop_table=apt)
        apt = simu.array_prop_table
        self.logger.debug(f"apt:\n{apt}")

        self.logger.debug(
            f'run {simu} with:{{}}\n'.format(
                pformat_yaml({
                    'perf_params': self._perf_params.to_dict(),
                    })))
        self.logger.debug(
            'mapping:\n{}\n\nsources:\n{}\n'.format(
                mapping_model,
                '\n'.join(str(s) for s in source_models)
                )
            )
        self.logger.debug(
            f'simu output dir: {simu_output_dir}\nsimu length={t_simu}'
            )
        # save the config file as YAML
        config_filepath = simu_output_dir.joinpath("tolteca.yaml")
        with open(config_filepath, 'w') as fo:
            # here we need to the config dict of the
            # underlying runtime context
            config = {
                "simu": {
                    "mapping": {
                        "type": "lmt_tcs",
                        "filepath": tel_filepath,
                    },
                    "sources": [{k: v for k, v in s.to_dict().items() if isinstance(v, (Path, str, int, float, dict, list))} for s in self._sources],
                    "perf_params": self._perf_params.to_dict()
                }
            }
            RuntimeBase.yaml_dump(config, fo)

        mapping_evaluator, mapping_eval_ctx = simu.mapping_evaluator(
            mapping=mapping_model, sources=source_models,
            erfa_interp_len=perf_params.mapping_erfa_interp_len,
            eval_interp_len=perf_params.mapping_eval_interp_len,
            catalog_model_render_pixel_size=(
                perf_params.catalog_model_render_pixel_size),
            )

        t_info = self._make_time_grids(mapping_model, output_by_nw, chunk_len=perf_params.chunk_len)
        t_chunks = t_info['t_chunks']
        t_grid_pre_eval = np.linspace(
                        0, t_simu.to_value(u.s),
                        perf_params.pre_eval_t_grid_size
                        ) << u.s
        # we run the mapping eval to get the det_sky_traj for the entire
        # simu
        mapping_info = mapping_evaluator(
            t_grid_pre_eval, mapping_only=True)
        # compute the extent for detectors
        bbox_padding = (
                perf_params.pre_eval_sky_bbox_padding_size,
                perf_params.pre_eval_sky_bbox_padding_size,
                )
        # here we add some padding to the bbox
        det_sky_traj = mapping_info['det_sky_traj']
        det_sky_bbox_icrs = SkyBoundingBox.from_lonlat(
            det_sky_traj['ra'], det_sky_traj['dec']).pad_with(
                *bbox_padding)
        det_sky_bbox_altaz = SkyBoundingBox.from_lonlat(
            det_sky_traj['az'], det_sky_traj['alt']).pad_with(
                *bbox_padding)
        self.logger.info(
            f"pre-eval sky bbox:\n"
            f"ra: {det_sky_bbox_icrs.w!s} - {det_sky_bbox_icrs.e!s}\n"
            f"dec: {det_sky_bbox_icrs.s!s} - {det_sky_bbox_icrs.n!s}\n"
            f"az: {det_sky_bbox_altaz.w!s} - {det_sky_bbox_altaz.e!s}\n"
            f"alt: {det_sky_bbox_altaz.s!s} - {det_sky_bbox_altaz.n!s}\n"
            f"size: {det_sky_bbox_icrs.width}, {det_sky_bbox_icrs.height}"
            )

        # iterative evaluator for each time
        with timeit("creating simulated data"):
            open_files = {}
            n_chunks = len(t_chunks)
            for ci, t in enumerate(t_chunks):
                self.logger.info(
                    f"simulate chunk {ci}/{n_chunks} "
                    f"t_min={t.min()} t_max={t.max()}")

                det_s, mapping_info = mapping_evaluator(
                    t,
                    lon_wrap_angle_altaz=det_sky_bbox_altaz.lon_wrap_angle,
                    lon_wrap_angle_icrs=det_sky_bbox_icrs.lon_wrap_angle,
                    )
                det_sky_traj = mapping_info['det_sky_traj']
                # here we use apt to convert det_s to x values directly
                det_x_simu = det_s.to_value(u.MJy/u.sr) * (apt['mJybeam_per_MJysr'] / apt['flxscale'])[:, np.newaxis]
                # now load the timestream data and compute the r and x values
                det_r= np.zeros_like(det_x_simu)
                det_x_raw= np.zeros_like(det_x_simu)
                det_x_tot= np.zeros_like(det_x_simu)
                det_I = np.zeros_like(det_x_simu)
                det_Q = np.zeros_like(det_x_simu)
                for nw, item in output_by_nw.items():
                    fsmp = item['time']['fsmp'] << u.Hz
                    accum_len = item['time']['accum_len']
                    norm = accum_len / 524288
                    nw_t0 = Time(item['time']['t0_grid'], format='unix')
                    # identify the segment that this time chunk applys to
                    nw_i0 = int(np.round(
                        ((t[0] + mapping_model.t0 - nw_t0) * fsmp).to_value(u.dimensionless_unscaled)))
                    nw_i1 = nw_i0 + len(t)
                    self.logger.info(f"write output {nw=} [{nw_i0}:{nw_i1}] {norm=}")
                    if nw not in open_files:
                        open_files[nw] = netCDF4.Dataset(item['filepath_out'], mode='a')
                    nc = open_files[nw]
                    m = apt['nw'] == nw
                    nw_iq_raw = (
                        nc['Data.Toltec.Is'][nw_i0:nw_i1, :]
                         + 1.j * nc['Data.Toltec.Qs'][nw_i0:nw_i1, :]
                         ) / norm
                    km = item['kids_model'].model
                    nw_iq_raw_derot = km.derotate(nw_iq_raw.T, km.f0.quantity).T
                    nw_rx_raw = 0.5 / nw_iq_raw_derot
                    nw_rx_tot = nw_rx_raw + 1.j * det_x_simu[m].T
                    nw_iq_tot = 0.5 / nw_rx_tot
                    nw_iq_readout = km.rotate(nw_iq_tot.T, km.f0.quantity).T * norm
                    # update the files
                    nc['Data.Toltec.Is'][nw_i0:nw_i1, :] = nw_iq_readout.real.astype(int)
                    nc['Data.Toltec.Qs'][nw_i0:nw_i1, :] = nw_iq_readout.imag.astype(int)
                    # update the variables
                    det_r[m] = nw_rx_tot.real.T
                    det_x_raw[m] = nw_rx_raw.imag.T
                    det_x_tot[m] = nw_rx_tot.imag.T
                    det_I[m] = nw_iq_readout.real.T
                    det_Q[m] = nw_iq_readout.imag.T

                det_array_name = apt['array_name']
                for array_name in toltec_info['array_names']:
                    m = (det_array_name == array_name)
                    _s = det_s[m]
                    _rs = det_r[m]
                    _x_simu = det_x_simu[m]
                    _x_raw = det_x_raw[m]
                    _x_tot = det_x_tot[m]
                    _i = det_I[m]
                    _q = det_Q[m]
                    summary_tbl = []
                    summary_rownames = ['S', 'x_simu', 'x_raw', 'x_tot', 'r', 'I', 'Q']
                    summary_vars = [_s, _x_simu, _x_raw, _x_tot, _rs, _i, _q]
                    summary_funcnames = ['min', 'max', 'med', 'mean', 'std']
                    summary_func = [np.min, np.max, np.median, np.mean, np.std]
                    for name, var in zip(summary_rownames, summary_vars):
                        row = [name, ]
                        for f in summary_func:
                            row.append(f(var))
                        summary_tbl.append(row)
                    summary_tbl = QTable(
                        rows=summary_tbl, names=['var'] + summary_funcnames)
                    for name in summary_funcnames:
                        summary_tbl[name].info.format = '.5g'
                    summary_tbl_str = '\n'.join(summary_tbl.pformat_all())
                    self.logger.info(
                        f"summary of simulated chunk for {array_name}:\n"
                        f"{summary_tbl_str}\n"
                        )
            for f in open_files.values():
                f.close()
        return simu_output_dir

    @classmethod
    def _make_time_grids(cls, mapping_model, output_by_nw, chunk_len):
        logger = get_logger()
        # this collects timing info and compute the simulation time grids
        ttbl = Table(rows=[
            {
                'nw': nw,
                't0_grid': d['time']['t0_grid'],
                't1_grid': d['time']['t1_grid'],               
                'fsmp': d['time']['fsmp']
            }
            for nw, d in output_by_nw.items()
        ])
        tel_t0 = mapping_model.t0.unix
        tel_t1 = (mapping_model.t0 + mapping_model.t_pattern).unix
        logger.debug(f"mapping model t0={mapping_model.t0} {tel_t0=} {tel_t1=} t_pattern={mapping_model.t_pattern}")
        nw_t0_max = ttbl['t0_grid'].max()
        nw_t1_min = ttbl['t1_grid'].min()
        logger.debug(f"t range {nw_t0_max=} {nw_t1_min=} n_times=")
        ttbl['delta_i0'] = (ttbl['t0_grid'] - nw_t0_max) * ttbl['fsmp']
        ttbl['delta_i1'] = (ttbl['t1_grid'] - nw_t1_min) * ttbl['fsmp']
        ttbl['n'] = (nw_t1_min - nw_t0_max) * ttbl['fsmp']
        ttbl['tel_delta_i0'] = (tel_t0 - nw_t0_max) * ttbl['fsmp']
        ttbl['tel_delta_i1'] = (tel_t1 - nw_t1_min) * ttbl['fsmp']
        logger.info(f"nw time grid table:\n{ttbl}")
        # generate time grid
        fsmp = np.unique(ttbl['fsmp'])
        if len(fsmp) != 1:
            raise ValueError("mismatch sample freqs.")
        simu_fsmp = fsmp[0]
        simu_t0 = Time(
            max([ttbl['tel_delta_i0'].max(), ttbl['delta_i0'].max()]) / simu_fsmp + nw_t0_max,
            format='unix')
        simu_t0.format = "isot"
        simu_fsmp = simu_fsmp << u.Hz
        simu_n_times = int(ttbl['n'].min()) + int(min([ttbl['delta_i1'].min(), ttbl['tel_delta_i1'].min()]))
        simu_t1 = simu_t0 + (simu_n_times - 1) / simu_fsmp
        simu_len = simu_t1 - simu_t0
        logger.info(f"simu time grid {simu_fsmp=} {simu_t0=} {simu_n_times=} {simu_t1=} {simu_len=}")
        t_chunks = make_time_grid(simu_len, simu_fsmp, chunk_len)
        # padd with the telescope time offset
        t_chunks = [
            t + (simu_t0 - mapping_model.t0).to(u.s)
            for t in t_chunks
        ]
        logger.info(f"padded time chunks:\n{t_chunks}")
        return locals()

    def _create_output_files(self, rootpath, input):
        output = []
        for item in input['data_items'] + input['cal_items']:
            meta = item.get('meta', None)
            if meta is None or meta.get("interface", None) is None:
                continue
            interface = meta['interface']
            filepath_in = Path(item['filepath'])
            filepath_out = rootpath.joinpath(filepath_in.name)
            if interface == 'lmt':
                filepath_out = rootpath.joinpath(filepath_in.name.replace("_recomputed.nc", ".nc"))
            d = {
                'interface': interface,
                'input': item,
                'filepath_in': filepath_in,
                'filepath_out': filepath_out,
            }
            _duplicate_file(filepath_in, filepath_out)
            if interface.startswith("toltec"):
                t_info = _load_time_grid(filepath_in)
                d['time'] = t_info
                kids_model = _load_kids_model(filepath_in)
                tune_filepath_in = kids_model.meta['tune_filepath']
                if tune_filepath_in is not None:
                    tune_filepath_out = rootpath.joinpath(tune_filepath_in.name)
                else:
                    tune_filepath_out = None
                if tune_filepath_out is not None:
                    _duplicate_file(tune_filepath_in, tune_filepath_out)
                d.update({
                    'kids_model': kids_model,
                    'tune_filepath_in': tune_filepath_in,
                    'tune_filepath_out': tune_filepath_out,
                })
            output.append(d)
        return output
    
def _load_kids_model(filepath):
    logger = get_logger()
    rootpath = filepath.parent
    bod = BasicObsData(filepath).open()
    meta = bod.meta
    interface = meta['interface']
    obsnum = meta['obsnum']
    subobsnum = meta['subobsnum']
    scannum = meta['scannum']
    if scannum == 2:
        tune_glob_patterns = [
            f"{interface}_{obsnum:06d}_{subobsnum:03d}_{scannum:04d}_*.txt",
            f"{interface}_{obsnum:06d}_{subobsnum:03d}_0001_*.txt",
        ]
    elif scannum == 1:
        tune_glob_patterns = [
            f"{interface}_{obsnum:06d}_{subobsnum:03d}_{scannum:04d}_*.txt",
        ]
    else:
        tune_glob_patterns = []
    tune_file = None
    for p in tune_glob_patterns:
        f = list(rootpath.glob(p))
        if len(f) == 1:
            tune_file = f[0]
    if tune_file is None:
        # load kmp from header
        kmp = KidsModelParamsIO(bod.get_model_params_table()).read()
        logger.debug(f"load built-in kmp for {filepath}:\n{kmp}")
    else:
        kmp = KidsModelParamsIO(tune_file).read()
        logger.debug(f"load kmp from {tune_file} for {filepath}:\n{kmp}")
    kmp.meta['tune_filepath'] = tune_file
    bod.close()
    return kmp


def _duplicate_file(source, source_new):
    if Path(source_new).exists():
        raise ValueError("file exists!")
    if source_new != source:
        try:
            shutil.copy(source, source_new)
        except Exception:
            raise ValueError(f"unable to create duplicated {source}")
    else:
        raise ValueError(f"invalid duplicate {source_new} filename")
    return source_new

def _load_time_grid(filepath):
    logger = get_logger()
    bod = BasicObsData(filepath).open()
    nm = bod.node_mappers['ncopen']
    fsmp = bod.meta['fsmp']
    ffpga = nm.getscalar("Header.Toltec.FpgaFreq")
    accum_len = nm.getscalar("Header.Toltec.AccumLen")
    v_ts = nm.nc_node.variables['Data.Toltec.Ts']
    # clock time
    sec0 = v_ts[:, 0]
    nsec0 = v_ts[:, 5]
    pps = v_ts[:, 1]
    msec = v_ts[:, 2]/ffpga
    count = v_ts[:, 3]
    pps_msec = v_ts[:, 4] / ffpga
    t0 = sec0 + nsec0 * 1e-9
    start_t = int(t0[0] - 0.5)
    dt = msec - pps_msec
    mdt = dt < 0
    dt[mdt] = dt[mdt] + (2 ** 32 - 1) / ffpga
    t_grid = start_t + pps + dt
    t0_grid = t_grid[0]
    t1_grid = t_grid[-1]
    bod.close()
    return {
        't_grid': t_grid,
        't0_grid': t0_grid,
        't1_grid': t1_grid,
        'fsmp': fsmp,
        'accum_len': accum_len,
    }