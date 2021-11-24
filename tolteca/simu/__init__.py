#!/usr/bin/env python

from tollan.utils import getobj
from tollan.utils import rupdate
from tollan.utils.registry import Registry, register_to
from tollan.utils.fmt import pformat_yaml
from tollan.utils.schema import (
        create_relpath_validator, make_nested_optional_defaults)
from tollan.utils.log import get_logger, logit, timeit, log_to_file
from tollan.utils.nc import ncopen, ncinfo
from tollan.utils.namespace import Namespace

from cached_property import cached_property
import re
import argparse
import yaml
import netCDF4
from tollan.utils.nc import NcNodeMapper
from datetime import datetime
from collections import UserDict
from pathlib import Path
from contextlib import contextmanager
import numpy as np
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from astroquery.utils import parse_coordinates
from astroquery.exceptions import InputWarning
from astropy.coordinates import SkyCoord, AltAz, Angle
from astropy.modeling.functional_models import GAUSSIAN_SIGMA_TO_FWHM
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve_fft
from astropy.io import fits
from astropy.wcs import WCS
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from schema import Optional, Or, Use, Schema, Hook

from ..utils import RuntimeContext, RuntimeContextError, get_pkg_data_path

# import these models as toplevel
from .base import (
        SkyRasterScanModel,
        SkyLissajousModel,
        SkyDoubleLissajousModel,
        SkyRastajousModel,
        SkyICRSTrajModel,
        SkyAltAzTrajModel,
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

    def get_array_prop_table(p):
        try:
            return Table.read(path_validator(p), format='ascii.ecsv')
        except Exception:
            logger.debug(
                    'invalid array prop table file path,'
                    ' fallback to default calobj')
            return None

    cfg = Schema({
        'name': 'toltec',
        Optional('calobj', default=get_calobj('')): Use(get_calobj),
        Optional('array_prop_table', default=None): Use(get_array_prop_table),
        Optional('select', default=None): str
        }).validate(cfg)

    logger.debug(f"simulator config: {cfg}")
    apt = cfg['array_prop_table']
    if cfg['array_prop_table'] is not None:
        logger.info(f"use user input array prop table:\n{apt}")
    else:
        logger.info(f"use array prop table from calobj {cfg['calobj']}")
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


@register_to(_simu_source_factory, 'toltec_array_loading')
def _ssf_toltec_array_loading(cfg, cfg_rt):
    """Handle simulator source for TolTEC LMT loading."""

    logger = get_logger()

    cfg = Schema({
        'type': 'toltec_array_loading',
        Optional('atm_model_name', default='am_q50'): Or(
            'am_q25', 'am_q50', 'am_q75', 'toast'),
        }).validate(cfg)
    logger.debug(f"source config: {cfg}")

    atm_model_name = cfg['atm_model_name']
    if atm_model_name == 'toast':
        # the toast model will be generated outside the context of
        # this model
        atm_model_name = None
        logger.info("atm model set to None to use TOAST3 atm model.")

    from .toltec import ArrayLoadingModel
    from .toltec import toltec_info
    m = {
            array_name: ArrayLoadingModel(
                atm_model_name=atm_model_name, array_name=array_name)
            for array_name in toltec_info['array_names']}
    logger.debug(f"toltec array loading models: {m}")
    return m


@register_to(_simu_source_factory, 'toltec_detector_readout_noise')
def _ssf_toltec_readout_noise(cfg, cfg_rt):
    """Handle simulator source for TolTEC readout noise."""

    logger = get_logger()

    cfg = Schema({
        'type': 'toltec_detector_readout_noise',
        Optional('scale_factor', default=1.): float,
        }).validate(cfg)

    logger.debug(f"source config: {cfg}")

    from .toltec import KidsReadoutNoiseModel
    m = KidsReadoutNoiseModel(scale_factor=cfg['scale_factor'])
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
    m = SourceCatalogModel.from_file(
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
        # ignore coordinte warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InputWarning)
            cfg = Schema({
                'type': Use(getobj),
                'target': str,
                'ref_frame': Use(_get_frame_class),
                't0': Use(Time),
                Optional(
                    'target_frame', default='icrs'): Use(_get_frame_class),
                Optional(str): object,
                }).validate(cfg)

        logger.debug(f"mapping model config: {cfg}")

        cls = cfg.pop('type')
        t0 = cfg.pop('t0')
        ref_frame = cfg.pop('ref_frame')

        target = cfg.pop('target')
        target_frame = cfg.pop('target_frame')
        if target_frame == 'icrs':
            target = parse_coordinates(target)
        else:
            target = SkyCoord(target, frame=target_frame)
        kwargs = {
                k: u.Quantity(v)
                for k, v in cfg.items()
                }
        m = cls(t0=t0, target=target, ref_frame=ref_frame, **kwargs)
        logger.debug(f"resolved mapping model: {m}")
        return m


_register_mapping_model_factory('tolteca.simu:SkyRasterScanModel')
_register_mapping_model_factory('tolteca.simu:SkyLissajousModel')
_register_mapping_model_factory('tolteca.simu:SkyDoubleLissajousModel')
_register_mapping_model_factory('tolteca.simu:SkyRastajousModel')


_simu_runtime_exporters = Registry.create()
"""This holds the exporters for the simulation runtime context."""


@register_to(_simu_runtime_exporters, 'lmtot')
def _sre_lmtot(rt):
    """Handle exporting of the simulator runtime to LMT observation tool."""
    logger = get_logger()

    cfg = rt.config['simu']

    simobj = rt.get_instrument_simulator()
    mapping = rt.get_mapping_model()

    ot_lines = []
    ot_lines.append(f"# LMT OT script created by tolteca.simu at {Time.now()}")
    ot_lines.append(f'ObsGoal Dcs; Dcs -ObsGoal "{cfg["jobkey"]}"')

    ref_coord = simobj.resolve_target(
            mapping.target,
            mapping.t0,
            ).transform_to('icrs')
    ot_lines.append(
        "Source Source;  Source  -BaselineList [] -CoordSys Eq"
        " -DecProperMotionCor 0"
        " -Dec[0] {Dec} -Dec[1] {Dec}"
        " -El[0] 0.000000 -El[1] 0.000000 -EphemerisTrackOn 0 -Epoch 2000.0"
        " -GoToZenith 0 -L[0] 0.0 -L[1] 0.0 -LineList [] -Planet None"
        " -RaProperMotionCor 0 -Ra[0] {RA} -Ra[1] {RA}"
        " -SourceName {Name} -VelSys Lsr -Velocity 0.000000 -Vmag 0.0".format(
            RA=ref_coord.ra.to_string(
                unit=u.hour, pad=True,
                decimal=False, fields=3, sep=':'),
            Dec=ref_coord.dec.to_string(
                unit=u.degree, pad=True, alwayssign=True,
                decimal=False, fields=3, sep=':'),
            Name=cfg['mapping']['target'],
            ))

    def _sky_lissajous_params_to_lmtot_params(
            x_length, y_length, x_omega, y_omega,
            delta, total_time
            ):
        v_x = x_length * x_omega.quantity.to(
                u.Hz, equivalencies=[(u.cy/u.s, u.Hz)])
        v_y = y_length * y_omega.quantity.to(
                u.Hz, equivalencies=[(u.cy/u.s, u.Hz)])
        return dict(
            XLength=x_length.quantity.to_value(u.arcmin),
            YLength=y_length.quantity.to_value(u.arcmin),
            XOmega=x_omega.quantity.to_value(u.rad/u.s),
            YOmega=y_omega.quantity.to_value(u.rad/u.s),
            XDelta=delta.quantity.to_value(u.rad),
            TScan=total_time,
            ScanRate=np.hypot(v_x, v_y).to_value(u.arcsec / u.s),
            )

    def _sky_raster_params_to_lmtot_params(
            length, space, n_scans, rot, speed, ref_frame
            ):
        if ref_frame == 'icrs' or ref_frame.name == 'icrs':
            MapCoord = 'Ra'
        elif ref_frame == 'altaz' or ref_frame.name == 'altaz':
            MapCoord = 'Az'
        else:
            raise NotImplementedError(f"invalid ref_frame {ref_frame}")
        return dict(
            MapCoord=MapCoord,
            ScanAngle=rot.quantity.to_value(u.deg),
            XLength=length.quantity.to_value(u.arcsec),
            XStep=speed.quantity.to_value(u.arcsec / u.s),
            YLength=(n_scans * space.quantity).to_value(u.arcsec),
            YStep=space.quantity.to_value(u.arcsec),
            )

    if isinstance(mapping, (SkyLissajousModel)):
        m = mapping
        ot_line = (
            "Lissajous -ExecMode 0 -RotateWithElevation 0 -TunePeriod 0"
            " -TScan {TScan} -ScanRate {ScanRate}"
            " -XLength {XLength} -YLength {YLength} -XOmega {XOmega}"
            " -YOmega {YOmega} -XDelta {XDelta}"
            " -XLengthMinor 0.0 -YLengthMinor 0.0 -XDeltaMinor 0.0"
            ).format(**_sky_lissajous_params_to_lmtot_params(
                x_length=m.x_length,
                y_length=m.y_length,
                x_omega=m.x_omega,
                y_omega=m.y_omega,
                delta=m.delta,
                total_time=m.get_total_time()
                ))
    elif isinstance(mapping, (SkyDoubleLissajousModel)):
        m = mapping
        major_params = _sky_lissajous_params_to_lmtot_params(
                x_length=m.x_length_0,
                y_length=m.y_length_0,
                x_omega=m.x_omega_0,
                y_omega=m.y_omega_0,
                delta=m.delta_0,
                total_time=m.get_total_time()
                )
        minor_params = _sky_lissajous_params_to_lmtot_params(
                x_length=m.x_length_1,
                y_length=m.y_length_1,
                x_omega=m.x_omega_1,
                y_omega=m.y_omega_1,
                delta=m.delta_1,
                total_time=0 << u.s
                )
        logger.warning(
                "some parameters will be ignored during the exporting "
                "and the result will different.")
        ot_line = (
            "Lissajous -ExecMode 0 -RotateWithElevation 0 -TunePeriod 0"
            " -TScan {TScan} -ScanRate {ScanRate}"
            " -XLength {XLength} -YLength {YLength} -XOmega {XOmega}"
            " -YOmega {YOmega} -XDelta {XDelta}"
            " -XLengthMinor {XLengthMinor} -YLengthMinor {YLengthMinor}"
            " -XDeltaMinor {XDeltaMinor}"
            ).format(
                XLengthMinor=minor_params['XLength'],
                YLengthMinor=minor_params['YLength'],
                XDeltaMinor=minor_params['XDelta'],
                **major_params)
    elif isinstance(mapping, (SkyRasterScanModel)):
        m = mapping
        ot_line = (
            "RasterMap Map; Map -ExecMode 0 -HPBW 1 -HoldDuringTurns 0"
            " -MapMotion Continuous -NumPass 1 -NumRepeats 1 -NumScans 0"
            " -RowsPerScan 1000000 -ScansPerCal 0 -ScansToSkip 0"
            " -TCal 0 -TRef 0 -TSamp 1"
            " -MapCoord {MapCoord} -ScanAngle {ScanAngle}"
            " -XLength {XLength} -XOffset 0 -XRamp 0 -XStep {XStep}"
            " -YLength {YLength} -YOffset 0 -YRamp 0 -YStep {YStep}"
            ).format(**_sky_raster_params_to_lmtot_params(
                length=m.length,
                space=m.space,
                n_scans=m.n_scans,
                rot=m.rot,
                speed=m.speed,
                ref_frame=m.ref_frame
                ))
    elif isinstance(mapping, (SkyRastajousModel)):
        m = mapping
        raster_params = _sky_raster_params_to_lmtot_params(
                length=m.length,
                space=m.space,
                n_scans=m.n_scans,
                rot=m.rot,
                speed=m.speed,
                ref_frame=m.ref_frame
                )
        major_params = _sky_lissajous_params_to_lmtot_params(
                x_length=m.x_length_0,
                y_length=m.y_length_0,
                x_omega=m.x_omega_0,
                y_omega=m.y_omega_0,
                delta=m.delta_0,
                total_time=m.get_total_time()
                )
        minor_params = _sky_lissajous_params_to_lmtot_params(
                x_length=m.x_length_1,
                y_length=m.y_length_1,
                x_omega=m.x_omega_1,
                y_omega=m.y_omega_1,
                delta=m.delta_1,
                total_time=0 << u.s
                )
        logger.warning(
                "some parameters will be ignored during the exporting "
                "and the result will different.")
        ot_line_lissajous = (
            "Lissajous -ExecMode 1 -RotateWithElevation 0 -TunePeriod 0"
            " -TScan {TScan} -ScanRate {ScanRate}"
            " -XLength {XLength} -YLength {YLength} -XOmega {XOmega}"
            " -YOmega {YOmega} -XDelta {XDelta}"
            " -XLengthMinor {XLengthMinor} -YLengthMinor {YLengthMinor}"
            " -XDeltaMinor {XDeltaMinor}"
            ).format(
                XLengthMinor=minor_params['XLength'],
                YLengthMinor=minor_params['YLength'],
                XDeltaMinor=minor_params['XDelta'],
                **major_params)
        ot_line_raster = (
            "RasterMap Map; Map -ExecMode 1 -HPBW 1 -HoldDuringTurns 0"
            " -MapMotion Continuous -NumPass 1 -NumRepeats 1 -NumScans 0"
            " -RowsPerScan 1000000 -ScansPerCal 0 -ScansToSkip 0"
            " -TCal 0 -TRef 0 -TSamp 1"
            " -MapCoord {MapCoord} -ScanAngle {ScanAngle}"
            " -XLength {XLength} -XOffset 0 -XRamp 0 -XStep {XStep}"
            " -YLength {YLength} -YOffset 0 -YRamp 0 -YStep {YStep}"
            ).format(**raster_params)
        ot_line = f"{ot_line_lissajous}\n{ot_line_raster}"
    else:
        raise NotImplementedError
    ot_lines.append(ot_line)
    ot_content = '\n'.join(ot_lines)
    print(ot_content)
    output_dir = rt.get_or_create_output_dir()
    with open(
            output_dir.joinpath(
                f'{output_dir.name}_exported.lmtot'), 'w') as fo:
        fo.write(ot_content)
    return ot_content


class SimulatorRuntimeError(RuntimeContextError):
    """Raise when errors occur in `SimulatorRuntime`."""
    pass


def _invalid_key_error(message):
    def wrapped(nkey, data, error):
        error = "Forbidden key encountered: %r in %r\n%r" % (nkey, data, message)
        raise RuntimeError(error)
    return wrapped


class SimulatorRuntime(RuntimeContext):
    """A class that manages the runtime of the simulator."""

    @classmethod
    def config_schema(cls):
        # this defines the subschema relevant to the simulator.
        simu_schema = {
            'jobkey': str,
            'instrument': Schema({
                'name': Or(*_instru_simu_factory.keys()),
                Optional(str): object
                }),
            'obs_params': {
                't_exp': Use(u.Quantity),
                Optional('f_smp_mapping', default=12 << u.Hz): Use(u.Quantity),
                Optional(
                    'f_smp_probing', default=122. << u.Hz): Use(u.Quantity),
                },
            Hook(
                'toast_atm',
                handler=_invalid_key_error('set in tolteca_array_loading, atm_model_name to "toast" to enable toast atm.')): object,
            Hook(
                'toast3_atm',
                handler=_invalid_key_error('set in tolteca_array_loading, atm_model_name to "toast" to enable toast atm.')): object,
            'sources': [{
                'type': Or(*_simu_source_factory.keys()),
                Optional(str): object
                }],
            'mapping': {
                'type': Or(*_mapping_model_factory.keys()),
                Optional(str): object
                },
            Optional('toast_array_padding', default=(1 * u.degree)): Use(u.Quantity),
            Optional('mapping_only', default=False): bool,
            Optional('coverage_only', default=False): bool,
            Optional('exports', default=[{'format': 'lmtot'}]): [{
                'format': Or(*_simu_runtime_exporters.keys()),
                Optional(str): object
                }, ],
            Optional('exports_only', default=False): bool,
            Optional('plot', default=False): bool,
            Optional('save', default=True): bool,
            }
        simu_schema.update(make_nested_optional_defaults({
            Optional('perf_params'): {
                Optional('chunk_size', default=10 << u.s): Use(u.Quantity),
                Optional('mapping_interp_len', default=1 << u.s): Use(
                    u.Quantity),
                Optional('erfa_interp_len', default=300 << u.s): Use(
                    u.Quantity),
                Optional('anim_frame_rate', default=12 << u.Hz): Use(
                    u.Quantity),
                },
            }, return_schema=False))
        return Schema({'simu': simu_schema, str: object})

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
        cfg_rt = self.config['runtime_info']
        mapping = _mapping_model_factory[cfg['mapping']['type']](
                cfg['mapping'], cfg_rt)
        return mapping

    def get_source_model(self):
        """Return the source model specified in the runtime config."""
        cfg = self.config['simu']
        cfg_rt = self.config['runtime_info']

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

        if not sources and not cfg['toast_atm']:
            raise SimulatorRuntimeError("no valid simulation sources found.")
        return sources

    def get_instrument_simulator(self):
        """Return the instrument simulator specified in the runtime config."""

        cfg = self.config['simu']
        cfg_rt = self.config['runtime_info']

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
        self.logger.info(f'simulator config:\n{pformat_yaml(cfg)}')
        simobj = self.get_instrument_simulator()
        obs_params = self.get_obs_params()

        self.logger.debug(
                pformat_yaml({
                    'simobj': simobj,
                    'obs_params': obs_params,
                    }))

        # resolve mapping
        mapping = self.get_mapping_model()

        self.logger.info(f"mapping:\n{mapping}")

        # resolve sources
        sources = self.get_source_model()

        self.logger.info("sources: n_sources={}\n{}".format(
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
            self.logger.info(f"resolve t_exp={t_exp} from count={ct_exp}")
        t = np.arange(
                0, t_exp.to_value(u.s),
                (1 / obs_params['f_smp_probing']).to_value(u.s)) * u.s

        # check if the toltec array loading model is specified
        # and if the atm model is None, which means to use toast
        from .toltec import ArrayLoadingModel
        if sources is not None:
            for source in sources:
                if isinstance(source, dict) and isinstance(
                        next(iter(source.values())), ArrayLoadingModel):
                    array_loading_model = source
                    break
            else:
                array_loading_model = None
        else:
            array_loading_model = None
        if array_loading_model is None:
            use_toast = False
        elif next(iter(array_loading_model.values())).has_atm_model:
            use_toast = False
        else:
            use_toast = True
            # raise ValueError("tolteca_array_loading atm_model_name has to be set to toast")
        # generic an instruction for older config with toast_atm
        if 'toast_atm' in cfg or 'toast3_atm' in cfg:
            raise ValueError("Invalid config. To enable toast, add in `tolteca_array_loading` source `atm_model_name: toast`")
        ### 
        ### toast atmosphere calculation
        ### 
        if use_toast:
            from .toltec.atm import ToastAtmosphereSimulation
            self.logger.info("generating the atmosphere/simulation using toast")
            # make t grid for atm (1 second intervals; high resolution not required)
            _t_atm = np.arange(0, t_exp.to_value(u.s), 1) * u.s

            _atm_time_obs = mapping.t0 + _t_atm

            _atm_ref_frame = resolve_sky_map_ref_frame(
                    AltAz, observer=simobj.observer, time_obs=_atm_time_obs)
            hold_flags = mapping.evaluate_holdflag(_t_atm)
            _atm_ref_coord = simobj.resolve_target(mapping.target, mapping.t0)
            _atm_obs_coords = mapping.evaluate_at(_atm_ref_coord, _t_atm)
            _atm_obs_coords = _atm_obs_coords.transform_to(_atm_ref_frame)

            m_proj_native = simobj.get_sky_projection_model(
                ref_coord=_atm_obs_coords,
                time_obs=_atm_time_obs,
                evaluate_frame=AltAz,
            )
            
            # get the array padding default 1 * u.deg
            array_size = 4 * u.arcmin
            a = _atm_obs_coords.alt.radian
            m_rot_m3 = np.array([
                [np.cos(a), -np.sin(a)],
                [np.sin(a),  np.cos(a)]
            ])
            x_t = u.Quantity(np.linspace(-1.0 * array_size.value, array_size.value, 10)) * array_size.unit
            y_t = x_t
            x = m_rot_m3[0, 0][:, np.newaxis] * x_t[np.newaxis, :] + m_rot_m3[0, 1][:, np.newaxis] * y_t[np.newaxis, :]
            y = m_rot_m3[1, 0][:, np.newaxis] * x_t[np.newaxis, :] + m_rot_m3[1, 1][:, np.newaxis] * y_t[np.newaxis, :]
            az, alt = m_proj_native(x, y)

            self.logger.info("calculated the boresight coordinates")
            # _atm_obs_coords should represent the boresight coordinates 
            # now obtain the bounding box and add padding extremes
            additional_padding = cfg['toast_array_padding']
            self.logger.debug(f"padding used: {additional_padding}")
            
            # altitude/elevation
            min_alt = np.min(alt) - additional_padding
            max_alt = np.max(alt) + additional_padding

            if min_alt < (0 * u.degree):
                min_alt = 0. * u.degree
            if max_alt > (90 * u.degree):
                max_alt = 90. * u.degree

            # azimuth (revise this procedure to account for wrapping)
            min_az = np.min(az) - additional_padding
            max_az = np.max(az) + additional_padding


            self.logger.debug(f'generated: min elevation: {min_alt}')
            self.logger.debug(f'generated: max elevation: {max_alt}')
            self.logger.debug(f'generated: min azimuth: {min_az}')
            self.logger.debug(f'generated: max azimuth: {max_az}')

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(_atm_obs_coords.az.to_value(u.degree), _atm_obs_coords.alt.to_value(u.degree), linewidth=0.5)
            # plt.axhline(min_alt.to_value(u.degree), color='red')
            # plt.axhline(max_alt.to_value(u.degree), color='red')
            # plt.axvline(min_az.to_value(u.degree), color='red')
            # plt.axvline(max_az.to_value(u.degree), color='red')
            # plt.xlabel('azimuth (deg)')
            # plt.ylabel('altitude/elevation (deg)')
            # plt.show()

            # import matplotlib.pyplot as plt
            # azel_fig, azel_subplots = plt.subplots(1, 1, dpi=100, subplot_kw={'projection': 'polar'})
            # azel_subplots.plot(_atm_obs_coords.az.to_value(u.radian), _atm_obs_coords.alt.to_value(u.radian), ',')
            # azel_subplots.set_rmax(np.pi / 2)
            # azel_subplots.set_rticks([])  # radial ticks
            # azel_subplots.set_rlabel_position(-22.5)  # get radial labels away from plotted line
            # azel_subplots.grid(True)
            # azel_subplots.set_theta_zero_location("N")  # theta = 0 at the top
            # #hwp_subfig.set_theta_direction(1)        # theta increasing clockwise
            # angle = np.deg2rad(67.5)
            # #azel_subplots.legend(fancybox=False, handletextpad=0.7, frameon=False, loc="lower left", bbox_to_anchor=(.5 + np.cos(angle)/2, .5 + np.sin(angle)/2))
            # azel_subplots.plot(np.linspace(0,2 * np.pi, 100), min_alt.to_value(u.radian)* np.ones_like(np.linspace(0,2 * np.pi, 100)), '-', linewidth=1, color='red')
            # azel_subplots.plot(np.linspace(0,2 * np.pi, 100), max_alt.to_value(u.radian)* np.ones_like(np.linspace(0,2 * np.pi, 100)), '-', linewidth=1, color='red')
            # azel_subplots.plot(min_az.to_value(u.radian) * np.ones_like(np.linspace(0, np.pi/2, 5)), np.linspace(0, np.pi/2, 5), '--', linewidth=1, color='red')
            # azel_subplots.plot(max_az.to_value(u.radian) * np.ones_like(np.linspace(0, np.pi/2, 5)), np.linspace(0, np.pi/2, 5), '--', linewidth=1, color='red')
            # plt.show()

            # debug_dir = self.rootpath.joinpath('debug_folder')
            # debug_dir.mkdir(parents=True, exist_ok=True)
            # simobj.debug_dir = debug_dir
            # np.savez(f'{debug_dir}/boresight.npz', alt=alt.to_value(u.degree), az=az.to_value(u.degree))
        
            # create the toast cache directory to cache files 
            # TODO: add option to specify a folder?
            toast_atm_cache_dir = self.rootpath.joinpath('toast_atm')
            if not toast_atm_cache_dir.exists():
                with logit(self.logger.debug, 'create cache output dir'):
                    toast_atm_cache_dir.mkdir(parents=True, exist_ok=True)
                    
            # propagate debug flag to toast so it also logs 
            # if self.logger.getEffectiveLevel() == 10:
            #     import toast.utils
            #     toast_env = toast.utils.Environment.get()
            #     toast_env.set_log_level('DEBUG')

            # generate the toast atmospheric simulation model 
            toast_atm_simulation = ToastAtmosphereSimulation(
                    _atm_time_obs[0], 
                    _atm_time_obs[0].unix, _atm_time_obs[-1].unix, 
                    min_az, max_az, min_alt, max_alt,
                    cachedir=toast_atm_cache_dir
                )

            toast_atm_simulation.generate_simulation()  

            # stick it into the simobj for easy access
            simobj.atm_simulation = toast_atm_simulation
        else:
            self.logger.info("skipping generation of the atmosphere/simulation")
            simobj.atm_simulation = None

        # make chunks
        chunk_size = cfg['perf_params']['chunk_size']
        if chunk_size is None:
            t_chunks = [t]
        else:
            n_times_per_chunk = int((
                    chunk_size * obs_params['f_smp_probing']).to_value(
                            u.dimensionless_unscaled))
            n_times = len(t)
            n_chunks = n_times // n_times_per_chunk + bool(
                    n_times % n_times_per_chunk)
            t_chunks = []
            for i in range(n_chunks):
                t_chunks.append(
                        t[i * n_times_per_chunk:(i + 1) * n_times_per_chunk])
            # merge the last chunk if it is too small
            if n_chunks >= 2:
                if len(t_chunks[-1]) * 10 < len(t_chunks[-2]):
                    last_chunk = t_chunks.pop()
                    t_chunks[-1] = np.hstack([t_chunks[-1], last_chunk])
            n_chunks = len(t_chunks)
            self.logger.info(
                    f"simulate with n_times_per_chunk={n_times_per_chunk}"
                    f" n_times={len(t)} n_chunks={n_chunks}")

        # construct the simulator payload
        def data_generator():
            with simobj.mapping_context(
                    mapping=mapping, sources=sources
                    ) as obs, simobj.probe_context(
                            fp=None, sources=sources,
                            f_smp=obs_params['f_smp_probing']) as probe:
                n_chunks = len(t_chunks)
                for i, t in enumerate(t_chunks):
                    s, obs_info = obs(t)
                    self.logger.info(
                        f'simulate chunk {i} of {n_chunks}:'
                        f' t=[{t.min()}, {t.max()}] '
                        f's=[{s.min()} {s.max()}]')
                    rs, xs, iqs, probe_info = probe(
                            s, alt=obs_info['alt'].T)
                    data = {
                        'time': t,
                        'flux': s,
                        'rs': rs,
                        'xs': xs,
                        'iqs': iqs,
                        'obs_info': obs_info,
                        'probe_info': probe_info,
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
        self.logger.info(f"mapping: {mapping}")

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

    def run_coverage_only(self, write_output=True, mask_with_holdflags=False):
        """Run the simualtor to generate an approximate coverage map."""
        simobj = self.get_instrument_simulator()
        self.logger.debug(f"simobj: {simobj}")
        mapping = self.get_mapping_model()
        self.logger.info(f"mapping: {mapping}")
        obs_params = self.get_obs_params()

        t0 = mapping.t0
        target_icrs = simobj.resolve_target(
                    mapping.target,
                    mapping.t0,
                    ).transform_to('icrs')

        # make -t grid
        f_smp = obs_params['f_smp_mapping']
        dt_smp = (1 / f_smp).to(u.s)
        t_exp = obs_params['t_exp']
        t_pattern = mapping.get_total_time()
        self.logger.debug(f"mapping pattern time: {t_pattern}")
        if t_exp.unit.is_equivalent(u.ct):
            ct_exp = t_exp.to_value(u.ct)
            t_exp = t_pattern * ct_exp
            self.logger.info(f"resolve t_exp={t_exp} from count={ct_exp}")
        t = np.arange(
                0, t_exp.to_value(u.s),
                dt_smp.to_value(u.s)) << u.s
        time_obs = t0 + t

        _ref_frame = simobj.resolve_sky_map_ref_frame(
            ref_frame=mapping.ref_frame, time_obs=time_obs)
        target_in_ref_frame = target_icrs.transform_to(_ref_frame)

        obs_coords = mapping.evaluate_at(target_in_ref_frame, t)
        hold_flags = mapping.evaluate_holdflag(t)
        obs_coords_icrs = obs_coords.transform_to('icrs')
        if isinstance(_ref_frame, AltAz):
            target_in_altaz = target_in_ref_frame
        else:
            target_in_altaz = target_icrs.transform_to(
                 simobj.resolve_sky_map_ref_frame(
                    ref_frame='altaz', time_obs=time_obs)
                    )

        apt = simobj.table

        def get_detector_coords(array_name, approximate=True):
            mapt = apt[apt['array_name'] == array_name]
            if approximate:
                m_proj = simobj.get_sky_projection_model(
                    ref_coord=target_icrs,
                    time_obs=np.mean(t) + t0)
                a_ra, a_dec = m_proj(mapt['x_t'], mapt['y_t'], frame='icrs')
            else:
                m_proj = simobj.get_sky_projection_model(
                    ref_coord=obs_coords,
                    time_obs=time_obs)
                n_samples = len(time_obs)
                x_t = np.tile(mapt['x_t'], (n_samples, 1))
                y_t = np.tile(mapt['y_t'], (n_samples, 1))
                a_ra, a_dec = m_proj(x_t, y_t, frame='icrs')
            return a_ra, a_dec

        def get_sky_bbox(lon, lat):
            lon = Angle(lon).wrap_at(360. << u.deg)
            lon_180 = Angle(lon).wrap_at(180. << u.deg)
            w, e = np.min(lon), np.max(lon)
            w1, e1 = np.min(lon_180), np.max(lon_180)
            if (e1 - w1) < (e - w):
                # use wrapping at 180d
                w = w1
                e = e1
                lon = lon_180
                self.logger.debug("re-wrapping coordinates at 180d")
            s, n = np.min(lat), np.max(lat)
            self.logger.debug(
                    f"data bbox: w={w} e={e} s={s} n={n} "
                    f"size=[{(e-w).to(u.arcmin)}, {(n-s).to(u.arcmin)}]")
            return w, e, s, n

        def make_wcs(pixscale, bbox):

            delta_pix = (1 << u.pix).to(u.arcsec, equivalencies=pixscale)
            w, e, s, n = bbox
            pad = 4 << u.arcmin
            nx = ((e - w + pad) / delta_pix).to_value(u.dimensionless_unscaled)
            ny = ((n - s + pad) / delta_pix).to_value(u.dimensionless_unscaled)
            nx = int(np.ceil(nx))
            ny = int(np.ceil(ny))
            self.logger.debug(f"wcs pixel shape: {nx=} {ny=} {delta_pix=}")
            # to avoid making too large map, we limit the output data
            # size to 200MB, which is 5000x5000
            # TODO add this to config
            size_max = 25e6
            if nx * ny > size_max:
                scale = nx * ny / size_max
                nx = nx / scale
                ny = ny / scale
                delta_pix = delta_pix * scale
                self.logger.debug(
                        f"wcs adjusted pixel shape: {nx=} {ny=} {delta_pix=}")
            # base the wcs on these values
            wcsobj = WCS(naxis=2)
            wcsobj.pixel_shape = (nx, ny)
            wcsobj.wcs.crpix = [nx / 2, ny / 2]
            wcsobj.wcs.cdelt = np.array([
                    -delta_pix.to_value(u.deg),
                    delta_pix.to_value(u.deg),
                    ])
            wcsobj.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            wcsobj.wcs.crval = [target_icrs.ra.degree, target_icrs.dec.degree]
            return wcsobj

        def make_cov_hdu(pixscale, array_name, approximate=True):
            a_ra, a_dec = get_detector_coords(
                    array_name, approximate=approximate)
            # this is ugly...
            # get the common bbox of the array and mapping pattern
            w, e, s, n = get_sky_bbox(a_ra, a_dec)
            w1, e1, s1, n1 = get_sky_bbox(
                    obs_coords_icrs.ra,
                    obs_coords_icrs.dec,
                    )
            bbox = get_sky_bbox(
                    list(map(
                        lambda v: v.to_value(u.deg),
                        [w, w, e, e, w1, w1, e1, e1])) << u.deg,
                    list(map(
                        lambda v: v.to_value(u.deg),
                        [s, n, s, n, s1, n1, s1, n1])) << u.deg)
            wcsobj = make_wcs(pixscale, bbox)

            if mask_with_holdflags:
                m = hold_flags
                xy_tel = wcsobj.world_to_pixel_values(
                    obs_coords_icrs.ra.degree[m == 0],
                    obs_coords_icrs.dec.degree[m == 0],
                    )
            else:
                xy_tel = wcsobj.world_to_pixel_values(
                    obs_coords_icrs.ra.degree,
                    obs_coords_icrs.dec.degree,
                    )
            xy_array = wcsobj.world_to_pixel_values(a_ra, a_dec)
            xbins = np.arange(wcsobj.pixel_shape[0])
            ybins = np.arange(wcsobj.pixel_shape[1])
            xbins_array = np.arange(
                    np.floor(xy_array[0].min()),
                    np.ceil(xy_array[0].max()) + 1
                    )
            ybins_array = np.arange(
                    np.floor(xy_array[1].min()),
                    np.ceil(xy_array[1].max()) + 1
                    )
            im_tel, _, _ = np.histogram2d(
                    xy_tel[1],
                    xy_tel[0],
                    bins=[ybins, xbins])
            im_tel *= dt_smp.to_value(u.s)  # scale to coverage

            im_array, _, _ = np.histogram2d(
                    xy_array[1],
                    xy_array[0],
                    bins=[ybins_array, xbins_array]
                    )
            # convolve
            with timeit("convolve with array layout"):
                im_cov = convolve_fft(
                    im_tel, im_array,
                    normalize_kernel=False, allow_huge=True)
            with timeit("convolve with beam"):
                fwhm_x = simobj.beam_model_cls.get_fwhm('x', array_name)
                fwhm_y = simobj.beam_model_cls.get_fwhm('y', array_name)
                g = Gaussian2DKernel(
                        (fwhm_x / GAUSSIAN_SIGMA_TO_FWHM).to_value(
                            u.pix, equivalencies=pixscale),
                        (fwhm_y / GAUSSIAN_SIGMA_TO_FWHM).to_value(
                            u.pix, equivalencies=pixscale),
                       )
                im_cov = convolve_fft(im_cov, g, normalize_kernel=False)
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(1, 3)
            # axes[0].imshow(im_tel)
            # axes[1].imshow(im_array)
            # axes[2].imshow(im_cov)
            # plt.show()
            self.logger.debug(
                    f'total time from coverage map: {im_cov.sum()} s')
            self.logger.debug(
                    f'total time expected: {im_array.sum() * t_exp}')

            imhdr = wcsobj.to_header()

            return fits.ImageHDU(data=im_cov, header=imhdr)

        # create output
        phdr = fits.Header()
        phdr.append((
            'ORIGIN', 'The TolTEC Project',
            'Organization generating this FITS file'
            ))
        phdr.append((
            'CREATOR', 'tolteca.simu',
            'The software used to create this FITS file'
            ))
        phdr.append((
            'TELESCOP', 'LMT',
            'Large Millimeter Telescope'
            ))
        phdr.append((
            'INSTRUME', 'TolTEC',
            'TolTEC Camera'
            ))
        phdr.append((
            'EXPTIME', f'{t_exp.to_value(u.s):.3g}',
            'Exposure time (s)'
            ))
        phdr.append((
            'OBSDUR', f'{t_exp.to_value(u.s):g}',
            'Observation duration (s)'
            ))
        phdr.append((
            'MEANALT', '{0:f}'.format(
                  target_in_altaz.alt.mean().to_value(u.deg)),
            'Mean altitude of the observation (deg)'))
        hdulist = [fits.PrimaryHDU(header=phdr)]

        pixscale = u.pixel_scale(1. << u.arcsec / u.pix)

        for array_name in apt.meta['array_names']:
            hdu = make_cov_hdu(pixscale, array_name, approximate=True)
            hdulist.append(hdu)
        hdulist = fits.HDUList(hdulist)
        if write_output:
            output_dir = self.get_or_create_output_dir()
            output_path = output_dir.joinpath(
                    f'{output_dir.name}_coverage.fits')
            hdulist.writeto(output_path, overwrite=True)
        return hdulist

    def update(self, config):
        cfg = self.config_backend._override_config
        rupdate(cfg, config)
        self.config_backend.set_override_config(cfg)
        if 'config' in self.__dict__:
            del self.__dict__['config']

    @cached_property
    def config(self):
        cfg = super().config
        return self.config_schema().validate(cfg)

    @timeit
    def cli_run(self, args=None):
        """Run the simulator and save the result.
        """
        if args is not None:
            self.logger.debug(f"update config with command line args: {args}")
            parser = argparse.ArgumentParser()
            n_args = len(args)
            re_arg = re.compile(r'^--(?P<key>[a-zA-Z_](\w|.|_)*)')
            for i, arg in enumerate(args):
                m = re_arg.match(arg)
                if m is None:
                    continue
                # g = m.groupdict()
                next_arg = args[i + 1] if i < n_args - 1 else None
                arg_kwargs = dict()
                if next_arg is None:
                    arg_kwargs['action'] = 'store_true'
                else:
                    arg_kwargs['type'] = yaml.safe_load
                parser.add_argument(arg, **arg_kwargs)
            args = parser.parse_args(args)
            self.logger.debug(f'parsed config: {pformat_yaml(args.__dict__)}')
            self.update({'simu': args.__dict__})
        cfg = self.config['simu']
        # configure the logging to log to file
        logfile = self.logdir.joinpath('simu.log')
        self.logger.info(f'setup logging to file {logfile}')
        with log_to_file(
                filepath=logfile, level='DEBUG', disable_other_handlers=False):
            mapping_only = cfg['mapping_only']
            coverage_only = cfg['coverage_only']
            exports_only = cfg['exports_only']
            if exports_only:
                exports = cfg['exports']
                if not exports:
                    raise ValueError("no export settings found.")
                result = list()
                for export_kwargs in exports:
                    result.append(self.export(**export_kwargs))
                return
            if coverage_only:
                result = self.run_coverage_only()
                return
            if mapping_only:
                result = self.run_mapping_only()
            else:
                result = self.run()
            if cfg['plot']:
                result.plot_animation()
            if cfg['save']:
                result.save(
                    self.get_or_create_output_dir(), mapping_only=mapping_only)

    def export(self, format, **kwargs):
        """Export the simulator context as various external formats.

        Supported `format`:

            * "lmtot": The script used by the LMT observation tool.

        """
        if format not in _simu_runtime_exporters:
            raise ValueError(f"invalid export format: {format}")
        return _simu_runtime_exporters[format](self)


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
                    state = yaml.safe_load(fo)
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
            self.logger.info(f"outdir state:\n{state}")
            output_config = make_output_filename('tolteca', state, '.yaml')
            self._save_config(output_config)
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

            if isinstance(
                    # TODO check what to do with multiple type of lissajous
                    mapping, (SkyLissajousModel, SkyDoubleLissajousModel)):
                nm_tel.setstr(
                        'Header.Dcs.ObsPgm',
                        'Lissajous')
            elif isinstance(mapping, (SkyRasterScanModel, SkyRastajousModel)):
                nm_tel.setstr(
                        'Header.Dcs.ObsPgm',
                        'Map')
            elif isinstance(mapping, (SkyICRSTrajModel, SkyAltAzTrajModel)):
                self.logger.debug(
                        f"mapping model meta:\n{pformat_yaml(mapping.meta)}")
                nm_tel.setstr(
                        'Header.Dcs.ObsPgm',
                        mapping.meta['mapping_type'])
            else:
                raise NotImplementedError

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
            # the _sky az alt and pa are the corrected positions of telescope
            # they are the same as the above when no pointing correction
            # is applied.
            v_alt_sky = nc_tel.createVariable(
                    'Data.TelescopeBackend.TelElSky', 'f8', (d_time, ))
            v_alt_sky.unit = 'rad'
            v_az_sky = nc_tel.createVariable(
                    'Data.TelescopeBackend.TelAzSky', 'f8', (d_time, ))
            v_az_sky.unit = 'rad'
            v_pa_sky = nc_tel.createVariable(
                    'Data.TelescopeBackend.ParAng', 'f8', (d_time, ))
            v_pa_sky.unit = 'rad'
            v_hold = nc_tel.createVariable(
                    'Data.TelescopeBackend.Hold', 'f8', (d_time, )
                    )
            # the len=2 is for mean and ref coordinates.
            d_coord = 'Header.Source.Ra_xlen'
            nc_tel.createDimension(d_coord, 2)
            v_source_ra = nc_tel.createVariable(
                    'Header.Source.Ra', 'f8', (d_coord, ))
            v_source_ra.unit = 'rad'
            v_source_dec = nc_tel.createVariable(
                    'Header.Source.Dec', 'f8', (d_coord, ))
            v_source_dec.unit = 'rad'
            v_source_alt = nc_tel.createVariable(
                    'Data.TelescopeBackend.SourceEl', 'f8', (d_time, ))
            v_source_alt.unit = 'rad'
            v_source_az = nc_tel.createVariable(
                    'Data.TelescopeBackend.SourceAz', 'f8', (d_time, ))
            v_source_az.unit = 'rad'

            ref_coord = simobj.resolve_target(
                    mapping.target,
                    mapping.t0,
                    ).transform_to('icrs')
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
                        obs_params['f_smp_probing'].to_value(u.Hz))
                nm_toltec.setscalar(
                        'Header.Toltec.LoCenterFreq', 0.)
                nm_toltec.setscalar(
                        'Header.Toltec.DriveAtten', 0.)
                nm_toltec.setscalar(
                        'Header.Toltec.SenseAtten', 0.)
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

            for ci, data in enumerate(self.reset_iterdata()):
                if ci == 0:
                    # we dump the apt table again with the additional
                    # info including flxscale
                    tbl = Table(simobj.table)
                    tbl['flxscale'] = data['probe_info']['flxscale']
                    tbl.write(
                            make_output_filename('apt', state, '.ecsv'),
                            format='ascii.ecsv')
                obs_coords_icrs = data['obs_info']['obs_coords_icrs']
                obs_coords_altaz = data['obs_info']['obs_coords_altaz']
                obs_coords_source_altaz = data['obs_info']['ref_coord_altaz']
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

                # no pointing model
                v_az_sky[idx:] = obs_coords_altaz.az.radian
                v_alt_sky[idx:] = obs_coords_altaz.alt.radian
                v_pa_sky[idx:] = obs_parallactic_angle.radian

                v_source_az[idx:] = obs_coords_source_altaz.az.radian
                v_source_alt[idx:] = obs_coords_source_altaz.alt.radian

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

    def _save_config(self, filepath):
        with open(filepath, 'w') as fo:
            self.simctx.yaml_dump(self.config, fo)

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
                    (obs_params['f_smp_probing'] / fps).to_value(
                        u.dimensionless_unscaled))))
        fps = (obs_params['f_smp_probing'] / t_slice.step).to_value(u.Hz)

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
