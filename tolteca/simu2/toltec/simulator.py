#!/usr/bin/env python

from .models import (
    ToltecArrayProjModel, ToltecSkyProjModel, pa_from_coords,
    ToltecPowerLoadingModel)
from .toltec_info import toltec_info
from ..utils import PersistentState, SkyBoundingBox, make_time_grid
from ..mapping import (PatternKind, LmtTcsTrajMappingModel)
# from ..mapping.utils import resolve_sky_coords_frame
from ..sources.base import (SurfaceBrightnessModel, )
from ..sources.models import (ImageSourceModel, CatalogSourceModel)

from ...utils.common_schema import PhysicalTypeSchema
from tollan.utils.nc import NcNodeMapper
from tollan.utils.log import get_logger, timeit
from tollan.utils.fmt import pformat_yaml
from tollan.utils.dataclass_schema import add_schema
from kidsproc.kidsmodel.simulator import KidsSimulator
from kidsproc.kidsmodel import ReadoutGainWithLinTrend

from scipy.interpolate import interp1d
import netCDF4
import astropy.units as u
from astropy.table import Column, QTable
import numpy as np
from contextlib import ExitStack, contextmanager
from datetime import datetime
import shutil
from dataclasses import dataclass, field
from astropy.coordinates.erfa_astrom import (
        erfa_astrom, ErfaAstromInterpolator)
from astropy.coordinates import Angle, Longitude, Latitude  # , AltAz, SkyCoord


__all__ = ['ToltecObsSimulator', 'ToltecHwpConfig']


@add_schema
@dataclass
class ToltecHwpConfig(object):
    """The config class for TolTEC half-wave plate and the rotator."""

    f_rot: u.Quantity = field(
        default=4. << u.Hz,
        metadata={
            'description': 'The rotator frequency',
            'schema': PhysicalTypeSchema("frequency"),
            }
        )
    f_smp: u.Quantity = field(
        default=20. << u.Hz,
        metadata={
            'description': 'The sampling frequency '
                           'of the position angle.',
            'schema': PhysicalTypeSchema("frequency"),
            }
        )
    rotator_enabled: bool = field(
        default=False,
        metadata={
            'description': 'True if use HWPR.'
            }
        )

    class Meta:
        schema = {
            'ignore_extra_keys': False,
            'description': 'The parameters related to HWP.'
            }


class ToltecObsSimulator(object):

    logger = get_logger()

    info = toltec_info
    site_info = info['site']
    observer = ToltecSkyProjModel.observer
    _m_array_proj = ToltecArrayProjModel()
    _m_sky_proj_cls = ToltecSkyProjModel
    _kids_readout_model_cls = ReadoutGainWithLinTrend

    def __init__(self, array_prop_table, polarized=False, hwp_config=None):

        apt = self._array_prop_table = self._prepare_array_prop_table(
            array_prop_table)
        self._polarized = polarized
        if hwp_config is None:
            hwp_config = ToltecHwpConfig()
        self._hwp_config = hwp_config
        # create low level models
        self._kids_simulator = KidsSimulator(
            fr=apt['fr'],
            Qr=apt['Qr'],
            background=apt['background'],
            responsivity=apt['responsivity']
            )
        self._kids_readout_model = self._kids_readout_model_cls(
            n_models=len(apt),
            **{
                c: apt[c]
                for c in self._kids_readout_model_cls.param_names
                }
            )
        self.logger.debug(f"kids_simulator: {self.kids_simulator}")
        self.logger.debug(f"kids_readout_model: {self.kids_readout_model}")

    @property
    def array_prop_table(self):
        """The table containing all detector properties."""
        return self._array_prop_table

    @property
    def array_names(self):
        return self.array_prop_table.meta['array_names']

    @property
    def polarized(self):
        """True if to simulate polarized signal."""
        return self._polarized

    @property
    def hwp_config(self):
        return self._hwp_config

    @property
    def kids_simulator(self):
        """The KIDs signal simulator to convert optical loading to
        KIDs timestream (I, Q)."""
        return self._kids_simulator

    @property
    def kids_readout_model(self):
        """The model to simulate specialties of the KIDs data readout system.
        """
        return self._kids_readout_model

    # these are some fiducial kids model params
    _default_kids_props = {
        'fp': 'f',  # column name of apt if string
        'fr': 'f',
        'Qr': 1e4,
        'g0': 200,
        'g1': 0,
        'g': 200,
        'phi_g': 0,
        'f0': 'f',
        'k0': 0 / u.Hz,
        'k1': 0 / u.Hz,
        'm0': 0,
        'm1': 0
            }

    @classmethod
    def _prepare_array_prop_table(cls, array_prop_table):
        """This function populates the `array_prop_table` with sensible
        defaults required to run the simulator"""
        tbl = array_prop_table.copy()
        # note that the apt passed to the function maybe a small portion
        # (both of row-wise and column-wise) of the full array_prop_table
        # of the TolTEC instrument. We check the column for the available
        # group names
        array_names = tbl.meta['array_names'] = np.unique(
            tbl['array_name']).tolist()
        # array props
        ap_to_cn_map = {
            'wl_center': 'wl_center',
            'a_fwhm': 'a_fwhm',
            'b_fwhm': 'b_fwhm',
            'background': 'background',
            'bkg_temp': 'bkg_temp',
            'responsivity': 'responsivity',
            'passband': 'passband',
            }
        for array_name in array_names:
            m = tbl['array_name'] == array_name
            props = {
                c: toltec_info[array_name][k]
                for k, c in ap_to_cn_map.items()
                }
            for c in props.keys():
                if c not in tbl.colnames:
                    tbl.add_column(Column(
                                np.empty((len(tbl), ), dtype='d'),
                                name=c, unit=props[c].unit))
                tbl[c][m] = props[c]

        # kids props
        for c, v in cls._default_kids_props.items():
            if c in tbl.colnames:
                continue
            cls.logger.debug(f"create kids prop column {c}")
            if isinstance(v, str) and v in tbl.colnames:
                tbl[c] = tbl[v]
                continue
            if isinstance(v, u.Quantity):
                value = v.value
                unit = v.unit
            else:
                value = v
                unit = None
            if np.isscalar(value):
                tbl.add_column(
                        Column(np.full((len(tbl),), value), name=c, unit=unit))
            else:
                raise ValueError('invalid kids prop')

        # calibration related
        # TODO need to revisit these assumptions
        if 'flxscale' not in tbl.colnames:
            tbl['flxscale'] = (1. / tbl['responsivity']).quantity.value

        # kids readout noise
        if 'sigma_readout' not in tbl.colnames:
            tbl['sigma_readout'] = 10.

        # detector locations in toltec frame
        if not {'x_t', 'y_t', 'pa_t'}.issubset(tbl.colnames):
            x_t, y_t, pa_t = cls._m_array_proj(
                tbl['x'].quantity,
                tbl['y'].quantity,
                tbl['array'], tbl['fg']
                )
            if not {"x_t", "y_t"}.issubset(tbl.colnames):
                tbl['x_t'] = x_t
                tbl['y_t'] = y_t
            if 'pa_t' not in tbl.colnames:
                tbl['pa_t'] = pa_t
        return QTable(tbl)

    def __str__(self):
        return (
            f'{self.__class__.__name__}('
            f'n_detectors={len(self.array_prop_table)}, '
            f'polarized={self.polarized})'
            )

    def output_context(self, dirpath):
        return ToltecSimuOutputContext(simulator=self, rootpath=dirpath)

    @timeit
    def _get_detector_sky_traj(
            self,
            time_obs,
            bs_coords_altaz,
            bs_coords_icrs,
            evaluate_interp_len=None):
        """Return the detector positions of shape [n_detectors, n_times]
        on sky.
        """
        logger = get_logger()
        if evaluate_interp_len is None:
            apt = self.array_prop_table
            x_t = apt['x_t']
            y_t = apt['y_t']
            pa_t = apt['pa_t']

            logger.debug(
                f'get {len(apt)} detector sky trajectories for '
                f'{len(time_obs)} time steps')

            m_sky_proj = self._m_sky_proj_cls(
                origin_coords_icrs=bs_coords_icrs,
                origin_coords_altaz=bs_coords_altaz)
            # this will do the altaz and icrs eval and save all
            # intermediate objects in the eval_ctx dict
            _, eval_ctx = m_sky_proj(
                x_t[np.newaxis, :],
                y_t[np.newaxis, :],
                pa_t[np.newaxis, :],
                evaluate_frame='icrs',
                use_evaluate_icrs_fast=False,
                return_eval_context=True
                )
            # unpack the eval_ctx
            # note that the detector id is dim0 and time_obs is dim1
            det_sky_traj = dict()
            det_sky_traj['az'] = eval_ctx['coords_altaz'].az
            det_sky_traj['alt'] = eval_ctx['coords_altaz'].alt
            det_sky_traj['pa_altaz'] = eval_ctx['pa_altaz']
            det_sky_traj['ra'] = eval_ctx['coords_icrs'].ra
            det_sky_traj['dec'] = eval_ctx['coords_icrs'].dec
            det_sky_traj['pa_icrs'] = eval_ctx['pa_icrs']
            # dpa_altaz_icrs = eval_ctx['dpa_altaz_icrs']
            return det_sky_traj

        # make a subset of parameters for faster evaluate
        # we need to make sure mjd_obs is sorted before hand
        logger.debug(
            f"evaluate sky_proj_model with "
            f"evaluate_interp_len={evaluate_interp_len}")
        mjd = time_obs.mjd << u.day
        if not np.all(np.diff(mjd) >= 0):
            raise ValueError('time_obs has to be sorted ascending.')
        # collect the subsample index
        s = [0]
        for i, t in enumerate(mjd):
            if t - mjd[s[-1]] < evaluate_interp_len:
                continue
            s.append(i)
        # ensure the last index is in the subsample
        if s[-1] != len(mjd) - 1:
            s.append(-1)
        logger.debug(
            f"prepare sky_proj_model for {len(s)}/{len(mjd)} time steps")
        time_obs_s = time_obs[s]
        bs_coords_altaz_s = bs_coords_altaz[s]
        bs_coords_icrs_s = bs_coords_icrs[s]
        # evaluate with the subsample data
        det_sky_traj_s = self._get_detector_sky_traj(
            time_obs=time_obs_s,
            bs_coords_altaz=bs_coords_altaz_s,
            bs_coords_icrs=bs_coords_icrs_s,
            evaluate_interp_len=None
            )
        # now build the interp along the time dim.
        mjd_day_s = mjd[s].to_value(u.day)
        interp_kwargs = dict(kind='linear', axis=1)
        az_deg_interp = interp1d(
                mjd_day_s,
                det_sky_traj_s['az'].degree, **interp_kwargs)
        alt_deg_interp = interp1d(
                mjd_day_s,
                det_sky_traj_s['alt'].degree, **interp_kwargs)
        pa_altaz_deg_interp = interp1d(
            mjd_day_s,
            det_sky_traj_s['pa_altaz'].to_value(u.deg), **interp_kwargs)

        ra_deg_interp = interp1d(
                mjd_day_s,
                det_sky_traj_s['ra'].degree, **interp_kwargs)
        dec_deg_interp = interp1d(
                mjd_day_s,
                det_sky_traj_s['dec'].degree, **interp_kwargs)
        pa_icrs_deg_interp = interp1d(
            mjd_day_s,
            det_sky_traj_s['pa_icrs'].to_value(u.deg), **interp_kwargs)
        # interp for full time steps
        mjd_day = mjd.to_value(u.day)
        det_sky_traj = dict()
        det_sky_traj['az'] = Longitude(az_deg_interp(mjd_day) << u.deg)
        det_sky_traj['alt'] = Latitude(alt_deg_interp(mjd_day) << u.deg)
        det_sky_traj['pa_altaz'] = Angle(pa_altaz_deg_interp(mjd_day) << u.deg)
        det_sky_traj['ra'] = Longitude(ra_deg_interp(mjd_day) << u.deg)
        det_sky_traj['dec'] = Latitude(dec_deg_interp(mjd_day) << u.deg)
        det_sky_traj['pa_icrs'] = Angle(pa_icrs_deg_interp(mjd_day) << u.deg)
        return det_sky_traj

    def probing_evaluator(
            self, f_smp, kids_fp=None,
            power_loading_model=None):
        """Return a function that can be used to get detector readout.

        When `power_loading_model` is given, the generated power loading
        will be the sum of the contribution from the astronomical source
        and the telescope and atmosphere:

            P_tot = P_src + P_bkg_fixture + P_atm(alt)

        We set the tune of the KidsSimulator,
        such that x=0 at P=P_bkg_fixture + P_atm(alt_of_tune_obs).

        Thus the measured detuning parameters is proportional to

            P_src + (P_atm(alt) - P_atm(alt_of_tune_obs))
        """
        apt = self.array_prop_table
        det_array_name = apt['array_name']
        kids_sim = self.kids_simulator
        kids_readout_model = self.kids_readout_model
        if kids_fp is None:
            kids_fp = kids_sim.fr
        if power_loading_model is not None:
            # make sure this is an instance of the toltec power loading model
            if not isinstance(power_loading_model, ToltecPowerLoadingModel):
                raise ValueError("invalid power loading model.")

        def evaluate(det_s=None, det_sky_traj=None):
            # make sure we have at least some input to the models
            if det_s is None and det_sky_traj is None:
                raise ValueError("one of det_s and det_sky_traj is required")
            if det_s is not None and det_sky_traj is not None:
                if det_s.shape != det_sky_traj['alt'].shape:
                    raise ValueError(
                        "mismatch shape in det_s and det_sky_traj")
            if det_s is not None:
                data_shape = det_s.shape
            else:
                data_shape = det_sky_traj['alt'].shape
            # make sure the data shape matches with apt shape
            if data_shape[0] != len(det_array_name):
                raise ValueError(
                    "mismatch shape in data shape and apt length.")
            if det_s is None:
                det_s = np.zeros(data_shape, dtype='d') << u.MJy / u.sr
            if power_loading_model is None:
                # in this case we just convert the detector surface brightness
                # to power with simple square passband.
                # convert det sb to pwr loading
                self.logger.debug(
                    "calculate power loading without loading model")
                det_pwr = np.zeroes(data_shape, dtype='d') << u.pW
                for array_name in self.array_names:
                    m = (det_array_name == array_name)
                    det_pwr[m] = self._sky_sb_to_pwr(
                        array_name=array_name,
                        det_s=det_s[m],
                        )
            else:
                if det_sky_traj is None:
                    raise ValueError(
                        "Power loading model requires det_sky_traj")
                with timeit(
                        f"calculate power loading with loading model "
                        f"{power_loading_model}"):
                    det_pwr = power_loading_model.evaluate_tod(
                        det_array_name=apt['array_name'],
                        det_s=det_s,
                        det_alt=det_sky_traj['alt'],
                        f_smp=f_smp,
                        noise_seed=None,
                        )
            self.logger.debug(
                f"power loading at detector: "
                f"min={det_pwr.min()} max={det_pwr.max()}")
            return det_pwr, locals()
        return evaluate

    def mapping_evaluator(
            self, mapping, sources=None,
            erfa_interp_len=300. << u.s,
            eval_interp_len=0.1 << u.s,
            catalog_model_render_pixel_size=0.5 << u.arcsec):
        if sources is None:
            sources = list()
        t0 = mapping.t0
        apt = self.array_prop_table

        hwp_cfg = self.hwp_config
        if hwp_cfg.rotator_enabled:
            def get_hwp_pa_t(t):
                # return the hwp position angle at time t
                return Angle(((hwp_cfg.f_rot * t).to_value(
                    u.dimensionless_unscaled) * 2. * np.pi) << u.rad)
        else:
            get_hwp_pa_t = None

        def evaluate(t, mapping_only=False):
            time_obs = t0 + t
            n_times = len(time_obs)
            self.logger.debug(
                f"evalute time_obs from {time_obs[0]} to "
                f"{time_obs[-1]} n_times={n_times}")
            if get_hwp_pa_t is None:
                hwp_pa_t = None
            else:
                hwp_pa_t = get_hwp_pa_t(t)
            # if True:
            with erfa_astrom.set(ErfaAstromInterpolator(erfa_interp_len)):
                with timeit("transform bore sight coords"):
                    # get bore sight trajectory and the hold flags
                    holdflag = mapping.evaluate_holdflag(t)
                    bs_coords = mapping.evaluate_coords(t)
                    bs_coords_icrs = bs_coords.transform_to('icrs')
                    bs_coords_altaz = bs_coords.transform_to('altaz')
                    bs_parallactic_angle = pa_from_coords(
                        observer=mapping.observer,
                        coords_altaz=bs_coords_altaz,
                        coords_icrs=bs_coords_icrs)
                    hwp_pa_altaz = hwp_pa_t + bs_coords_altaz.alt
                    hwp_pa_icrs = hwp_pa_altaz + bs_parallactic_angle
                    bs_sky_bbox_icrs = SkyBoundingBox.from_lonlat(
                        bs_coords_icrs.ra, bs_coords_icrs.dec)
                    bs_sky_bbox_altaz = SkyBoundingBox.from_lonlat(
                        bs_coords_altaz.az, bs_coords_altaz.alt)
                    self.logger.debug(
                        f"sky_bbox icrs={bs_sky_bbox_icrs} "
                        f"altaz={bs_sky_bbox_altaz}")
                # make the model to project detector positions
                det_sky_traj = self._get_detector_sky_traj(
                    time_obs=time_obs,
                    bs_coords_altaz=bs_coords_altaz,
                    bs_coords_icrs=bs_coords_icrs,
                    evaluate_interp_len=eval_interp_len)
                det_ra = det_sky_traj['ra']
                det_dec = det_sky_traj['dec']
                det_pa_icrs = det_sky_traj['pa_icrs']
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots(1, 1)
                # for i in range(400):
                #     for j in range(0, det_ra.shape[1], 10):
                #         plt.plot(
                #             [det_ra.degree[i, j]],
                #             [det_dec.degree[i, j]],
                #             marker=(2, 0, det_pa_icrs.degree[i, j]),
                #             markersize=5, linestyle=None)
                # plt.show()
                if mapping_only:
                    return locals()
                # get source flux from models
                s_additive = list()
                for m_source in sources:
                    # TODO maybe there is a faster way of handling the
                    # catalog source model directly. For now
                    # we just convert it to image model
                    if isinstance(m_source, CatalogSourceModel):
                        # get fwhms from toltec_info
                        fwhms = dict()
                        for array_name in self.array_names:
                            fwhms[array_name] = toltec_info[
                                array_name]['a_fwhm']
                        m_source = m_source.make_image_model(
                            fwhms=fwhms,
                            pixscale=catalog_model_render_pixel_size / u.pix
                            )
                    if isinstance(m_source, ImageSourceModel):
                        # TODO support more types of wcs. For now
                        # only ICRS is supported
                        with timeit(
                                "extract flux from source image model"):
                            s = m_source.evaluate_tod_icrs(
                                apt['array_name'],
                                det_ra,
                                det_dec,
                                det_pa_icrs,
                                hwp_pa_icrs=hwp_pa_icrs
                                )
                        s_additive.append(s)
                if len(s_additive) <= 0:
                    self.logger.debug("no surface brightness model available")
                    s = np.zeros(det_ra.shape) << u.MJy / u.sr
                else:
                    s = s_additive[0]
                    for _s in s_additive[1:]:
                        s += _s
                self.logger.debug(
                    f"surface brightness at detector: "
                    f"min={s.min()} max={s.max()}")
                return s, locals()
        return evaluate

    @contextmanager
    def iter_eval_context(self, simu_config):
        """Run the simuation defined by `simu_config`."""
        mapping_model = simu_config.mapping_model
        source_models = simu_config.source_models
        obs_params = simu_config.obs_params
        perf_params = simu_config.perf_params
        t_simu = simu_config.t_simu

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
        self.logger.debug(f"power loading model:\n{power_loading_model}")
        self.logger.warning(f"ignored sources:\n{sources_unknown}")

        # create the time grids
        # this is the iterative eval time grids
        t_chunks = make_time_grid(
            t=t_simu,
            f_smp=obs_params.f_smp_probing,
            chunk_len=perf_params.chunk_len)
        # this is used for doing pre-eval calcuation.
        t_grid_pre_eval = np.linspace(
                        0, t_simu.to_value(u.s),
                        perf_params.pre_eval_t_grid_size
                        ) << u.s

        # create the evaluators
        mapping_evaluator = self.mapping_evaluator(
            mapping=mapping_model, sources=sources_sb,
            erfa_interp_len=perf_params.mapping_erfa_interp_len,
            eval_interp_len=perf_params.mapping_eval_interp_len,
            catalog_model_render_pixel_size=(
                perf_params.catalog_model_render_pixel_size),
            )
        probing_evaluator = self.probing_evaluator(
            kids_fp=None,
            f_smp=obs_params.f_smp_probing,
            power_loading_model=power_loading_model,
            )

        # this context es is to hold any contexts during the iterative
        # eval
        es = ExitStack()
        # we run the mapping eval to get the det_sky_traj for the entire
        # simu
        mapping_info = mapping_evaluator(
            t_grid_pre_eval, mapping_only=True)
        # compute the extent for detectors
        bbox_padding = (1 << u.arcmin, 1 << u.arcmin)
        # here we add some padding to the bbox
        det_sky_traj = mapping_info['det_sky_traj']
        det_sky_bbox_icrs = SkyBoundingBox.from_lonlat(
            det_sky_traj['ra'], det_sky_traj['dec']).pad_with(
                *bbox_padding)
        det_sky_bbox_altaz = SkyBoundingBox.from_lonlat(
            det_sky_traj['az'], det_sky_traj['alt']).pad_with(
                *bbox_padding)
        self.logger.debug(
            f"det sky bbox: icrs={det_sky_bbox_icrs} "
            f"altaz={det_sky_bbox_altaz}")
        # setup interp for power loading model
        if power_loading_model is not None:
            alt_deg_step = perf_params.atm_eval_interp_alt_step.to_value(
                u.deg)
            interp_alt_grid = np.arange(
                det_sky_bbox_altaz.s.degree,
                det_sky_bbox_altaz.n.degree + alt_deg_step,
                alt_deg_step,
                ) << u.deg
            if len(interp_alt_grid) < 5:
                raise ValueError('atm_eval_interp_alt_step too small.')
            es.enter_context(
                power_loading_model.atm_eval_interp_context(
                    alt_grid=interp_alt_grid))
            # also we setup the toast slabs if atm_model_name is set to
            # toast
            if power_loading_model.atm_model_name == 'toast':
                es.enter_context(
                    power_loading_model.toast_atm_eval_context(
                        sky_bbox_altaz=det_sky_bbox_altaz
                        )
                    )
        # import matplotlib.pyplot as plt
        # import matplotlib.patches as patches
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(
        #     det_sky_traj['az'].degree, det_sky_traj['alt'].degree,
        #     linestyle='none', marker='o')
        # p = patches.Rectangle(
        #     (
        #         det_sky_bbox_altaz.w.to_value(u.deg),
        #         det_sky_bbox_altaz.s.to_value(u.deg),
        #         ),
        #     det_sky_bbox_altaz.width.to_value(u.deg),
        #     det_sky_bbox_altaz.height.to_value(u.deg),
        #     linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(p)
        # plt.show()

        # now we are ready to return the iterative evaluator
        def evaluate(t):
            det_s, mapping_info = mapping_evaluator(t)
            det_sky_traj = mapping_info['det_sky_traj']
            det_p = probing_evaluator(det_s=det_s, det_sky_traj=det_sky_traj)
            return locals()

        yield evaluate, t_chunks
        # release the contexts
        es.close()


class ToltecSimuOutputContext(ExitStack):
    """A context class to manage TolTEC simulator output files.

    """

    logger = get_logger()
    _lockfile = 'simu.lock'
    _statefile = 'simustate.yaml'

    def __init__(self, simulator, rootpath):
        super().__init__()
        self._simulator = simulator
        self._rootpath = rootpath
        self._state = None
        self._nms = dict()

    @property
    def simulator(self):
        return self._simulator

    @property
    def rootpath(self):
        return self._rootpath

    @property
    def state(self):
        return self._state

    @property
    def nms(self):
        """The dict of nc file mappers."""
        return self._nms

    def _create_nm(self, interface, suffix):
        if interface in self.nms:
            raise ValueError(f"NcNodeMapper already exists for {interface}")
        output_filepath = self.make_output_filename(interface, suffix)
        nm = self.nms[interface] = NcNodeMapper(
            source=output_filepath, mode='w')
        return nm

    def _get_tel_interface(self):
        return 'tel'

    def write_mapping_meta(self, mapping, simu_config):
        """Save the mapping model to tel.nc."""
        # create tel.nc
        nm_tel = self._create_nm(self._get_tel_interface(), '.nc')
        self.logger.debug(
            f"save mapping model {mapping} to {nm_tel}")
        nc_tel = nm_tel.nc_node
        # populate the headers
        # add some common settings to the nc tel header
        rootpath = simu_config.runtime_info.config_info.runtime_context_dir
        nm_tel.setstr(
                'Header.File.Name',
                nm_tel.file_loc.path.relative_to(rootpath).as_posix())
        nm_tel.setstr(
                'Header.Source.SourceName',
                simu_config.jobkey)
        if isinstance(mapping, (LmtTcsTrajMappingModel, )):
            self.logger.debug(
                    f"mapping model meta:\n{pformat_yaml(mapping.meta)}")
            nm_tel.setstr(
                    'Header.Dcs.ObsPgm',
                    mapping.meta['mapping_type'])
        elif mapping.pattern_kind & PatternKind.lissajous:
            # TODO handle lissajous
            nm_tel.setstr(
                    'Header.Dcs.ObsPgm',
                    'Lissajous')
        elif mapping.pattern_kind & PatternKind.raster_like:
            nm_tel.setstr(
                    'Header.Dcs.ObsPgm',
                    'Map')
        else:
            raise NotImplementedError
        # the len=2 is for mean and ref coordinates.
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

        # setup data variables for write_data
        d_time = 'time'
        nc_tel.createDimension(d_time, None)
        m = dict()  # this get added to the node mapper
        m['time'] = nc_tel.createVariable(
                'Data.TelescopeBackend.TelTime', 'f8', (d_time, ))
        v_ra = m['ra'] = nc_tel.createVariable(
                'Data.TelescopeBackend.TelRaAct', 'f8', (d_time, ))
        v_ra.unit = 'rad'
        v_dec = m['dec'] = nc_tel.createVariable(
                'Data.TelescopeBackend.TelDecAct', 'f8', (d_time, ))
        v_dec.unit = 'rad'
        v_alt = m['alt'] = nc_tel.createVariable(
                'Data.TelescopeBackend.TelElAct', 'f8', (d_time, ))
        v_alt.unit = 'rad'
        v_az = m['az'] = nc_tel.createVariable(
                'Data.TelescopeBackend.TelAzAct', 'f8', (d_time, ))
        v_az.unit = 'rad'
        v_pa = m['pa'] = nc_tel.createVariable(
                'Data.TelescopeBackend.ActParAng', 'f8', (d_time, ))
        v_pa.unit = 'rad'
        # the _sky az alt and pa are the corrected positions of telescope
        # they are the same as the above when no pointing correction
        # is applied.
        v_alt_sky = m['alt_sky'] = nc_tel.createVariable(
                'Data.TelescopeBackend.TelElSky', 'f8', (d_time, ))
        v_alt_sky.unit = 'rad'
        v_az_sky = m['az_sky'] = nc_tel.createVariable(
                'Data.TelescopeBackend.TelAzSky', 'f8', (d_time, ))
        v_az_sky.unit = 'rad'
        v_pa_sky = m['pa_sky'] = nc_tel.createVariable(
                'Data.TelescopeBackend.ParAng', 'f8', (d_time, ))
        v_pa_sky.unit = 'rad'
        m['hold'] = nc_tel.createVariable(
                'Data.TelescopeBackend.Hold', 'f8', (d_time, )
                )
        v_source_alt = m['source_alt'] = nc_tel.createVariable(
                'Data.TelescopeBackend.SourceEl', 'f8', (d_time, ))
        v_source_alt.unit = 'rad'
        v_source_az = m['source_az'] = nc_tel.createVariable(
                'Data.TelescopeBackend.SourceAz', 'f8', (d_time, ))
        v_source_az.unit = 'rad'
        nm_tel.update(m)
        return nm_tel

    def _get_kidsdata_interface(self, nw):
        return f'toltec{nw}'

    def _make_kidsdata_nc(self, nw, simu_config):
        sim = self._simulator
        state = self.state
        apt = sim.array_prop_table
        mapt = apt[apt['nw'] == nw]
        interface = self._get_kidsdata_interface(nw)
        nm_toltec = self._create_nm(interface, '_timestream.nc')
        nc_toltec = nm_toltec.nc_node
        # add meta data
        rootpath = simu_config.runtime_info.config_info.runtime_context_dir
        obs_params = simu_config.obs_params
        nm_toltec.setstr(
                'Header.Toltec.Filename',
                nm_toltec.file_loc.path.relative_to(rootpath).as_posix())
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
                obs_params.f_smp_probing.to_value(u.Hz))
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
        nc_toltec.createDimension('toneFreqLen', len(mapt))
        v_tones = nc_toltec.createVariable(
                'Header.Toltec.ToneFreq',
                'f8', ('numSweeps', 'toneFreqLen')
                )
        v_tones[0, :] = mapt['fp']
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
                    v = mapt[v]
                v_mp[0, i, :] = v

        # data variables
        m = dict()
        nc_toltec.createDimension('loclen', len(mapt))
        nc_toltec.createDimension('iqlen', len(mapt))
        nc_toltec.createDimension('tlen', 6)
        nc_toltec.createDimension('time', None)
        m['flo'] = nc_toltec.createVariable(
                'Data.Toltec.LoFreq', 'i4', ('time', ))
        m['time'] = nc_toltec.createVariable(
                'Data.Toltec.Ts', 'i4', ('time', 'tlen'))
        m['I'] = nc_toltec.createVariable(
                'Data.Toltec.Is', 'i4', ('time', 'iqlen'))
        m['Q'] = nc_toltec.createVariable(
                'Data.Toltec.Qs', 'i4', ('time', 'iqlen'))
        nm_toltec.update(m)
        return nm_toltec

    def _get_hwp_interface(self):
        return 'hwp'

    def _make_hwp_nc(self, simu_config):
        sim = self._simulator
        hwp_cfg = sim.hwp_config
        if not hwp_cfg.rotator_enabled:
            return None
        state = self.state
        interface = self._get_hwp_interface()
        nm_hwp = self._create_nm(interface, '.nc')
        nc_hwp = nm_hwp.nc_node
        # add meta data
        rootpath = simu_config.runtime_info.config_info.runtime_context_dir
        nm_hwp.setstr(
                'Header.Toltec.Filename',
                nm_hwp.file_loc.path.relative_to(rootpath).as_posix())
        nm_hwp.setscalar(
                'Header.Toltec.ObsType', 1, dtype='i4')  # Timestream
        nm_hwp.setscalar(
                'Header.Toltec.Master', 0, dtype='i4')
        nm_hwp.setscalar(
                'Header.Toltec.RepeatLevel', 0, dtype='i4')
        nm_hwp.setscalar(
                'Header.Toltec.ObsNum', state['obsnum'], dtype='i4')
        nm_hwp.setscalar(
                'Header.Toltec.SubObsNum',
                state['subobsnum'], dtype='i4')
        nm_hwp.setscalar(
                'Header.Toltec.ScanNum',
                state['scannum'], dtype='i4')
        nm_hwp.setscalar(
                'Header.Toltec.TargSweepObsNum',
                state['cal_obsnum'], dtype='i4')
        nm_hwp.setscalar(
                'Header.Toltec.TargSweepSubObsNum',
                state['cal_subobsnum'], dtype='i4')
        nm_hwp.setscalar(
                'Header.Toltec.TargSweepScanNum',
                state['cal_scannum'], dtype='i4')
        nm_hwp.setscalar(
                'Header.Hwp.SampleFreq',
                hwp_cfg.f_smp.to_value(u.Hz))
        # data variables
        m = dict()
        nc_hwp.createDimension('tlen', 6)
        nc_hwp.createDimension('time', None)
        m['pa'] = nc_hwp.createVariable(
                'Data.Hwp.', 'f8', ('time', ))
        m['time'] = nc_hwp.createVariable(
                'Data.Toltec.Ts', 'i4', ('time', 'tlen'))
        nm_hwp.update(m)
        return nm_hwp

    def write_sim_meta(self, simu_config):
        # apt.ecsv
        apt = self._simulator.array_prop_table
        apt.write(self.make_output_filename('apt', '.ecsv'),
                  format='ascii.ecsv', overwrite=True)
        # toltec*.nc
        for nw in np.unique(apt['nw']):
            self._make_kidsdata_nc(nw, simu_config)
        # hwp.nc
        self._make_hwp_nc(simu_config)

    def write_sim_data(self, data):
        pass

    def open(self, overwrite=False):
        """Open files to save data.

        This increments the obsnum state and makes available a set of opened
        file handlers in :attr:`nms`.

        Parameters
        ----------
        overwrite : bool, optional
            If True, the obsnum is not incremented.
        """
        state = self._state = self.enter_context(self.writelock())
        # check if the previous data is valid
        valid = state.get('valid', False)
        if not valid:
            self.logger.warning(f"overwrite invalid state entry:\n{state}")
            overwrite = True
        if not overwrite:
            state['obsnum'] += 1
            state['cal_obsnum'] += 1
        state['ut'] = datetime.utcnow()
        state['valid'] = False
        state.sync()
        self.logger.info(f"simulator output state:\n{state}")
        return self

    def __exit__(self, *args):
        self.state['valid'] = True
        self.state.sync()
        super().__exit__(*args)
        # make a copy of the state file for this obsnum
        shutil.copy(self.state.filepath, self.make_output_filename(
            'simustate', '.yaml'))
        # clean up the contexts
        self._state = None
        # close all files and clear the mapper
        for nm in self.nms.values():
            nm.close()
        self.nms.clear()

    @contextmanager
    def writelock(self):
        outdir = self.rootpath
        lockfile = outdir.joinpath(self._lockfile)
        if lockfile.exists():
            raise RuntimeError(f"cannot acquire write lock for {outdir}")
        state = PersistentState(
                outdir.joinpath(self._statefile),
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
                self.logger.debug(f'create lock file: {lockfile}')
                pass
            yield state.sync()
        finally:
            try:
                lockfile.unlink()
                self.logger.debug(f'unlink lock file: {lockfile}')
            except Exception:
                self.logger.debug("failed release write lock", exc_info=True)

    def make_output_filename(self, interface, suffix):
        state = self.state
        filename = (
                f'{interface}_{state["obsnum"]:06d}_'
                f'{state["subobsnum"]:03d}_'
                f'{state["scannum"]:04d}_'
                f'{state["ut"].strftime("%Y_%m_%d_%H_%M_%S")}'
                f'{suffix}'
                )
        return self.rootpath.joinpath(filename)
