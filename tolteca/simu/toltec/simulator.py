#!/usr/bin/env python

from .models import (
    ToltecArrayProjModel, ToltecSkyProjModel, pa_from_coords,
    ToltecPowerLoadingModel)
from .toltec_info import toltec_info
from ..utils import (
    PersistentState, SkyBoundingBox, get_lon_extent, make_time_grid)
from ..mapping import (PatternKind, LmtTcsTrajMappingModel)
from ..mapping.utils import resolve_sky_coords_frame
from ..sources.base import (SurfaceBrightnessModel, )
from ..sources.models import (ImageSourceModel, CatalogSourceModel)

from ...utils.common_schema import PhysicalTypeSchema
from tollan.utils.nc import NcNodeMapper
from tollan.utils.log import get_logger, timeit
from tollan.utils.fmt import pformat_yaml
from tollan.utils.dataclass_schema import add_schema
from tollan.utils import rupdate
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
from astropy.modeling.functional_models import GAUSSIAN_SIGMA_TO_FWHM


__all__ = ['ToltecObsSimulator', 'ToltecHwpConfig']


@add_schema
@dataclass
class ToltecHwpConfig(object):
    """The config class for TolTEC half-wave plate and the rotator."""

    f_rot: u.Quantity = field(
        default=2. << u.Hz,
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
        default=True,
        metadata={
            'description': 'True if rotate the HWPR.'
            }
        )
    installed: bool = field(
        default=True,
        metadata={
            'description': 'True if HWP installed in the beam.'
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
        self.logger.info(f"normalized apt:\n{apt}")
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
        'm0': 2e6,
        'm1': 2e6
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
        if 'array_name' not in tbl.colnames:
            # generate array_name from array
            array_names = [toltec_info["array_names"][a] for a in tbl['array']]
            tbl['array_name'] = array_names
        array_names = tbl.meta['array_names'] = np.unique(
            tbl['array_name']).tolist()
        if 'f' not in tbl.colnames:
            if 'kids_fr' in tbl.colnames:
                tbl['f'] = tbl['kids_fr']
            else:
                raise ValueError("apt table missing KIDs frequency info.")
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
        # if 'flxscale' not in tbl.colnames:
        #     tbl['flxscale'] = (1. / tbl['responsivity']).quantity.value

        # kids readout noise
        if 'sigma_readout' not in tbl.colnames:
            tbl['sigma_readout'] = 10.

        # detector locations in toltec frame
        if not {'x_t', 'y_t'}.issubset(tbl.colnames):
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
        if 'pa_t' not in tbl.colnames:
            tbl['pa_t'] = 0. << u.deg
        # add more columns for citlali v2:
        for c in ["tone_freq", "x_t_raw", "y_t_raw", "x_t_derot", "y_t_derot"]:
            if c not in tbl.colnames:
                tbl[c] = 0.
        return QTable(tbl)

    def __str__(self):
        return (
            f'{self.__class__.__name__}('
            f'n_detectors={len(self.array_prop_table)}, '
            f'polarized={self.polarized})'
            )

    def output_context(self, dirpath, **kwargs):
        return ToltecSimuOutputContext(
            simulator=self, rootpath=dirpath, **kwargs)

    @timeit
    def _get_detector_sky_traj(
            self,
            time_obs,
            bs_coords_altaz,
            bs_coords_icrs,
            evaluate_interp_len=None,
            lon_wrap_angle_altaz=None,
            lon_wrap_angle_icrs=None):
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
            # set the lon wrap angle if specified
            if lon_wrap_angle_altaz is not None:
                det_sky_traj['az'].wrap_angle = lon_wrap_angle_altaz
            if lon_wrap_angle_icrs is not None:
                det_sky_traj['ra'].wrap_angle = lon_wrap_angle_icrs
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
        # note that the longitude interp has to work outside of the
        # wrap angle. We determine the wrap angle and re-wrap it for interp
        det_az = det_sky_traj_s['az']
        _, _, det_az_wrap_angle = get_lon_extent(det_az)
        det_az = Longitude(det_az, wrap_angle=det_az_wrap_angle)
        det_ra = det_sky_traj_s['ra']
        _, _, det_ra_wrap_angle = get_lon_extent(det_ra)
        det_ra = Longitude(det_ra, wrap_angle=det_ra_wrap_angle)

        mjd_day_s = mjd[s].to_value(u.day)
        interp_kwargs = dict(kind='linear', axis=1)
        az_deg_interp = interp1d(
                mjd_day_s,
                det_az.degree, **interp_kwargs)
        alt_deg_interp = interp1d(
                mjd_day_s,
                det_sky_traj_s['alt'].degree, **interp_kwargs)
        pa_altaz_deg_interp = interp1d(
            mjd_day_s,
            det_sky_traj_s['pa_altaz'].to_value(u.deg), **interp_kwargs)

        ra_deg_interp = interp1d(
                mjd_day_s,
                det_ra.degree, **interp_kwargs)
        dec_deg_interp = interp1d(
                mjd_day_s,
                det_sky_traj_s['dec'].degree, **interp_kwargs)
        pa_icrs_deg_interp = interp1d(
            mjd_day_s,
            det_sky_traj_s['pa_icrs'].to_value(u.deg), **interp_kwargs)
        # interp for full time steps
        # we use the wrap angle when it is not specified
        if lon_wrap_angle_altaz is None:
            lon_wrap_angle_altaz = det_az_wrap_angle
        if lon_wrap_angle_icrs is None:
            lon_wrap_angle_icrs = det_ra_wrap_angle
        mjd_day = mjd.to_value(u.day)
        det_sky_traj = dict()
        det_sky_traj['az'] = Longitude(
            az_deg_interp(mjd_day) << u.deg,
            wrap_angle=lon_wrap_angle_altaz)
        det_sky_traj['alt'] = Latitude(alt_deg_interp(mjd_day) << u.deg)
        det_sky_traj['pa_altaz'] = Angle(pa_altaz_deg_interp(mjd_day) << u.deg)
        det_sky_traj['ra'] = Longitude(
            ra_deg_interp(mjd_day) << u.deg,
            wrap_angle=lon_wrap_angle_icrs)
        det_sky_traj['dec'] = Latitude(dec_deg_interp(mjd_day) << u.deg)
        det_sky_traj['pa_icrs'] = Angle(pa_icrs_deg_interp(mjd_day) << u.deg)
        return det_sky_traj

    def probing_evaluator(
            self,
            f_smp,
            kids_p_tune,
            kids_fp=None,
            power_loading_model=None):
        """Return a function that can be used to calculate detector readout.

        When `power_loading_model` is given, the detector signal will be the
        sum of the contribution from the astronomical source and the power
        loading model which includes the telescope and atmosphere::

            P_det = P_src + P_bkg_fixture + P_atm(alt)

        We set the tune of the KidsSimulator,such that x=0 at P=kids_p_tune,
        where kids_p_tune is typically calculated with the loading model
        at some altitude P_bkg_fixture + P_atm(alt_tune)

        Thus the measured detuning parameters is proportional to

            P_src + (P_atm(alt) - P_atm(alt_tune))
        """

        apt = self.array_prop_table
        det_array_name = apt['array_name']
        kids_simu = self.kids_simulator.copy()
        kids_simu._background = kids_p_tune
        kids_readout_model = self.kids_readout_model
        if kids_fp is None:
            kids_fp = kids_simu.fr
        if power_loading_model is not None:
            # make sure this is an instance of the toltec power loading model
            if not isinstance(power_loading_model, ToltecPowerLoadingModel):
                raise ValueError("invalid power loading model.")

        def kids_probe_p(det_p):
            nonlocal kids_simu
            return kids_simu.probe_p(
                det_p,
                fp=kids_fp, readout_model=kids_readout_model)

        def sky_sb_to_pwr(det_s):
            if power_loading_model is None:
                # in this case we just convert the detector surface brightness
                # to power with simple square passband.
                # convert det sb to pwr loading
                return (
                    det_s.to(
                            u.K,
                            equivalencies=u.brightness_temperature(
                                apt['wl_center'][:, np.newaxis])).to(
                                u.J,
                                equivalencies=u.temperature_energy())
                            * apt['passband'][:, np.newaxis]
                    ).to(u.pW)
            return power_loading_model.sky_sb_to_pwr(
                det_array_name=det_array_name,
                det_s=det_s,
                )

        def evaluate(
                det_s=None,
                det_sky_traj=None,
                time_obs=None,
                det_add_pwr=None,
                det_scale_pwr=None,
                ):
            # make sure we have at least some input for eval
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
                det_pwr = sky_sb_to_pwr(det_s)
            else:
                if det_sky_traj is None:
                    raise ValueError(
                        "Power loading model requires det_sky_traj")
                with timeit(
                        f"calculate power loading with loading model "
                        f"{power_loading_model}"):
                    det_pwr = power_loading_model.evaluate_tod(
                        det_array_name=det_array_name,
                        det_s=det_s,
                        det_az=det_sky_traj['az'],
                        det_alt=det_sky_traj['alt'],
                        f_smp=f_smp,
                        noise_seed=None,
                        time_obs=time_obs,
                        )
            if det_add_pwr is not None:
                det_pwr += det_add_pwr
            if det_scale_pwr is not None:
                det_pwr = det_pwr * det_scale_pwr
            # calculate the kids signal
            nonlocal kids_probe_p
            rs, xs, iqs = kids_probe_p(det_pwr)

            for array_name in self.array_names:
                m = (det_array_name == array_name)
                _p = det_pwr[m]
                _s = det_s[m]
                _rs = rs[m]
                _xs = xs[m]
                _iqs = iqs[m]
                summary_tbl = []
                summary_rownames = ['S', 'P', 'x', 'r', 'I', 'Q']
                summary_vars = [_s, _p, _xs, _rs, _iqs.real, iqs.imag]
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

            # self.logger.info(
            #     f"power loading at detector: "
            #     f"min={det_pwr.min()} max={det_pwr.max()}")
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1, 1)
            # ax.set_aspect('equal')
            # for i in [0, ]:
            #     ax.plot([0, ], [0, ])
            #     ax.plot(
            #         iqs.real[i], iqs.imag[i], linestyle='none', marker='.')
            # plt.show()
            return det_pwr, locals()
        return evaluate, locals()

    def mapping_evaluator(
            self, mapping, sources=None,
            pointing_model_altaz=None,
            erfa_interp_len=300. << u.s,
            eval_interp_len=0.1 << u.s,
            catalog_model_render_pixel_size=0.5 << u.arcsec):
        if sources is None:
            sources = list()
        t0 = mapping.t0
        apt = self.array_prop_table

        hwp_cfg = self.hwp_config

        def get_hwp_pa_t(t):
            # return the hwp position angle at time t
            # if enabled
            if hwp_cfg.rotator_enabled:
                return Angle(((hwp_cfg.f_rot * t).to_value(
                    u.dimensionless_unscaled) * 2. * np.pi) << u.rad)
            # just a constant
            return Angle(np.full(t.shape, 0.) << u.rad)

        # convert the catalog source model to image source model, if any
        source_models_for_eval = list()
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
            source_models_for_eval.append(m_source)

        def evaluate(
                t, mapping_only=False,
                lon_wrap_angle_altaz=None,
                lon_wrap_angle_icrs=None,
                ):
            time_obs = t0 + t
            n_times = len(time_obs)
            self.logger.debug(
                f"evaluate time_obs from {time_obs[0]} to "
                f"{time_obs[-1]} n_times={n_times}")
            self.logger.debug(
                f'use lon_wrap_angle_icrs={lon_wrap_angle_icrs} '
                f'lon_wrap_angle_altaz={lon_wrap_angle_altaz}')
            # TODO add more control for the hwp position
            hwp_pa_t = get_hwp_pa_t(t)
            # if True:
            with erfa_astrom.set(ErfaAstromInterpolator(erfa_interp_len)):
                with timeit("transform bore sight coords"):
                    # get bore sight trajectory and the hold flags
                    holdflag = mapping.evaluate_holdflag(t)
                    bs_coords = mapping.evaluate_coords(t)
                    bs_coords_icrs = bs_coords.transform_to('icrs')
                    # for the altaz we need to resolve the frame for the
                    # toltec observer and time
                    bs_coords_altaz = bs_coords.transform_to(
                        resolve_sky_coords_frame(
                            'altaz',
                            observer=mapping.observer,
                            time_obs=time_obs
                            ))
                    if pointing_model_altaz is not None:
                        po_az, po_alt = pointing_model_altaz(t)
                        self.logger.debug(
                            f"applying pointing offset in altaz: {po_az.mean()} {po_alt.mean()}")
                        bs_coords_altaz = bs_coords_altaz.spherical_offsets_by(
                            po_az,
                            po_alt
                        )
                        bs_coords_icrs = bs_coords_altaz.transform_to("icrs")
                    bs_parallactic_angle = pa_from_coords(
                        observer=mapping.observer,
                        coords_altaz=bs_coords_altaz,
                        coords_icrs=bs_coords_icrs)
                    target_altaz = mapping.target.transform_to(
                        bs_coords_altaz.frame)
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
                    evaluate_interp_len=eval_interp_len,
                    lon_wrap_angle_altaz=lon_wrap_angle_altaz,
                    lon_wrap_angle_icrs=lon_wrap_angle_icrs,
                    )
                det_ra = det_sky_traj['ra']
                det_dec = det_sky_traj['dec']
                det_pa_icrs = det_sky_traj['pa_icrs']
                det_sky_bbox_icrs = SkyBoundingBox.from_lonlat(
                    det_ra, det_dec)
                det_sky_bbox_altaz = SkyBoundingBox.from_lonlat(
                    det_sky_traj['az'], det_sky_traj['alt'])
                self.logger.info(
                    f"det sky traj bbox:\n"
                    f"ra: {det_sky_bbox_icrs.w!s} - {det_sky_bbox_icrs.e!s}\n"
                    f"dec: {det_sky_bbox_icrs.s!s} - {det_sky_bbox_icrs.n!s}\n"
                    f"az: {det_sky_bbox_altaz.w!s} "
                    f"- {det_sky_bbox_altaz.e!s}\n"
                    f"alt: {det_sky_bbox_altaz.s!s} "
                    f"- {det_sky_bbox_altaz.n!s}\n"
                    f"size: {det_sky_bbox_icrs.width}, "
                    f"{det_sky_bbox_icrs.height}"
                    )

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
                for m_source in source_models_for_eval:
                    # TODO maybe there is a faster way of handling the
                    # catalog source model directly. For now
                    # we converted it to image model
                    if isinstance(m_source, ImageSourceModel):
                        # TODO support more types of wcs. For now
                        # only ICRS is supported
                        with timeit(
                                "extract flux from source image model"):
                            # we only pass the pa and hwp when we want
                            # them to be eval for polarimetry
                            source_eval_kw = dict()
                            if self.polarized:
                                source_eval_kw['det_pa_icrs'] = det_pa_icrs
                                if hwp_cfg.installed:
                                    source_eval_kw['hwp_pa_icrs'] = hwp_pa_icrs
                            s = m_source.evaluate_tod_icrs(
                                apt['array_name'],
                                det_ra,
                                det_dec,
                                **source_eval_kw
                                )
                        s_additive.append(s)
                if len(s_additive) <= 0:
                    self.logger.debug("no surface brightness model available")
                    s = np.zeros(det_ra.shape) << u.MJy / u.sr
                else:
                    s = s_additive[0]
                    for _s in s_additive[1:]:
                        s += _s
                self.logger.info(
                    f"source surface brightness at detector: "
                    f"min={s.min()} max={s.max()}")
                return s, locals()
        return evaluate, locals()

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
        if sources_unknown:
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

        # create the mapping evaluator first so that we can get full
        # sky bbox for the observation
        # the altitude is needed to create the probing evaluator
        mapping_evaluator, mapping_eval_ctx = self.mapping_evaluator(
            mapping=mapping_model, sources=sources_sb,
            erfa_interp_len=perf_params.mapping_erfa_interp_len,
            eval_interp_len=perf_params.mapping_eval_interp_len,
            catalog_model_render_pixel_size=(
                perf_params.catalog_model_render_pixel_size),
            )
        # this context es is to hold any contexts during the iterative
        # eval
        es = ExitStack()
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
        # this apt will be modified and saved in the eval context
        apt = self.array_prop_table.copy()
        det_array_name = apt['array_name']
        # setup interp for power loading model
        if power_loading_model is not None:
            alt_deg_step = perf_params.aplm_eval_interp_alt_step.to_value(
                u.deg)
            n_min_steps = 5
            if (alt_deg_step * (n_min_steps - 1)) \
                    > det_sky_bbox_altaz.height.degree:
                alt_step_new = (
                    det_sky_bbox_altaz.height / (n_min_steps - 1)).to(
                        perf_params.aplm_eval_interp_alt_step.unit)
                # self.logger.warning(
                raise ValueError(
                    f"perf_params.aplm_eval_interp_alt_step="
                    f"{perf_params.aplm_eval_interp_alt_step} is too large, "
                    f"The value should be adjusted to at most {alt_step_new} "
                    f"to ensure at least {n_min_steps} steps.")
                alt_deg_step = alt_step_new.to_value(u.deg)
            interp_alt_grid = np.arange(
                det_sky_bbox_altaz.s.degree,
                det_sky_bbox_altaz.n.degree + alt_deg_step,
                alt_deg_step,
                ) << u.deg
            # if len(interp_alt_grid) < 5:
            #     raise ValueError(
            #         'aplm_eval_interp_alt_step too large.')
            es.enter_context(
                power_loading_model.aplm_eval_interp_context(
                    t0=mapping_model.t0,
                    t_grid=t_grid_pre_eval,
                    sky_bbox_altaz=det_sky_bbox_altaz,
                    alt_grid=interp_alt_grid,
                    ))

        # figure out the tune power  and flxscale
        # we use the closest point on the boresight to the target
        # for the tune obs
        # when each detector are tuned at different
        # power, the time stream x values will have different
        # zero point. The effect has to taken into account
        # before making the map
        # in the simulator, we simplify this effect by
        # setting the tune power at the mean value for all the
        # detectors
        bs_coords_icrs = mapping_info['bs_coords_altaz']
        target_icrs = mapping_model.target.transform_to('icrs')
        i_closest = np.argmin(
            target_icrs.separation(bs_coords_icrs))

        det_az_tune = det_sky_traj['az'][:, i_closest]
        det_alt_tune = det_sky_traj['alt'][:, i_closest]
        t_tune = t_grid_pre_eval[i_closest]

        self.logger.debug(
            f"use tune at detector alt={det_alt_tune.mean()} t={t_tune}")
        if (
                power_loading_model is None
                or power_loading_model.atm_model_name is None):
            # when power loading model is not set, we use the apt default
            # in this case we also need to add to the det power the
            # additional background loading
            kids_p_tune_det = apt['background']
            det_add_background_loading = kids_p_tune_det[:, np.newaxis]
        else:
            det_add_background_loading = None
            kids_p_tune_det = power_loading_model.get_P(
                det_array_name=det_array_name,
                det_az=Angle(np.full(
                    len(det_array_name), det_az_tune.degree) << u.deg),
                det_alt=Angle(np.full(
                    len(det_array_name), det_alt_tune.degree) << u.deg),
                time_obs=mapping_model.t0 + t_tune,
                )
        kids_p_tune = kids_p_tune_det.copy()
        for array_name in self.array_names:
            m = (det_array_name == array_name)
            p = kids_p_tune_det[m].mean()
            kids_p_tune[m] = p
            self.logger.debug(f"set tune of {array_name} at P={p}")

        # TODO allow adjust kids_fp
        probing_evaluator, probing_eval_ctx = self.probing_evaluator(
            kids_p_tune=kids_p_tune,
            kids_fp=None,
            f_smp=obs_params.f_smp_probing,
            power_loading_model=power_loading_model,
            )
        # compute the detector flxscale for sb = 1MJy
        kids_probe_p = probing_eval_ctx['kids_probe_p']
        sky_sb_to_pwr = probing_eval_ctx['sky_sb_to_pwr']
        p_sb_unity = np.squeeze(sky_sb_to_pwr(
            det_s=np.ones((len(det_array_name), 1)) << u.MJy / u.sr
            ))
        # the flxscale is (p_sb_unity / responsivity)
        flxscale = 1 / (p_sb_unity * apt['responsivity']).to_value(
            u.dimensionless_unscaled)
        # p_norm = p_sb_unity + kids_p_tune
        # _, x_norm, _ = kids_probe_p(p_norm[:, np.newaxis])
        # x_norm = np.squeeze(x_norm)
        # flxscale = 1. / x_norm
        # update the apt
        apt['background'] = kids_p_tune
        # change flxscale unit to mJy/beam/x
        apt['mJybeam_per_MJysr'] = np.nan
        for array_name in toltec_info['array_names']:
            fwhm = toltec_info[array_name]['a_fwhm']
            beam_area = 2 * np.pi * (fwhm / GAUSSIAN_SIGMA_TO_FWHM) ** 2
            conv = (1 << u.mJy/u.beam).to_value(u.MJy / u.sr, equivalencies=u.beam_angular_area(beam_area))
            apt['mJybeam_per_MJysr'][apt['array_name'] == array_name] = 1 / conv
        # only extract good detectors for computation
        flxscale = flxscale * apt['mJybeam_per_MJysr'] << u.mJy/u.beam
        # compare the internel flxscale with desired flxscale
        # this is additional factor that goes to the generated x values
        if 'flxscale' not in apt.colnames:
            apt['flxscale'] = flxscale
        flxscale2 = apt['flxscale'] / flxscale
        self.logger.debug(f"rescale flxscale {flxscale2.mean()}")

        for array_name in self.array_names:
            m = (det_array_name == array_name)
            _kids_p_tune = kids_p_tune[m].mean()
            _p_sb_unity = p_sb_unity[m].mean()
            # _p_norm = p_norm[m].mean()
            # _x_norm = x_norm[m].mean()
            _flxscale = flxscale[m].mean()
            if det_add_background_loading is not None:
                _add_bkg_pwr = det_add_background_loading[m].mean()
            else:
                _add_bkg_pwr = None
            self.logger.debug(
                f"summary of probing setup for {array_name}:\n"
                f"    kids_p_tune={_kids_p_tune}\n"
                f"    p_sb_unity={_p_sb_unity}\n"
                f"    add_bkg_pwr={_add_bkg_pwr}\n"
                # f"    p_norm={_p_norm}\n"
                # f"    x_norm={_x_norm}\n"
                f"    flxscale={_flxscale}\n"
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
        # det_az_tune = det_sky_traj['az'][:, i_closest]
        # c = kids_p_tune.to_value(u.pW)[m]
        # ax.scatter(
        #     det_az_tune[m], det_alt_tune[m],
        #     c=c, s=100, vmin=c.min(), vmax=c.max())
        # plt.show()

        # now we are ready to return the iterative evaluator
        def evaluate(t):
            det_s, mapping_info = mapping_evaluator(
                t,
                lon_wrap_angle_altaz=det_sky_bbox_altaz.lon_wrap_angle,
                lon_wrap_angle_icrs=det_sky_bbox_icrs.lon_wrap_angle,
                )
            det_sky_traj = mapping_info['det_sky_traj']
            det_pwr, probing_info = probing_evaluator(
                det_s=det_s,
                det_sky_traj=det_sky_traj,
                time_obs=mapping_info['time_obs'],
                det_add_pwr=det_add_background_loading,
                det_scale_pwr=flxscale2[:, np.newaxis],
                )
            return locals()

        self._eval_context = locals()
        yield evaluate, t_chunks
        # release the contexts
        es.close()
        self._eval_context = None


class ToltecSimuOutputContext(ExitStack):
    """A context class to manage TolTEC simulator output files.

    """

    logger = get_logger()
    _lockfile = 'simu.lock'
    _statefile = 'simustate.yaml'

    def __init__(self, simulator, rootpath, state_init=None):
        super().__init__()
        self._simulator = simulator
        self._rootpath = rootpath
        self._state = None
        self._nms = dict()
        self._state_init = state_init

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
                'Header.Sim.Jobkey',
                simu_config.jobkey)
        nm_tel.setstr(
                'Header.Source.SourceName',
                simu_config.jobkey)
        nm_tel.setstr(
                'Header.Dcs.ProjectId',
                simu_config.jobkey)
        nm_tel.setstr(
                'Header.Dcs.ObsGoal',
                'Other')

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
            nm_tel.setscalar(
                    'Header.Map.ExecMode',
                    1, dtype=int)
        elif mapping.pattern_kind & PatternKind.raster_like:
            nm_tel.setstr(
                    'Header.Dcs.ObsPgm',
                    'Map')
            if mapping.pattern_kind & PatternKind.raster:
                nm_tel.setscalar(
                        'Header.Map.ExecMode',
                        0, dtype=int)
            else:
                nm_tel.setscalar(
                        'Header.Map.ExecMode',
                        1, dtype=int)
            if mapping.ref_frame == "icrs":
                nm_tel.setstr("Header.Map.MapCoord", "Ra")
            else:
                nm_tel.setstr("Header.Map.MapCoord", "Az")
            nm_tel.setscalar("Header.Map.XLength", mapping.offset_mapping_model.length.quantity.to_value(u.rad))
            nm_tel.setscalar("Header.Map.YLength", min(mapping.offset_mapping_model.n_scans.value - 1, 1) * mapping.offset_mapping_model.space.quantity.to_value(u.rad))
            nm_tel.setscalar("Header.Map.ScanAngle", mapping.offset_mapping_model.rot.quantity.to_value(u.rad))
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
        # tau values
        self._set_tel_tau(
                nm_tel, tau_values={'tau225': 0., 'tau_a1100': 0., 'tau_a1400': 0., 'tau_a2000': 0.})

        # setup data variables for write_data
        d_time = 'time'
        nc_tel.createDimension(d_time, None)
        m = dict()  # this get added to the node mapper
        m['time'] = nc_tel.createVariable(
                'Data.TelescopeBackend.TelTime', 'f8', (d_time, ))
        m['pps_time'] = nc_tel.createVariable(
                'Data.TelescopeBackend.PpsTime', 'f8', (d_time, ))

        v_ra = m['ra'] = nc_tel.createVariable(
                'Data.TelescopeBackend.TelRaAct', 'f8', (d_time, ))
        v_ra.unit = 'rad'
        v_dec = m['dec'] = nc_tel.createVariable(
                'Data.TelescopeBackend.TelDecAct', 'f8', (d_time, ))
        v_dec.unit = 'rad'

        v_ra_src = m['ra_src'] = nc_tel.createVariable(
                'Data.TelescopeBackend.SourceRaAct', 'f8', (d_time, ))
        v_ra_src.unit = 'rad'
        v_dec_src = m['dec_src'] = nc_tel.createVariable(
                'Data.TelescopeBackend.SourceDecAct', 'f8', (d_time, ))
        v_dec_src.unit = 'rad'

        v_alt = m['alt'] = nc_tel.createVariable(
                'Data.TelescopeBackend.TelElAct', 'f8', (d_time, ))
        v_alt.unit = 'rad'
        v_az = m['az'] = nc_tel.createVariable(
                'Data.TelescopeBackend.TelAzAct', 'f8', (d_time, ))
        v_az.unit = 'rad'

        v_alt_cor = m['alt_cor'] = nc_tel.createVariable(
                'Data.TelescopeBackend.TelElCor', 'f8', (d_time, ))
        v_alt_cor.unit = 'rad'
        v_az_cor = m['az_cor'] = nc_tel.createVariable(
                'Data.TelescopeBackend.TelAzCor', 'f8', (d_time, ))
        v_az_cor.unit = 'rad'

        v_alt_des = m['alt_des'] = nc_tel.createVariable(
                'Data.TelescopeBackend.TelElDes', 'f8', (d_time, ))
        v_alt_des.unit = 'rad'
        v_az_des = m['az_des'] = nc_tel.createVariable(
                'Data.TelescopeBackend.TelAzDes', 'f8', (d_time, ))
        v_az_des.unit = 'rad'

        v_alt_map = m['alt_map'] = nc_tel.createVariable(
                'Data.TelescopeBackend.TelElMap', 'f8', (d_time, ))
        v_alt_map.unit = 'rad'
        v_az_map = m['az_map'] = nc_tel.createVariable(
                'Data.TelescopeBackend.TelAzMap', 'f8', (d_time, ))
        v_az_map.unit = 'rad'

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

    def _make_kidsdata_nc(self, apt, nw, simu_config):
        state = self.state
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
        nm_toltec.setscalar(
                'Header.Toltec.FpgaFreq',
                2.56e8)

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
                'f_centered': ('fp', u.Hz),
                'f_out': ('fr', u.Hz),
                'f_in': ('fp', u.Hz),
                'flag': None,
                'fp': ('fp', u.Hz),
                'Qr': 'Qr',
                'Qc': 1.,
                'fr': ('fr', u.Hz),
                'A': 0.,
                'normI': 'g0',
                'normQ': 'g1',
                'slopeI': ('k0', u.s),
                'slopeQ': ('k1', u.s),
                'interceptI': 'm0',
                'interceptQ': 'm1',
                }
        for i, (k, v) in enumerate(mp_map.items()):
            v_mph[i, :] = netCDF4.stringtochar(np.array(k, '|S32'))
            if v is not None:
                if isinstance(v, str):
                    v = mapt[v]
                elif isinstance(v, tuple):
                    v = mapt[v[0]].to_value(v[1])
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
        return 'hwpr'

    def _make_hwp_nc(self, simu_config):
        sim = self._simulator
        hwp_cfg = sim.hwp_config
        # if not hwp_cfg.rotator_enabled:
        #     return None
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
        nm_hwp.setscalar(
                'Header.Hwp.RotatorEnabled',
                hwp_cfg.rotator_enabled, dtype='i4')
        nm_hwp.setscalar(
                'Header.Hwp.Installed',
                hwp_cfg.installed, dtype='i4')
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

    def _ensure_sim_eval_context(self):
        if getattr(self._simulator, '_eval_context', None) is None:
            raise ValueError(
                "Cannot write sim meta without eval context open")
        return self._simulator._eval_context

    def write_sim_meta(self, simu_config):
        eval_ctx = self._ensure_sim_eval_context()
        # apt.ecsv
        apt = eval_ctx['apt']
        # write the apt
        apt.write(self.make_output_filename('apt', '.ecsv'),
                  format='ascii.ecsv', overwrite=True)
        # toltec*.nc
        for nw in np.unique(apt['nw']):
            self._make_kidsdata_nc(apt, nw, simu_config)
        # hwp.nc
        self._make_hwp_nc(simu_config)
        # tel.nc
        nm_tel = self.nms['tel']
        nc_tel = nm_tel.nc_node
        power_loading_model = eval_ctx['power_loading_model']
        if power_loading_model is not None:
            self._set_tel_tau(self.nms['tel'], tau_values=power_loading_model._atm_model_tau)

    def _set_tel_tau(self, nm_tel, tau_values):
        nm_tel.setscalar(
                'Header.Radiometer.Tau', tau_values['tau225'], dtype='f8', exist_ok=True)
        nm_tel.setscalar(
                'Header.Sim.Tau_a1100', tau_values['tau_a1100'], dtype='f8', exist_ok=True)
        nm_tel.setscalar(
                'Header.Sim.Tau_a1400', tau_values['tau_a1400'], dtype='f8', exist_ok=True)
        nm_tel.setscalar(
                'Header.Sim.Tau_a2000', tau_values['tau_a2000'], dtype='f8', exist_ok=True)

    def write_sim_data(self, data):
        eval_ctx = self._ensure_sim_eval_context()
        # apt.ecsv
        apt = eval_ctx['apt']

        nm_tel = self.nms[self._get_tel_interface()]
        nc_tel = nm_tel.nc_node
        d_time = 'time'

        bs_coords_icrs = data['mapping_info']['bs_coords_icrs']
        bs_coords_altaz = data['mapping_info']['bs_coords_altaz']
        bs_parallactic_angle = data['mapping_info']['bs_parallactic_angle']
        target_altaz = data['mapping_info']['target_altaz']
        # hwp_pa_altaz = data['mapping_info']['hwp_pa_altaz']
        hwp_pa_t = data['mapping_info']['hwp_pa_t']
        # hwp_pa_icrs = data['mapping_info']['hwp_pa_icrs']
        time_obs = data['mapping_info']['time_obs']
        holdflag = data['mapping_info']['holdflag']
        t_grid = data['t']
        iqs = data['probing_info']['iqs']

        idx = nc_tel.dimensions[d_time].size
        nm_tel.getvar('time')[idx:] = time_obs.unix

        t_grid_sec = t_grid.to_value(u.s)
        t_grid_sec_int = t_grid_sec.astype(int)
        nm_tel.getvar('pps_time')[idx:] = t_grid_sec_int + time_obs.unix[0]

        nm_tel.getvar('ra')[idx:] = bs_coords_icrs.ra.radian
        nm_tel.getvar('dec')[idx:] = bs_coords_icrs.dec.radian

        nm_tel.getvar('ra_src')[idx:] = bs_coords_icrs.ra.radian
        nm_tel.getvar('dec_src')[idx:] = bs_coords_icrs.dec.radian

        nm_tel.getvar('az')[idx:] = bs_coords_altaz.az.radian
        nm_tel.getvar('alt')[idx:] = bs_coords_altaz.alt.radian
        nm_tel.getvar('pa')[idx:] = bs_parallactic_angle.radian
        # no pointing model
        nm_tel.getvar('az_cor')[idx:] = 0.
        nm_tel.getvar('alt_cor')[idx:] = 0.
        nm_tel.getvar('az_des')[idx:] = bs_coords_altaz.az.radian
        nm_tel.getvar('alt_des')[idx:] = bs_coords_altaz.alt.radian
        nm_tel.getvar('az_map')[idx:] = 0.
        nm_tel.getvar('alt_map')[idx:] = 0.

        nm_tel.getvar('az_sky')[idx:] = bs_coords_altaz.az.radian
        nm_tel.getvar('alt_sky')[idx:] = bs_coords_altaz.alt.radian
        nm_tel.getvar('pa_sky')[idx:] = bs_parallactic_angle.radian

        nm_tel.getvar('source_az')[idx:] = target_altaz.az.radian
        nm_tel.getvar('source_alt')[idx:] = target_altaz.alt.radian
        nm_tel.getvar('hold')[idx:] = holdflag
        self.logger.info(
                f'write [{idx}:{idx + len(time_obs)}] to'
                f' {nc_tel.filepath()}')
        for nw in np.unique(apt['nw']):
            nm_toltec = self.nms[self._get_kidsdata_interface(nw)]
            nc_toltec = nm_toltec.nc_node
            idx = nc_toltec.dimensions[d_time].size
            self.logger.info(
                f'write [{nc_toltec.dimensions["iqlen"].size}]'
                f'[{idx}:{idx + len(time_obs)}] to'
                f' {nc_toltec.filepath()}')
            m = (apt['nw'] == nw)
            nm_toltec.getvar('flo')[idx:] = 0
            # 0  ClockTime (sec)
            # 1  PpsCount (pps ticks)
            # 2  ClockCount (clock ticks)
            # 3  PacketCount (packet ticks)
            # 4  PpsTime (clock ticks)
            # 5  ClockTimeNanoSec (nsec)
            # The clock time and clock nanosec are the unix time stamp
            # of the system startup, which is constant.
            # The actual time stamp of each sample is to be computed
            # with clock count and clock frequency.
            t00 = time_obs[0].unix
            nm_toltec.getvar('time')[idx:, 0] = np.full(
                t_grid.shape, int(t00))
            nm_toltec.getvar('time')[idx:, 5] = np.full(
                t_grid.shape, int((t00 - int(t00)) * 1e9))
            # package count, increment for each sample
            nm_toltec.getvar('time')[idx:, 3] = range(len(t_grid))
            # pps count and pps time
            # since we set t00 to be the t0, this is just simple
            # int sequence of t_grid
            t_grid_sec = t_grid.to_value(u.s)
            t_grid_sec_int = t_grid_sec.astype(int)
            nm_toltec.getvar('time')[idx:, 1] = t_grid_sec_int
            f_fpga_Hz = nm_toltec.getany('Header.Toltec.FpgaFreq')
            # pps time is the clock count of the pps tick
            nm_toltec.getvar('time')[idx:, 4] = (t_grid_sec_int * f_fpga_Hz)
            # clock count
            nm_toltec.getvar('time')[idx:, 2] = t_grid_sec * f_fpga_Hz

            nm_toltec.getvar('I')[idx:, :] = iqs.real[m, :].T
            nm_toltec.getvar('Q')[idx:, :] = iqs.imag[m, :].T

        # hwp
        nm_hwp = self.nms[self._get_hwp_interface()]
        nc_hwp = nm_hwp.nc_node
        idx = nc_hwp.dimensions[d_time].size
        self.logger.info(
            f'write '
            f'[{idx}:{idx + len(time_obs)}] to'
            f' {nc_hwp.filepath()}')
        nm_hwp.getvar('time')[idx:, 0] = time_obs.unix
        # nm_hwp.getvar('pa')[idx:] = hwp_pa_altaz.radian
        nm_hwp.getvar('pa')[idx:] = hwp_pa_t.radian
        # nm_hwp.getvar('pa')[idx:] = hwp_pa_icrs.radian

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
        state_init = {
            'obsnum': 1,
            'subobsnum': 0,
            'scannum': 0,
            }
        if self._state_init is not None:
            rupdate(state_init, self._state_init)
        # make sure the obsnum is always >= 1
        if state_init['obsnum'] < 1:
            state_init['obsnum'] = 1
        # set cal obs to be same as obsnum, because we use
        # built-in kids params
        for k in ['obsnum', 'subobsnum', 'scannum']:
            state_init[f'cal_{k}'] = state_init[k]
        state = PersistentState(
                outdir.joinpath(self._statefile),
                init=state_init)
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
