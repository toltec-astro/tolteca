#! /usr/bin/env python

from contextlib import contextmanager
import numpy as np
import datetime
# import functools
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.modeling import models, Parameter, Model
from astropy.modeling.functional_models import GAUSSIAN_SIGMA_TO_FWHM
from astropy import coordinates as coord
from astropy.time import Time
from astropy.table import Column, QTable, Table
from astropy.wcs.utils import celestial_frame_to_wcs
from astropy.coordinates import SkyCoord, AltAz
from astropy.coordinates.erfa_astrom import (
        erfa_astrom, ErfaAstromInterpolator)
from astropy.cosmology import default_cosmology
from astropy import constants as const
from astropy.utils.decorators import classproperty

from gwcs import coordinate_frames as cf

from kidsproc.kidsmodel import ReadoutGainWithLinTrend
from kidsproc.kidsmodel.simulator import KidsSimulator
from tollan.utils.log import timeit, get_logger

from ..base import (
        _Model,
        ProjModel, _get_skyoffset_frame,
        SourceImageModel, SourceCatalogModel)
from ..base import resolve_sky_map_ref_frame as _resolve_sky_map_ref_frame
from ...utils import get_pkg_data_path
from ...common.toltec import info as toltec_info  # noqa: F401
from .lmt import info as site_info
from .lmt import get_lmt_atm_models

__all__ = [
        'toltec_info',
        'site_info',
        'get_default_passbands',
        'get_default_cosmology',
        'get_observer',
        'ArrayProjModel',
        ]


def get_default_passbands():
    """Return the default passband tables as a dict.
    """
    from ...cal.toltec import ToltecPassband
    calobj = ToltecPassband.from_indexfile(get_pkg_data_path().joinpath(
        'cal/toltec_passband/index.yaml'
        ))
    result = dict()
    for array_name in calobj.array_names:
        result[array_name] = calobj.get(array_name=array_name)
    return result


def get_default_cosmology():
    """Return the default cosmology."""
    return default_cosmology.get()


def get_observer():
    """Return the `astroplan.Observer` object for LMT TolTEC."""
    return site_info['observer']


class ArrayProjModel(ProjModel):
    """A model that transforms the TolTEC detector locations per array to
    a common instrument coordinate system.

    """
    # TODO we need a unified management for such facts, in the package
    # level
    toltec_instru_spec = {
            'a1100': {
                'rot_from_a1100': 0. * u.deg
                },
            'a1400': {
                'rot_from_a1100': 180. * u.deg
                },
            'a2000': {
                'rot_from_a1100': 180. * u.deg
                },
            'toltec': {
                'rot_from_a1100': 90. * u.deg,
                'array_names': ('a1100', 'a1400', 'a2000'),
                # 'plate_scale': ()
                'fov_diam': 4. * u.arcmin,
                'array_diam': 127049.101 * u.um  # from a1100 design spec.
                },
            }
    input_frame = cf.Frame2D(
                name='array',
                axes_names=("x", "y"),
                unit=(u.um, u.um))
    output_frame = cf.Frame2D(
                name='toltec',
                axes_names=("az_offset", "alt_offset"),
                unit=(u.deg, u.deg))
    _name = f'{output_frame.name}_proj'

    n_inputs = 3
    n_outputs = 2

    def __init__(self, **kwargs):
        spec = self.toltec_instru_spec

        # The mirror put array on the perspective of an observer.
        m_mirr = np.array([[1, 0], [0, -1]])

        plate_scale = spec['toltec']['fov_diam'] / spec['toltec']['array_diam']

        m_projs = dict()

        for array_name in self.array_names:

            # this rot and scale put the arrays on the on the sky in Altaz
            rot = (
                    spec['toltec']['rot_from_a1100'] -
                    spec[array_name]['rot_from_a1100']
                    )
            m_rot = models.Rotation2D._compute_matrix(
                    angle=rot.to_value('rad'))

            m_projs[array_name] = models.AffineTransformation2D(
                (m_rot @ m_mirr) * u.cm,
                translation=(0., 0.) * u.cm) | (
                    models.Multiply(plate_scale) &
                    models.Multiply(plate_scale))
        self._m_projs = m_projs
        super().__init__(
                inputs=('axes_names', ) + self.input_frame.axes_names,
                **kwargs)

    @timeit(_name)
    def evaluate(self, array_name, x, y):
        out_unit = u.deg
        x_out = np.empty(x.shape) * out_unit
        y_out = np.empty(y.shape) * out_unit
        for n in self.array_names:
            m = array_name == n
            xx, yy = self._m_projs[n](x[m], y[m])
            x_out[m] = xx.to(out_unit)
            y_out[m] = yy.to(out_unit)
        return x_out, y_out

    @property
    def array_names(self):
        return self.toltec_instru_spec['toltec']['array_names']

    def prepare_inputs(self, array_name, *inputs, **kwargs):
        # this is necessary to handle the array_name inputs
        array_name_idx = np.arange(array_name.size).reshape(array_name.shape)
        inputs_new, broadcasts = super().prepare_inputs(
                array_name_idx, *inputs, **kwargs)
        inputs_new[0] = np.ravel(array_name)[inputs_new[0].astype(int)]
        return inputs_new, broadcasts


class ArrayPolarizedProjModel(ArrayProjModel):
    """A model that transforms the TolTEC detector locations per array to
    a common instrument coordinate system.

    This model is different from ArrayProjModel that it also projects
    the polarization angles of each detector, taking into account
    the parity caused by the mirror reflection.

    """
    _pa_frame = cf.CoordinateFrame(
                    naxes=1,
                    axes_type='SPATIAL',
                    axes_order=(2, ),
                    unit=(u.deg, ),
                    axes_names=("pa", ),
                    name='polarimetry'
                    )
    input_frame = cf.CompositeFrame(
            frames=[
                ArrayProjModel.input_frame,
                _pa_frame
                ],
            name=ArrayProjModel.input_frame.name + '_polarimetry'
            )
    output_frame = cf.CompositeFrame(
            frames=[
                ArrayProjModel.output_frame,
                _pa_frame
                ],
            name=ArrayProjModel.input_frame.name + '_polarimetry'
            )
    _name = f'{output_frame.name}_proj'

    n_inputs = 4
    n_outputs = 3

    @timeit(_name)
    def evaluate(self, array_name, x, y, pa):
        x_out, y_out = super().evaluate(array_name, x, y)

        spec = self.toltec_instru_spec
        out_unit = u.deg

        pa_out = np.empty(x_out.shape) * out_unit
        for n in self.array_names:
            m = array_name == n
            pa_out[m] = (
                    pa[m]
                    + spec['toltec']['rot_from_a1100']
                    - spec[n]['rot_from_a1100'])
        return x_out, y_out, pa_out

    @property
    def array_names(self):
        return self.toltec_instru_spec['toltec']['array_names']

    def prepare_inputs(self, array_name, *inputs, **kwargs):
        # this is necessary to handle the array_name inputs
        array_name_idx = np.arange(array_name.size).reshape(array_name.shape)
        inputs_new, broadcasts = super().prepare_inputs(
                array_name_idx, *inputs, **kwargs)
        inputs_new[0] = np.ravel(array_name)[inputs_new[0].astype(int)]
        return inputs_new, broadcasts


class SkyProjModel(ProjModel):
    """A sky projection model for TolTEC.

    Parameters
    ----------
    ref_coord: 2-tuple of `astropy.units.Quantity`
        The coordinate of the TolTEC frame origin on the sky.
    """

    input_frame = ArrayProjModel.output_frame
    output_frame = cf.Frame2D(
                name='sky',
                axes_names=("lon", "lat"),
                unit=(u.deg, u.deg))
    _name = f'{output_frame.name}_proj'

    n_inputs = 2
    n_outputs = 2

    crval0 = Parameter(default=180., unit=output_frame.unit[0])
    crval1 = Parameter(default=30., unit=output_frame.unit[1])
    mjd_obs = Parameter(default=Time(2000.0, format='jyear').mjd, unit=u.day)

    observer = get_observer()
    logger = get_logger()

    def __init__(
            self, ref_coord=None, time_obs=None,
            evaluate_frame=None, **kwargs):
        if ref_coord is not None:
            if 'crval0' in kwargs or 'crval1' in kwargs:
                raise ValueError(
                        "ref_coord cannot be specified along with crvals")
            if isinstance(ref_coord, coord.SkyCoord):
                _ref_coord = (
                        ref_coord.data.lon.degree,
                        ref_coord.data.lat.degree) * u.deg
                kwargs['crval0'] = _ref_coord[0]
                kwargs['crval1'] = _ref_coord[1]
                kwargs['n_models'] = np.asarray(_ref_coord[0]).size
                self.crval_frame = ref_coord.frame
        if time_obs is not None:
            if 'mjd_obs' in kwargs:
                raise ValueError(
                        "time_obs cannot be specified along with mjd_obs")
            kwargs['mjd_obs'] = time_obs.mjd << u.day
        self.evaluate_frame = evaluate_frame
        super().__init__(**kwargs)

    @classmethod
    def _get_native_frame(cls, mjd_obs):
        return cls.observer.altaz(time=Time(mjd_obs, format='mjd'))

    def get_native_frame(self):
        return self._get_native_frame(self.mjd_obs)

    @classmethod
    def _get_projected_frame(
            cls, crval0, crval1, crval_frame,
            mjd_obs, also_return_native_frame=False):
        ref_frame = cls._get_native_frame(mjd_obs)
        ref_coord = coord.SkyCoord(
                crval0.value << u.deg, crval1.value << u.deg,
                frame=crval_frame).transform_to(ref_frame)
        ref_offset_frame = _get_skyoffset_frame(ref_coord)
        if also_return_native_frame:
            return ref_offset_frame, ref_frame
        return ref_offset_frame

    @timeit
    def get_projected_frame(self, **kwargs):
        return self._get_projected_frame(
                self.crval0, self.crval1, self.crval_frame,
                self.mjd_obs, **kwargs)

    @timeit(_name)
    def __call__(self, *args, frame=None, eval_interp_len=None, **kwargs):
        if frame is None:
            frame = self.evaluate_frame
        old_evaluate_frame = self.evaluate_frame
        self.evaluate_frame = frame
        if eval_interp_len is None:
            result = super().__call__(*args, **kwargs)
        else:
            x, y = args
            mjd_obs = self.mjd_obs.quantity
            ref_coord = coord.SkyCoord(
                self.crval0.value << u.deg, self.crval1.value << u.deg,
                frame=self.crval_frame)

            # make a subset of crvals for fast evaluate
            # we need to make sure mjd_obs is sorted before hand
            if not np.all(np.diff(mjd_obs) >= 0):
                raise ValueError('mjd_obs has to be sorted ascending.')
            s = [0]
            for i, t in enumerate(mjd_obs):
                if t - mjd_obs[s[-1]] <= eval_interp_len:
                    continue
                s.append(i)
            s.append(-1)
            self.logger.debug(f"evaluate {len(s)}/{len(mjd_obs)} times")
            ref_coord_s = ref_coord[s]
            mjd_obs_s = mjd_obs[s]
            mdl_s = self.__class__(
                    ref_coord=ref_coord_s, mjd_obs=mjd_obs_s,
                    evaluate_frame=self.evaluate_frame)
            lon_s, lat_s = mdl_s(x[s, :], y[s, :])
            # now build the spline interp
            lon_interp = interp1d(
                    mjd_obs_s, lon_s.degree, axis=0, kind='cubic')
            lat_interp = interp1d(
                    mjd_obs_s, lat_s.degree, axis=0, kind='cubic')
            lon = lon_interp(mjd_obs) << u.deg
            lat = lat_interp(mjd_obs) << u.deg
            result = (lon, lat)
        self.evaluate_frame = old_evaluate_frame
        return result

    @timeit
    def evaluate(self, x, y, crval0, crval1, mjd_obs):

        ref_offset_frame, ref_frame = self._get_projected_frame(
                crval0, crval1, self.crval_frame,
                mjd_obs, also_return_native_frame=True)
        det_coords_offset = coord.SkyCoord(x, y, frame=ref_offset_frame)
        with timeit("transform det coords to altaz"):
            det_coords = det_coords_offset.transform_to(ref_frame)

        frame = self.evaluate_frame
        if frame is None or frame == 'native':
            return det_coords.az, det_coords.alt
        with timeit(f"transform det coords to {frame}"):
            det_coords = det_coords.transform_to(frame)
            attrs = list(
                    det_coords.get_representation_component_names().keys())
            return (getattr(det_coords, attrs[0]),
                    getattr(det_coords, attrs[1]))

    def mpl_axes_params(self):
        w = celestial_frame_to_wcs(coord.ICRS())
        ref_coord = SkyCoord(
                self.crval0, self.crval1, frame=self.crval_frame
                ).transform_to('icrs')
        w.wcs.crval = [
                ref_coord.ra.degree,
                ref_coord.dec.degree,
                ]
        return dict(super().mpl_axes_params(), projection=w)


class BeamModel(Model):
    """A model that describes the TolTEC beam shapes.
    """
    beam_props = {
            'array_names': ('a1100', 'a1400', 'a2000'),
            'model': models.Gaussian2D,
            'x_fwhm_a1100': 5 * u.arcsec,
            'y_fwhm_a1100': 5 * u.arcsec,
            'a1100': {
                'wl_center': 1.1 * u.mm
                },
            'a1400': {
                'wl_center': 1.4 * u.mm
                },
            'a2000': {
                'wl_center': 2.0 * u.mm
                },
            }
    n_inputs = 3
    n_outputs = 1

    @classmethod
    def get_fwhm(cls, axis, array_name):
        beam_props = cls.beam_props
        if axis in ['x', 'a']:
            key = 'x_fwhm_a1100'
        elif axis in ['y', 'b']:
            key = 'y_fwhm_a1100'
        else:
            raise ValueError("invalid axis.")
        return (
                beam_props[key] *
                beam_props[array_name]['wl_center'] /
                beam_props['a1100']['wl_center'])

    def __init__(self, **kwargs):
        beam_props = self.beam_props
        m_beams = dict()

        for array_name in beam_props['array_names']:
            x_fwhm = self.get_fwhm('a', array_name)
            y_fwhm = self.get_fwhm('b', array_name)
            x_stddev = x_fwhm / GAUSSIAN_SIGMA_TO_FWHM
            y_stddev = y_fwhm / GAUSSIAN_SIGMA_TO_FWHM
            beam_area = 2 * np.pi * x_stddev * y_stddev
            m_beams[array_name] = beam_props['model'](
                    amplitude=1. / beam_area,
                    x_mean=0. * u.arcsec,
                    y_mean=0. * u.arcsec,
                    x_stddev=x_stddev,
                    y_stddev=y_stddev,
                    )
        self._m_beams = m_beams
        super().__init__(**kwargs)
        self.inputs = ('array_name', ) + m_beams['a1100'].inputs
        self.outputs = m_beams['a1100'].outputs

    def evaluate(self, array_name, x, y):
        out_unit = self._m_beams['a1100'].amplitude.unit
        out = np.empty(x.shape) << out_unit
        for n in self.beam_props['array_names']:
            m = array_name == n
            mm = self._m_beams[n]
            (b, t), (l, r) = mm.bounding_box
            x_m = x[m]
            y_m = y[m]
            g = (y_m >= b) & (y_m <= t) & (x_m >= l) & (x_m <= r)
            m_out = np.zeros(x_m.shape) << out_unit
            m_out[g] = mm(x_m[g], y_m[g])
            out[m] = m_out.to(out_unit)
        return out

    @property
    def models(self):
        return self._m_beams

    def prepare_inputs(self, array_name, *inputs, **kwargs):
        # this is necessary to handle the array_name inputs
        array_name_idx = np.arange(array_name.size).reshape(array_name.shape)
        inputs_new, broadcasts = super().prepare_inputs(
                array_name_idx, *inputs, **kwargs)
        inputs_new[0] = np.ravel(array_name)[inputs_new[0].astype(int)]
        return inputs_new, broadcasts


class ArrayLoadingModel(_Model):
    """
    A model of the LMT optical loading at the TolTEC arrays.

    This is based on the Mapping-speed-caluator
    """

    # TODO allow overwriting these per instance.
    _toltec_passbands = get_default_passbands()
    _cosmo = get_default_cosmology()

    logger = get_logger()

    n_inputs = 1
    n_outputs = 2

    @property
    def input_units(self):
        return {self.inputs[0]: u.deg}

    def __init__(self, array_name, atm_model_name='am_q50', *args, **kwargs):
        super().__init__(name=f'{array_name}_loading', *args, **kwargs)
        self._inputs = ('alt', )
        self._outputs = ('P', 'nep')
        self._array_name = array_name
        self._passband = self._toltec_passbands[array_name]
        self._f = self._passband['f'].quantity
        # check the f step, they shall be uniform
        df = np.diff(self._f).value
        if np.std(df) / df[0] > 1e-7:
            raise ValueError(
                "invalid passband format, frequency grid has to be uniform")
        self._df = self._f[1] - self._f[0]
        self._throughput = self._passband['throughput']
        self._atm_model, self._atm_tx_model = get_lmt_atm_models(
                name=atm_model_name)

    @classproperty
    def _internal_params(cls):
        """Lower level instrument parameters for LMT/TolTEC.

        Note that all these values does not take into account the
        passbands, and are frequency independent.
        """
        # TODO merge this to the instrument fact yaml file?
        p = {
                'det_optical_efficiency': 0.8,
                'det_noise_factor': 0.334,
                'horn_aperture_efficiency': 0.35,
                'tel_diameter': 48. << u.m,
                'tel_surface_rms': 76. << u.um,
                'tel_emissivity': 0.06,
                'T_coldbox': 5.75 << u.K,
                'T_tel': 273. << u.K,  # telescope ambient temperature
                'T_coupling_optics': 290. << u.K,  # coupling optics
                }
        # derived values
        p['tel_area'] = np.pi * (p['tel_diameter'] / 2.) ** 2
        # effective optics temperature due to telescope and the coupling
        p['T_warm'] = (
                p['tel_emissivity'] * p['T_tel']
                # TODO add documents for the numbers here
                + 3. * p['T_coupling_optics'] * 0.01
                )
        # cold efficiency is the efficiency inside the cold box.
        p['cold_efficiency'] = (
                p['det_optical_efficiency'] * p['horn_aperture_efficiency'])
        # effetive temperature at detectors for warm components through
        # the cold box
        p['T_det_warm'] = (p['T_warm'] * p['cold_efficiency'])
        # effetive temperature at detectors for cold box
        # note that the "horn aperture efficiency" is actually the
        # internal system aperture efficiency since it includes the
        # truncation of the lyot stop and the loss to the cold optics
        p['T_det_coldbox'] = (
                p['T_coldbox'] * p['det_optical_efficiency']
                * (1. - p['horn_aperture_efficiency'])
                )
        return p

    @property
    def _tel_primary_surface_optical_efficiency(self):
        """The telescope optical efficiency due to RMS of the
        primary surface over the passband.

        This is just the Ruze formula.
        """
        tel_surface_rms = self._internal_params['tel_surface_rms']
        f = self._f
        return np.exp(-((4.0 * np.pi * tel_surface_rms)/(const.c / f)) ** 2)

    @property
    def _system_efficiency(self):
        """The overall system efficiency over the passband."""
        return (
                self._tel_primary_surface_optical_efficiency
                * self._internal_params['cold_efficiency']
                * self._throughput
                )

    @staticmethod
    def _wsum(q, w):
        """Return weighted sum of some quantity.

        q : `astropy.units.Quantity`
            The quantity.

        w : float
            The wegith.
        """
        if w.ndim > 1:
            raise ValueError("weight has to be 1d")
        return np.nansum(q * w, axis=-1) / np.nansum(w)

    def _get_T_atm(
            self, alt,
            return_avg=False):
        """Return the atmosphere temperature.

        This is the "true" temperature without taking into account the system
        efficiency.

        Parameters
        ----------
        alt : `astropy.units.Quantity`
            The altitude.
        return_avg : bool, optional
            If True, return the weighted sum over the passband instead.
        """
        atm_model = self._atm_model
        # here we put the alt on the first axis for easier reduction on f.
        T_atm = atm_model(*np.meshgrid(self._f, alt, indexing='ij')).T
        if return_avg:
            T_atm = self._wsum(T_atm, self._throughput)
        T_atm = np.squeeze(T_atm)
        return T_atm

    def _get_tx_atm(self, alt):
        """Return the atmosphere transmission.

        Parameters
        ----------
        alt : `astropy.units.Quantity`
            The altitude.
        """
        atm_tx_model = self._atm_tx_model
        # here we put the alt on the first axis for easier reduction on f.
        tx_atm = atm_tx_model(*np.meshgrid(self._f, alt, indexing='ij')).T
        tx_atm = np.squeeze(tx_atm)
        return tx_atm

    def _get_T(
            self, alt,
            return_avg=False
            ):
        """Return the effective temperature at altitude `alt`, as seen
        by the cryostat.

        Parameters
        ----------
        alt : `astropy.units.Quantity`
            The altitude.
        return_avg : bool, optional
            If True, return the weighted sum over the passband instead.
        """
        T_atm = self._get_T_atm(alt, return_avg=False)
        # add the telescope warm component temps
        T_tot = T_atm + self._internal_params['T_warm']
        if return_avg:
            T_tot = self._wsum(T_tot, self._system_efficiency)
        return T_tot

    def _get_T_det(
            self, alt,
            return_avg=True):
        """Return the effective temperature seen by the detectors
        at altitude `alt`.

        Parameters
        ----------
        alt : `astropy.units.Quantity`
            The altitude.
        return_avg : bool, optional
            If True, return the weighted sum over the passband instead.
        """
        T_atm = self._get_T_atm(alt, return_avg=False)
        # TODO why no telescope efficiency term?
        T_det = (
                T_atm * self._internal_params['cold_efficiency']
                + self._internal_params['T_det_warm']
                + self._internal_params['T_det_coldbox']
                ) * self._throughput
        if return_avg:
            # note this is different from the Detector.py in that
            # does not mistakenly (?) average over the passband again
            T_det = np.mean(T_det)
        return T_det

    def _T_to_dP(self, T):
        """Return the Rayleigh-Jeans power for the passband frequency bins.

        Parameters
        ----------
        T : `astropy.units.Quantity`
            The temperature.
        """
        # power from RJ source in frequency bin df
        # TODO this can be done this way because we ensured df is contant
        # over the passband.
        # we may change this to trapz to allow arbitrary grid?
        return const.k_B * T * self._df

    def _T_to_dnep(self, T):
        """Return the photon noise equivalent power in W / sqrt(Hz) for
        the passband frequency bins.
        """
        f = self._f
        df = self._df
        dP = self._T_to_dP(T)

        shot = 2. * const.k_B * T * const.h * f * df
        wave = 2. * dP ** 2 / df
        return np.sqrt(shot + wave)

    def _T_to_dnet_cmb(self, T, tx_atm):
        """Return the noise equivalent CMB temperature in K / sqrt(Hz) for
        the passband frequency bins.

        Parameters
        ----------
        T : `astropy.units.Quantity`
            The temperature.
        tx_atm : array
            The atmosphere transmission.
        """
        f = self._f
        df = self._df
        Tcmb = self._cosmo.Tcmb(0)

        dnep = self._T_to_dnep(T)
        x = const.h * f / (const.k_B * Tcmb)
        net_integrand = (
                (const.k_B * x) ** 2.
                * (1. / const.k_B)
                * np.exp(x) / (np.expm1(x)) ** 2.
                )
        dnet = dnep / (
                np.sqrt(2.0)
                * self._system_efficiency
                * net_integrand
                * df)
        # scale by the atmosphere transmission so this is comparable
        # to astronomical sources.
        return dnet / tx_atm

    def _dnep_to_dnefd(self, dnep, tx_atm):
        """Return the noise equivalent flux density in Jy / sqrt(Hz) for
        the passband frequency bins.

        Parameters
        ----------
        T : `astropy.units.Quantity`
            The temperature.
        tx_atm : array
            The atmosphere transmission.
        """
        df = self._df
        A = self._internal_params['tel_area']
        # TODO Z. Ma: I combined the sqrt(2) term. need to check the eqn here.
        dnefd = (
                dnep
                / (A * df)
                / self._system_efficiency
                * np.sqrt(2.))
        # scale by the atmosphere transmission so this is comparable
        # to astronomical sources.
        return dnefd / tx_atm  # Jy / sqrt(Hz)

    def _get_P(self, alt):
        """Return the detector power loading at altitude `alt`.

        """
        T_det = self._get_T_det(alt=alt, return_avg=False)
        return np.nansum(self._T_to_dP(T_det), axis=-1).to(u.pW)

    def _get_noise(self, alt, return_avg=True):
        """Return the noise at altitude `alt`.

        Parameters
        ----------
        alt : `astropy.units.Quantity`
            The altitude.
        return_avg : bool, optional
            If True, return the value integrated for the passband.
        """
        # noise calculations
        # strategy is to do this for each frequency bin and then do a
        # weighted average across the band.  This is copied directly from
        # Sean's python code.
        T_det = self._get_T_det(alt=alt, return_avg=False)
        dnep_phot = self._T_to_dnep(T_det)

        # detector noise factor coefficient
        det_noise_coeff = np.sqrt(
                1. + self._internal_params['det_noise_factor'])

        dnep = dnep_phot * det_noise_coeff

        # atm transmission
        tx_atm = self._get_tx_atm(alt)
        # the equivalent noise in astronomical units
        dnet_cmb = (
                self._T_to_dnet_cmb(T_det, tx_atm=tx_atm)
                * det_noise_coeff
                )
        dnefd = self._dnep_to_dnefd(dnep, tx_atm=tx_atm)

        if return_avg:
            # integrate these up
            net_cmb = np.sqrt(1.0 / np.nansum(dnet_cmb ** (-2.0), axis=-1))
            nefd = np.sqrt(1.0 / np.nansum(dnefd ** (-2.0), axis=-1))
            # nep is sum of squares
            nep = np.sqrt(np.nansum(dnep ** 2.0, axis=-1))
            # power just adds
            return {
                    'net_cmb': net_cmb.to(u.mK * u.Hz ** -0.5),
                    'nefd': nefd.to(u.mJy * u.Hz ** -0.5),
                    'nep': nep.to(u.aW * u.Hz ** -0.5)
                    }
        return {
                    'dnet_cmb': net_cmb.to(u.mK * u.Hz ** -0.5),
                    'dnefd': nefd.to(u.mJy * u.Hz ** -0.5),
                    'dnep': nep.to(u.aW * u.Hz ** -0.5)
                    }

    def make_summary_table(self, alt=None):
        """Return a summary for a list of altitudes.

        """
        if alt is None:
            alt = [50., 60., 70.] << u.deg
        result = dict()
        result['P'] = self._get_P(alt)
        result.update(self._get_noise(alt, return_avg=True))
        return Table(result)

    def evaluate(self, alt):
        P = self._get_P(alt)
        nep = self._get_noise(alt, return_avg=True)['nep']
        return P, nep


class ToltecObsSimulator(object):
    """A class that make simulated observations for TolTEC.

    The simulator makes use of a suite of models::

                        telescope pointing (lon, lat)
                                                |
                                                v
        detectors positions (x, y) -> [SkyProjectionModel]
                                                |
                                                v
                            projected detectors positions (lon, lat)
                                                |
         sky/atmosphere model (lon, lat flux) ->|
                                                v
        source catalogs (lon, lat, flux) -> [BeamModel]
                                                |
                            [filter passband] ->|
                                                v
                                        detector loading (pwr)
                                                |
                                                v
                                           [KidsProbeModel]
                                                |
                                                v
                                detector raw readout (I, Q)

    Parameters
    ----------
    array_prop_table: astropy.table.Table
        The array property table that contains all necessary information
        for the detectors.
    """

    # these are generated from Grant's Mapping-Speed-Calculator code
    # The below is for elev 45 deg, atm 25 quantiles
    array_optical_props = {
        'a1100': {
            'background': 10.01 * u.pW,
            'bkg_temp': 9.64 * u.K,
            'responsivity': 5.794e-5 / u.pW,
            'passband': 65 * u.GHz,
            },
        'a1400': {
            'background': 7.15 * u.pW,
            'bkg_temp': 9.43 * u.K,
            'responsivity': 1.1e-4 / u.pW,
            'passband': 50 * u.GHz,
            },
        'a2000': {
            'background': 5.29 * u.pW,
            'bkg_temp': 8.34 * u.K,
            'responsivity': 1.1e-4 / u.pW,
            'passband': 42 * u.GHz,
            },
        }

    # these are some fiducial kids model params
    kids_props = {
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
    readout_model_cls = ReadoutGainWithLinTrend
    beam_model_cls = BeamModel
    erfa_interp_len = 300 << u.s
    _observer = site_info['observer']

    @property
    def observer(self):
        return self._observer

    def __init__(self, array_prop_table):

        tbl = self._table = self._prepare_table(array_prop_table)
        self._m_beam = self.beam_model_cls()

        # create the simulator
        self._kidssim = KidsSimulator(
                fr=tbl['fr'],
                Qr=tbl['Qr'],
                background=tbl['background'],
                responsivity=tbl['responsivity'])
        # create the gain model
        self._readout_model = self.readout_model_cls(
                n_models=len(tbl),
                **{
                    c: tbl[c]
                    for c in [
                        'g0', 'g1', 'g', 'phi_g', 'f0', 'k0', 'k1', 'm0', 'm1']
                    },
                )
        # get detector position on the sky in the toltec frame
        if 'x_t' not in tbl.colnames:
            x_a = tbl['x'].to(u.cm)
            y_a = tbl['y'].to(u.cm)
            x_t, y_t = ArrayProjModel()(tbl['array_name'], x_a, y_a)
            tbl.add_column(Column(x_t, name='x_t', unit=x_t.unit))
            tbl.add_column(Column(y_t, name='y_t', unit=y_t.unit))

    @property
    def table(self):
        return self._table

    @staticmethod
    def get_sky_projection_model(**kwargs):
        """Return the model that project TolTEC detectors on the sky."""
        m_proj = SkyProjModel(**kwargs)
        return m_proj

    @property
    def kidssim(self):
        return self._kidssim

    @property
    def kids_readout_model(self):
        return self._readout_model

    @contextmanager
    def probe_context(
            self, fp=None,
            sources=None,
            f_smp=None,
            ):
        """Return a function that can be used to get IQ for given flux.

        When `with_array_loading` is True, the generated power loading
        will be the sum of the contribution from the astronomical source
        and the telescope and atmosphere:

            P_tot = P_src + P_bkg_fixture + P_atm(alt)

        We set the tune of the KidsSimulator,
        such that x=0 at P=P_bkg_fixture + P_atm(alt_of_tune_obs).

        Thus the measured detuning parameters is proportional to

            P_src + (P_atm(alt) - P_atm(alt_of_tune_obs))
        """
        tbl = self.table
        # make a copy here because
        # we'll adjust the bkg
        kidssim = self._kidssim
        readout = self._readout_model
        if fp is None:
            fp = kidssim._fr
        # check the sources for array loading model
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
        logger = get_logger()
        logger.debug(
                f"evaluate array loading model: {array_loading_model}")
        if array_loading_model is not None:
            # the loading model is per-array
            # this holds the power at which x=0
            # this has to be a constant for all evaluate calls
            p_tune = dict()

            for array_name, alm in array_loading_model.items():
                p_tune[array_name] = None

            def evaluate(s, alt=None):
                if alt is None:
                    raise ValueError(
                            "need altitudes to evaluate array loading model")

                def _evaluate(s, wl_center, pb_width, alt, alm):
                    # per-array eval
                    # we also need to ravel s and alt so that they become
                    # 1d
                    # brightness temperature
                    logger.debug(
                            f"evaluate loading model name={alm}")
                    tbs = np.ravel(
                            s.to(
                                u.K,
                                equivalencies=u.brightness_temperature(
                                    wl_center)))
                    pwrs = (
                        tbs.to(
                            u.J,
                            equivalencies=u.temperature_energy())
                        * pb_width
                        ).to(u.pW)
                    # note that we cannot afford
                    # doing this for each frequency bin.
                    # we'll just assume a square passband with width=df
                    # overall sys_eff
                    sys_eff = alm._wsum(
                            alm._system_efficiency, alm._throughput
                            )
                    pwrs = pwrs * sys_eff

                    # add the loading temperate from non astro source
                    alt = np.ravel(alt)
                    # again, this is too slow to do for each sample,
                    # we'll just use interpolation.
                    alt_min = np.min(alt)
                    alt_max = np.max(alt)
                    alt_grid = np.arange(
                            alt_min.to_value(u.deg),
                            alt_max.to_value(u.deg) + 0.1,
                            0.1
                            ) << u.deg
                    if len(alt_grid) < 10:
                        # make sure we have enough elevation points
                        alt_grid = np.linspace(
                            alt_min.to_value(u.deg),
                            alt_max.to_value(u.deg),
                            10
                            ) << u.deg
                    p_interp = interp1d(
                            alt_grid, alm._get_P(alt_grid).to_value(u.pW),
                            kind='cubic'
                            )
                    dp_interp = interp1d(
                            alt_grid,
                            (
                                alm._get_noise(alt_grid)['nep']
                                * np.sqrt(f_smp / 2.)).to_value(u.pW),
                            kind='cubic'
                            )

                    pwrs_non_src = p_interp(alt) << u.pW
                    dpwr = dp_interp(alt)
                    dpwr = np.random.normal(0., dpwr) << u.pW
                    # print(np.min(pwrs), np.max(pwrs))
                    # print(np.min(pwrs_non_src), np.max(pwrs_non_src))
                    # print(np.min(dpwr), np.max(dpwr))
                    # import matplotlib.pyplot as plt
                    # fig, axes = plt.subplots(3, 1, constrained_layout=True)
                    # axes[0].imshow(pwrs.reshape(s.shape))
                    # axes[1].imshow(pwrs_non_src.reshape(s.shape))
                    # axes[2].imshow(dpwr.reshape(s.shape))
                    # plt.show()
                    # make a realization of the pwrs
                    pwrs = pwrs + pwrs_non_src + dpwr
                    return pwrs.reshape(s.shape)

                # the alm has to be evaluated on a per array basis
                pwrs = np.empty(s.shape)
                # adjust the bkg
                _kidssim = kidssim.copy()

                for array_name, alm in array_loading_model.items():
                    m = tbl['array_name'] == array_name
                    if p_tune[array_name] is None:
                        # take the mean of  alt as the tune position
                        p_tune[array_name] = alm._get_P(np.mean(alt))
                        logger.debug(
                            f"set P_tune[{array_name}]={p_tune[array_name]}")
                    else:
                        logger.debug(
                            f"use P_tune[{array_name}]={p_tune[array_name]}")
                    with timeit(f"calc array loading for {array_name}"):
                        pwrs[m, :] = _evaluate(
                            s[m, :], wl_center=tbl[m]['wl_center'][0],
                            pb_width=tbl[m]['passband'][0],
                            alt=alt[m, :], alm=alm,
                            ).to_value(u.pW)
                    _kidssim._background[m] = p_tune[array_name]
                pwrs = pwrs << u.pW
                rs, xs, iqs = _kidssim.probe_p(
                        pwrs,
                        fp=fp, readout_model=readout)
                # compute the flxscale
                pwr_norm = np.empty((s.shape[0], ))
                for array_name, alm in array_loading_model.items():
                    m = tbl['array_name'] == array_name
                    wl_center = tbl[m]['wl_center'][0]
                    pb_width = tbl[m]['passband'][0]
                    tb = (1. << u.MJy / u.sr).to(
                            u.K,
                            equivalencies=u.brightness_temperature(
                                wl_center)
                            )
                    pwr = (
                            tb.to(
                                u.J,
                                equivalencies=u.temperature_energy())
                            * pb_width).to(u.pW)
                    sys_eff = alm._wsum(
                            alm._system_efficiency, alm._throughput
                            )
                    pwr_norm[m] = (
                            pwr * sys_eff + p_tune[array_name]).to_value(u.pW)
                pwr_norm = pwr_norm << u.pW
                _, x_norm, _ = _kidssim.probe_p(
                        pwr_norm[:, np.newaxis],
                        fp=fp, readout_model=readout)
                flxscale = 1. / np.squeeze(x_norm)
                logger.debug(f"flxscale: {flxscale.mean()}")
                return rs, xs, iqs, locals()
        else:
            # when loading model is not specified, we just use
            # the pre-defined values.
            def evaluate(s, alt=None):
                # convert to brightness temperature and
                # assuming a square pass band, we can get the power loading
                # TODO use the real passbands.
                tbs = s.to(
                        u.K,
                        equivalencies=u.brightness_temperature(
                            tbl['wl_center'][:, np.newaxis]))
                pwrs = (
                        tbs.to(
                            u.J,
                            equivalencies=u.temperature_energy())
                        * tbl['passband'][:, np.newaxis]
                        ).to(u.pW)
                rs, xs, iqs = kidssim.probe_p(
                        pwrs + tbl['background'][:, np.newaxis],
                        fp=fp, readout_model=readout)
                # compute flxscale
                pwr_norm = (
                        (
                            np.ones((s.shape[0], 1)) << u.MJy / u.sr).to(
                                u.K,
                                equivalencies=u.brightness_temperature(
                                    tbl['wl_center'][:, np.newaxis])).to(
                                    u.J,
                                    equivalencies=u.temperature_energy())
                                * tbl['passband'][:, np.newaxis]
                        ).to(u.pW)
                _, x_norm, _ = kidssim.probe_p(
                        pwr_norm + tbl['background'][:, np.newaxis],
                        fp=fp, readout_model=readout)
                flxscale = 1. / np.squeeze(x_norm)
                logger.debug(f"flxscale: {flxscale.mean()}")
                return rs, xs, iqs, locals()
        # check the sources for kids noise model
        if sources is not None:
            for source in sources:
                if isinstance(source, KidsReadoutNoiseModel):
                    readout_noise_model = source
                    break
            else:
                readout_noise_model = None
        else:
            readout_noise_model = None
        if readout_noise_model is not None:
            def evaluate_with_readout_noise(*args, **kwargs):
                logger.debug(f"readout noise model {readout_noise_model}")
                rs, xs, iqs, info = evaluate(*args, **kwargs)
                diqs = readout_noise_model.evaluate_tod(tbl, iqs)
                # info['diqs'] = diqs
                iqs += diqs
                return rs, xs, iqs, info
            yield evaluate_with_readout_noise
        else:
            yield evaluate

    @timeit
    def resolve_sky_map_ref_frame(self, ref_frame, time_obs):
        return _resolve_sky_map_ref_frame(
                    ref_frame, observer=self.observer, time_obs=time_obs)

    def resolve_target(self, target, time_obs):
        if isinstance(target.frame, AltAz):
            target = SkyCoord(
                    target.data, frame=self.resolve_sky_map_ref_frame(
                            'altaz', time_obs=time_obs))
        return target

    @contextmanager
    def mapping_context(self, mapping, sources):
        """
        Return a function that can be used to get
        input flux at each detector for given time.

        Parameters
        ----------
        mapping : tolteca.simu.base.SkyMapModel

            The model that defines the on-the-fly mapping trajectory.

        sources : tolteca.simu.base.SourceModel

            The list of models that define the input signal and noise.
        """
        tbl = self.table
        x_t = tbl['x_t']
        y_t = tbl['y_t']

        ref_frame = mapping.ref_frame
        t0 = mapping.t0
        ref_coord = self.resolve_target(mapping.target, t0)

        def evaluate(t):
            with erfa_astrom.set(ErfaAstromInterpolator(self.erfa_interp_len)):
                time_obs = t0 + t
                # transform ref_coord to ref_frame
                # need to re-set altaz frame with frame attrs
                with timeit("transform bore sight coords to projected frame"):
                    _ref_frame = self.resolve_sky_map_ref_frame(
                            ref_frame, time_obs=time_obs)
                    with timeit(
                            f'transform ref coords to {len(time_obs)} times'):
                        _ref_coord = ref_coord.transform_to(_ref_frame)
                    obs_coords = mapping.evaluate_at(_ref_coord, t)
                    hold_flags = mapping.evaluate_holdflag(t)
                    m_proj_icrs = self.get_sky_projection_model(
                            ref_coord=obs_coords,
                            time_obs=time_obs,
                            evaluate_frame='icrs',
                            )
                    m_proj_native = self.get_sky_projection_model(
                            ref_coord=obs_coords,
                            time_obs=time_obs,
                            evaluate_frame='native',
                            )
                    projected_frame, native_frame = \
                        m_proj_icrs.get_projected_frame(
                            also_return_native_frame=True)

                    # there is weird cache issue so we cannot
                    # just do the transform easily
                    if hasattr(obs_coords, 'ra'):  # icrs
                        obs_coords_icrs = SkyCoord(
                                obs_coords.ra, obs_coords.dec,
                                frame='icrs'
                                )
                        _altaz_frame = self.resolve_sky_map_ref_frame(
                                'altaz', time_obs=time_obs)
                        obs_coords_altaz = obs_coords_icrs.transform_to(
                                _altaz_frame)
                    elif hasattr(obs_coords, 'alt'):  # altaz
                        obs_coords_icrs = obs_coords.transform_to('icrs')
                        obs_coords_altaz = obs_coords
                    obs_parallactic_angle = \
                        self.observer.parallactic_angle(
                                time_obs, obs_coords_icrs)

                # get detector positions, which requires absolute time
                # to get the altaz to equatorial transformation

                with timeit("transform det coords to projected frame"):
                    # this has to take into account
                    # the rotation of det coord by alt due to M3
                    a = obs_coords_altaz.alt.radian
                    m_rot_m3 = np.array([
                        [np.cos(a), -np.sin(a)],
                        [np.sin(a), np.cos(a)]
                        ])
                    # there should be more clever way of this but for now
                    # we just spell out the rotation
                    x = m_rot_m3[0, 0][:, np.newaxis] * x_t[np.newaxis, :] \
                        + m_rot_m3[0, 1][:, np.newaxis] * y_t[np.newaxis, :]
                    y = m_rot_m3[1, 0][:, np.newaxis] * x_t[np.newaxis, :] \
                        + m_rot_m3[1, 1][:, np.newaxis] * y_t[np.newaxis, :]
                    lon, lat = m_proj_icrs(
                        x, y, eval_interp_len=0.1 << u.s)
                    az, alt = m_proj_native(x, y, eval_interp_len=0.1 << u.s)

                # TODO: also put all of this stuff somewhere else?

                # spam me even if (when) the c++ part doesnt fail
                import toast
                # from toast.utils import Environment
                # env = Environment.get()
                # env.set_log_level('VERBOSE')
        
                mpi_comm = None
                # TODO: figure out from Ted what the expectations for tmin ---> tmax are
                # we have python datetimes 
                tmin = 0
                tmax = len(time_obs) + 1
                with timeit(f"instantiate tmin:{tmin} and tmax:{tmax}"):

                    # TODO: expose relevant parameters through the configuration file 
                    # there are sooooo mannnnnny
                    toast_atmsim_model = toast.atm.AtmSim(
                        # TODO: what are these units? Below they seem like they needed to be radians
                        # these are in degrees (* u.degree)
                        azmin=np.min(az), azmax=np.max(az),
                        elmin=np.min(alt), elmax=np.max(alt),
                        tmin=tmin, tmax=tmax,
                        lmin_center=0.001 * u.meter,
                        lmin_sigma=0.001*  u.meter,
                        lmax_center=1.0 * u.meter,
                        lmax_sigma=10* u.meter,
                        w_center=10 * (u.km / u.second),
                        w_sigma=10 * (u.km / u.second),
                        wdir_center=10 * u.degree,
                        wdir_sigma=10 * u.degree,
                        z0_center=2000 * u.meter,
                        z0_sigma=0 * u.meter,
                        T0_center=100 * u.Kelvin,
                        T0_sigma=1 * u.Kelvin,
                        zatm=40000.0 * u.meter,
                        zmax=200.0 * u.meter,
                        xstep=105.0 * u.meter,
                        ystep=105.0 * u.meter,
                        zstep=105.0 * u.meter,
                        nelem_sim_max=10000,
                        comm=mpi_comm,
                        key1=0,
                        key2=0,
                        counterval1=0,
                        counterval2=0,
                        cachedir='./toast_cache',
                        rmin=0.0 * u.meter,
                        rmax=1000.0 * u.meter,
                        write_debug=False
                    )
                with timeit("simulating toast atmosphere (for this time chunk)"):
                    # TODO: seems slow / ensure that the chosen atmosphere params aren't messing with runtime
                    err = toast_atmsim_model.simulate(use_cache=True)
                    if err != 0:
                        raise RuntimeError("toast atmosphere simulation failed")
                    pass
                
                atm_result = []
                with timeit("observe the toast atmosphere with detector (for this time chunk)"):
                    
                    # same for all the detectors in this time chunk
                    atm_times = np.linspace(tmin, tmax, len(time_obs))
                    
                    # loop through each detector (it be fairly quick)
                    # (compared with the atmosphere generation step)
                    for az_single, alt_single in zip(az.T, alt.T): 
                        # place to store the atmosphere data
                        atmtod = np.zeros_like(az_single.value)
                        err = toast_atmsim_model.observe(
                            times=atm_times, az=az_single.to(u.radian).value, 
                            el=alt_single.to(u.radian).value, tod=atmtod, fixed_r=0
                        )
                        # what are the units of atm pwr?
                        if err != 0:
                            raise RuntimeError("toast atmosphere detector observation failed")
                        atm_result.append(atmtod)
                    atm_result = np.array(atm_result)
                    atm_result.dump(f'toast_atm_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

                    # format atm_result to something that can be used 
                    # see 's_additive' and 's' 

                    import subprocess
                    subprocess.check_output("mpiexec -n 4 python /Users/dennislee/Documents/repos/toltec_dr_scripts/atm_mpi.py",
                        stderr=subprocess.STDOUT,
                        shell=True
                    )
                    # plz stop (breakpoint)
                    raise RuntimeError("plz stop")

                # combine the array projection with sky projection
                # and evaluate with source frame
                s_additive = []
                for m_source in sources:
                    if isinstance(m_source, SourceCatalogModel):
                        m_source = m_source.make_image_model(
                                beam_models=self._m_beam.models,
                                pixscale=1 << u.arcsec / u.pix
                                )
                    if isinstance(m_source, SourceImageModel):
                        # TODO support more types of wcs. For now
                        # only ICRS is supported
                        # the projected lon lat
                        # extract the flux
                        # detector is required to be the first dimension
                        # for the evaluate_tod
                        with timeit("extract flux from source image"):
                            s = m_source.evaluate_tod(tbl, lon.T, lat.T)
                        s_additive.append(s)
                    # TODO revisit the performance issue here
                    elif False and isinstance(m_source, SourceCatalogModel):
                        with timeit("transform src coords to projected frame"):
                            src_pos = m_source.pos[:, np.newaxis].transform_to(
                                        native_frame).transform_to(
                                            projected_frame)
                        # evaluate with beam_model and reduce on sources axes
                        with timeit("convolve with beam"):
                            dx = x_t[np.newaxis, :, np.newaxis] - \
                                src_pos.lon[:, np.newaxis, :]
                            dy = y_t[np.newaxis, :, np.newaxis] - \
                                src_pos.lat[:, np.newaxis, :]
                            an = np.moveaxis(
                                    np.tile(
                                        tbl['array_name'],
                                        src_pos.shape + (1, )),
                                    1, 2)
                            s = self._m_beam(an, dx, dy)
                            # weighted sum with flux at each detector
                            # assume no polarization
                            w = np.vstack([
                                m_source.data[a] for a in tbl['array_name']]).T
                            s = np.sum(s * w[:, :, np.newaxis], axis=0)
                        s_additive.append(s)
                if len(s_additive) <= 0:
                    s = np.zeros(lon.T.shape) << u.MJy / u.sr
                else:
                    s = s_additive[0]
                    for _s in s_additive[1:]:
                        s += _s
                return s, locals()
        yield evaluate

    @contextmanager
    def obs_context(self, obs_model, sources, ref_coord=None, ref_frame=None):
        """
        Return a function that can be used to get
        input flux at each detector for given time."""
        m_obs = obs_model
        tbl = self.table
        x_t = tbl['x_t']
        y_t = tbl['y_t']

        sources = sources[0]
        # TODO: implement handling of other source model
        if not isinstance(sources, (QTable, Table)):
            raise NotImplementedError

        if ref_coord is None:
            # define a field center
            # here we use the first object in the sources catalog
            # and realize the obs pattern around this center
            # we need to take into acount the ref_frame and
            # prepare ref_coord such that it is in the ref_frame
            ref_coord = coord.SkyCoord(
                    ra=sources['ra'].quantity[0],
                    dec=sources['dec'].quantity[0],
                    frame='icrs')

        def evaluate(t0, t):

            time_obs = t0 + t

            # transform ref_coord to ref_frame
            # need to re-set altaz frame with frame attrs
            _ref_frame = self.resolve_sky_map_ref_frame(
                    ref_frame, time_obs=time_obs)
            _ref_coord = ref_coord.transform_to(_ref_frame)
            obs_coords = m_obs.evaluate_at(_ref_coord, t)
            # get detector positions, which requires absolute time
            # to get the altaz to equatorial transformation
            # here we only project in alt az, and we transform the source coord
            # to alt az for faster computation.

            # combine the array projection with sky projection
            m_proj = self.get_sky_projection_model(
                    ref_coord=obs_coords,
                    time_obs=time_obs
                    )
            # logger.debug(f"proj model:\n{m_proj}")

            projected_frame, native_frame = m_proj.get_projected_frame(
                also_return_native_frame=True)

            # transform the sources on to the projected frame this has to be
            # done in two steps due to limitation in astropy
            with timeit("transform src coords to projected frame"):
                src_coords = coord.SkyCoord(
                    ra=sources['ra'][:, np.newaxis],
                    dec=sources['dec'][:, np.newaxis],
                    frame='icrs').transform_to(
                            native_frame).transform_to(
                                projected_frame)
            # evaluate with beam_model and reduce on sources axes
            with timeit("compute detector pwr loading"):
                dx = x_t[np.newaxis, :, np.newaxis] - \
                    src_coords.lon[:, np.newaxis, :]
                dy = y_t[np.newaxis, :, np.newaxis] - \
                    src_coords.lat[:, np.newaxis, :]
                an = np.moveaxis(
                        np.tile(tbl['array_name'], src_coords.shape + (1, )),
                        1, 2)
                s = self._m_beam(an, dx, dy)
                # weighted sum with flux at each detector
                # assume no polarization
                s = np.squeeze(
                        np.moveaxis(s, 0, -1) @ sources['flux_a1100'][
                            :, np.newaxis],
                        axis=-1)

            # transform all obs_coords to equitorial
            obs_coords_icrs = obs_coords.transform_to('icrs')
            return s, locals()
        yield evaluate

    @classmethod
    def _prepare_table(cls, tbl):
        logger = get_logger()
        # make columns for additional array properties to be used
        # for the kids simulator
        tbl = tbl.copy()
        meta_keys = ['wl_center', ]
        # array props
        for array_name in tbl.meta['array_names']:
            m = tbl['array_name'] == array_name
            props = dict(
                    cls.array_optical_props[array_name],
                    **{k: tbl.meta[array_name][k] for k in meta_keys})
            for c in props.keys():
                if c not in tbl.colnames:
                    tbl.add_column(Column(
                                np.empty((len(tbl), ), dtype=float),
                                name=c, unit=props[c].unit))
                tbl[c][m] = props[c]
            for k, c in [('a', 'a_fwhm'), ('b', 'b_fwhm')]:
                if c not in tbl.colnames:
                    tbl.add_column(Column(
                                np.empty((len(tbl), ), dtype=float),
                                name=c, unit=u.arcsec))
                tbl[c][m] = cls.beam_model_cls.get_fwhm(
                        k, array_name).to_value(u.arcsec)

        # kids props
        for c, v in cls.kids_props.items():
            if c in tbl.colnames:
                continue
            logger.debug(f"create kids prop column {c}")
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
        # calibration factor
        # TODO need to revisit these assumptions
        if 'flxscale' not in tbl.colnames:
            tbl['flxscale'] = (1. / tbl['responsivity']).quantity.value
        if 'sigma_readout' not in tbl.colnames:
            tbl['sigma_readout'] = 10.
        return QTable(tbl)


class KidsReadoutNoiseModel(_Model):
    """
    A model of the TolTEC KIDs readout noise.

    """
    logger = get_logger()

    n_inputs = 1
    n_outputs = 1

    # @property
    # def input_units(self):
    #     return {self.inputs[0]: }

    def __init__(self, scale_factor=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inputs = ('S21', )
        self._outputs = ('dS21', )
        self._scale_factor = scale_factor

    def evaluate(self, S21):
        n = self._scale_factor
        shape = S21.shape
        dI = np.random.normal(0, n, shape)
        dQ = np.random.normal(0, n, shape)
        return dI + 1.j * dQ

    def evaluate_tod(self, tbl, S21):
        """Make readout noise in ADU."""

        dS21 = self(S21)
        dS21 = dS21 * tbl['sigma_readout'][:, np.newaxis]
        return dS21
