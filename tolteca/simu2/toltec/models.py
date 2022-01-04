#!/usr/bin/env python


from gwcs import coordinate_frames as cf
import astropy.units as u
from astropy.time import Time
from astropy.modeling import models, Parameter, Model
from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table
from astropy.cosmology import default_cosmology
from astropy import constants as const
from astropy.utils.decorators import classproperty
from scipy.interpolate import interp1d

import numpy as np

from tollan.utils.log import timeit, get_logger
from kidsproc.kidsmodel import _Model as ComplexModel
from contextlib import contextmanager, ExitStack

from ...utils import get_pkg_data_path
from .toltec_info import toltec_info
from .lmt import get_lmt_atm_models

from ..base import ProjModel, LabelFrame
from ..sources.base import PowerLoadingModel
from ..mapping.utils import rotation_matrix_2d, _get_skyoffset_frame


__all__ = [
    'pa_from_coords',
    'ToltecArrayProjModel', 'ToltecSkyProjModel',
    'KidsReadoutNoiseModel',
    'ToltecArrayPowerLoadingModel',
    'ToltecPowerLoadingModel'
    ]


def pa_from_coords(observer, coords_altaz, coords_icrs):
    """Calculate parallactic angle at coords.

    """
    # TODO: revisit this
    # http://star-www.st-and.ac.uk/~fv/webnotes/chapter7.htm
    # note that their are issues with these values
    # where cosha^2 + sinha^2 is off from 1. by 0.1%. This
    # gives about 0.7 deg of deviation from the direct
    # calculation using LST from time_obs
    cosha = (
        np.sin(coords_altaz.alt.radian)
        - np.sin(coords_icrs.dec.radian)
        * np.sin(observer.location.lat.radian)) / (
            np.cos(coords_icrs.dec.radian)
            * np.cos(observer.location.lat.radian)
            )
    sinha = (
        -np.sin(coords_altaz.az.radian)
        * np.cos(coords_altaz.alt.radian)
        / np.cos(coords_icrs.dec.radian)
        )
    # print(sinha ** 2 + cosha ** 2 - 1)
    parallactic_angle = Angle(np.arctan2(
        sinha,
        (
            np.tan(observer.location.lat.radian)
            * np.cos(coords_icrs.dec.radian)
            - np.sin(coords_icrs.dec.radian)
            * cosha)
        ) << u.rad)
    return parallactic_angle


class ToltecArrayProjModel(ProjModel):
    """
    A model to transform TolTEC detector locations and orientations on the
    each array to a common TolTEC instrument frame defined in offset angle
    unit, with the extent of arrays normalized to the size of the on-sky
    field of view.

    The TolTEC frame is attached to the TolTEC instrument body and describes
    the projected positions and orientations of all detectors on the sky. The
    origin of the TolTEC frame is fixed at the telescope bore sight.

    The two axes az_offset and alt_offset is aligned with the telescope
    Az/Alt at altitude of 0 deg, and they rotate by the value of the altitude
    following the left hand rule.

    The orientations of detectors also get projected to the TolTEC frame,
    where the P.A = 0 is set to be the +alt_offset and the sign convention
    follows the left hand rule.
    """

    input_frame = cf.CompositeFrame([
        cf.Frame2D(
            name='det_pos',
            axes_names=("x", "y"),
            unit=(u.um, u.um),
            ),
        LabelFrame(
            axes_names=['array', 'fg'], axes_order=(2, 3),
            name='det_prop'),
        ], name='focal_plane')
    output_frame = cf.CompositeFrame([
        cf.Frame2D(
            name='sky_offset',
            axes_names=("az_offset", "alt_offset"),
            unit=(u.deg, u.deg)),
        cf.CoordinateFrame(
            naxes=1,
            axes_type='SPATIAL',
            axes_order=(2, ),
            unit=(u.deg, ),
            axes_names=("pa", ),
            name='det_pa'),
        ], name='toltec')
    n_inputs = input_frame.naxes
    n_outputs = output_frame.naxes

    _array_index_to_mounting_angle = {
        toltec_info[array_name]['index']:
        toltec_info[array_name]['array_mounting_angle']
        for array_name in toltec_info['array_names']
        }

    _fg_to_det_pa = {
        toltec_info[fg_name]['index']:
        toltec_info[fg_name]['det_pa']
        for fg_name in toltec_info['fg_names']
        }

    _plate_scale = toltec_info['fov_diameter'] \
        / toltec_info['array_physical_diameter']
    # this is need to make the affine transform work correctly
    _plate_unit = toltec_info['array_physical_diameter'].unit

    _mat_refl = np.array([[1, 0], [0, -1]], dtype='d')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # build the models for transforming x and y
        m_pos = dict()
        m_pa = dict()
        for ai, rot in self._array_index_to_mounting_angle.items():
            m_pos[ai] = models.AffineTransformation2D(
                (
                    rotation_matrix_2d(rot.to_value(u.rad)) @ self._mat_refl
                    ) << self._plate_unit,
                translation=(0., 0.) << self._plate_unit
                ) | (
                    models.Multiply(self._plate_scale) &
                    models.Multiply(self._plate_scale)
                    )
            for fg, pa in self._fg_to_det_pa.items():
                m_pa[(ai, fg)] = models.Const1D(pa + rot)
        m_proj = dict()
        for k, m in m_pa.items():
            # build the full proj model
            # k[0] is array index
            m_proj[k] = models.Mapping((0, 1, 0)) | m_pos[k[0]] & m_pa[k]
        self._m_pos = m_pos
        self._m_pa = m_pa
        self._m_proj = m_proj

    def evaluate(self, x, y, array, fg):
        # note that both array and fg are coerced to double and
        # we need to make them int before creating the masks
        array = array.astype(int)
        fg = fg.astype(int)
        # loop over proj models and populate result
        result = np.empty((self.n_outputs, ) + x.shape, dtype='d') << u.deg
        # this is used to check if all values are covered
        not_computed = np.ones(x.shape, dtype=bool)
        for k, m in self._m_proj.items():
            mask = (array == k[0]) & (fg == k[1])
            result[0, mask], result[1, mask], result[2, mask] = m(
                x[mask], y[mask])
            not_computed[mask] = False
        if np.sum(not_computed) > 0:
            invalid = np.unique(
                np.vstack([
                    array[not_computed],
                    fg[not_computed]]),
                axis=1
                ).T
            raise ValueError(
                f"Invalid (array, fg) in input: {invalid}")
        # apply the transformation for each unit
        return result


class ToltecSkyProjModel(ProjModel):
    """
    A model to transform TolTEC detector positions and orientations
    expressed in offset angular unit in the TolTEC frame to
    absolute world coordinates for given telescope bore sight target
    and time of obs.

    The output coordinate frame is a generic sky lon/lat frame which
    can represent any of the valid celestial coordinate frames supported,
    by specifying the ``evaluate_frame`` keyword argument.
    """

    logger = get_logger()

    def __init__(
            self,
            origin_coords_icrs=None,
            origin_coords_altaz=None,
            time_obs=None):
        origin_coords_icrs, origin_coords_altaz, \
            origin_az, origin_alt, mjd = self._make_origin_coords(
                origin_coords_icrs=origin_coords_icrs,
                origin_coords_altaz=origin_coords_altaz,
                time_obs=time_obs,
                ensure_altaz=True,
                ensure_icrs=True,
                return_params=True,
                )
        if np.isscalar(mjd):
            n_models = 1
        else:
            n_models = len(mjd)
        super().__init__(
            origin_az=origin_az, origin_alt=origin_alt, mjd=mjd,
            n_models=n_models)
        self._origin_coords_icrs = origin_coords_icrs
        self._origin_coords_altaz = origin_coords_altaz
        # this is to be overridden by the __call__ so that we can
        # ensure the evaluation is always done with __call__
        self._eval_context = None

    def __setattr__(self, attr, value):
        # since we cache the origin coords and we need to disallow
        # changing the params to make all of the values in-sync.
        if attr in ('origin_az', 'origin_alt', 'mjd'):
            raise AttributeError(f'{attr} is read-only')
        return super().__setattr__(attr, value)

    @classmethod
    def _make_origin_coords(
            cls,
            origin_coords_icrs, origin_coords_altaz, time_obs,
            ensure_altaz=True,
            ensure_icrs=True,
            return_params=True,
            ):
        if sum([origin_coords_altaz is None, origin_coords_icrs is None]) == 2:
            raise ValueError(
                "at least one of origin_coords_{altaz,icrs} is needed.")
        if origin_coords_altaz is None and (ensure_altaz or return_params):
            # compute origin altaz from icrs and time_obs
            if time_obs is None:
                raise ValueError("time is need to transform to altaz.")
            with timeit("transform origin from icrs to altaz"):
                origin_coords_altaz = origin_coords_icrs.transform_to(
                    cls.observer.altaz(time=time_obs))
        if origin_coords_icrs is None and ensure_icrs:
            # compute origin icrs from altaz
            with timeit("transform origin from altaz to icrs"):
                origin_coords_icrs = origin_coords_altaz.transform_to("icrs")
        if return_params:
            origin_az = origin_coords_altaz.az
            origin_alt = origin_coords_altaz.alt
            mjd = (origin_coords_altaz.frame.obstime.mjd) << u.day
            return (
                origin_coords_icrs, origin_coords_altaz,
                origin_az, origin_alt, mjd)
        return (origin_coords_icrs, origin_coords_altaz)

    input_frame = ToltecArrayProjModel.output_frame
    output_frame = cf.CompositeFrame([
        cf.Frame2D(
            name='sky',
            axes_names=("lon", "lat"),
            unit=(u.deg, u.deg)),
        cf.CoordinateFrame(
            naxes=1,
            axes_type='SPATIAL',
            axes_order=(2, ),
            unit=(u.deg, ),
            axes_names=("pa", ),
            name='det_pa'),
        ], name='sky')

    n_inputs = input_frame.naxes
    n_outputs = output_frame.naxes

    origin_az = Parameter(
        default=180.,
        unit=output_frame.unit[0],
        description='The Az of the telescope bore sight.'
        )
    origin_alt = Parameter(
        default=60.,
        unit=output_frame.unit[1],
        description='The Alt of the telescope bore sight.'
        )
    mjd = Parameter(
        default=Time(2022.0, format='jyear').mjd,
        unit=u.day,
        description='The UT of observation expressed in MJD.'
        )

    observer = toltec_info['site']['observer']
    """The observer (LMT)."""

    @classmethod
    def _get_altaz_frame(cls, mjd):
        return cls.observer.altaz(time=Time(mjd, format='mjd'))

    @classmethod
    def _get_origin_coords_altaz(cls, origin_az, origin_alt, mjd):
        """Return the origin coordinates in AltAz."""
        return SkyCoord(
            origin_az,
            origin_alt,
            frame=cls._get_altaz_frame(mjd)
            )

    @classmethod
    @timeit
    def _get_altaz_offset_frame(cls, origin_coords_altaz):
        """Return the sky offset frame in AltAz centered at origin."""
        return _get_skyoffset_frame(origin_coords_altaz)

    @classmethod
    @timeit
    def evaluate_altaz(
            cls, x, y, pa,
            origin_coords_icrs=None,
            origin_coords_altaz=None,
            time_obs=None):
        """Compute the projected coordinates in AltAz using full
        transformation.
        """
        _, origin_coords_altaz = cls._make_origin_coords(
            origin_coords_icrs=origin_coords_icrs,
            origin_coords_altaz=origin_coords_altaz,
            time_obs=time_obs,
            ensure_altaz=True,
            ensure_icrs=False,
            return_params=False,
            )
        # now we always have origin_coords_altaz
        with timeit("apply rotation to detector offset coords"):
            origin_alt = origin_coords_altaz.alt
            # The first step has to be rotation the toltec frame by
            # the amount of origin_coords_altaz.alt, due to the M3 mirror.
            mat_rot_m3 = rotation_matrix_2d(origin_alt.to_value(u.rad))

            # there should be more clever way of this but for now
            # we just spell out the rotation because x and y are already
            # separated arrays
            x_offset_altaz = mat_rot_m3[0, 0] * x + mat_rot_m3[0, 1] * y
            y_offset_altaz = mat_rot_m3[1, 0] * x + mat_rot_m3[1, 1] * y
            # y_offset_altaz = mat_rot_m3[1, 0][:, np.newaxis] \
            #     * x[np.newaxis, :] \
            #     + mat_rot_m3[1, 1][:, np.newaxis] * y[np.newaxis, :]
            # the pa get rotated by the value of alt
            pa_altaz = (pa + origin_alt).to(u.deg)

        # now do the coordinate transformation
        with timeit("transform detector offset coords to altaz"):
            altaz_offset_frame = cls._get_altaz_offset_frame(
                origin_coords_altaz)
            det_coords_altaz_offset = SkyCoord(
                x_offset_altaz, y_offset_altaz, frame=altaz_offset_frame)
            det_coords_altaz = det_coords_altaz_offset.transform_to(
                origin_coords_altaz.frame)
        return det_coords_altaz.az, det_coords_altaz.alt, pa_altaz

    @classmethod
    @timeit
    def evaluate_icrs_fast(
            cls, x, y, pa,
            origin_coords_icrs=None,
            origin_coords_altaz=None,
            time_obs=None):
        """Compute the projected coordinates in ICRS with small field
        approximation (TolTEC FOV is small ~4 arcmin) directly.
        """
        origin_coords_icrs, origin_coords_altaz = cls._make_origin_coords(
            origin_coords_icrs=origin_coords_icrs,
            origin_coords_altaz=origin_coords_altaz,
            time_obs=time_obs,
            ensure_altaz=True,
            ensure_icrs=True,
            return_params=False,
            )
        with timeit("compute rotation angle from toltec frame to icrs"):
            origin_par_angle = cls.observer.parallactic_angle(
                        origin_coords_altaz.obstime,
                        origin_coords_icrs)
            # now we can rotate the x y and pa by alt + par_ang
            rot = origin_coords_altaz.alt + origin_par_angle

        with timeit("apply rotation to detector offset coords"):
            # The first step has to be rotation the toltec frame by
            # the amount of origin_alt, due to the M3 mirror.
            mat_rot_m3 = rotation_matrix_2d(rot.to_value(u.rad))

            # there should be more clever way of this but for now
            # we just spell out the rotation because x and y are already
            # separated arrays
            x_offset_icrs = mat_rot_m3[0, 0][:, np.newaxis] \
                * x[np.newaxis, :] \
                + mat_rot_m3[0, 1][:, np.newaxis] * y[np.newaxis, :]
            y_offset_icrs = mat_rot_m3[1, 0][:, np.newaxis] \
                * x[np.newaxis, :] \
                + mat_rot_m3[1, 1][:, np.newaxis] * y[np.newaxis, :]
            # the pa get rotated by the value of rot
            pa_icrs = pa + rot

        with timeit("transform detector offset coords to icrs"):
            # now we need to build the icrs offset frame and transform back to
            # absolute coordinates
            icrs_offset_frame = _get_skyoffset_frame(origin_coords_icrs)

            det_coords_icrs_offset = SkyCoord(
                x_offset_icrs, y_offset_icrs, frame=icrs_offset_frame)
            det_coords_icrs = det_coords_icrs_offset.transform_to(
                origin_coords_icrs.frame)
            return det_coords_icrs.ra, det_coords_icrs.dec, pa_icrs

    @staticmethod
    def _check_frame_by_name(frame, frame_name):
        if isinstance(frame, str):
            return frame == frame_name
        return frame.name == frame_name

    @timeit
    def evaluate(
            self,
            x, y, pa, origin_az, origin_alt, mjd):
        # make sure we have _eval_context set before proceed
        eval_ctx = self._eval_context
        if eval_ctx is None:
            raise ValueError("This model can only be evaluated with __call__")
        evaluate_frame = eval_ctx['evaluate_frame']

        # create origin coords in altaz
        origin_coords_altaz = self._get_origin_coords_altaz(
            origin_az=origin_az, origin_alt=origin_alt,
            mjd=mjd)

        result_altaz = self.evaluate_altaz(
            x, y, pa, origin_coords_altaz=origin_coords_altaz)

        # update evaluate_context
        result_az, result_alt, pa_altaz = result_altaz
        coords_altaz = SkyCoord(
            az=result_az, alt=result_alt, frame=origin_coords_altaz.frame
            )
        eval_ctx['pa_altaz'] = pa_altaz
        eval_ctx['coords_altaz'] = coords_altaz

        if self._check_frame_by_name(evaluate_frame, 'altaz'):
            return result_altaz
        elif self._check_frame_by_name(evaluate_frame, 'icrs'):
            # TODO the handling of other frame for the PA has to be on a
            # per-frame basis? So we only implement for now the ICRS
            with timeit("transform detector coords from altaz to icrs"):
                coords_icrs = coords_altaz.transform_to('icrs')
                # calculate the par angle between the two set of coords
                dpa_altaz_icrs = pa_from_coords(
                    observer=self.observer,
                    coords_altaz=coords_altaz,
                    coords_icrs=coords_icrs)
                pa_icrs = pa_altaz + dpa_altaz_icrs
            eval_ctx['pa_icrs'] = pa_icrs
            eval_ctx['coords_icrs'] = coords_icrs
            eval_ctx['dpa_altaz_icrs'] = dpa_altaz_icrs
            return coords_icrs.ra, coords_icrs.dec, pa_icrs
        else:
            raise ValueError(f"invalid evaluate_frame {evaluate_frame}")

    @timeit('toltec_sky_proj_evaluate')
    def __call__(
            self, *args,
            evaluate_frame='icrs',
            use_evaluate_icrs_fast=False,
            return_eval_context=False):

        result_eval_context = dict(
                evaluate_frame=evaluate_frame,
                )

        @contextmanager
        def _set_eval_context():
            nonlocal result_eval_context
            self._eval_context = result_eval_context
            yield
            self._eval_context = None

        def wrap_return(result):
            nonlocal result_eval_context
            if return_eval_context:
                return result, result_eval_context
            return result

        with _set_eval_context():
            if self._check_frame_by_name(evaluate_frame, 'icrs') and \
                    use_evaluate_icrs_fast:
                # use the fast icrs eval
                return wrap_return(self.evaluate_icrs_fast(
                    *args,
                    origin_coords_altaz=self._origin_coords_altaz,
                    origin_coords_icrs=self._origin_coords_icrs,
                    ))
            return wrap_return(super().__call__(*args))

    # TODO this is to override the default behavior of checking the model
    # axis. We allow the model axis to broadcasted with size=1.
    def _validate_input_shape(
            self, _input, idx, argnames, model_set_axis, check_model_set_axis):
        """
        Perform basic validation of a single model input's shape
            -- it has the minimum dimensions for the given model_set_axis

        Returns the shape of the input if validation succeeds.
        """
        input_shape = np.shape(_input)
        # Ensure that the input's model_set_axis matches the model's
        # n_models
        if input_shape and check_model_set_axis:
            # Note: Scalar inputs *only* get a pass on this
            if len(input_shape) < model_set_axis + 1:
                raise ValueError(
                    f"For model_set_axis={model_set_axis},"
                    f" all inputs must be at "
                    f"least {model_set_axis + 1}-dimensional.")
            if input_shape[model_set_axis] > 1 and (
                    input_shape[model_set_axis] != self._n_models):
                try:
                    argname = argnames[idx]
                except IndexError:
                    # the case of model.inputs = ()
                    argname = str(idx)

                raise ValueError(
                    f"Input argument '{argname}' does not have the correct "
                    f"dimensions in model_set_axis={model_set_axis} for a "
                    f"model set with "
                    f"n_models={self._n_models}.")
        return input_shape


class KidsReadoutNoiseModel(ComplexModel):
    """
    A model of the TolTEC KIDs readout noise.

    """
    logger = get_logger()

    n_inputs = 1
    n_outputs = 1

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

    def evaluate_tod(self, apt, S21):
        """Make readout noise in ADU."""

        dS21 = self(S21)
        dS21 = dS21 * apt['sigma_readout'][:, np.newaxis]
        return dS21


def _get_default_passbands():
    """Return the default TolTEC passband tables as a dict.
    """
    from ...cal.toltec import ToltecPassband
    calobj = ToltecPassband.from_indexfile(get_pkg_data_path().joinpath(
        'cal/toltec_passband/index.yaml'
        ))
    result = dict()
    for array_name in calobj.array_names:
        result[array_name] = calobj.get(array_name=array_name)
    return result


class ToltecArrayPowerLoadingModel(Model):
    """
    A model of the LMT optical loading at the TolTEC arrays.

    This is based on the Mapping-speed-calculator
    """

    # TODO allow overwriting these per instance.
    _toltec_passbands = _get_default_passbands()
    _cosmo = default_cosmology.get()

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
        self._array_info = toltec_info[array_name]
        self._passband = self._toltec_passbands[array_name]
        self._f = self._passband['f'].quantity
        # check the f step, they shall be uniform
        df = np.diff(self._f).value
        if np.std(df) / df[0] > 1e-7:
            raise ValueError(
                "invalid passband format, frequency grid has to be uniform")
        self._df = self._f[1] - self._f[0]
        self._throughput = self._passband['throughput']
        if atm_model_name is not None:
            self._atm_model, self._atm_tx_model = get_lmt_atm_models(
                name=atm_model_name)
        else:
            self._atm_model = None
            # TODO revisit this
            _, self._atm_tx_model = get_lmt_atm_models(
                name='am_q50')

    @property
    def has_atm_model(self):
        return self._atm_model is not None

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
        if atm_model is None:
            return np.squeeze(np.zeros((alt.size, self._f.size)) << u.K)
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

    def _get_dP(self, alt, f_smp):
        """Return the detector power loading uncertainty according to the nep
        """
        return (
            self._get_noise(alt)['nep']
            * np.sqrt(f_smp / 2.)).to(u.pW)

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

    def sky_sb_to_pwr(self, det_s):
        """Return detector power loading for given on-sky surface brightness.
        """
        # note that this is approximate using a square passband.
        wl_center = self._array_info['wl_center']
        pb_width = self._array_info['passband']
        tb = det_s.to(
                    u.K,
                    equivalencies=u.brightness_temperature(
                        wl_center))
        p = (
            tb.to(
                u.J,
                equivalencies=u.temperature_energy())
            * pb_width
            ).to(u.pW)
        # the sys eff is also approximate
        sys_eff = self._wsum(
                self._system_efficiency, self._throughput
                )
        return p * sys_eff

    @contextmanager
    def atm_eval_interp_context(self, alt_grid):
        interp_kwargs = dict(kind='linear')
        with timeit(
            f"setup atm eval interp context with "
            f"alt_grid=[{alt_grid.min()}:{alt_grid.max()}] "
                f"size={len(alt_grid)}"):
            self._p_pW_interp = interp1d(
                    alt_grid.to_value(u.deg),
                    self._get_P(alt_grid).to_value(u.pW),
                    **interp_kwargs
                    )
            one_Hz = 1 << u.Hz
            self._dp_pW_interp_unity_f_smp = interp1d(
                    alt_grid.to_value(u.deg),
                    self._get_dP(alt_grid, one_Hz).to_value(u.pW),
                    **interp_kwargs
                    )
        yield self
        self._p_pW_interp = None
        self._dp_pW_interp_unity_f_smp = None

    def evaluate_tod(
            self,
            det_alt,
            f_smp,
            random_seed=None,
            ):
        """Return the atmosphere power loading along with the noise."""

        if self._p_pW_interp is None:
            # no interp, direct eval
            alt = np.ravel(det_alt)
            det_pwr_atm = self._get_P(alt).to(u.pW).reshape(det_alt.shape),
            det_delta_pwr_atm_pW = self._get_dP(alt, f_smp).reshape(
                det_alt.shape),
        else:
            det_pwr_atm = self._p_pW_interp(det_alt.degree) << u.pW
            one_Hz = 1. << u.Hz
            det_delta_pwr_atm_pW = self._dp_pW_interp_unity_f_smp(
                det_alt.degree) * np.sqrt(f_smp / one_Hz)
        rng = np.random.default_rng(seed=random_seed)
        det_dp_atm = rng.normal(0., det_delta_pwr_atm_pW) << u.pW
        # calc the median P and dP for logging purpose
        med_alt = np.median(det_alt)
        med_P = self._get_P(med_alt).to(u.pW)
        med_dP = self._get_dP(med_alt, f_smp).to(u.aW)
        self.logger.debug(f"atm at med_alt={med_alt} P={med_P} dp={med_dP}")
        return det_pwr_atm + det_dp_atm


class ToltecPowerLoadingModel(PowerLoadingModel):
    """
    A wrapper model to calculate power loading for all the TolTEC arrays.

    This model in-corporates both the "static" am_qxx models and the toast
    model.
    """

    logger = get_logger()
    array_names = toltec_info['array_names']

    n_inputs = 3
    n_outputs = 1

    def __init__(self, atm_model_name):
        if atm_model_name is None or atm_model_name == 'toast':
            _atm_model_name = 'am_q50'
        else:
            _atm_model_name = atm_model_name
        # we need the ToltecArrayPowerLoadingModel for converting sb to
        # pwr, so we need it anyway.
        self._array_power_loading_models = {
            array_name: ToltecArrayPowerLoadingModel(
                array_name=array_name,
                atm_model_name=_atm_model_name)
            for array_name in self.array_names
            }
        super().__init__(name='toltec_power_loading')
        self.inputs = ('array_name', 'S', 'alt')
        self.outputs = ('P', )
        self._atm_model_name = atm_model_name

    @property
    def atm_model_name(self):
        return self._atm_model_name

    def evaluate(self):
        # TODO
        # implement the default behavior for the model
        return NotImplemented

    def atm_eval_interp_context(self, alt_grid):
        es = ExitStack()
        for m in self._array_power_loading_models.values():
            es.enter_context(m.atm_eval_interp_context(alt_grid))
        return es

    def evaluate_tod(
            self, det_array_name, det_s, det_alt,
            f_smp,
            noise_seed=None,
            ):
        p_out = np.zeros(det_s.shape) << u.pW
        for array_name in self.array_names:
            mask = (det_array_name == array_name)
            aplm = self._array_power_loading_models[array_name]
            # compute the power loading from on-sky surface brightness
            p = aplm.sky_sb_to_pwr(det_s=det_s[mask])
            if self.atm_model_name is None:
                # atm is disabled
                pass
            elif self.atm_model_name == 'toast':
                p_atm = calc_toast_atm_pwr(det_alt=det_alt[mask])
                p += p_atm
            else:
                # use the ToltecArrayPowerLoadingModel atm
                p_atm = aplm.evaluate_tod(
                    det_alt=det_alt[mask],
                    f_smp=f_smp,
                    random_seed=noise_seed,
                    )
                p += p_atm
            p_out[mask] = p
        return p_out

    def __str__(self):
        return (
            f'{self.__class__.__name__}(atm_model_name={self.atm_model_name})')


def calc_toast_atm_pwr(det_alt):
    # TODO
    # implement this
    raise NotImplementedError("toast atm is not implemented yet")
