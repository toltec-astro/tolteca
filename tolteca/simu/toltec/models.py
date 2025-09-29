#!/usr/bin/env python


from gwcs import coordinate_frames as cf
import astropy.units as u
from astropy.time import Time
from astropy.modeling import models, Parameter, Model
from astropy.modeling.functional_models import GAUSSIAN_SIGMA_TO_FWHM
from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table
from astropy.cosmology import default_cosmology
from astropy import constants as const
from astropy.utils.decorators import classproperty
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
import numpy as np

from tollan.utils.dataclass_schema import add_schema
from tollan.utils.log import timeit, get_logger
from tollan.utils.fmt import pformat_yaml
from kidsproc.kidsmodel import _Model as ComplexModel
from contextlib import contextmanager, ExitStack

from ...utils.common_schema import PhysicalTypeSchema
from ...utils import get_pkg_data_path
from .toltec_info import toltec_info
from ..lmt import get_lmt_atm_models

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


def _get_atm_model_tau(atm_model_name):
    data = {
        'am_q25': {
            'tau225': 0.051,
            'tau_a1100': 0.072,
            'tau_a1400': 0.047,
            'tau_a2000': 0.026,
            },
        'am_q50': {
            'tau225': 0.090,
            'tau_a1100': 0.126,
            'tau_a1400': 0.083,
            'tau_a2000': 0.042,
            },
        'am_q75': {
            'tau225': 0.161,
            'tau_a1100': 0.227,
            'tau_a1400': 0.149,
            'tau_a2000': 0.072,
            },
        'am_q95': {
            'tau225': 0.310,
            'tau_a1100': 0.438,
            'tau_a1400': 0.286,
            'tau_a2000': 0.135,
            },
        'toast': {
            'tau225': 0.,
            'tau_a1100': 0.,
            'tau_a1400': 0.,
            'tau_a2000': 0.,
            },
        }
    return data.get(atm_model_name, {'tau225': 0., 'tau_a1100': 0., 'tau_a1400': 0., 'tau_a2000': 0})


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

    def __init__(
            self,
            array_name,
            atm_model_name='am_q50',
            tel_surface_rms=None,
            det_noise_factor=None,
            mapping_speed=None,
            mapping_speed_alt=None,
            mapping_speed_n_dets=None,
            *args, **kwargs):
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
            # we still need the atm transmission for calculating efficiency
            # TODO revisit this
            _, self._atm_tx_model = get_lmt_atm_models(
                name='am_q50')
        self._internal_params = self._internal_params_default.copy()
        if tel_surface_rms is None:
            tel_surface_rms = 76. << u.um
        self._internal_params['tel_surface_rms'] = tel_surface_rms
        if det_noise_factor is None:
            # this is to re-define det_noise_factor to linear unit
            det_noise_factor = 0.334 ** 0.5

        self._internal_params['det_noise_factor'] = det_noise_factor 

        if mapping_speed_alt is None:
            mapping_speed_alt = 70. << u.deg
        if mapping_speed_n_dets is None:
            mapping_speed_n_dets = 7000
        # calculate ms_value without scaling first
        self._internal_params['global_noise_factor'] = 1.
        ms_value = self.get_mapping_speed(alt=mapping_speed_alt, n_dets=mapping_speed_n_dets)
        self.logger.debug(f"internal default unscaled mapping speed: {ms_value} at alt={mapping_speed_alt} n_dets={mapping_speed_n_dets}")
        if mapping_speed is None:
            pass
        else:
            global_noise_factor = ((ms_value / mapping_speed) ** 0.5).to_value(u.dimensionless_unscaled)
            self.logger.info(f"use {global_noise_factor=} for target mapping speed {mapping_speed}")
            self._internal_params['global_noise_factor'] = global_noise_factor 
        self.logger.debug(f"power loading model internal parameters:\n{pformat_yaml(self._internal_params)}")

    @property
    def has_atm_model(self):
        return self._atm_model is not None

    @classproperty
    def _internal_params_default(cls):
        """Lower level instrument parameters for LMT/TolTEC.

        Note that all these values does not take into account the
        passbands, and are frequency independent.
        """
        # TODO merge this to the instrument fact yaml file?
        p = {
                'det_optical_efficiency': 0.8,
                'horn_aperture_efficiency': 0.35,
                'tel_diameter': 48. << u.m,
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

    def _get_syseff_sky_to_det(self, return_avg=False):
        """The overall system efficiency over the passband."""
        syseff = (
                self._tel_primary_surface_optical_efficiency
                * self._internal_params['cold_efficiency']
                * self._throughput
                )
        if return_avg:
            return self._wsum(syseff, self._throughput)
        return syseff

    def _get_syseff_window_to_det(self, return_avg=False):
        """The system efficiency over the passband from window to detectors."""
        syseff = (
                self._internal_params['cold_efficiency']
                * self._throughput
                )
        if return_avg:
            return self._wsum(syseff, self._throughput)
        return syseff

    @property
    def _system_efficiency(self):
        return self._get_syseff_sky_to_det(return_avg=False)

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
            If True, return the weighted sum over the system efficiency
            instead.
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
                1. + self._internal_params['det_noise_factor'] ** 2)
        # scale it further by global noise factor
        det_noise_coeff = self._internal_params["global_noise_factor"] * det_noise_coeff

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
        result['alt'] = alt
        result['P'] = self._get_P(alt)
        result.update(self._get_noise(alt, return_avg=True))
        return Table(result)

    def get_mapping_speed(self, alt, n_dets):

        sens = self._get_noise(alt, return_avg=True)
        array_name = self._array_name
        a_stddev = toltec_info[array_name]['a_fwhm'] / GAUSSIAN_SIGMA_TO_FWHM
        b_stddev = toltec_info[array_name]['b_fwhm'] / GAUSSIAN_SIGMA_TO_FWHM
        beam_area = 2 * np.pi * a_stddev * b_stddev
        mapping_speed = (
            (1 / sens['nefd']) ** 2 * n_dets * beam_area
            ).to(u.deg ** 2 / u.mJy ** 2 / u.hr)
        return mapping_speed

    def get_map_rms_depth(self, alt, n_dets, map_area, t_exp):

        mapping_speed = self.get_mapping_speed(alt, n_dets)
        rms_depth = np.sqrt(map_area / (t_exp * mapping_speed)).to(u.mJy)
        return rms_depth

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
        syseff = self._get_syseff_sky_to_det(return_avg=True)
        return p * syseff

    @contextmanager
    def eval_interp_context(self, alt_grid):
        interp_kwargs = dict(kind='linear')
        with timeit(
            f"setup power loading model for {self._array_name} "
            f"eval interp context with "
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
            f_smp=1 << u.Hz,
            random_seed=None,
            return_realized_noise=True,
            ):
        """Return the array power loading along with the noise."""

        if getattr(self, '_p_pW_interp', None) is None:
            # no interp, direct eval
            alt = np.ravel(det_alt)
            det_pwr = self._get_P(alt).to(u.pW).reshape(det_alt.shape),
            det_delta_pwr = self._get_dP(alt, f_smp).reshape(
                det_alt.shape).to(u.pW),
        else:
            det_pwr = self._p_pW_interp(det_alt.degree) << u.pW
            one_Hz = 1. << u.Hz
            det_delta_pwr = (self._dp_pW_interp_unity_f_smp(
                det_alt.degree) << u.pW) * np.sqrt(f_smp / one_Hz)
        if not return_realized_noise:
            return det_pwr, det_delta_pwr
        # realize noise
        rng = np.random.default_rng(seed=random_seed)
        det_noise = rng.normal(0., det_delta_pwr.to_value(u.pW)) << u.pW
        # calc the median P and dP for logging purpose
        med_alt = np.median(det_alt)
        med_P = self._get_P(med_alt).to(u.pW)
        med_dP = self._get_dP(med_alt, f_smp).to(u.aW)
        self.logger.debug(
            f"array power loading at med_alt={med_alt} P={med_P} dP={med_dP}")
        return det_pwr, det_noise


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

    def __init__(
            self, atm_model_name, atm_model_params=None,
            atm_cache_dir=None,
            tel_surface_rms=None,
            det_noise_factor=None,
            mapping_speed=None,
            mapping_speed_alt=None,
            mapping_speed_n_dets=None,
            ):
        if atm_model_name is None or atm_model_name == 'toast':
            # this will disable the atm component in the power loading model
            # but still create one for system efficiency calculation
            _atm_model_name = None
        else:
            _atm_model_name = atm_model_name

        if isinstance(det_noise_factor, dict):
            if not set(det_noise_factor.keys()) == set(self.array_names):
                raise ValueError("invalid det noise factor.")
        else:
            det_noise_factor = {array_name: det_noise_factor for array_name in self.array_names}

        if isinstance(mapping_speed, dict):
            if not set(mapping_speed.keys()) == set(self.array_names):
                raise ValueError("invalid mapping speed.")
        else:
            mapping_speed = {array_name: mapping_speed for array_name in self.array_names}

        self._array_power_loading_models = {
            array_name: ToltecArrayPowerLoadingModel(
                array_name=array_name,
                atm_model_name=_atm_model_name,
                tel_surface_rms=tel_surface_rms,
                det_noise_factor=det_noise_factor[array_name],
                mapping_speed=mapping_speed[array_name],
                mapping_speed_alt=mapping_speed_alt,
                mapping_speed_n_dets=mapping_speed_n_dets,
                )
            for array_name in self.array_names
            }
        if atm_model_name == 'toast':
            self._toast_atm_evaluator = ToastAtmEvaluator(
                cache_dir=atm_cache_dir,
                params=atm_model_params)
        else:
            self._toast_atm_evaluator = None
        super().__init__(name='toltec_power_loading')
        self.inputs = ('array_name', 'S', 'alt')
        self.outputs = ('P', )
        self._atm_model_name = atm_model_name
        self._atm_model_tau = _get_atm_model_tau(atm_model_name)

    @property
    def atm_model_name(self):
        return self._atm_model_name

    def evaluate(self):
        # TODO
        # implement the default behavior for the model
        return NotImplemented

    def aplm_eval_interp_context(
            self, t0, t_grid,
            sky_bbox_altaz, alt_grid):
        """Context manager that pre-calculate the interp for array power
        loading model.
        """
        es = ExitStack()
        for m in self._array_power_loading_models.values():
            es.enter_context(m.eval_interp_context(alt_grid))
        # setup the toast eval context
        if self._toast_atm_evaluator is not None:
            es.enter_context(self._toast_atm_evaluator.setup(
                t0=t0,
                t_grid=t_grid,
                sky_bbox_altaz=sky_bbox_altaz,
                alt_grid=alt_grid,
                ))
        return es

    def _get_toast_P(self, array_name, det_az, det_alt, time_obs):
        """Return the toast power loading with TolTEC system efficiency."""
        aplm = self._array_power_loading_models[array_name]
        if time_obs is None:
            raise ValueError("time_obs is required for toast atm")
        p = self._toast_atm_evaluator.calc_toast_atm_pwr_for_array(
            array_name=array_name,
            det_az=det_az,
            det_alt=det_alt,
            time_obs_unix=time_obs.unix,
            )
        return p * aplm._get_syseff_window_to_det(return_avg=True)

    def get_P(self, det_array_name, det_az, det_alt, time_obs=None):
        """Evaluate the power loading model only and without noise."""
        p_out = np.zeros(det_alt.shape) << u.pW
        for array_name in self.array_names:
            mask = (det_array_name == array_name)
            aplm = self._array_power_loading_models[array_name]
            if self.atm_model_name == 'toast':
                p = self._get_toast_P(
                    array_name=array_name,
                    det_az=det_az[mask],
                    det_alt=det_alt[mask],
                    time_obs=time_obs,
                    )
                # this power is only for the atm so we still need
                # to add the telescope warm components and system efficiency
                # note in this case aplm does not have the am_qxx models
                # enabled.
                p_tel, _ = aplm.evaluate_tod(
                    det_alt=det_alt[mask],
                    return_realized_noise=False,
                    )
                p += p_tel
            else:
                # use the ToltecArrayPowerLoadingModel
                p, _ = aplm.evaluate_tod(
                    det_alt[mask], return_realized_noise=False)
            p_out[mask] = p
        return p_out

    def sky_sb_to_pwr(self, det_array_name, det_s):
        p_out = np.zeros(det_s.shape) << u.pW
        for array_name in self.array_names:
            mask = (det_array_name == array_name)
            aplm = self._array_power_loading_models[array_name]
            # compute the power loading from on-sky surface brightness
            p_out[mask] = aplm.sky_sb_to_pwr(det_s=det_s[mask])
        return p_out

    def evaluate_tod(
            self, det_array_name, det_s, det_az, det_alt,
            f_smp,
            noise_seed=None,
            time_obs=None,
            ):
        p_out = self.sky_sb_to_pwr(det_array_name, det_s)
        for array_name in self.array_names:
            mask = (det_array_name == array_name)
            aplm = self._array_power_loading_models[array_name]
            if self.atm_model_name is None:
                # atm is disabled
                pass
            elif self.atm_model_name == 'toast':
                p_atm = self._get_toast_P(
                    array_name=array_name,
                    det_az=det_az[mask],
                    det_alt=det_alt[mask],
                    time_obs=time_obs,
                    )
                # this power is only for the atm so we still need
                # to add the telescope warm components and system efficiency
                # note in this case aplm does not have the am_qxx models
                # enabled.
                p_tel, p_noise = aplm.evaluate_tod(
                    det_alt=det_alt[mask],
                    f_smp=f_smp,
                    random_seed=noise_seed,
                    return_realized_noise=True,
                    )
                p_out[mask] += p_atm + p_tel + p_noise
            else:
                # use the ToltecArrayPowerLoadingModel atm
                p, p_noise = aplm.evaluate_tod(
                    det_alt=det_alt[mask],
                    f_smp=f_smp,
                    random_seed=noise_seed,
                    return_realized_noise=True,
                    )
                p_out[mask] += (p + p_noise)
        return p_out

    def __str__(self):
        return (
            f'{self.__class__.__name__}(atm_model_name={self.atm_model_name})')


@add_schema
@dataclass
class ToastAtmConfig(object):
    """The config class for TOAST atm model."""
    median_weather: bool = field(
        default=True,
        metadata={
            'description': 'use median weather information'
        }
    )
    lmin_center: u.Quantity = field(
        default=0.01 << u.meter,
        metadata={
            'description': 'The lmin_center value',
            'schema': PhysicalTypeSchema('length')
            }
        )
    lmin_sigma: u.Quantity = field(
        default=0.001 << u.meter,
        metadata={
            'description': 'The lmin_sigma value',
            'schema': PhysicalTypeSchema('length')
            }
        )
    lmax_center: u.Quantity = field(
        default=10.0 << u.meter,
        metadata={
            'description': 'The lmax_center value',
            'schema': PhysicalTypeSchema('length')
            }
        )
    lmax_sigma: u.Quantity = field(
        default=10.0 << u.meter,
        metadata={
            'description': 'The lmax_sigma value',
            'schema': PhysicalTypeSchema('length')
            }
        )
    z0_center: u.Quantity = field(
        default=2000.0 << u.meter,
        metadata={
            'description': 'The z0_center value',
            'schema': PhysicalTypeSchema('length')
            }
        )
    z0_sigma: u.Quantity = field(
        default=0.0 << u.meter,
        metadata={
            'description': 'The z0_sigma value',
            'schema': PhysicalTypeSchema('length')
            }
        )
    zatm: u.Quantity = field(
        default=40000.0 << u.meter,
        metadata={
            'description': 'The zatm value',
            'schema': PhysicalTypeSchema('length')
            }
        )
    zmax: u.Quantity = field(
        default=2000.0 << u.meter,
        metadata={
            'description': 'The zmax value',
            'schema': PhysicalTypeSchema('length')
            }
        )
    w_sigma: u.Quantity = field(
        default= 0 << (u.km / u.second),
        metadata = {
            'description': 'w_sigma value: (w_center is set by the weather)'
        }
    )
    wdir_sigma: u.Quantity = field(
        default= 0 << u.radian,
        metadata = {
            'description': 'wdir_sigma value: (wdir_center is set by the weather)'
        }
    )
    T0_sigma: u.Quantity = field(
        default=10 * u.Kelvin, 
        metadata = {
            'description': 'T0_sigma value: (T0_center is set by the weather)'
        }
    )
    rmin: u.Quantity = field(
        default=0 << u.meter,
        metadata={
            'description': 'The rmin value'
        }
    )
    rmax: u.Quantity = field(
        default=100 << u.meter,
        metadata={
            'description': 'The rmin value'
        }
    )
    scale: np.float64 = field(
        default=10.0,
        metadata = {
            'description': 'scale value'
        }
    )
    xstep: u.Quantity = field(
        default=5 << u.meter,
        metadata={
            'description': 'The xstep value'
        }
    )
    ystep: u.Quantity = field(
        default=5 << u.meter,
        metadata={
            'description': 'The ystep value'
        }
    )
    zstep: u.Quantity = field(
        default=5 << u.meter,
        metadata={
            'description': 'The zstep value'
        }
    )
    nelem_sim_max: int = field(
        default=20000,
        metadata={
            'description': 'The nelem_sim_max value',
            }
        )
    key1: int = field(
        default=0,
        metadata={
            'description': 'key1 randomization',
            }
        )
    key2: int = field(
        default=0,
        metadata={
            'description': 'key2 randomization',
            }
        )

    class Meta:
        schema = {
            'ignore_extra_keys': False,
            'description': 'The parameters related to TOAST atm model.'
            }


class ToastAtmEvaluator(object):
    """A helper class to work with the Toast Atm model class."""

    def __init__(self, cache_dir=None, params=None):
        self._cache_dir = cache_dir
        if params is None:
            params = ToastAtmConfig()
        self._params = params
        self._toast_atm_simu = None

    @contextmanager
    def setup(self, t0, t_grid, sky_bbox_altaz, alt_grid):
        """A context for TOAST atm calculation."""
        # initialize the toast atm model
        # create the ToastAtmosphereSimulation instance here with
        # self._params and the sky bbox, and compute the atm slabs
        from . import toast_atm

        self.logger = get_logger()
        init_kwargs = {
            't0': t0,
            'tmin': t0.unix,
            'tmax': (t0 + t_grid[-1]).unix,
            'azmin': sky_bbox_altaz.w,
            'azmax': sky_bbox_altaz.e,
            'elmin': sky_bbox_altaz.s,
            'elmax': sky_bbox_altaz.n,
            'cachedir': self._cache_dir
            }
        self.logger.debug(
            f"init toast atm simulation with:\n{pformat_yaml(init_kwargs)}"
            )
        toast_atm_simu = self._toast_atm_simu = \
            toast_atm.ToastAtmosphereSimulation(**init_kwargs)
        # here we can pass the atm params to toast for generating the slabs
        setup_params = self._params

        self.logger.debug(
            f"setup toast atm simulation slabs with params:\n"
            f"{pformat_yaml(setup_params)}")
        toast_atm_simu.generate_simulation(self._params.to_dict())
        yield
        # clean up the context
        self._toast_atm_simu = None

    def calc_toast_atm_pwr_for_array(
            self, array_name, det_az, det_alt, time_obs_unix):
        toast_atm_simu = self._toast_atm_simu
        if toast_atm_simu is None:
            raise RuntimeError(
                "The toast atm simulator is not setup.")

        # to contain each individual slab
        atm_pW_additive = list()

        # ravels the 2D array into one 1D array as opposed
        # to iterating over all detectors
        # this is now directly impacted by the chunk size
        # this also covers the 1d case because ravel is a no-op.
        original_shape = det_alt.shape
        det_alt_observe = det_alt.ravel()
        det_az_observe = det_az.ravel()
        time_obs_unix_observe = np.tile(
            time_obs_unix, original_shape[0])  # time steps * no of detectors

        # iterate through all generated slabs
        for slab_id, atm_slab in toast_atm_simu.atm_slabs.items():
            self.logger.debug(
                f"integrating {array_name=} "
                f"({det_az_observe.size} discrete steps) on {slab_id=}")

            # returns atmospheric brightness temperature (Kelvin)
            atmtod = np.zeros_like(time_obs_unix_observe)
            # print(atmtod.size, det_az.to(u.radian).value.size,
            #       det_alt.to(u.radian).value.size, time_obs_unix.size)
            # print(det_az, det_alt)
            err = atm_slab.observe(
                times=time_obs_unix_observe,
                az=det_az_observe.to_value(u.radian),
                el=det_alt_observe.to_value(u.radian),
                tod=atmtod,
                fixed_r=0,
            )
            if err != 0:
                self.logger.error(f"toast slab observation failed {err=}")
                raise RuntimeError("toast slab observation failed")
            self.logger.debug('toast slab observation observation success')

            absorption_det = toast_atm_simu.absorption[array_name]
            loading_det = toast_atm_simu.loading[array_name]
            atm_gain = 1e-3  # this value is used to bring down the bandpass

            # calibrate the atmopsheric fluctuations to appropriate bandpass
            atmtod *= atm_gain * absorption_det

            # add the elevation-dependent atmospheric loading component
            atmtod += loading_det / np.sin(det_alt_observe.to_value(u.radian))

            atmtod *= 5e-2  # bring it down again

            # convert from antenna temperature (Kelvin) to MJy/sr
            # conversion_equiv = u.brightness_temperature(
            #     info_single['wl_center'])
            # atm_Mjy_sr = (atmtod * u.Kelvin).to_value(
            #     u.MJy / u.sr, equivalencies=conversion_equiv),

            # convert from antenna temperature (Kelvin) to pW
            pb_width = toltec_info[array_name]['passband']
            atm_pW = ((atmtod << u.Kelvin).to(
                u.J, equivalencies=u.temperature_energy()) * pb_width).to(u.pW)
            atm_pW_additive.append(atm_pW)

        # sum over all slabs
        result = np.sum(atm_pW_additive, axis=0) << u.pW

        # reshape back if that is required
        if original_shape != result.shape:
            result = result.reshape(original_shape)
        return result
