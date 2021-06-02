#! /usr/bin/env python

from contextlib import contextmanager
import numpy as np
from pytz import timezone
import functools
from scipy.interpolate import interp1d
from astroplan import Observer
import astropy.units as u
from astropy.modeling import models, Parameter, Model
from astropy.modeling.functional_models import GAUSSIAN_SIGMA_TO_FWHM
from astropy import coordinates as coord
from astropy.time import Time
from astropy.table import Column, QTable, Table
from astropy.wcs.utils import celestial_frame_to_wcs
from astropy.coordinates import SkyCoord
from astropy.coordinates.erfa_astrom import (
        erfa_astrom, ErfaAstromInterpolator)

from gwcs import coordinate_frames as cf

from kidsproc.kidsmodel import ReadoutGainWithLinTrend
from kidsproc.kidsmodel.simulator import KidsSimulator
from tollan.utils.log import timeit, get_logger

from ..base import (
        _Model,
        ProjModel, _get_skyoffset_frame,
        SourceImageModel, SourceCatalogModel)
from ..base import resolve_sky_map_ref_frame as _resolve_sky_map_ref_frame


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


class SiteInfo(object):
    name = 'LMT'
    name_long = 'Large Millimeter Telescope'
    location = coord.EarthLocation.from_geodetic(
            "-97d18m53s", '+18d59m06s', 4600 * u.m)
    timezone = timezone('America/Mexico_City')
    observer = Observer(
        name=name_long,
        location=location,
        timezone=timezone,
        )


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

    @staticmethod
    def _get_native_frame(mjd_obs):
        return SiteInfo.observer.altaz(time=Time(mjd_obs, format='mjd'))

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
    """A model that describes the beam shape.
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

    def __init__(self, **kwargs):
        beam_props = self.beam_props
        m_beams = dict()

        for array_name in beam_props['array_names']:
            x_fwhm = (
                    beam_props['x_fwhm_a1100'] *
                    beam_props[array_name]['wl_center'] /
                    beam_props['a1100']['wl_center'])
            y_fwhm = (
                    beam_props['y_fwhm_a1100'] *
                    beam_props[array_name]['wl_center'] /
                    beam_props['a1100']['wl_center'])
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

    @property
    def observer(self):
        return SiteInfo.observer

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
    def probe_context(self, fp=None):
        """Return a function that can be used to get IQ for given flux
        """
        tbl = self.table
        kidssim = self._kidssim
        readout = self._readout_model
        if fp is None:
            fp = kidssim._fr

        def evaluate(s):
            # convert to brightness temperature and
            # assuming a square pass band, we can get the power loading
            tbs = s.to(
                    u.K,
                    equivalencies=u.brightness_temperature(
                        self.table['wl_center'][:, np.newaxis]))
            pwrs = (
                    tbs.to(
                        u.J,
                        equivalencies=u.temperature_energy())
                    * self.table['passband'][:, np.newaxis]
                    ).to(u.pW)
            return kidssim.probe_p(
                    pwrs + tbl['background'][:, np.newaxis],
                    fp=fp, readout_model=readout)

        yield evaluate

    @timeit
    def resolve_sky_map_ref_frame(self, ref_frame, time_obs):
        return _resolve_sky_map_ref_frame(
                    ref_frame, observer=self.observer, time_obs=time_obs)

    @contextmanager
    def mapping_context(self, mapping, sources):
        """
        Return a function that can be used to get
        input flux at each detector for given time.

        Parameters
        ==========
        mapping : tolteca.simu.base.SkyMapModel

            The model that defines the on-the-fly mapping trajectory.

        sources : tolteca.simu.base.SourceModel

            The list of models that define the input signal and noise.
        """
        tbl = self.table
        x_t = tbl['x_t']
        y_t = tbl['y_t']

        ref_coord = mapping.target
        ref_frame = mapping.ref_frame
        t0 = mapping.t0

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
                        SiteInfo.observer.parallactic_angle(
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
                            print(src_pos.shape)
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
                            print(s.shape, w.shape)
                        s_additive.append(s)
                if len(s_additive) <= 0:
                    raise ValueError("no additive source found in source list")
                s = functools.reduce(np.sum, s_additive)

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
        # kids props
        for c, v in cls.kids_props.items():
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
        tbl['flxscale'] = (1. / tbl['responsivity']).quantity.value
        return QTable(tbl)


class KidsReadoutNoiseModel(_Model):
    """
    A model of the TolTEC KIDs readout noise.

    """
    pass
