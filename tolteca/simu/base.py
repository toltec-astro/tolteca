#! /usr/bin/env python

import numpy as np
import inspect
from pathlib import Path
# import functools

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, ICRS, AltAz, Angle
from astropy.coordinates.baseframe import frame_transform_graph
from astropy.modeling import Model, Parameter
import astropy.units as u
from astropy.modeling import models
# from astropy.table import Table
from astropy.io import fits
from astropy.table import Table, QTable

from gwcs import coordinate_frames as cf
from scipy import interpolate

from tollan.utils.log import get_logger, timeit
from tollan.utils import getobj
from tollan.utils.fmt import pformat_yaml
from tollan.utils.namespace import NamespaceMixin
from tollan.utils.registry import Registry


def _get_skyoffset_frame(c):
    """This function creates a skyoffset_frame and ensures
    the cached origin frame attribute is the correct instance.
    """
    frame = c.skyoffset_frame()
    frame_transform_graph._cached_frame_attributes['origin'] = \
        frame.frame_attributes['origin']
    return frame


class _Model(Model, NamespaceMixin):

    _namespace_type_key = 'model'

    @classmethod
    def _namespace_from_dict_op(cls, d):
        # we resolve the model here so that we can allow
        # one use only the model class name to specify a model class.
        if cls._namespace_type_key not in d:
            raise ValueError(
                    f'unable to load model: '
                    f'missing required key "{cls._namespace_type_key}"')
        model_cls = cls._resolve_model_cls(d[cls._namespace_type_key])
        return dict(d, **{cls._namespace_type_key: model_cls})

    @staticmethod
    def _resolve_model_cls(arg):
        """Return a template class specified by `arg`.

        If `arg` is string, it is resolved using `tollan.utils.getobj`.
        """
        logger = get_logger()

        _arg = arg  # for logging
        if isinstance(arg, str):
            arg = getobj(arg)
        # check if _resolve_template_cls attribute is present
        if inspect.ismodule(arg):
            raise ValueError(f"cannot resolve model class from {arg}")
        if not isinstance(arg, Model):
            raise ValueError(f"cannot resolve model class from {arg}")
        model_cls = arg
        logger.debug(
                f"resolved model {_arg} as {model_cls}")
        return model_cls


# class SkyOffsetModel(_Model):
#     """This computes the relative offsets between two sets of coordinates.

#     """
#     n_inputs = 2
#     n_outputs = 2

#     def __init__(self, ref_frame=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._t0 = t0
#         self._target = target
#         self._ref_frame = ref_frame

#     def evaluate(self, x, y):
#         return NotImplemented

#     def evaluate_at(self, ref_coord, *args):
#         """Returns the mapping pattern as evaluated at given coordinates.
#         """
#         frame = _get_skyoffset_frame(ref_coord)
#         return coord.SkyCoord(*self(*args), frame=frame).transform_to(
#                 ref_coord.frame)

#     # TODO these three method seems to be better live in a wrapper
#     # model rather than this model
#     @property
#     def t0(self):
#         return self._t0

#     @property
#     def target(self):
#         return self._target

#     input_frame = cf.CelestialFrame(
#             name='icrs',
#             reference_frame=coord.ICRS(),
#             unit=(u.deg, u.deg)
#             )

#     def __init__(self, *args, **kwargs):
#         inputs = kwargs.pop('inputs', self.input_frame.axes_names)
#         super().__init__(*args, **kwargs)
#         # self.inputs =


class SourceModel(_Model):
    """Base class for models that compute the optical signal.
    """

    _subclasses = Registry.create()

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls._subclasses.register(cls, cls)

    def __init__(self, *args, **kwargs):
        inputs = kwargs.pop('inputs', self.input_frame.axes_names)
        outputs = ('S', )
        super().__init__(*args, **kwargs)
        self.inputs = inputs
        self.outputs = outputs

    @property
    def data(self):
        return self._data


class SourceImageModel(SourceModel):
    """
    A model given by 2-d images.
    """

    logger = get_logger()

    n_inputs = 2
    n_outputs = 1
    input_frame = cf.CelestialFrame(
            name='icrs',
            reference_frame=ICRS(),
            unit=(u.deg, u.deg)
            )

    def __init__(self, data=None, grouping=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = data
        self._grouping = grouping

    @classmethod
    def from_fits(cls, filepath, extname_map=None, **kwargs):
        """
        Parameters
        ----------
        filepath : str, pathlib.Path
            The path to the FITS file.

        extname_map : dict, optional
            Specify the extensions to included in the returned data.
            If None, an educated guess will be made.
        """
        filepath = Path(filepath)
        hdulist = fits.open(filepath)

        extname_hdu = dict()
        for i, hdu in enumerate(hdulist):
            cls.logger.debug('HDU {}: {}'.format(
                i, hdu.header.tostring(sep='\n')
                ))
            label = hdu.header.get('EXTNAME', i)
            extname_hdu[label] = hdu

        if extname_map is None:
            extname_map = {
                    k: k for k in extname_hdu.keys()
                    }
        cls.logger.debug(f"use extname_map: {pformat_yaml(extname_map)}")

        data = dict()
        for k, extname in extname_map.items():
            data[k] = extname_hdu[extname]
        cls.logger.debug(f'data keys: {list(data.keys())}')
        return cls(data=data, name=filepath.as_posix(), **kwargs)

    def evaluate_tod(self, tbl, lon, lat):
        """Extract flux for given array property table.

        Parameters
        ==========
        tbl : astropy.table.Table
            The array property table for mapping the data keys.

        """
        data = self._data
        grouping = self._grouping
        if grouping not in tbl.colnames:
            raise ValueError(
                    "unable to map data keys to array property table.")
        # make masks
        data_groups = []
        for g in np.unique(tbl[grouping]):
            if g not in data:
                self.logger.debug(f"group {g} not found in data")
                continue
            d = self._data[g]
            m = tbl[grouping] == g
            data_groups.append([d, m])
            self.logger.debug(f"group {g}: {m.sum()}/{len(m)}")
        self.logger.debug(f"evaluate {len(data_groups)} groups")

        s_out = np.zeros(lon.shape) << u.MJy / u.sr
        lon = lon.to_value(u.deg)
        lat = lat.to_value(u.deg)
        for d, m in data_groups:
            wcsobj = WCS(d.header)
            s_out_unit = u.Unit(d.header.get('SIGUNIT', 'adu'))
            # check lon lat range
            # because here we check longitude ranges
            # we need to take into account wrapping issue
            lon_m = lon[m, :]
            lat_m = lat[m, :]
            # s_out_m = s_out[m, :]
            # w, e = np.min(lon_m), np.max(lon_m)
            # s, n = np.min(lat_m), np.max(lat_m)
            # check pixel range
            ny, nx = d.data.shape
            # lon lat range of pixel edges
            lon_e, lat_e = wcsobj.wcs_pix2world(
                    np.array([0, 0, nx - 1, nx - 1]),
                    np.array([0, ny - 1, 0, ny - 1]),
                    0)
            xx, yy = wcsobj.wcs_world2pix(lon_e, lat_e, 0)
            # fix potential wrapping issue by check at 360 and 180 wrapping
            lon_e = Angle(lon_e << u.deg).wrap_at(360. << u.deg).degree
            lon_e_180 = Angle(lon_e << u.deg).wrap_at(180. << u.deg).degree

            w_e, e_e = np.min(lon_e), np.max(lon_e)
            w_e_180, e_e_180 = np.min(lon_e_180), np.max(lon_e_180)
            s_e, n_e = np.min(lat_e), np.max(lat_e)
            # take the one with smaller size as the coordinte
            if (e_e_180 - w_e_180) < (e_e - w_e):
                # use wrapping at 180.d
                w_e = w_e_180
                e_e = e_e_180
                lon_m = Angle(lon_m << u.deg).wrap_at(180. << u.deg).degree
                self.logger.debug("re-wrapping coordinates at 180d")
            self.logger.debug(f"data bbox: w={w_e} e={e_e} s={s_e} n={n_e}")
            self.logger.debug(f'data shape: {d.data.shape}')

            # mask to include in range lon lat
            g = (
                    (lon_m > w_e) & (lon_m < e_e)
                    & (lat_m > s_e) & (lat_m < n_e)
                    )
            self.logger.debug(f"data mask {g.sum()}/{lon_m.size}")
            if g.sum() == 0:
                continue
            # convert all lon lat to x y
            x_g, y_g = wcsobj.wcs_world2pix(lon_m[g], lat_m[g], 0)
            ii = np.rint(y_g).astype(int)
            jj = np.rint(x_g).astype(int)
            self.logger.debug(
                    f"pixel range: [{ii.min()}, {ii.max()}] "
                    f"[{jj.min()}, {jj.max()}]")
            # check ii and jj for valid pixel range
            gp = (ii >= 0) & (ii < ny) & (jj >= 0) & (jj < nx)
            # update g to include only valid pixels
            g[g] = gp
            # convert all lon lat to x y
            x_g, y_g = wcsobj.wcs_world2pix(lon_m[g], lat_m[g], 0)
            ii = np.rint(y_g).astype(int)
            jj = np.rint(x_g).astype(int)
            self.logger.debug(
                    f"pixel range updated: [{ii.min()}, {ii.max()}] "
                    f"[{jj.min()}, {jj.max()}]")
            # take values in data
            # ii, jj = np.meshgrid(
            #         np.rint(y_g).astype(int),
            #         np.rint(x_g).astype(int), indexing='ij')

            # x, y = w.wcs_world2pix(
            #         lon[m, :].ravel(), lat[m, :].ravel(), 0)
            # g = (x >= 0) & (y < imshape[0]) & (jj >=0) & (jj < imshape[1])
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            # ax.scatter(ii, jj, c=d.data[ii, jj])
            # ax.set_aspect('equal')
            # plt.show()
            ig, jg = np.where(g)
            s_out[np.flatnonzero(m)[ig], jg] = d.data[ii, jj] << s_out_unit
        self.logger.debug(
                f'signal range: [{s_out.min()}, {s_out.max()}]')
        return s_out

    def evaluate(self, lon, lat):
        pass


class SourceCatalogModel(SourceModel):
    """
    A model with point sources.
    """
    logger = get_logger()

    n_inputs = 2
    n_outputs = 1
    input_frame = cf.CelestialFrame(
            name='icrs',
            reference_frame=ICRS(),
            unit=(u.deg, u.deg)
            )

    def __init__(self, pos, data, grouping=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos = pos
        self._data = data
        self._grouping = grouping

    def make_image_model(self, beam_models, pixscale):

        pixscale = u.pixel_scale(pixscale)
        delta_pix = (1. << u.pix).to(u.arcsec, equivalencies=pixscale)

        # convert beam_models to pixel unit
        if next(iter(beam_models.values())).x_mean.unit.is_equivalent(u.pix):
            m_beams = beam_models
        else:
            m_beams = dict()
            for k, m in beam_models.items():
                m_beams[k] = m.__class__(
                        amplitude=m.amplitude.quantity,
                        x_mean=m.x_mean.quantity.to_value(
                            u.pix, equivalencies=pixscale),
                        y_mean=m.y_mean.quantity.to_value(
                            u.pix, equivalencies=pixscale),
                        x_stddev=m.x_stddev.quantity.to_value(
                            u.pix, equivalencies=pixscale),
                        y_stddev=m.y_stddev.quantity.to_value(
                            u.pix, equivalencies=pixscale),
                        )
        # use the first pos as the reference
        ref_coord = self.pos[0]

        wcsobj = WCS(naxis=2)
        wcsobj.wcs.crpix = [1.5, 1.5]
        wcsobj.wcs.cdelt = np.array([
            -delta_pix.to_value(u.deg),
            delta_pix.to_value(u.deg),
            ])
        wcsobj.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcsobj.wcs.crval = [ref_coord.ra.degree, ref_coord.dec.degree]

        # compute the pixel range
        x, y = wcsobj.wcs_world2pix(self.pos.ra, self.pos.dec, 0)
        l, r = np.min(x), np.max(x)
        b, t = np.min(y), np.max(y)
        w, h = r - l, t - b
        # size of the square bbox, with added padding on the edge
        s = int(np.ceil(np.max([w, h]) + 10 * np.max(
                [m.x_fwhm for m in m_beams.values()])))
        self.logger.debug(f'source image size: {s}')
        # figure out center coord
        c_ra, c_dec = wcsobj.wcs_pix2world((l + r) / 2, (b + t) / 2, 0)
        # re-center the wcs to pixel center
        wcsobj.wcs.crpix = [s / 2 + 1, s / 2 + 1]
        wcsobj.wcs.crval = c_ra, c_dec
        header = wcsobj.to_header()
        # compute the pixel positions
        x, y = wcsobj.wcs_world2pix(self.pos.ra, self.pos.dec, 0)
        assert ((x < 0) | (x > s)).sum() == 0
        assert ((y < 0) | (y > s)).sum() == 0
        # get the pixel
        # render the image
        hdus = dict()
        for k, m in m_beams.items():
            amp = (self.data[k] * m.amplitude).to(u.MJy / u.sr)
            img = np.zeros((s, s), dtype=float) << u.MJy / u.sr
            m = m.copy()
            for xx, yy, aa in zip(x, y, amp):
                m.amplitude = aa
                m.x_mean = xx
                m.y_mean = yy
                m.render(img)
            hdu = fits.ImageHDU(img.to_value(u.MJy / u.sr), header=header)
            hdu.header['SIGUNIT'] = 'MJy / sr'
            hdus[k] = hdu
            self.logger.debug('HDU {}: {}'.format(
                k, hdu.header.tostring(sep='\n')
                ))
        return SourceImageModel(
                data=hdus, grouping=self._grouping,
                name=self.name,
                )

    @property
    def pos(self):
        return self._pos

    @classmethod
    def from_file(cls, filepath, **kwargs):
        """Create instance from file path.

        Parameters
        ----------
        filepath : str, `pathlib.Path`
            The path to the catalog file.

        **kwargs
            Arguments passed to `from_table`.
        """
        tbl = Table.read(filepath, format='ascii')
        # use this to keep track of the original table filepath
        tbl.meta['_source'] = Path(filepath)
        return cls.from_table(tbl, **kwargs)

    @classmethod
    def from_table(cls, tbl, colname_map=None, **kwargs):
        """
        Parameters
        ----------
        tbl : `astropy.table.Table`
            The table containing the source catalog.

        colname_map : dict, optional
            Specify the column names to included in the returned data.
            If None, an educated guess will be made.
        """
        # TODO: add code to guess colname map
        if colname_map is None:
            colname_map = dict()
        colname_map = dict(**colname_map)
        cls.logger.debug(f"use colname_map: {pformat_yaml(colname_map)}")

        def getcol_quantity(tbl, colname, unit):
            col = tbl[colname]
            if col.unit is None and unit is not None:
                cls.logger.debug(f"assume unit {unit} for column {colname}")
                col.unit = unit
            if col.unit is None:
                return col
            return col.quantity

        # we use a qtable to hold the data internally
        data = QTable()
        for k, c in colname_map.items():
            if c.startswith('flux'):
                unit = u.mJy
            elif c in ['ra', 'dec']:
                unit = u.deg
            else:
                unit = None
            data[k] = getcol_quantity(tbl, c, unit)
        cls.logger.debug(f'data keys: {data.colnames}')

        # figure out the position
        # TODO: add handling of different coordinate system in input
        pos = SkyCoord(
                data['ra'], data['dec'],
                frame=ICRS()).transform_to(cls.input_frame.reference_frame)

        filepath = tbl.meta.get('_source', None)
        if filepath is not None:
            name = filepath.as_posix()
        else:
            name = None
        return cls(pos=pos, data=data, name=name, **kwargs)

    def evaluate(self, lon, lat):
        coo = self.input_frame.coordinates(lon, lat)
        sep = self._source_pos.separation(coo)
        return self._psfmodel(sep) * self._source_flux


class ProjModel(_Model):
    """Base class for models that transform the detector locations.
    """

    def __init__(self, t0=None, target=None, *args, **kwargs):
        inputs = kwargs.pop('inputs', self.input_frame.axes_names)
        outputs = kwargs.pop('outputs', self.output_frame.axes_names)
        kwargs.setdefault('name', self._name)
        super().__init__(*args, **kwargs)
        self.inputs = inputs
        self.outputs = outputs
        self._t0 = t0
        self._target = target

    def mpl_axes_params(self):
        return dict(aspect='equal')


class SkyMapModel(_Model):
    """A model that describes mapping patterns on the sky.

    It computes the sky coordinates as a function of the time.
    """

    n_inputs = 1
    n_outputs = 2

    def __init__(self, t0=None, target=None, ref_frame=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._t0 = t0
        self._target = target
        self._ref_frame = ref_frame

    def evaluate(self, x, y):
        return NotImplemented

    @timeit
    def evaluate_at(self, ref_coord, *args):
        """Returns the mapping pattern as evaluated at given coordinates.
        """
        frame = _get_skyoffset_frame(ref_coord)
        return SkyCoord(*self(*args), frame=frame).transform_to(
                ref_coord.frame)

    # TODO these three method seems to be better live in a wrapper
    # model rather than this model
    @property
    def t0(self):
        return self._t0

    @property
    def target(self):
        return self._target

    @property
    def ref_frame(self):
        return self._ref_frame


class RasterScanModelMeta(SkyMapModel.__class__):
    """A meta class that defines a raster scan pattern.

    This is implemented as a meta class so that we can reuse it
    in any map model of any coordinate frame.
    """

    @staticmethod
    def _evaluate(
            inst, t, length, space, n_scans, rot, speed, t_turnover,
            return_holdflag_only=False):
        """This computes a raster patten around the origin.

        This assumes a circular turn over trajectory where the
        speed of the turn over is implicitly controlled by `t_turnover`.
        """
        t_per_scan = length / speed

        holdflag = np.zeros(t.shape, dtype=bool)
        if n_scans == 1:
            # TODO this is ugly, will revisit later
            if return_holdflag_only:
                return holdflag
            # have to make a special case here
            x = (t / t_per_scan - 0.5) * length
            y = np.zeros(t.shape) << length.unit
        else:
            n_spaces = n_scans - 1

            # bbox_width = length
            # bbox_height = space * n_spaces
            # # (x0, y0, w, h)
            # bbox = (
            #         -bbox_width / 2., -bbox_height / 2.,
            #         bbox_width, bbox_height)
            t_per_scan = length / speed
            ratio_scan_to_si = (
                    t_per_scan / (t_turnover + t_per_scan))
            ratio_scan_to_turnover = (t_per_scan / t_turnover)

            # scan index
            _si = (t / (t_turnover + t_per_scan))
            si = _si.astype(int)
            si_frac = _si - si

            # get scan and turnover part
            scan_frac = np.empty_like(si_frac)
            turnover_frac = np.empty_like(si_frac)

            turnover = si_frac > ratio_scan_to_si
            if return_holdflag_only:
                holdflag[turnover] = True
                return holdflag
            scan_frac[turnover] = 1.
            scan_frac[~turnover] = si_frac[~turnover] / ratio_scan_to_si
            turnover_frac[turnover] = si_frac[turnover] - (
                    1. - si_frac[turnover]) * ratio_scan_to_turnover
            turnover_frac[~turnover] = 0.

            x = (scan_frac - 0.5) * length
            y = (si / n_spaces - 0.5) * n_spaces * space

            # turnover part
            radius_t = space / 2
            theta_t = turnover_frac[turnover] * np.pi * u.rad
            dy = radius_t * (1 - np.cos(theta_t))
            dx = radius_t * np.sin(theta_t)
            x[turnover] = x[turnover] + dx
            y[turnover] = y[turnover] + dy
            # make continuous
            x = x * (-1) ** si

        # apply rotation
        m_rot = models.AffineTransformation2D(
            models.Rotation2D._compute_matrix(
                angle=rot.to_value('rad')) * inst.frame_unit,
            translation=(0., 0.) * inst.frame_unit)
        xx, yy = m_rot(x, y)
        return xx, yy

    def __new__(meta, name, bases, attrs):
        frame = attrs['frame']
        frame_unit = frame.unit[0]

        attrs.update(dict(
            frame_unit=frame_unit,
            length=Parameter(default=10., unit=frame_unit),
            space=Parameter(default=1., unit=frame_unit),
            n_scans=Parameter(default=10., unit=u.dimensionless_unscaled),
            rot=Parameter(default=0., unit=u.deg),
            speed=Parameter(default=1., unit=frame_unit / u.s),
            # accel=Parameter(default=1., unit=cls.frame_unit / u.s ** 2),
            t_turnover=Parameter(default=1., unit=u.s),
            pattern='raster',
                ))

        def get_total_time(self):
            return (self.length / self.speed * self.n_scans
                    + self.t_turnover * (self.n_scans - 1.)).to(u.s)

        attrs['get_total_time'] = get_total_time

        @timeit(name)
        def evaluate(self, t, *args, **kwargs):
            t = np.asarray(t) * t.unit
            return meta._evaluate(self, t, *args, **kwargs)

        attrs['evaluate'] = evaluate
        # TODO refactor this part
        attrs['evaluate_holdflag'] = lambda self, t: evaluate(
                self, t, self.length, self.space, self.n_scans, self.rot,
                self.speed, self.t_turnover, return_holdflag_only=True
                )
        return super().__new__(meta, name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        inst = super().__call__(*args, **kwargs)
        inst.inputs = ('t', )
        inst.outputs = cls.frame.axes_names
        return inst


class LissajousModelMeta(SkyMapModel.__class__):
    """A meta class that defines a Lissajous scan pattern.

    This is implemented as a meta class so that we can reuse it
    in any map model of any coordinate frame.
    """

    def __new__(meta, name, bases, attrs):
        frame = attrs['frame']
        frame_unit = frame.unit[0]

        attrs.update(dict(
            frame_unit=frame_unit,
            x_length=Parameter(default=10., unit=frame_unit),
            y_length=Parameter(default=10., unit=frame_unit),
            x_omega=Parameter(default=1. * u.rad / u.s),
            y_omega=Parameter(default=1. * u.rad / u.s),
            delta=Parameter(default=0., unit=u.rad),
            rot=Parameter(default=0., unit=u.deg),
            pattern='lissajous',
                ))

        def get_total_time(self):
            t_x = 2 * np.pi * u.rad / self.x_omega
            t_y = 2 * np.pi * u.rad / self.y_omega
            r = (t_y / t_x).to_value(u.dimensionless_unscaled)
            s = 100
            r = np.lcm(int(r * s), s) / s
            return (t_x * r).to(u.s)

        attrs['get_total_time'] = get_total_time

        @timeit(name)
        def evaluate(
                self, t, x_length, y_length, x_omega, y_omega, delta, rot):
            """This computes a lissajous pattern around the origin.

            """
            t = np.asarray(t) * t.unit

            x = x_length * 0.5 * np.sin(x_omega * t + delta)
            y = y_length * 0.5 * np.sin(y_omega * t)

            m_rot = models.AffineTransformation2D(
                models.Rotation2D._compute_matrix(
                    angle=rot.to_value('rad')) * self.frame_unit,
                translation=(0., 0.) * self.frame_unit)
            xx, yy = m_rot(x, y)
            return xx, yy

        attrs['evaluate'] = evaluate
        attrs['evaluate_holdflag'] = \
            lambda self, t: np.zeros(t.shape, dtype=bool)
        return super().__new__(meta, name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        inst = super().__call__(*args, **kwargs)
        inst.inputs = ('t', )
        inst.outputs = cls.frame.axes_names
        return inst


class DoubleLissajousModelMeta(SkyMapModel.__class__):
    """A meta class that defines a Double Lissajous scan pattern.

    """

    @staticmethod
    def _evaluate(
            inst, t,
            x_length_0, y_length_0, x_omega_0, y_omega_0, delta_0,
            x_length_1, y_length_1, x_omega_1, y_omega_1, delta_1,
            delta, rot):
        """This computes a double lissajous pattern around the origin.

        """
        x_0 = x_length_0 * 0.5 * np.sin(x_omega_0 * t + delta + delta_0)
        y_0 = y_length_0 * 0.5 * np.sin(y_omega_0 * t + delta)
        x_1 = x_length_1 * 0.5 * np.sin(x_omega_1 * t + delta_1)
        y_1 = y_length_1 * 0.5 * np.sin(y_omega_1 * t)

        x = x_0 + x_1
        y = y_0 + y_1

        m_rot = models.AffineTransformation2D(
            models.Rotation2D._compute_matrix(
                angle=rot.to_value('rad')) * inst.frame_unit,
            translation=(0., 0.) * inst.frame_unit)
        xx, yy = m_rot(x, y)
        return xx, yy

    def __new__(meta, name, bases, attrs):
        frame = attrs['frame']
        frame_unit = frame.unit[0]

        attrs.update(dict(
            frame_unit=frame_unit,
            x_length_0=Parameter(default=10., unit=frame_unit),
            y_length_0=Parameter(default=10., unit=frame_unit),
            x_omega_0=Parameter(default=1. * u.rad / u.s),
            y_omega_0=Parameter(default=1. * u.rad / u.s),
            delta_0=Parameter(default=0., unit=u.rad),
            x_length_1=Parameter(default=5., unit=frame_unit),
            y_length_1=Parameter(default=5., unit=frame_unit),
            x_omega_1=Parameter(default=1. * u.rad / u.s),
            y_omega_1=Parameter(default=1. * u.rad / u.s),
            delta_1=Parameter(default=0., unit=u.rad),
            delta=Parameter(default=0., unit=u.rad),
            rot=Parameter(default=0., unit=u.deg),
            pattern='double_lissajous',
                ))

        def get_total_time(self):
            # make the total time the longer one among the two
            def _get_total_time(x_omega, y_omega):
                t_x = 2 * np.pi * u.rad / x_omega
                t_y = 2 * np.pi * u.rad / y_omega
                r = (t_y / t_x).to_value(u.dimensionless_unscaled)
                s = 100
                r = np.lcm(int(r * s), s) / s
                return (t_x * r).to(u.s)
            t0 = _get_total_time(self.x_omega_0, self.y_omega_0)
            t1 = _get_total_time(self.x_omega_1, self.y_omega_1)
            return t0 if t0 > t1 else t1

        attrs['get_total_time'] = get_total_time

        @timeit(name)
        def evaluate(self, t, *args, **kwargs):
            t = np.asarray(t) * t.unit
            return meta._evaluate(self, t, *args, **kwargs)

        attrs['evaluate'] = evaluate
        attrs['evaluate_holdflag'] = \
            lambda self, t: np.zeros(t.shape, dtype=bool)
        return super().__new__(meta, name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        inst = super().__call__(*args, **kwargs)
        inst.inputs = ('t', )
        inst.outputs = cls.frame.axes_names
        return inst


class RastajousModelMeta(SkyMapModel.__class__):
    """A meta class that defines a Rastajous scan pattern.

    """

    def __new__(meta, name, bases, attrs):
        frame = attrs['frame']
        frame_unit = frame.unit[0]

        attrs.update(dict(
            frame_unit=frame_unit,
            length=Parameter(default=10., unit=frame_unit),
            space=Parameter(default=1., unit=frame_unit),
            n_scans=Parameter(default=10., unit=u.dimensionless_unscaled),
            rot=Parameter(default=0., unit=u.deg),
            speed=Parameter(default=1., unit=frame_unit / u.s),
            t_turnover=Parameter(default=1., unit=u.s),
            x_length_0=Parameter(default=10., unit=frame_unit),
            y_length_0=Parameter(default=10., unit=frame_unit),
            x_omega_0=Parameter(default=1. * u.rad / u.s),
            y_omega_0=Parameter(default=1. * u.rad / u.s),
            delta_0=Parameter(default=0., unit=u.rad),
            x_length_1=Parameter(default=5., unit=frame_unit),
            y_length_1=Parameter(default=5., unit=frame_unit),
            x_omega_1=Parameter(default=1. * u.rad / u.s),
            y_omega_1=Parameter(default=1. * u.rad / u.s),
            delta_1=Parameter(default=0., unit=u.rad),
            delta=Parameter(default=0., unit=u.rad),
            pattern='rastajous',
                ))

        def get_total_time(self):
            # make the total time based on the raster
            return (self.length / self.speed * self.n_scans
                    + self.t_turnover * (self.n_scans - 1.)).to(u.s)

        attrs['get_total_time'] = get_total_time

        @timeit(name)
        def evaluate(
                self, t,
                length, space, n_scans, rot, speed, t_turnover,
                x_length_0, y_length_0, x_omega_0, y_omega_0, delta_0,
                x_length_1, y_length_1, x_omega_1, y_omega_1, delta_1,
                delta):
            """This computes a rastajous pattern around the origin.

            """
            t = np.asarray(t) * t.unit

            x_r, y_r = RasterScanModelMeta._evaluate(
                    inst=self, t=t, length=length, space=space,
                    n_scans=n_scans, rot=0. << u.deg,
                    speed=speed, t_turnover=t_turnover,
                    return_holdflag_only=False)

            x_l, y_l = DoubleLissajousModelMeta._evaluate(
                    inst=self, t=t,
                    x_length_0=x_length_0, y_length_0=y_length_0,
                    x_omega_0=x_omega_0, y_omega_0=y_omega_0, delta_0=delta_0,
                    x_length_1=x_length_1, y_length_1=y_length_1,
                    x_omega_1=x_omega_1, y_omega_1=y_omega_1, delta_1=delta_1,
                    delta=delta, rot=0. << u.deg
                    )
            x = x_r + x_l
            y = y_r + y_l
            m_rot = models.AffineTransformation2D(
                models.Rotation2D._compute_matrix(
                    angle=rot.to_value('rad')) * self.frame_unit,
                translation=(0., 0.) * self.frame_unit)
            xx, yy = m_rot(x, y)
            return xx, yy

        attrs['evaluate'] = evaluate
        # TODO use the raster hold flag
        attrs['evaluate_holdflag'] = \
            lambda self, t: RasterScanModelMeta._evaluate(
                self, t, self.length, self.space, self.n_scans, self.rot,
                self.speed, self.t_turnover, return_holdflag_only=True
                )
        return super().__new__(meta, name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        inst = super().__call__(*args, **kwargs)
        inst.inputs = ('t', )
        inst.outputs = cls.frame.axes_names
        return inst


class TrajectoryModelMeta(SkyMapModel.__class__):
    """A meta class that defines a trajectory.

    This is implemented as a meta class so that we can reuse it
    in any map model of any coordinate frame.
    """

    def __new__(meta, name, bases, attrs):
        frame = attrs['frame']
        frame_unit = frame.unit[0]

        attrs.update(dict(
            frame_unit=frame_unit,
                ))

        def get_total_time(self):
            return self._time[-1]

        attrs['get_total_time'] = get_total_time

        lon_attr, lat_attr = {
                'icrs': ('_ra', '_dec'),
                'altaz': ('_az', '_alt'),
                }[frame.name]

        @property
        def _lon(self):
            return getattr(self, lon_attr)

        attrs['_lon_attr'] = lon_attr
        attrs['_lon'] = _lon

        @property
        def _lat(self):
            return getattr(self, lat_attr)

        attrs['_lat_attr'] = lat_attr
        attrs['_lat'] = _lat

        @timeit(name)
        def evaluate(
                self, t):
            """This computes the position based on interpolation.

            """
            return self._lon_interp(t), self._dec_interp(t)

        attrs['evaluate'] = evaluate
        return super().__new__(meta, name, bases, attrs)

    def __call__(
            cls, *args, **kwargs):
        data = dict()
        for attr in ('time', 'ra', 'dec', 'az', 'alt'):
            data[f'_{attr}'] = kwargs.pop(attr, None)
        inst = super().__call__(*args, **kwargs)
        inst.__dict__.update(data)
        inst.inputs = ('t', )
        inst.outputs = cls.frame.axes_names
        inst._lon_interp = interpolate.interp1d(inst._time, inst._lon)
        inst._lat_interp = interpolate.interp1d(inst._time, inst._lat)
        return inst


class SkyRasterScanModel(SkyMapModel, metaclass=RasterScanModelMeta):
    frame = cf.Frame2D(
            name='skyoffset', axes_names=('lon', 'lat'),
            unit=(u.deg, u.deg))


class SkyLissajousModel(SkyMapModel, metaclass=LissajousModelMeta):
    frame = cf.Frame2D(
            name='skyoffset', axes_names=('lon', 'lat'),
            unit=(u.deg, u.deg))


class SkyDoubleLissajousModel(SkyMapModel, metaclass=DoubleLissajousModelMeta):
    frame = cf.Frame2D(
            name='skyoffset', axes_names=('lon', 'lat'),
            unit=(u.deg, u.deg))


class SkyRastajousModel(SkyMapModel, metaclass=RastajousModelMeta):
    frame = cf.Frame2D(
            name='skyoffset', axes_names=('lon', 'lat'),
            unit=(u.deg, u.deg))


class SkyICRSTrajModel(SkyMapModel, metaclass=TrajectoryModelMeta):
    frame = cf.CelestialFrame(
            name='icrs',
            reference_frame=ICRS(),
            unit=(u.deg, u.deg)
            )


class SkyAltAzTrajModel(SkyMapModel, metaclass=TrajectoryModelMeta):
    frame = cf.CelestialFrame(
            name='altaz',
            reference_frame=AltAz(),
            unit=(u.deg, u.deg)
            )


def resolve_sky_map_ref_frame(ref_frame, observer=None, time_obs=None):
    """
    Return a frame with respect to which sky map offset model can be
    rendered.
    """
    if isinstance(ref_frame, str):
        # this is not public API so be careful for future changes.
        from astropy.coordinates.sky_coordinate_parsers import (
                _get_frame_class)
        ref_frame = _get_frame_class(ref_frame)
    if ref_frame is AltAz:
        return observer.altaz(time=time_obs)
    return ref_frame
