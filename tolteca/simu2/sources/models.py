#!/usr/bin/env python

from astropy.io import fits
import astropy.units as u
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table, QTable
from astropy.modeling import models
from astropy.modeling.functional_models import GAUSSIAN_SIGMA_TO_FWHM

# from tollan.utils.fmt import pformat_yaml
from tollan.utils.log import get_logger
from tollan.utils import ensure_abspath

from .base import SurfaceBrightnessModel


__all__ = ['ImageSourceModel', 'CatalogSourceModel']


class ImageSourceModel(SurfaceBrightnessModel):
    """The class for simulator source from FITS image.

    Parameters
    ----------
    data : dict
        A dict of HDUs for the data. The dict keys are labels that
        expected as the first argument of the model evaluation.
        The dict values can alternatively be a dict of keys I, Q and
        U for polarized data.
    """

    logger = get_logger()

    n_inputs = 4
    n_outputs = 1

    def __init__(self, hdulist, data_exts=None, **kwargs):
        super().__init__(**kwargs)
        self.inputs = ('label', 'lon', 'lat', 'pa')
        self._hdulist = hdulist

        # resolve data exts by replacing ext names with the fits extensions
        data_items = list()

        # a dict view of hdus keyed by extname
        extname_hdu = dict()
        for i, hdu in enumerate(hdulist):
            self.logger.debug('HDU {}: {}'.format(
                i, hdu.header.tostring(sep='\n')
                ))
            label = hdu.header.get('EXTNAME', i)
            # TODO handling stokes WCS here to generate I Q U map
            extname_hdu[label] = hdu
        for item in data_exts:
            extname = item['extname']
            if extname not in extname_hdu.keys():
                raise ValueError(f"missing ext {extname} in FITS data.")
            data_items.append(dict(**item, hdu=extname_hdu[extname]))
        # store the data item as table for masked access
        self._data = Table(rows=data_items)
        self.logger.debug(f"FITS image data:\n{self.data}")

        # if extname_map is None:
        #     extname_map = {
        #             k: k for k in extname_hdu.keys()
        #             }
        # cls.logger.debug(f"use extname_map: {pformat_yaml(extname_map)}")

        # data = dict()

        # def resolve_extname(d):
        #     result = dict()
        #     for k, v in d.items():
        #         if isinstance(v, dict):
        #             result[k] = resolve_extname(v)
        #         elif v in extname_hdu:
        #             result[k] = extname_hdu[v]
        #         else:
        #             cls.logger.warning(f"ignore invalid extname {v}")
        #             continue
        #     return result

        # data = resolve_extname(extname_map)
        # cls.logger.debug(f'source data items: {pformat_yaml(data)}')
        # return cls(data=data, name=filepath.as_posix(), **kwargs)

    @property
    def data(self):
        return self._data

    @staticmethod
    def _get_data_sky_bbox(wcsobj, data_shape):
        logger = get_logger()
        # check lon lat range
        # because here we check longitude ranges
        # we need to take into account wrapping issue
        # check pixel range
        ny, nx = data_shape
        # lon lat edge of pixel edges
        lon_e, lat_e = wcsobj.wcs_pix2world(
                np.array([0, 0, nx - 1, nx - 1]),
                np.array([0, ny - 1, 0, ny - 1]),
                0)
        # fix potential wrapping issue by check at 360 and 180 wrapping
        lon_e = Angle(lon_e << u.deg).wrap_at(360. << u.deg).degree
        lon_e_180 = Angle(lon_e << u.deg).wrap_at(180. << u.deg).degree
        # this is done by taking the one with smaller span
        w_e, e_e = np.min(lon_e), np.max(lon_e)
        w_e_180, e_e_180 = np.min(lon_e_180), np.max(lon_e_180)
        s_e, n_e = np.min(lat_e), np.max(lat_e)
        if (e_e_180 - w_e_180) < (e_e - w_e):
            # use wrapping at 180.d
            w_e = w_e_180
            e_e = e_e_180
            lon_wrap_at = 180. << u.deg
            logger.debug(f"re-wrapping data bbox coords at {lon_wrap_at}")
        else:
            lon_wrap_at = 360. << u.deg
        logger.debug(f"data bbox: w={w_e} e={e_e} s={s_e} n={n_e}")
        logger.debug(f'data shape: {data_shape}')
        return [(w_e, e_e), (s_e, n_e)], lon_wrap_at

    @classmethod
    def _set_data_for_group(cls, hdu, group_mask, lon, lat, s_out):
        """Populate `s_out` for `group_mask` from `hdu`."""
        logger = get_logger()
        # check lon lat range of hdu and re-wrap longitude for proper
        # range check
        s_out_unit = u.Unit(hdu.header.get('SIGUNIT', 'adu'))
        wcsobj = WCS(hdu.header)
        ny, nx = data_shape = hdu.data.shape
        (w_e, e_e), (s_e, n_e), lon_wrap_at = cls._get_data_sky_bbox(
            wcsobj, data_shape)
        lon = Angle(lon << u.deg).wrap_at(lon_wrap_at).degree
        g = (
                (lon > w_e) & (lon < e_e)
                & (lat > s_e) & (lat < n_e)
                )
        logger.debug(f"data group mask {g.sum()}/{lon.size}")
        if g.sum() == 0:
            # skip because no overlap
            return
        # convert all lon lat to x y
        x_g, y_g = wcsobj.wcs_world2pix(lon[g], lat[g], 0)
        ii = np.rint(y_g).astype(int)
        jj = np.rint(x_g).astype(int)
        logger.debug(
                f"pixel range: [{ii.min()}, {ii.max()}] "
                f"[{jj.min()}, {jj.max()}]")
        # check ii and jj for valid pixel range
        gp = (ii >= 0) & (ii < ny) & (jj >= 0) & (jj < nx)
        # update g to include only valid pixels
        g[g] = gp
        # TODO this seems to be unnecessary
        # convert all lon lat to x y
        x_g, y_g = wcsobj.wcs_world2pix(lon[g], lat[g], 0)
        ii = np.rint(y_g).astype(int)
        jj = np.rint(x_g).astype(int)
        logger.debug(
                f"pixel range updated: [{ii.min()}, {ii.max()}] "
                f"[{jj.min()}, {jj.max()}]")
        ig = np.where(g)
        m = np.flatnonzero(group_mask)[ig]
        s_out[m] = hdu.data[ii, jj] << s_out_unit
        logger.debug(
            f'signal range: [{s_out[m].min()}, {s_out[m].max()}]')

    def evaluate_tod_icrs(
            self, array_name, det_ra, det_dec,
            det_pa_icrs, hwp_pa_icrs=None):
        data_tbl = self.data
        sp = self.data
        s_outs = {
            'I': None,
            'Q': None,
            'U': None
            }
        for g in group_names:
            if g not in data:
                self.logger.debug(f"group {g} not found in data")
                continue
            data_g = self._data[g]
            m = (label == g)
            data_groups.append([data_g, m])
            self.logger.debug(f"group {g}: {m.sum()}/{len(m)}")
            for sk in s_outs.keys():
                if sk in data_g and s_outs[sk] is None:
                    # create this s_out data
                    s_outs[sk] = np.zeros(lon.shape) << u.MJy / u.sr
        used_stokes_keys = set(
            sk for sk, arr in s_outs.items() if arr is not None)
        self.logger.debug(
            f"evaluate {len(data_groups)} groups "
            f"for Stokes {used_stokes_keys}")

        lon = lon.to_value(u.deg)
        lat = lat.to_value(u.deg)

        for data_g, m in data_groups:
            for sk, hdu in data_g.items():
                self._set_data_for_group(hdu, m, lon[m], lat[m], s_outs[sk])
        # now we do the polarimetry handling
        if used_stokes_keys == {'I', }:
            # this is un-polarized
            return s_outs['I']
        # mix i, Q, and U
        I = s_outs['I']  # noqa: E741
        Q = s_outs['Q']
        U = s_outs['U']
        return 0.5 * (I + np.cos(2. * pa) * Q + np.sin(2. * pa) * U)



    def evaluate(self, label, lon, lat, pa):
        # make group_masks for each label value
        data = self._data
        data_groups = []
        group_names = np.unique(label)

        # note the data_g is a dict of hdus keyed off with stokes I Q U
        # and they may not be present so we can save some memory here
        # by only create those that are needed
        s_outs = {
            'I': None,
            'Q': None,
            'U': None
            }
        for g in group_names:
            if g not in data:
                self.logger.debug(f"group {g} not found in data")
                continue
            data_g = self._data[g]
            m = (label == g)
            data_groups.append([data_g, m])
            self.logger.debug(f"group {g}: {m.sum()}/{len(m)}")
            for sk in s_outs.keys():
                if sk in data_g and s_outs[sk] is None:
                    # create this s_out data
                    s_outs[sk] = np.zeros(lon.shape) << u.MJy / u.sr
        used_stokes_keys = set(
            sk for sk, arr in s_outs.items() if arr is not None)
        self.logger.debug(
            f"evaluate {len(data_groups)} groups "
            f"for Stokes {used_stokes_keys}")

        lon = lon.to_value(u.deg)
        lat = lat.to_value(u.deg)

        for data_g, m in data_groups:
            for sk, hdu in data_g.items():
                self._set_data_for_group(hdu, m, lon[m], lat[m], s_outs[sk])
        # now we do the polarimetry handling
        if used_stokes_keys == {'I', }:
            # this is un-polarized
            return s_outs['I']
        # mix i, Q, and U
        I = s_outs['I']  # noqa: E741
        Q = s_outs['Q']
        U = s_outs['U']
        return 0.5 * (I + np.cos(2. * pa) * Q + np.sin(2. * pa) * U)

    @classmethod
    def from_file(
            cls, filepath, **kwargs):
        """
        Return source model from FITS image file.

        Parameters
        ----------
        filepath : str, `pathlib.Path`
            The path to the FITS file.

        **kwargs :
            Arguments passed to constructor.
        """

        # we may be working with large files so let's just keep this
        # open during the program.
        # TODO use an exit stack to manage the resource.
        filepath = ensure_abspath(filepath)
        hdulist = fits.open(filepath)
        kwargs.setdefault('name', filepath.as_posix())
        return cls(hdulist=hdulist, **kwargs)


class CatalogSourceModel(SurfaceBrightnessModel):
    """The class for simulator source specified by catalog.

    Parameters
    ----------
    catalog : `astropy.table.Table`
        The source catalog.

    pos_cols : tuple, optional
        A 2-tuple specifying the column names of source positions.

    data_cols : list of dict, optional
        A list of dicts that tag the flux column names with labels.
        All items shall have matching dict keys, and have the `colname`
        key referring to the column name of the catalog.
        An educated guess will be made when missing.

    name_col : str, optional
        Specify the column names to included in the returned data.
        If None, an educated guess will be made.
    """

    logger = get_logger()

    n_inputs = 4
    n_outputs = 1

    def __init__(
            self,
            catalog,
            pos_cols=('ra', 'dec'),
            name_col='source_name',
            data_cols=None,
            **kwargs):
        super().__init__(**kwargs)
        self.inputs = ('label', 'lon', 'lat', 'pa')
        self._catalog = catalog
        # extract and normalize some useful props from the catalog
        prop_tbl = self._prop_tbl = self._get_prop_table(
            catalog, pos_cols, name_col)
        self.logger.debug(f"catalog source props:\n{prop_tbl}")
        # TODO: add handling of different coordinate system in input
        pos = SkyCoord(
                prop_tbl['ra'], prop_tbl['dec'], frame='icrs')
        self._pos = pos
        # resolve data cols by replacing column names with the catalog column
        data_items = list()
        for item in data_cols:
            colname = item['colname']
            if colname not in catalog.colnames:
                raise ValueError("missing column {colname} in catalog.")
            data_items.append(dict(
                **item,
                flux=self._get_col_quantity(catalog, colname, u.mJy)))
        # store the data item as table for masked access
        self._data = QTable(rows=data_items)
        self.logger.debug(f"catalog data:\n{self.data}")

    @classmethod
    def _get_col_quantity(cls, tbl, colname, unit):
        # get quantity from table column
        col = tbl[colname]
        if col.unit is None and unit is not None:
            cls.logger.debug(f"assume unit {unit} for column {colname}")
            col.unit = unit
        if col.unit is None:
            return col
        return col.quantity

    @classmethod
    def _get_prop_table(cls, catalog, pos_cols, name_col='source_name'):
        # we use a qtable to hold name and pos
        prop_tbl = QTable()
        if name_col in catalog.colnames:
            prop_tbl['name'] = cls._get_col_quantity(catalog, name_col, None)
        else:
            prop_tbl['name'] = [f'src{i}' for i in range(len(catalog))]
        for k, c in zip(('lon', 'lat'), pos_cols):
            prop_tbl[k] = cls._get_col_quantity(catalog, c, u.deg)
        # keep a record of the original colnames
        for k, c in zip(('lon', 'lat'), pos_cols):
            prop_tbl[c] = prop_tbl[k]
        return prop_tbl

    @property
    def pos(self):
        return self._pos

    @property
    def data(self):
        return self._data

    @classmethod
    def from_file(cls, filepath, **kwargs):
        """Return source model from catalog file.

        Parameters
        ----------
        filepath : str, `pathlib.Path`
            The path to the catalog file.

        **kwargs
            Arguments passed to constructor.
        """
        filepath = ensure_abspath(filepath)
        catalog = Table.read(filepath, format='ascii')
        kwargs.setdefault('name', filepath.as_posix())
        return cls(catalog, **kwargs)

    def evaluate(self, *args, **kwargs):
        # TODO a trivial implementation of this is too slow. Need to figure
        # out a better one
        return NotImplemented

    def make_image_model(self, fwhms, pixscale):
        # fwhms is a dict keyed by the array_names
        pixscale = u.pixel_scale(pixscale)
        delta_pix = (1. << u.pix).to(u.arcsec, equivalencies=pixscale)
        data_tbl = self.data

        # we convert all fwhms to pixels and create a gaussian PSF model
        # for each data table entry

        def make_gauss_psf_model(a_fwhm, b_fwhm):
            # here we keep the flux in unit of surface brightness but
            # x and y are in pix
            a_stddev = (a_fwhm / GAUSSIAN_SIGMA_TO_FWHM)
            b_stddev = (b_fwhm / GAUSSIAN_SIGMA_TO_FWHM)
            beam_area = 2 * np.pi * a_stddev * b_stddev
            return models.Gaussian2D(
                    amplitude=1. / beam_area,
                    x_mean=0.,
                    y_mean=0.,
                    x_stddev=a_stddev.to_value(
                        u.pix, equivalencies=pixscale
                        ),
                    y_stddev=b_stddev.to_value(
                        u.pix, equivalencies=pixscale
                        ),
                    )

        # keep a record of max fwhm in pix so we can use it as the padding
        fwhm_max = np.max(u.Quantity(list(fwhms.values())))

        d_psf_models = {
            array_name: make_gauss_psf_model(fwhm, fwhm)
            for array_name, fwhm in fwhms.items()
            }

        # make a list of models that matches with the data table
        psf_models = list()
        for entry in data_tbl:
            psf_models.append(d_psf_models[entry['array_name']])

        # Here we create the fits image with the same WCS for all extensions
        # TODO: maybe it is more efficient to use different WCS per ext?
        # use the first pos as the reference to build the image wcs
        ref_coord = self.pos[0]
        wcsobj = WCS(naxis=2)
        wcsobj.wcs.crpix = [1.5, 1.5]
        wcsobj.wcs.cdelt = np.array([
            -delta_pix.to_value(u.deg),
            delta_pix.to_value(u.deg),
            ])
        wcsobj.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcsobj.wcs.crval = [ref_coord.ra.degree, ref_coord.dec.degree]

        # compute the pixel range of the catalog for this wcs
        x, y = wcsobj.wcs_world2pix(self.pos.ra, self.pos.dec, 0)
        l, r = np.min(x), np.max(x)
        b, t = np.min(y), np.max(y)
        w, h = r - l, t - b
        # size of the square bbox, with added padding on the edge
        # from the psf model
        padding = 10 * fwhm_max.to_value(u.pix, equivalencies=pixscale)
        s = int(np.ceil(np.max([w, h]) + padding))
        self.logger.debug(f'source image size: {s}')
        # figure out center coord on sky and re-center the wcs
        c_x = (l + r) / 2
        c_y = (b + t) / 2
        c_ra, c_dec = wcsobj.wcs_pix2world(c_x, c_y, 0)
        wcsobj.wcs.crpix = [s / 2 + 1, s / 2 + 1]
        wcsobj.wcs.crval = c_ra, c_dec
        header = wcsobj.to_header()
        # compute the pixel positions with the new wcs
        x, y = wcsobj.wcs_world2pix(self.pos.ra, self.pos.dec, 0)
        # just a sanity check of the pixel values of all catalog sources
        # are within the footprint
        assert ((x < 0) | (x > s)).sum() == 0
        assert ((y < 0) | (y > s)).sum() == 0

        # render the image for each data table entry

        hdus = list()
        data_exts = list()

        for i, (entry, psf_model) in enumerate(zip(data_tbl, psf_models)):
            # make hdu for model psf_model with flux in entry['flux']
            # note flux is a vector
            data_ext = {
                c: entry[c]
                for c in data_tbl.colnames
                }
            flux = data_ext.pop('flux')
            extname = data_ext['extname'] = f'{i}_{data_ext["colname"]}'
            amp = (flux * psf_model.amplitude).to(u.MJy / u.sr)
            img = np.zeros((s, s), dtype=float) << u.MJy / u.sr
            # make a copy of the model so we update the info
            m = psf_model.copy()
            for xx, yy, aa in zip(x, y, amp):
                m.amplitude = aa
                m.x_mean = xx
                m.y_mean = yy
                m.render(img)
            # create the hdu with wcs and data
            hdu = fits.ImageHDU(
                img.to_value(u.MJy / u.sr), header=header)
            hdu.header['SIGUNIT'] = 'MJy / sr'
            hdu.header['EXTNAME'] = extname
            hdus.append(hdu)
            data_exts.append(data_ext)

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(3, 3)
        # for i, hdu in enumerate(hdus):
        #     ax = axes.ravel()[i]
        #     ax.imshow(hdu.data)
        #     ax.set_title(hdu.header['EXTNAME'])
        # plt.show()
        return ImageSourceModel(
                hdulist=hdus,
                data_exts=data_exts,
                name=self.name,
                )
