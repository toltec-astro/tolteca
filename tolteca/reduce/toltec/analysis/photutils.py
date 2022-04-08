#!/usr/bin/env python


from photutils.psf import DAOGroup
from photutils.psf import (
            IntegratedGaussianPRF,
            BasicPSFPhotometry)
# from photutils.background import MMMBackground
from astropy.modeling.fitting import LevMarLSQFitter
# from astropy.stats import SigmaClip
# from photutils.background import Background2D
# from photutils import CircularAperture


from dataclasses import dataclass, field
from pathlib import Path
from schema import Or
from typing import Union
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import astropy.units as u
from astropy.modeling.functional_models import GAUSSIAN_SIGMA_TO_FWHM

from tollan.utils.dataclass_schema import add_schema
from tollan.utils.log import get_logger

from ....utils.common_schema import RelPathSchema
from ....datamodels.dp.base import DataProd, DataItemKind
from ... import steps_registry
from ....simu.toltec.toltec_info import toltec_info


@steps_registry.register('photutils')
@add_schema
@dataclass
class PhotUtilsStepConfig():
    '''
    The config class for the photutils analysis.
    '''

    enabled: bool = field(
        default=True,
        metadata={
            'description': 'Enable/disable this pipeline step.'
            }
        )
    input_source_catalog_path: Union[None, Path] = field(
        default=None,
        metadata={
            'description': 'The source catalog to use for flux extraction.',
            'schema': Or(RelPathSchema(), None)}
        )
    array_name_for_detection: str = field(
        default='a1100',
        metadata={
            'description': (
                'The array used for detection, when input'
                ' source catalog is not provided.'),
            'schema': Or(*toltec_info['array_names'])
            }
        )
    detection_params: dict = field(
        default_factory=dict,
        metadata={
            'description': 'The config dict for the detection module.'
            }
        )
    extraction_params: dict = field(
        default_factory=dict,
        metadata={
            'description': 'The config dict for the extraction module.'
            }
        )

    def __post_init__(self):
        # some additional initialization
        fwhms = dict()
        for array_name in toltec_info['array_names']:
            fwhms[array_name] = toltec_info[array_name]['a_fwhm']
        self._fwhms = fwhms
        self.logger = get_logger()

    def __call__(self, cfg):
        # create the photometry executor
        def photutils_executor(dp, output_dir):

            # do detection, if needed
            cat_in_path = self.input_source_catalog_path
            if cat_in_path is not None:
                cat_in = Table.read(
                    cat_in_path, format='ascii.commented_header')
            else:
                cat_in = NotImplemented
            # do photometry
            fwhms = self._fwhms
            results = list()
            for i, item in enumerate(dp.index_table):
                array_name = item['array_name']
                filepath = Path(item['filepath'])
                # TODO implement IO support for data items in data prod
                hl = fits.open(filepath)
                hdu = hl[1]  # signal
                # hdu_wht = item.get_hdu(name='weight')
                wcsobj = WCS(hdu.header)
                # source catalog for extract flux
                x_src, y_src = wcsobj.all_world2pix(
                    cat_in['ra'], cat_in['dec'], 0)
                xy = Table(names=['x_0', 'y_0'], data=[x_src, y_src])
                pixscale = wcsobj.proj_plane_pixel_scales()[0] / u.pix
                # convert the data from MJy/sr to mJy/pix
                fwhm_pix = fwhms[array_name].to_value(
                    u.pix, equivalencies=u.pixel_scale(pixscale))
                beam_area = 2 * np.pi * (
                    fwhms[array_name] / GAUSSIAN_SIGMA_TO_FWHM) ** 2
                beam_area_pix2 = 2 * np.pi * (
                    fwhm_pix / GAUSSIAN_SIGMA_TO_FWHM) ** 2
                data = (hdu.data << u.MJy/u.sr).to_value(
                    u.mJy / u.beam,
                    equivalencies=u.beam_angular_area(beam_area)
                    ) / beam_area_pix2
                psf_model = IntegratedGaussianPRF(
                    sigma=fwhm_pix / GAUSSIAN_SIGMA_TO_FWHM)
                daogroup = DAOGroup(0.5 * fwhm_pix)
                fit_size = int(fwhm_pix * 3.)  # fit box of 3 * fwhm_pix
                if fit_size % 2 == 0:
                    fit_size += 1
                photometry = BasicPSFPhotometry(
                                    group_maker=daogroup,
                                    bkg_estimator=None,
                                    psf_model=psf_model,
                                    fitter=LevMarLSQFitter(),
                                    fitshape=(fit_size, fit_size))
                catalog = photometry(
                    image=data,
                    init_guesses=xy)
                # get normalization from kernel map
                hdu_kernel = hl[3]
                corr = hdu_kernel.data.max()
                catalog[f'flux_{array_name}'] = catalog['flux_fit'] / corr
                catalog[f'fluxerr_{array_name}'] = catalog['flux_unc'] / corr
                catalog.meta['array_name'] = array_name
                catalog.meta['image_filepath'] = filepath
                results.append(catalog)
            # prepare output catalog, which is merged
            cat_out = cat_in[['name', 'ra', 'dec']]
            for cat in results:
                array_name = cat.meta['array_name']
                fcol = f'flux_{array_name}'
                ferrcol = f'fluxerr_{array_name}'
                cat_out[fcol] = cat[fcol]
                cat_out[ferrcol] = cat[ferrcol]
            # write to output
            # cat_out.meta['context'] = self.to_dict()
            # cat_out.meta['inputs'] = [dp.index]
            output_path = output_dir.joinpath(
                dp.meta['name'] + '_photutils.cat')
            cat_out.write(
                output_path,
                overwrite=True, format='ascii.ecsv')
            self.logger.info(f"output catalog written to: {output_path}")
            return cat_out
        return photutils_executor

    def run(self, cfg, inputs=None):
        """Run this reduction step."""
        if inputs is None:
            inputs = cfg.load_input_data()
        dps = [
            input for input in inputs
            if (
                isinstance(input, DataProd)
                and (input.data_item_kinds & DataItemKind.Image))
            ]
        if len(dps) == 0:
            self.logger.debug("no valid input for this step, skip")
            return None

        assert len(dps) == 1
        dp = dps[0]
        output_dir = cfg.get_or_create_output_dir()
        photutils_executor = self(cfg)
        return photutils_executor(
            dp=dp,
            output_dir=output_dir,
            )
