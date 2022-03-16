#!/usr/bin/env python

from dash_component_template import ComponentTemplate
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

from dasha.web.templates.common import (
        LabeledChecklist,
        DownloadButton
        )
import dash_js9 as djs9

import astropy.units as u
from astropy.modeling.functional_models import GAUSSIAN_SIGMA_TO_FWHM
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from base64 import b64encode

from tollan.utils.log import get_logger, timeit

from ....simu.sequoia.sequoia_info import sequoia_info
from ....simu import instrument_registry
from ....simu.utils import SkyBoundingBox, make_wcs
from .base import ObsInstru, ObsSite


class Sequoia(ObsInstru, name='sequoia'):
    """An `ObsInstru` for SEQUOIA."""

    info = sequoia_info
    display_name = info['name_long']

    class ResultPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Container

        def __init__(self, instru, **kwargs):
            kwargs.setdefault("fluid", True)
            super().__init__(**kwargs)
            self._instru = instru
            container = self
            fitsview_container, info_container = container.colgrid(1, 2)
            self._fitsview_loading = fitsview_container.child(
                dbc.Spinner,
                show_initially=False, color='primary',
                spinner_style={"width": "5rem", "height": "5rem"}
                )
            self._fitsview = self._fitsview_loading.child(
                djs9.DashJS9,
                style={
                    'width': '100%',
                    'min-height': '500px',
                    'height': '40vh'
                    },
                )
            self._info_loading = info_container.child(
                dbc.Spinner,
                show_initially=False, color='primary',
                spinner_style={"width": "5rem", "height": "5rem"}
                )

        def make_callbacks(self, app, exec_info_store_id):
            fitsview = self._fitsview
            info = self._info_loading.child(
                html.Div, style={'min-height': '500px'})

            app.clientside_callback(
                '''
                function(exec_info) {
                    // console.log(exec_info);
                    if (!exec_info) {
                        return Array(1).fill(window.dash_clientside.no_update);
                    }
                    return [exec_info.instru.fits_images];
                }
                ''',
                [
                    Output(fitsview.id, 'data'),
                    ],
                [
                    Input(exec_info_store_id, 'data'),
                    ]
                )

            app.clientside_callback(
                '''
                function(exec_info) {
                    // console.log(exec_info);
                    if (!exec_info) {
                        return Array(1).fill(window.dash_clientside.no_update);
                    }
                    return [exec_info.instru.info];
                }
                ''',
                [
                    Output(info.id, 'children'),
                    ],
                [
                    Input(exec_info_store_id, 'data'),
                    ]
                )

        @property
        def loading_indicators(self):
            return {
                'outputs': [
                    Output(self._fitsview_loading.id, 'color'),
                    Output(self._info_loading.id, 'color')
                    ],
                'states': [
                    State(self._fitsview_loading.id, 'color'),
                    State(self._info_loading.id, 'color'),
                    ]
                }

    class ResultControlPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Container

        def __init__(self, instru, **kwargs):
            kwargs.setdefault("fluid", True)
            super().__init__(**kwargs)
            container = self
            dlbtn_props = {'disabled': True}

            self._lmtot_download = container.child(
                DownloadButton(
                    button_text='LMT OT Script',
                    className='me-2',
                    button_props=dlbtn_props,
                    tooltip=(
                        'Download LMT Observation Tool script to '
                        'execute the observation at LMT.')
                    )
                )
            self._fits_download = container.child(
                DownloadButton(
                    button_text='Coverage Map',
                    className='me-2',
                    button_props=dlbtn_props,
                    tooltip=(
                        'Download the generated FITS (approximate) coverage '
                        'image for the observation.'
                        )
                    )
                )

        def make_callbacks(self, app, exec_info_store_id):

            for dl in [
                    self._lmtot_download,
                    self._fits_download]:
                app.clientside_callback(
                    '''
                    function(exec_info) {
                        if (!exec_info) {
                            return true;
                        }
                        return false;
                    }
                    ''',
                    Output(dl.button.id, 'disabled'),
                    [
                        Input(exec_info_store_id, 'data'),
                        ]
                    )

            app.clientside_callback(
                r'''
                function(n_clicks, exec_info) {
                    // console.log(exec_info);
                    if (!exec_info) {
                        return window.dash_clientside.no_update;
                    }
                    target = exec_info.exec_config.mapping.target

                    filename = (
                        'target_' + target + '.lmtot').replace(/\s+/g, '-');
                    return {
                        content: exec_info.instru.lmtot,
                        base64: false,
                        filename: filename,
                        type: 'text/plain;charset=UTF-8'
                        };
                }
                ''',
                Output(self._lmtot_download.download.id, 'data'),
                [
                    Input(self._lmtot_download.button.id, 'n_clicks'),
                    State(exec_info_store_id, 'data'),
                    ]
                )

            app.clientside_callback(
                '''
                function(n_clicks, exec_info) {
                    // console.log(exec_info);
                    if (!exec_info) {
                        return window.dash_clientside.no_update;
                    }
                    im = exec_info.instru.fits_images[0]
                    return {
                        content: im.blob,
                        base64: true,
                        filename: im.options.file,
                        type: 'application/fits'
                        };
                }
                ''',
                Output(self._fits_download.download.id, 'data'),
                [
                    Input(self._fits_download.button.id, 'n_clicks'),
                    State(exec_info_store_id, 'data'),
                    ]
                )

    class ControlPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Form

        def __init__(self, instru, **kwargs):
            super().__init__(**kwargs)
            self._instru = instru
            container = self
            self._info_store = container.child(dcc.Store, data={
                'name': instru.name
                })

        @property
        def info_store(self):
            return self._info_store

        def setup_layout(self, app):
            sequoia_info = self._instru.info
            container = self.child(dbc.Row, className='gy-2')
            band_select = container.child(
                    LabeledChecklist(
                        label_text='SEQUOIA band',
                        className='w-auto',
                        size='sm',
                        # set to true to allow multiple check
                        multi=False,
                        input_props={
                            'style': {
                                'text-transform': 'none'
                                }
                            }
                        )).checklist
            band_select.options = [
                    {
                        'label': str(sequoia_info[b]['name']),
                        'value': b,
                        }
                    for b in sequoia_info['band_names']
                    ]
            band_select.value = sequoia_info['band_names'][0]

            covtype_select = container.child(
                    LabeledChecklist(
                        label_text='Coverage Unit',
                        className='w-auto',
                        size='sm',
                        # set to true to allow multiple check
                        multi=False
                        )).checklist
            covtype_select.options = [
                    {
                        'label': 's/pixel',
                        'value': 'time',
                        },
                    ]
            covtype_select.value = 'time'
            super().setup_layout(app)

            # collect inputs to store
            app.clientside_callback(
                """
                function(band_select_value, covtype_select_value, data_init) {
                    data = {...data_init}
                    data['band_name'] = band_select_value
                    data['coverage_map_type'] = covtype_select_value
                    return data
                }
                """,
                Output(self.info_store.id, 'data'),
                [
                    Input(band_select.id, 'value'),
                    Input(covtype_select.id, 'value'),
                    State(self.info_store.id, 'data')
                    ]
                )

    @staticmethod
    def _hdulist_to_base64(hdulist):
        fo = BytesIO()
        hdulist.writeto(fo, overwrite=True)
        return b64encode(fo.getvalue()).decode("utf-8")

    @classmethod
    def make_traj_data(cls, exec_config, bs_traj_data):
        logger = get_logger()
        logger.debug("make traj data for instru sequoia")
        # get observer from site name
        observer = ObsSite.get_observer(exec_config.site_data['name'])
        mapping_model = exec_config.mapping.get_model(observer=observer)
        instru = instrument_registry.schema.validate({
            'name': 'sequoia',
            'mode': 's_wide',
            }, create_instance=True)
        simulator = instru.simulator
        band_name = exec_config.instru_data['band_name']
        apt = simulator.array_prop_table
        det_dlon = apt['x_t']
        det_dlat = apt['y_t']

        # apply the footprint on target
        # to do so we find the closest point in the trajectory to
        # the target and do the transformation
        bs_coords_icrs = SkyCoord(
                bs_traj_data['ra'], bs_traj_data['dec'], frame='icrs')
        target_icrs = mapping_model.target.transform_to('icrs')
        i_closest = np.argmin(
                target_icrs.separation(bs_coords_icrs))
        # the center of the array overlay in altaz
        az1 = bs_traj_data['az'][i_closest]
        alt1 = bs_traj_data['alt'][i_closest]
        t1 = bs_traj_data['time_obs'][i_closest]
        c1 = SkyCoord(
                az=az1, alt=alt1, frame=observer.altaz(time=t1)
                )
        det_altaz = SkyCoord(
                det_dlon, det_dlat,
                frame=c1.skyoffset_frame()).transform_to(c1.frame)
        det_icrs = det_altaz.transform_to("icrs")

        det_sky_bbox_icrs = SkyBoundingBox.from_lonlat(
            det_icrs.ra,
            det_icrs.dec
            )
        # make coverage fits image in s_per_pix
        # we'll init the power loading model to estimate the conversion factor
        # of this to mJy/beam
        cov_hdulist_s_per_pix = cls._make_cov_hdulist(ctx=locals())
        # overlay traces
        # each trace is for one polarimetry group
        offset_traces = [{
            'x': det_dlon.to_value(u.arcmin),
            'y': det_dlat.to_value(u.arcmin),
            'mode': 'markers',
            'marker': {
                'symbol': 'square',
                'color': 'gray',
                'size': 6,
                },
            'legendgroup': 'sequoia_fov',
            'showlegend': True,
            'name': f"Toggle FOV: {cls.info['name_long']}"
            }]

        # skyview layers
        skyview_layers = list()
        n_dets = len(det_icrs)
        det_tbl = pd.DataFrame.from_dict({
            "ra": det_icrs.ra.degree,
            "dec": det_icrs.dec.degree,
            "color": ["blue"] * n_dets,
            "type": ["circle"] * n_dets,
            "radius": [
                cls.info[band_name]["hpbw"].to_value(u.deg) * 0.5
                ] * n_dets,
            })
        skyview_layers.extend([
            {
                "type": "overlay",
                "data": det_tbl.to_dict(orient="records"),
                "options": {
                    'name': f"Detectors: {cls.info['name_long']}",
                    "show": True,
                    }
                },
            ])

        # make cov hdulist depending on the instru_data cov unit settings
        if exec_config.instru_data['coverage_map_type'] == 'depth':
            raise NotImplementedError
        else:
            cov_hdulist = cov_hdulist_s_per_pix

        # from the cov image we can create a coutour showing the outline of
        # the observation on the skyview
        cov_ctr = cov_hdulist[1].data.copy()
        cov_wcs = WCS(cov_hdulist[1].header)
        m_cov = cov_ctr > 0.02 * cov_ctr.max()
        cov_ctr[~m_cov] = 0
        im = cv2.normalize(
            src=cov_ctr,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        cxy = cv2.findContours(
            im,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[0][0]
        cxy_s = cv2.approxPolyDP(
            cxy,
            0.002 * cv2.arcLength(cxy, True), True
            ).squeeze()
        cra, cdec = cov_wcs.pixel_to_world_values(
            cxy_s[:, 0], cxy_s[:, 1]
            )
        skyview_layers.extend([
            {
                'type': "overlay",
                "data": [{
                    "type": "polygon",
                    "data": list(zip(cra, cdec)),
                    }],
                "options": {
                    'name': "Coverage Outline",
                    "color": "#66cccc",
                    "show": True,
                    "lineWidth": 4,
                    }
                },
            ])

        return {
            "dlon": det_dlon,
            "dlat": det_dlat,
            "az": det_altaz.az,
            "alt": det_altaz.alt,
            "ra": det_icrs.ra,
            "dec": det_icrs.dec,
            "sky_bbox_icrs": det_sky_bbox_icrs,
            'overlay_traces': {
                'offset': offset_traces
                },
            'skyview_layers': skyview_layers,
            'results': {
                'fits_images': [
                    {
                        'options': {
                            'file': f"obsplanner_sequoia_{band_name}_cov.fits",
                            },
                        'blob': cls._hdulist_to_base64(cov_hdulist),
                        }
                    ],
                'lmtot': 'N/A',
                'simu_config': 'N/A',
                'info': 'N/A',
                }
            }

    @classmethod
    def _make_cov_hdu_approx(cls, ctx):
        logger = get_logger()
        # unpack the cxt
        bs_traj_data = ctx['bs_traj_data']
        det_icrs = ctx['det_icrs']
        det_sky_bbox_icrs = ctx['det_sky_bbox_icrs']
        dt_smp = bs_traj_data['time_obs'][1] - bs_traj_data['time_obs'][0]
        band_name = ctx['band_name']

        # create the wcs
        pixscale = u.pixel_scale(4. << u.arcsec / u.pix)
        # the pixsize will be int factor of 2 arcsec.
        adaptive_pixscale_factor = 0.5
        n_pix_max = 1e6  # 8 MB of data
        bs_sky_bbox_icrs = bs_traj_data['sky_bbox_icrs']
        sky_bbox_wcs = bs_sky_bbox_icrs.pad_with(
            det_sky_bbox_icrs.width + (2 << u.arcmin),
            det_sky_bbox_icrs.height + (2 << u.arcmin),
            )
        wcsobj = make_wcs(
            sky_bbox=sky_bbox_wcs, pixscale=pixscale, n_pix_max=n_pix_max,
            adaptive_pixscale_factor=adaptive_pixscale_factor)

        bs_xy = wcsobj.world_to_pixel_values(
            bs_traj_data['ra'].degree,
            bs_traj_data['dec'].degree,
            )
        det_xy = wcsobj.world_to_pixel_values(
            det_icrs.ra.degree,
            det_icrs.dec.degree,
            )
        # because these are bin edges, we add 1 at end to
        # makesure the nx and ny are included in the range.
        xbins = np.arange(wcsobj.pixel_shape[0] + 1)
        ybins = np.arange(wcsobj.pixel_shape[1] + 1)
        det_xbins = np.arange(
                np.floor(det_xy[0].min()),
                np.ceil(det_xy[0].max()) + 1 + 1
                )
        det_ybins = np.arange(
                np.floor(det_xy[1].min()),
                np.ceil(det_xy[1].max()) + 1 + 1
                )
        # note the axis order ij -> yx
        bs_im, _, _ = np.histogram2d(
                bs_xy[1],
                bs_xy[0],
                bins=[ybins, xbins])
        # scale to coverage image of unit s / pix
        bs_im *= dt_smp.to_value(u.s)

        det_im, _, _ = np.histogram2d(
                det_xy[1],
                det_xy[0],
                bins=[det_ybins, det_xbins]
                )
        # convolve boresignt image with the detector image
        with timeit("convolve with array layout"):
            cov_im = convolve_fft(
                bs_im, det_im,
                normalize_kernel=False, allow_huge=True)
        with timeit("convolve with beam"):
            g_stddev = cls.info[band_name]['hpbw'] / GAUSSIAN_SIGMA_TO_FWHM
            g = Gaussian2DKernel(
                    g_stddev.to_value(
                        u.pix, equivalencies=pixscale),
                    g_stddev.to_value(
                        u.pix, equivalencies=pixscale),
                   )
            cov_im = convolve_fft(cov_im, g, normalize_kernel=False)
        logger.debug(
                f'total exp time on coverage map: '
                f'{(cov_im.sum() / det_im.sum() << u.s).to(u.min)}')
        logger.debug(
                f'total time of observation: '
                f'{bs_traj_data["t_exp"].to(u.min)}')
        cov_hdr = wcsobj.to_header()
        cov_hdr['BUNIT'] = 's / pix'
        cov_hdr.append((
            "BANDNAME", band_name,
            "The name of the band"))
        cov_hdr.append((
            "BAND", band_name,
            "The name of the band"))
        return fits.ImageHDU(data=cov_im, header=cov_hdr)

    @classmethod
    def _make_cov_hdulist(cls, ctx):
        bs_traj_data = ctx['bs_traj_data']
        t_exp = bs_traj_data['t_exp']
        target_alt = bs_traj_data['target_alt']
        site_info = cls.info['site']
        phdr = fits.Header()
        phdr.append((
            'ORIGIN', 'The TolTEC Project',
            'Organization generating this FITS file'
            ))
        phdr.append((
            'CREATOR', cls.__qualname__,
            'The software used to create this FITS file'
            ))
        phdr.append((
            'TELESCOP', site_info['name'],
            site_info['name_long']
            ))
        phdr.append((
            'INSTRUME', cls.info['name'],
            cls.info['name_long']
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
                  target_alt.mean().to_value(u.deg)),
            'Mean altitude of the target during observation (deg)'))
        hdulist = [
            fits.PrimaryHDU(header=phdr),
            cls._make_cov_hdu_approx(ctx)
            ]
        hdulist = fits.HDUList(hdulist)
        return hdulist
