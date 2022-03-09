#!/usr/bin/env python

from dash_component_template import ComponentTemplate, NullComponent
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash
import dash_bootstrap_components as dbc
import dash_aladin_lite as dal
import dash_js9 as djs9

from dasha.web.templates.common import (
        CollapseContent,
        LabeledChecklist,
        LabeledInput,
        DownloadButton
        )
from dasha.web.templates.utils import PatternMatchingId, make_subplots
import astropy.units as u
# from astropy.coordinates import get_icrs_coordinates
from astroquery.utils import parse_coordinates
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.coordinates.erfa_astrom import (
        erfa_astrom, ErfaAstromInterpolator)
from astroplan import FixedTarget
from astroplan import (AltitudeConstraint, AtNightConstraint)
from astroplan import observability_table
from astropy.table import QTable
from astropy.modeling.functional_models import GAUSSIAN_SIGMA_TO_FWHM
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve_fft
from astropy.io import fits
from astropy.wcs import WCS

from dataclasses import dataclass, field
import numpy as np
import functools
import bisect
from typing import Union
from io import StringIO, BytesIO
from base64 import b64encode
from schema import Or
import jinja2
import json
import pandas as pd
import cv2

from tollan.utils.log import get_logger, timeit
from tollan.utils.fmt import pformat_yaml
from tollan.utils import rupdate
from tollan.utils.dataclass_schema import add_schema
from tollan.utils.namespace import Namespace

from ....utils import yaml_load, yaml_dump
from ....simu.toltec.lmt import lmt_info
from ....simu.toltec.toltec_info import toltec_info
from ....simu.exports import LmtOTExporterConfig
from ....simu.utils import SkyBoundingBox, make_wcs
from ....simu import (
    mapping_registry,
    instrument_registry, SimulatorRuntime, ObsParamsConfig)
from ....simu.mapping.utils import resolve_sky_coords_frame

from .preset import PresetsConfig


_j2env = jinja2.Environment()
"""A jinja2 environment for generating clientside callbacks."""


def _add_from_name_factory(cls):
    """A helper decorator to add ``from_name`` factory method to class."""
    cls._subclasses = dict()

    def _init_subclass(cls, name):
        cls._subclasses[name] = cls
        cls.name = name

    cls.__init_subclass__ = classmethod(_init_subclass)

    def from_name(cls, name):
        """Return the site instance for `name`."""
        if name not in cls._subclasses:
            raise ValueError(f"invalid obs site name {name}")
        subcls = cls._subclasses[name]
        return subcls()

    cls.from_name = classmethod(from_name)

    return cls


class ObsPlannerModule(object):
    """Base class for modules in obsplanner."""

    def make_results_display(self, container, **kwargs):
        return container.child(self.ResultPanel(self, **kwargs))

    def make_results_controls(self, container, **kwargs):
        return container.child(self.ResultControlPanel(self, **kwargs))

    def make_controls(self, container, **kwargs):
        return container.child(self.ControlPanel(self, **kwargs))

    def make_info_display(self, container, **kwargs):
        return container.child(self.InfoPanel(self, **kwargs))

    InfoPanel = NotImplemented
    """
    Subclass should implement this as a `ComponentTemplate` to generate UI
    for display read-only custom info for this module.
    """

    ControlPanel = NotImplemented
    """
    Subclass should implement this class as a ComponentTemplate
    to generate UI for collecting user inputs.

    The template should have an attribute ``info_store`` of type
    dcc.Store, which is updated when user input changes.

    The data in info_store is typically consumed by `ObsplannerExecConfig` to
    create the exec config object.
    """

    ResultPanel = NotImplemented
    """
    Subclass should implement this as a `ComponentTemplate` to generate UI
    for presenting the obs planning results.
    """

    ResultControlPanel = NotImplemented
    """
    Subclass should implement this as a `ComponentTemplate` to generate UI
    for operating with the obs planning results.
    """

    display_name = NotImplemented
    """Name to display as title of the module."""


@_add_from_name_factory
class ObsSite(ObsPlannerModule):
    """A module base class for an observation site"""

    observer = NotImplemented
    """The astroplan.Observer instance for the site."""

    @classmethod
    def get_observer(cls, name):
        return cls._subclasses[name].observer


@_add_from_name_factory
class ObsInstru(ObsPlannerModule):
    """A module base class for an observation instrument."""

    @classmethod
    def make_traj_data(cls, exec_config):
        """Subclass should implement this to generate traj_data
        as part of the result of `ObsPlannerExecConfig.make_traj_data`.
        """
        return NotImplemented


class Lmt(ObsSite, name='lmt'):
    """An `ObsSite` for LMT."""

    info = lmt_info
    display_name = info['name_long']
    observer = info['observer']

    class ControlPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Form

        _atm_q_values = [25, 50, 75]
        _atm_q_default = 25
        _tel_surface_rms_default = 76 << u.um

        def __init__(self, site, **kwargs):
            super().__init__(**kwargs)
            self._site = site
            container = self
            self._info_store = container.child(dcc.Store, data={
                'name': site.name
                })

        @property
        def info_store(self):
            return self._info_store

        def setup_layout(self, app):
            container = self.child(dbc.Row, className='gy-2')
            atm_select = container.child(
                    LabeledChecklist(
                        label_text='Atm. Quantile',
                        className='w-auto',
                        size='sm',
                        # set to true to allow multiple check
                        multi=False
                        )).checklist
            atm_select.options = [
                    {
                        'label': f"{q} %",
                        'value': q,
                        }
                    for q in self._atm_q_values]
            atm_select.value = self._atm_q_default

            tel_surface_input = container.child(
                    LabeledInput(
                        label_text='Tel. Surface RMS',
                        className='w-auto',
                        size='sm',
                        input_props={
                            # these are the dbc.Input kwargs
                            'type': 'number',
                            'min': 0,
                            'max': 200,
                            'placeholder': '0-200',
                            'style': {
                                'flex': '0 1 5rem'
                                },
                            },
                        suffix_text='μm'
                        )).input
            tel_surface_input.value = self._tel_surface_rms_default.to_value(
                u.um)
            super().setup_layout(app)

            # collect inputs to store
            app.clientside_callback(
                """
                function(atm_select_value, tel_surface_value, data_init) {
                    data = {...data_init}
                    data['atm_model_name'] = 'am_q' + atm_select_value
                    data['tel_surface_rms'] = tel_surface_value + ' um'
                    return data
                }
                """,
                Output(self.info_store.id, 'data'),
                [
                    Input(atm_select.id, 'value'),
                    Input(tel_surface_input.id, 'value'),
                    State(self.info_store.id, 'data')
                    ]
                )

    class InfoPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Container

        def __init__(self, site, **kwargs):
            kwargs.setdefault('fluid', True)
            super().__init__(**kwargs)
            self._site = site

        def setup_layout(self, app):
            container = self.child(
                    CollapseContent(button_text='Current Site Info ...')
                ).content
            info = self._site.info
            location = info['location']
            timezone_local = info['timezone_local']
            info_store = container.child(
                dcc.Store, data={
                    'name': self._site.name,
                    'display_name': self._site.display_name,
                    'lon': location.lon.degree,
                    'lat': location.lat.degree,
                    'height_m': location.height.to_value(u.m),
                    'timezone_local': timezone_local.zone,
                    })
            pre_kwargs = {
                'className': 'mb-0',
                'style': {
                    'font-size': '80%'
                    }
                }
            loc_display = container.child(
                html.Pre,
                f'Location: {location.lon.to_string()} '
                f'{location.lat.to_string()}',
                # f' {location.height:.0f}'
                **pre_kwargs)
            loc_display.className += ' mb-0'
            time_display = container.child(
                html.Pre, **pre_kwargs)
            timer = container.child(dcc.Interval, interval=500)
            super().setup_layout(app)
            app.clientside_callback(
                """
function(t_interval, info) {
    // console.log(info)
    // current datetime
    var dt = new Date()
    let ut_fmt = new Intl.DateTimeFormat(
        'sv-SE', {
            timeZone: 'UTC',
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false,
            timeZoneName: 'short'})
    result = 'Time: ' + ut_fmt.format(dt)
    let lmt_lt_fmt = new Intl.DateTimeFormat(
        'sv-SE', {
            timeZone: info["timezone_local"],
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false,
            timeZoneName: 'short'})
    result = result + '\\nLocal Time: ' + lmt_lt_fmt.format(dt)
    // LST
    var au = window.dash_clientside.astro_utils
    var lst = au.getLST(dt, info["lon"])
    // convert to hh:mm::ss
    var n = new Date(0,0)
    n.setSeconds(+lst * 60 * 60)
    lmt_lst = n.toTimeString().slice(0, 8)
    result = result + '\\nLST: ' + lmt_lst
    return result
}
""",
                Output(time_display.id, 'children'),
                Input(timer.id, 'n_intervals'),
                Input(info_store.id, 'data'),
                prevent_initial_call=True
                )


class Toltec(ObsInstru, name='toltec'):
    """An `ObsInstru` for TolTEC."""

    info = toltec_info
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
                style={'width': '100%', 'min-height': '500px', 'height': '40vh'}
                )
            self._info_loading = info_container.child(
                dbc.Spinner,
                show_initially=False, color='primary',
                spinner_style={"width": "5rem", "height": "5rem"}
                )
            
        def setup_layout(self, app):
            return super().setup_layout(app)

        def make_callbacks(self, app, exec_info_store_id):
            fitsview = self._fitsview
            info = self._info_loading.child(html.Div, style={'min-height': '500px'})

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
                    tooltip='Download LMT Observation Tool script to execute the observation at LMT.'
                    ) 
                )
            self._simuconfig_download = container.child(
                DownloadButton(
                    button_text='Simu. Config',
                    className='me-2',
                    button_props=dlbtn_props,
                    tooltip='Download tolteca.simu 60_simu.yaml config file to run the observation simulator.'
                    ) 
                )
            self._fits_download = container.child(
                DownloadButton(
                    button_text='Coverage Map',
                    className='me-2',
                    button_props=dlbtn_props,
                    tooltip='Download the generated FITS (approximate) coverage image for the observation.'
                    ) 
                )

        def make_callbacks(self, app, exec_info_store_id):

            for dl in [self._lmtot_download, self._simuconfig_download, self._fits_download]:
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
                '''
                function(n_clicks, exec_info) {
                    // console.log(exec_info);
                    if (!exec_info) {
                        return window.dash_clientside.no_update;
                    }
                    target = exec_info.exec_config.mapping.target
                    
                    filename = ('target_' + target + '.lmtot').replace(/\s+/g, '-');
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
                    filename = '60_simu.yaml'
                    return {
                        content: exec_info.instru.simu_config,
                        base64: false,
                        filename: filename,
                        type: 'text/plain;charset=UTF-8'
                        };
                }
                ''',
                Output(self._simuconfig_download.download.id, 'data'),
                [
                    Input(self._simuconfig_download.button.id, 'n_clicks'),
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
            toltec_info = self._instru.info
            container = self.child(dbc.Row, className='gy-2')
            band_select = container.child(
                    LabeledChecklist(
                        label_text='TolTEC band',
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
                        'label': str(toltec_info[a]['wl_center']),
                        'value': a,
                        }
                    for a in toltec_info['array_names']
                    ]
            band_select.value = toltec_info['array_names'][0]

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
                        'label': 'mJy/beam',
                        'value': 'depth',
                        },
                    {
                        'label': 's/pixel',
                        'value': 'time',
                        },
                    ]
            covtype_select.value = 'depth'
            super().setup_layout(app)

            # collect inputs to store
            app.clientside_callback(
                """
                function(band_select_value, covtype_select_value, data_init) {
                    data = {...data_init}
                    data['array_name'] = band_select_value
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
        logger.debug("make traj data for instru toltec")
        # get observer from site name
        observer = ObsSite.get_observer(exec_config.site_data['name'])
        mapping_model = exec_config.mapping.get_model(observer=observer)


        instru = instrument_registry.schema.validate({
            'name': 'toltec',
            'polarized': False
            }, create_instance=True)
        simulator = instru.simulator
        array_name = exec_config.instru_data['array_name']
        apt = simulator.array_prop_table
        # apt_0 is the apt for the current selected array
        apt_0 = apt[apt['array_name'] == array_name]
        # this is the apt including only detectors on the edge
        # useful for making the footprint outline
        ei = apt.meta[array_name]["edge_indices"]
        
        det_dlon = apt_0['x_t']
        det_dlat = apt_0['y_t']

        # apply the footprint on target
        # to do so we find the closest poinit in the trajectory to
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
        offset_traces = list()
        for i, (pg, marker) in enumerate([(0, 'cross'), (1, 'x')]):
            mask = apt_0['pg'] == pg
            offset_traces.append({
                'x': det_dlon[mask].to_value(u.arcmin),
                'y': det_dlat[mask].to_value(u.arcmin),
                'mode': 'markers',
                'marker': {
                    'symbol': marker,
                    'color': 'gray',
                    'size': 6,
                    },
                'legendgroup': 'toltec_array_fov',
                'showlegend': i == 0,
                'name': f"Toggle FOV: {cls.info[array_name]['name_long']}"
                })

        # skyview layers
        skyview_layers = list()
        n_dets = len(det_icrs)
        det_tbl = pd.DataFrame.from_dict({
            "ra": det_icrs.ra.degree,
            "dec": det_icrs.dec.degree,
            "color": ["blue"] * n_dets,
            "type": ["circle"] * n_dets, 
            "radius": [cls.info[array_name]["a_fwhm"].to_value(u.deg) * 0.5] * n_dets,
            })
        skyview_layers.extend([
            # {
            #     'type': "catalog",
            #     "data": det_tbl.to_dict(orient="records"),
            #     "options": {
            #         'name': f"Detectors: {cls.info[array_name]['name_long']}",
            #         "color": "#3366cc",
            #         "show": False,
            #         }
            #     },
            {
                "type": "overlay",
                "data": det_tbl.to_dict(orient="records"),
                "options": {
                    'name': f"Detectors: {cls.info[array_name]['name_long']}",
                    "show": False, 
                }
            },
            {
                'type': "overlay",
                "data": [{
                    "type": "polygon",
                    "data": list(zip(det_icrs.ra.degree[ei], det_icrs.dec.degree[ei])), 
                    }],
                "options": {
                    'name': f"FOV: {cls.info[array_name]['name_long']}",
                    "color": "#cc66cc",
                    "show": True,
                    "lineWidth": 8,
                    }
                },
            ])
        
        # tolteca.simu
        simrt = exec_config.get_simulator_runtime()
        simu_config = simrt.config
        simu_config_yaml = yaml_dump(simrt.config.to_config_dict())
        
        
        # use power loading model to infer the sensitivity
        # this is rough esitmate based on the mean altitude of the observation.
        tplm = simu_config.sources[0].get_power_loading_model()
        target_alt = bs_traj_data['target_alt']
        alt_mean = target_alt.mean()
        t_exp = bs_traj_data['t_exp']
        # for this purpose we generate the info for all the three arrays
        sens_coeff = np.sqrt(2.)
        sens_tbl = list()
        array_names = cls.info['array_names']

        for an in array_names:
            # TODO fix the api
            aplm = tplm._array_power_loading_models[an]
            result = {
                'array_name': an,
                'alt_mean': alt_mean,
                'P': aplm._get_P(alt_mean)
                } 
            result.update(aplm._get_noise(alt_mean, return_avg=True))
            result['dsens'] = sens_coeff * result['nefd'].to(u.mJy * u.s ** 0.5)
            sens_tbl.append(result)
        sens_tbl = QTable(rows=sens_tbl)
        logger.debug(f"summary table for all arrays:\n{sens_tbl}")

        # for the current array we get the mapping area from the cov map
        # and convert to mJy/beam if requested
        def _get_entry(an):
            return sens_tbl[sens_tbl['array_name'] == an][0]

        sens_entry = _get_entry(array_name)
        cov_data = cov_hdulist_s_per_pix[1].data
        cov_wcs = WCS(cov_hdulist_s_per_pix[1].header)
        cov_pixarea = cov_wcs.proj_plane_pixel_area()
        cov_max = cov_data.max()
        m_cov = (cov_data > 0.02 * cov_max)
        m_cov_01 = (cov_data > 0.1 * cov_max)
        map_area = (m_cov_01.sum() * cov_pixarea).to(u.deg ** 2)
        a_stddev = cls.info[array_name]['a_fwhm'] / GAUSSIAN_SIGMA_TO_FWHM
        b_stddev = cls.info[array_name]['b_fwhm'] / GAUSSIAN_SIGMA_TO_FWHM
        beam_area = 2 * np.pi * a_stddev * b_stddev
        beam_area_pix2 = (beam_area / cov_pixarea).to_value(u.dimensionless_unscaled)

        cov_data_mJy_per_beam = np.zeros(cov_data.shape, dtype='d')
        cov_data_mJy_per_beam[m_cov] = sens_coeff * sens_entry['nefd'] / np.sqrt(cov_data[m_cov] * beam_area_pix2)
        # calculate rms depth from the depth map
        depth_rms = np.median(cov_data_mJy_per_beam[m_cov_01]) << u.mJy
        # scale the depth rms to all arrays and update the sens tbl
        sens_tbl['depth_rms'] = depth_rms / sens_entry['nefd'] * sens_tbl['nefd']
        sens_tbl['t_exp'] = t_exp
        sens_tbl['map_area'] = map_area

        # make cov hdulist depending on the instru_data cov unit settings
        if exec_config.instru_data['coverage_map_type'] == 'depth':
            cov_hdulist = cov_hdulist_s_per_pix.copy()
            cov_hdulist[1].header['BUNIT'] = 'mJy / beam'
            cov_hdulist[1].data = cov_data_mJy_per_beam
        else:
            cov_hdulist = cov_hdulist_s_per_pix

        # from the cov image we can create a countour showing the outline of the observation
        # on the skyview
        cov_ctr = cov_hdulist[1].data.copy()
        cov_ctr[~m_cov] = 0
        im = cv2.normalize(
            src=cov_ctr, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        cxy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        cxy_s = cv2.approxPolyDP(cxy, 0.002 * cv2.arcLength(cxy, True), True).squeeze()
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
                    'name': f"Coverage Outline",
                    "color": "#66cccc",
                    "show": True,
                    "lineWidth": 4,
                    }
                },
            ])
 
        # create the layout to display the sensitivity info table 
        def _make_sens_tab_content(an):
            entry = _get_entry(an) 
            def _fmt(v):
                if isinstance(v, str):
                    return v
                return f'{v.value:.3g} {v.unit:unicode}'
            key_labels = {
                'array_name': 'Array Name',
                'alt_mean': 'Mean Alt.',
                't_exp': 'Total Exp. Time',
                'dsens': 'Detector Sens.',
                'map_area': 'Map Area',
                'depth_rms': 'Median RMS sens.'
            }
            data = {v: _fmt(entry[k]) for k, v in key_labels.items()}
            data['Coverage Map Unit'] = cov_hdulist[1].header['BUNIT']
            df = pd.DataFrame(data.items(), columns=['', ''])
            t = dbc.Table.from_dataframe(
                    df, striped=True, bordered=True, hover=True, className='mx-0 my-0')
            # get rid of the first child which is the header
            t.children = t.children[1:]
            return dbc.Card(
                [
                    dbc.CardBody(
                        t,
                        className='py-0 px-0',
                        style={'border-width': '0px'}
                    ),
                ],
            )
        sens_tbl_layout = dbc.Tabs(
            [
                dbc.Tab(
                    _make_sens_tab_content(an),
                    label=str(cls.info[an]['wl_center']),
                    tab_id=an,
                    activeTabClassName="fw-bold",
                    )
                for an in array_names 
                ],
            active_tab=array_name
            )
        
        # lmtot script export
        lmtot_exporter = LmtOTExporterConfig(save=False)
        lmtot_content = lmtot_exporter(simu_config)
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
                            'file': f"obsplanner_toltec_{array_name}_cov.fits",
                            },
                        'blob': cls._hdulist_to_base64(cov_hdulist),
                        }                
                    ],
                'lmtot': lmtot_content,
                'simu_config': simu_config_yaml,
                'info': sens_tbl_layout,
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
        array_name = ctx['array_name']
       
        # create the wcs
        pixscale = u.pixel_scale(4. << u.arcsec / u.pix)
        adaptive_pixscale_factor = 0.5  # the pixsize will be int factor of 2 arcsec.
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
        bs_im *= dt_smp.to_value(u.s)  # scale to coverage image of unit s / pix

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
            a_stddev = cls.info[array_name]['a_fwhm'] / GAUSSIAN_SIGMA_TO_FWHM
            b_stddev = cls.info[array_name]['b_fwhm'] / GAUSSIAN_SIGMA_TO_FWHM
            g = Gaussian2DKernel(
                    a_stddev.to_value(
                        u.pix, equivalencies=pixscale),
                    b_stddev.to_value(
                        u.pix, equivalencies=pixscale),
                   )
            cov_im = convolve_fft(cov_im, g, normalize_kernel=False)
        logger.debug(
                f'total exp time on coverage map: {(cov_im.sum() / det_im.sum() << u.s).to(u.min)}')
        logger.debug(
                f'total time of observation: {bs_traj_data["t_exp"].to(u.min)}')
        cov_hdr = wcsobj.to_header()
        cov_hdr['BUNIT'] = 's / pix'
        cov_hdr.append((
            "ARRAYNAM", array_name,
            "The name of the TolTEC array"))
        cov_hdr.append((
            "BAND", array_name,
            "The name of the TolTEC array"))
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


class ObsPlanner(ComponentTemplate):
    """An observation Planner."""
    # TODO we should be able to generate the __init__ function

    class Meta:
        component_cls = dbc.Container

    logger = get_logger()

    def __init__(
            self,
            raster_model_length_max=3 << u.deg,
            lissajous_model_length_max=20 << u.deg,
            t_exp_max=1 << u.hour,
            site_name='lmt',
            instru_name='toltec',
            pointing_catalog_path=None,
            presets_config_path=None,
            js9_config_path=None,
            title_text='Obs Planner',
            **kwargs):
        kwargs.setdefault('fluid', True)
        super().__init__(**kwargs)
        self._raster_model_length_max = raster_model_length_max,
        self._lissajous_model_length_max = lissajous_model_length_max,
        self._t_exp_max = t_exp_max,
        self._site = ObsSite.from_name(site_name)
        self._instru = ObsInstru.from_name(instru_name)
        self._pointing_catalog_path = pointing_catalog_path
        self._presets_config_path = presets_config_path
        self._js9_config_path = js9_config_path 
        self._title_text = title_text

    def setup_layout(self, app):

        # this is required to locate the js9
        djs9.setup_js9(app, config_path=self._js9_config_path)

        container = self
        header, body = container.grid(2, 1)
        # Header
        title_container = header.child(
            html.Div, className='d-flex align-items-baseline')
        title_container.child(html.H2(self._title_text, className='my-2'))
        app_details = title_container.child(
                CollapseContent(button_text='Details ...', className='ms-4')
            ).content
        app_details.child(html.Pre(pformat_yaml(self.__dict__)))
        header.child(html.Hr(className='mt-0 mb-3'))
        # Body
        controls_panel, results_panel = body.colgrid(1, 2, width_ratios=[1, 3])
        controls_panel.style = {
                    'width': '375px'
                    }
        controls_panel.parent.style = {
            'flex-wrap': 'nowrap'
            }
        # make the plotting area auto fill the available space.
        results_panel.style = {
            'flex-grow': '1',
            'flex-shrink': '1',
            }
        # Left panel, these are containers for the input controls
        target_container, mapping_container, \
            obssite_container, obsinstru_container = \
            controls_panel.colgrid(4, 1, gy=3)

        target_container = target_container.child(self.Card(
            title_text='Target')).body_container
        target_select_panel = target_container.child(
            ObsPlannerTargetSelect(site=self._site, className='px-0')
            )
        target_info_store = target_select_panel.info_store

        mapping_container = mapping_container.child(self.Card(
            title_text='Mapping')).body_container
        mapping_info_store = mapping_container.child(
            ObsPlannerMappingPresetsSelect(
                presets_config_path=self._presets_config_path,
                className='px-0')
            ).info_store

        # mapping execution
        exec_button_container = mapping_container.child(
            html.Div,
            className=(
                'd-flex justify-content-between align-items-start mt-2'
                )
            )
        exec_details = exec_button_container.child(
                CollapseContent(button_text='Details ...')
            ).content.child(
                html.Pre,
                'N/A',
                className='mb-0',
                style={
                    'font-size': '80%'
                    })

        # mapping execute button and execute result data store
        exec_button_disabled_color = 'primary'
        exec_button_enabled_color = 'danger'
        # this is to save some configs to clientside for enable/disable
        # the exec button.
        exec_button_config_store = exec_button_container.child(
            dcc.Store, data={
                'disabled_color': exec_button_disabled_color,
                'enabled_color': exec_button_enabled_color
                })
        exec_button = exec_button_container.child(
            dbc.Button,
            "Execute",
            size='sm',
            color=exec_button_disabled_color, disabled=True)
        exec_info_store = exec_button_container.child(dcc.Store)

        # site config panel
        obssite_title = f'Site: {self._site.display_name}'
        obssite_container = obssite_container.child(self.Card(
            title_text=obssite_title)).body_container
        site_info_store = self._site.make_controls(
            obssite_container).info_store

        # instru config panel
        obsinstru_title = f'Instrument: {self._instru.display_name}'
        obsinstru_container = obsinstru_container.child(self.Card(
            title_text=obsinstru_title)).body_container
        instru_info_store = self._instru.make_controls(
            obsinstru_container).info_store

        # Right panel, for plotting
        # mapping_plot_container, dal_container = \
        # dal_container, mapping_plot_container, js9_container = \
        #     plots_panel.colgrid(3, 1, gy=3)
        dal_container, results_controls_container, mapping_plot_container, instru_results_container = \
            results_panel.colgrid(4, 1, gy=3)
        
        instru_results_controls = self._instru.make_results_controls(results_controls_container, className='px-0 d-flex')

        mapping_plot_container.className = 'my-0'
        mapping_plot_collapse = instru_results_controls.child(
            CollapseContent(
                button_text='Show Trajs in Horizontal Coords...',
                button_props={
                    # 'color': 'primary',
                    'disabled': True,
                    'style': {
                        'text-transform': 'none',
                        }
                    },
                content=mapping_plot_container.child(dbc.Collapse, is_open=False, className='mt-3')),
            )
        mapping_plot_container = mapping_plot_collapse.content

        instru_results = self._instru.make_results_display(instru_results_container, className='px-0')

        mapping_plotter_loading = mapping_plot_container.child(
            dbc.Spinner,
            show_initially=False, color='primary',
            spinner_style={"width": "5rem", "height": "5rem"}
            )
        mapping_plotter = mapping_plotter_loading.child(
            ObsPlannerMappingPlotter(
                site=self._site,
            ))
            
        skyview = dal_container.child(
            dal.DashAladinLite,
            survey='P/DSS2/color',
            target='M1',
            fov=(10 << u.arcmin).to_value(u.deg),
            style={"width": "100%", "height": "40vh"},
            options={
                "showLayerBox": True,
                "showSimbadPointerControl": True,
                "showReticle": False,
                }
            )
        
        super().setup_layout(app)

        # connect the target name to the dal view
        app.clientside_callback(
            """
            function(target_info) {
                if (!target_info) {
                    return window.dash_clientside.no_update;
                }
                var ra = target_info.ra_deg;
                var dec = target_info.dec_deg;
                return ra + ' ' + dec
            }
            """,
            Output(skyview.id, "target"),
            Input(target_select_panel.target_info_store.id, 'data')
        )

        # this toggles the exec button enabled only when
        # user has provided valid mapping data and target data through
        # the forms
        app.clientside_callback(
            """
            function (mapping_data, target_data, btn_cfg) {
                var disabled_color = btn_cfg['disabled_color']
                var enabled_color = btn_cfg['enabled_color']
                if ((mapping_data === null) || (target_data === null)){
                    return [disabled_color, true]
                }
                return [enabled_color, false]
            }
            """,
            [
                Output(exec_button.id, 'color'),
                Output(exec_button.id, 'disabled'),
                ],
            [
                Input(mapping_info_store.id, 'data'),
                Input(target_info_store.id, 'data'),
                State(exec_button_config_store.id, 'data')
                ]
            )

        # this callback collects all data from info stores and
        # create the exec config. The exec config is then used
        # to make the traj data, which is consumed by the plotter
        # to create figure. The returned data is stored on
        # clientside and the graphs are updated with the figs

        instru_results_loading = instru_results.loading_indicators

        @app.callback(
            [
                Output(exec_info_store.id, 'data'),
                Output(mapping_plotter_loading.id, 'color'),
                ] + instru_results_loading['outputs'],
            [
                Input(exec_button.id, 'n_clicks'),
                State(mapping_info_store.id, 'data'),
                State(target_info_store.id, 'data'),
                State(site_info_store.id, 'data'),
                State(instru_info_store.id, 'data'),
                State(mapping_plotter_loading.id, 'color'),
                ] + instru_results_loading['states']
            )
        def make_exec_info_data(
                n_clicks, mapping_data, target_data, site_data, instru_data,
                mapping_loading, *instru_loading):
            """Collect data for the planned observation."""
            if mapping_data is None or target_data is None:
                return (None, mapping_loading) + instru_loading
            exec_config = ObsPlannerExecConfig.from_data(
                mapping_data=mapping_data,
                target_data=target_data,
                site_data=site_data,
                instru_data=instru_data)
            # generate the traj. Note that this is cached for performance.
            traj_data = _make_traj_data_cached(exec_config)

            # create figures to be displayed
            mapping_figs = mapping_plotter.make_figures(
                exec_config=exec_config,
                traj_data=traj_data)
            skyview_params = mapping_plotter.make_skyview_params(
                exec_config=exec_config,
                traj_data=traj_data
                )

            # copy the instru traj data results 
            instru_results = None
            if 'instru' in traj_data:
                instru_results = traj_data['instru'].get('results', None)

            # send the items to clientside for displaying 
            return (
                {
                    'mapping_figs': {
                        name: fig.to_dict()
                        for name, fig in mapping_figs.items()
                        },
                    "skyview_params": skyview_params,
                    'exec_config': exec_config.to_yaml_dict(),
                    # instru specific data, this is consumed by the instru results
                    # display
                    'instru': instru_results 
                    }, mapping_loading) + instru_loading
                
        app.clientside_callback(
            '''
            function(exec_info) {
                if (!exec_info) {
                    return true;
                }
                return false;
            }
            ''',
            Output(mapping_plot_collapse.button.id, 'disabled'),
            [
                Input(exec_info_store.id, 'data'),
                ]
            )

        app.clientside_callback(
            '''
            function(exec_info) {
                // console.log(exec_info)
                return JSON.stringify(
                    exec_info && exec_info.exec_config,
                    null,
                    2
                    );
            }
            ''',
            Output(exec_details.id, 'children'),
            [
                Input(exec_info_store.id, 'data'),
                ]
            )
        
        # update the sky map layers
        # with the traj data
        app.clientside_callback(
            '''
            function(exec_info) {
                // console.log(exec_info);
                if (!exec_info) {
                    return Array(2).fill(window.dash_clientside.no_update);
                }
                var svp = exec_info.skyview_params;
                var fov = svp.fov || window.dash_clientside.no_update;
                var layers = svp.layers || window.dash_clientside.no_update;
                return [fov, layers]
            }
            ''',
            [
                Output(skyview.id, 'fov'),
                Output(skyview.id, 'layers'),
                ],
            [
                Input(exec_info_store.id, 'data'),
                ]
            )

        # connect exec info with plotter
        mapping_plotter.make_mapping_plot_callbacks(
            app, exec_info_store_id=exec_info_store.id)
        
        # connect exec info with instru results
        instru_results.make_callbacks(
            app, exec_info_store_id=exec_info_store.id
            )
        instru_results_controls.make_callbacks(
            app, exec_info_store_id=exec_info_store.id
            )

    class Card(ComponentTemplate):
        class Meta:
            component_cls = dbc.Card

        def __init__(self, title_text, **kwargs):
            super().__init__(**kwargs)
            container = self
            container.child(html.H6(title_text, className='card-header'))
            self.body_container = container.child(dbc.CardBody)


def _collect_mapping_config_tooltips():
    # this function generates a dict for tooltip string used for mapping
    # pattern config fields.
    logger = get_logger()
    result = dict()
    for key, info in mapping_registry._register_info.items():
        # get the underlying entry key and value schema
        s = info['dispatcher_schema']
        tooltips = dict()
        mapping_type = key
        for key_schema, value_schema in s._schema.items():
            item_key = key_schema._schema
            desc = key_schema.description
            tooltips[item_key] = desc
        result[mapping_type] = tooltips
    logger.debug(
        f"collected tooltips for mapping configs:\n{pformat_yaml(result)}")
    result['common'] = {
        't_exp': 'Exposure time of the observation.'
        }
    return result


@add_schema
@dataclass
class ObsPlannerExecConfig(object):
    """A class for obs planner execution config.
    """
    # this class looks like the SimuConfig but it is actually independent
    # from it. The idea is that the obs planner template will define
    # components that fill in this object. And all actual planning functions
    # happens by consuming this object.
    mapping: dict = field(
        metadata={
            'description': "The simulator mapping trajectory config.",
            'schema': mapping_registry.schema,
            'pformat_schema_type': f'<{mapping_registry.name}>'
            }
        )
    obs_params: ObsParamsConfig = field(
        metadata={
            'description': 'The dict contains the observation parameters.',
            })
    # TODO since we now only support LMT/TolTEC, we do not have a
    # good measure of the schema to use for the generic site and instru
    # we just save the raw data dict here.
    site_data: Union[None, dict] = field(
        default=None,
        metadata={
            'description': "The data dict for the site.",
            'schema': Or(dict, None)
            }
        )
    instru_data: Union[None, dict] = field(
        default=None,
        metadata={
            'description': "The data dict for the instrument.",
            'schema': Or(dict, None)
            }
        )

    @classmethod
    def from_data(
            cls, mapping_data, target_data, site_data=None, instru_data=None):
        mapping_dict = cls._make_mapping_config_dict(mapping_data, target_data)
        obs_params_dict = {
            'f_smp_mapping': '10 Hz',
            'f_smp_probing': '100 Hz',
            't_exp': mapping_data.pop('t_exp', None)
            }
        return cls.from_dict({
            'mapping': mapping_dict,
            'obs_params': obs_params_dict,
            'site_data': site_data,
            'instru_data': instru_data,
            })

    @staticmethod
    def _make_mapping_config_dict(mapping_data, target_data):
        '''Return mapping config dict from mapping and target data store.'''
        cfg = dict(**mapping_data)
        cfg['t0'] = f"{target_data['date']} {target_data['time']}"
        cfg['target'] = target_data['name']
        # for mapping config, we discard the t_exp.
        cfg.pop("t_exp", None)
        return cfg

    def to_yaml_dict(self):
        # this differes from to_dict in that it only have basic serializable types.
        return yaml_load(StringIO(yaml_dump(self.to_dict())))

    def get_simulator_runtime(self):
        """Return a simulator runtime object from this config."""
        # dispatch site/instru to create parts of the simu config dict
        if self.site_data['name'] == 'lmt' and self.instru_data['name'] == 'toltec':
            simu_cfg = self._make_lmt_toltec_simu_config_dict(
                lmt_data=self.site_data, toltec_data=self.instru_data)
        else:
            raise ValueError("unsupported site/instruments for simu.")
        # add to simu cfg the mapping and obs_params
        exec_cfg = self.to_yaml_dict()
        rupdate(simu_cfg, {
            'simu': {
                'obs_params': exec_cfg['obs_params'],
                'mapping': exec_cfg['mapping'],
                }
            })
        return SimulatorRuntime(simu_cfg)

    @staticmethod
    def _make_lmt_toltec_simu_config_dict(lmt_data, toltec_data):
        """Return simu config dict segment for LMT TolTEC."""
        atm_model_name = lmt_data['atm_model_name']
        return {
            'simu': {
                'jobkey': 'obs_planner_simu',
                'obs_params': {
                    'f_smp_probing': '122 Hz',
                    'f_smp_mapping': '20 Hz'
                    },
                'instrument': {
                    'name': 'toltec',
                    },
                'sources': [
                    {
                        'type': 'toltec_power_loading',
                        'atm_model_name': atm_model_name,
                        'atm_cache_dir': None,
                        },
                    ],
                }
            }

    def __hash__(self):
        # allow this object to be hashed so we can cache it to avoid
        # repeated calculation
        exec_cfg_yaml = yaml_dump(self.to_dict())
        return exec_cfg_yaml.__hash__()

    @timeit
    def make_traj_data(self):
        logger = get_logger()
        logger.info(f'make traj data for {pformat_yaml(self.to_dict())}')
        if self.site_data is None:
            logger.warning("no site data found for generating trajectory.")
            return None
        # get observer from site name
        observer = ObsSite.get_observer(self.site_data['name'])
        logger.debug(f"observer: {observer}")

        mapping_model = self.mapping.get_model(observer=observer)

        t_exp = self.obs_params.t_exp or mapping_model.t_pattern
        dt_smp_s = (
            1. / self.obs_params.f_smp_mapping).to_value(u.s)
        t = np.arange(0, t_exp.to_value(u.s) + dt_smp_s, dt_smp_s) << u.s
        n_pts = t.size
        # ensure we at least have 100 points
        if n_pts < 100:
            n_pts = 100
            t = np.linspace(0, t_exp.to_value(u.s), n_pts) << u.s
        logger.debug(f"create {n_pts} sampling points for t_exp={t_exp}")

        # this is to show the mapping in offset unit
        offset_model = mapping_model.offset_mapping_model
        dlon, dlat = offset_model(t)
        time_obs = mapping_model.t0 + t

        # then evaluate the mapping model in mapping.ref_frame
        # to get the bore sight coords
        bs_coords = mapping_model.evaluate_coords(t)

        # and we can convert the bs_coords to other frames if needed
        altaz_frame = resolve_sky_coords_frame(
            'altaz', observer=observer, time_obs=time_obs
            )
        # this interpolator can speeds things up a bit
        erfa_interp_len = 300. << u.s
        with erfa_astrom.set(ErfaAstromInterpolator(erfa_interp_len)):
            # and we can convert the bs_coords to other frames if needed
            bs_coords_icrs = bs_coords.transform_to('icrs')

            bs_coords_altaz = bs_coords.transform_to(altaz_frame)
            # also for the target coord
            target_coord = self.mapping.target_coord
            target_coords_altaz = target_coord.transform_to(altaz_frame)

        bs_sky_bbox_icrs = SkyBoundingBox.from_lonlat(
            bs_coords_icrs.ra,
            bs_coords_icrs.dec,
            )
        bs_traj_data = {
            't_exp': t_exp,
            'time_obs': time_obs,
            'dlon': dlon,
            'dlat': dlat,
            'ra': bs_coords_icrs.ra,
            'dec': bs_coords_icrs.dec,
            'az': bs_coords_altaz.az,
            'alt': bs_coords_altaz.alt,
            'target_az': target_coords_altaz.az,
            'target_alt': target_coords_altaz.alt,
            "sky_bbox_icrs": bs_sky_bbox_icrs,
            }
        # make instru specific traj data
        obsinstru = ObsInstru.from_name(self.instru_data['name'])
        instru_traj_data = obsinstru.make_traj_data(self, bs_traj_data)

        return {
            'site': bs_traj_data,
            'instru': instru_traj_data,
            }


@functools.lru_cache(maxsize=8)
def _make_traj_data_cached(exec_config):
    return exec_config.make_traj_data()


class ObsPlannerMappingPresetsSelect(ComponentTemplate):

    class Meta:
        component_cls = dbc.Container

    logger = get_logger()

    # we inspect the simu mapping patterns to collect default
    # tooltip strings for mapping presets
    _mapping_config_tooltips = _collect_mapping_config_tooltips()

    def __init__(self, presets_config_path=None, t_exp_max=1 << u.h, **kwargs):
        kwargs.setdefault('fluid', True)
        super().__init__(**kwargs)
        with open(presets_config_path, 'r') as fo:
            self._presets = PresetsConfig.from_dict(yaml_load(fo))
        self._t_exp_max = t_exp_max
        container = self
        self._info_store = container.child(dcc.Store)

    @property
    def info_store(self):
        return self._info_store

    @property
    def presets(self):
        return self._presets

    def make_mapping_preset_options(self):
        result = []
        for preset in self.presets:
            result.append({
                'label': preset.label,
                'value': preset.key
                })
        return result

    def setup_layout(self, app):
        container = self
        controls_form = container.child(dbc.Form)
        controls_form_container = controls_form.child(
            dbc.Row, className='gx-2 gy-2')
        # mapping_preset_select = controls_form_container.child(
        #     LabeledDropdown(
        #         label_text='Mapping Pattern',
        #         # className='w-auto',
        #         size='sm',
        #         placeholder='Select a mapping pattern template ...'
        #         )).dropdown
        mapping_preset_select = controls_form_container.child(
            dbc.Select,
            size='sm',
            placeholder='Choose a mapping pattern template to edit ...',
            )
        mapping_preset_feedback = controls_form_container.child(
            dbc.FormFeedback)
        mapping_preset_select.options = self.make_mapping_preset_options()
        mapping_preset_tooltip = controls_form_container.child(
            html.Div)
        mapping_preset_form_container = controls_form_container.child(
            html.Div)
        mapping_preset_data_store = controls_form_container.child(dcc.Store)
        mapping_ref_frame_select = controls_form_container.child(
            LabeledChecklist(
                label_text='Mapping Reference Frame',
                className='w-auto',
                size='sm',
                # set to true to allow multiple check
                multi=False,
                checklist_props={
                    'options': [
                        {
                            'label': "AZ/Alt",
                            'value': "altaz",
                            },
                        {
                            'label': "RA/Dec",
                            'value': "icrs",
                            }
                        ],
                    'value': 'altaz'
                    }
                )).checklist
        super().setup_layout(app)
        self.make_mapping_preset_form_callbacks(
            app, parent_id=mapping_preset_form_container.id,
            preset_select_id=mapping_preset_select.id,
            datastore_id=mapping_preset_data_store.id
            )

        @app.callback(
            [
                Output(mapping_preset_select.id, 'valid'),
                Output(mapping_preset_select.id, 'invalid'),
                Output(mapping_preset_feedback.id, 'children'),
                Output(mapping_preset_feedback.id, 'type'),
                ],
            [
                Input(self.info_store.id, 'data')
                ]
            )
        def validate_mapping(mapping_data):
            if mapping_data is None:
                return [False, True, 'Invalid mapping settings.', 'invalid']
            # we use some fake data to initialize the mapping config object
            target_data = {
                'name': '180d 0d',
                'date': '2022-02-02',
                'time': '06:00:00'
                }
            # this should never fail since both mapping_data and the
            # target_data should be always correct at this point
            exec_config = ObsPlannerExecConfig.from_data(
                mapping_data, target_data
                )
            mapping_config = exec_config.mapping
            t_pattern = mapping_config.get_offset_model().t_pattern
            # for those have t_exp, we report the total exposure time
            if 't_exp' in mapping_data:
                t_exp = mapping_data['t_exp']
                feedback_content = (
                    f' Total exposure time: {t_exp}. '
                    f'Time to finish one pass: {t_pattern:.2f}.')
                return [True, False, feedback_content, 'valid']
            # raster like patterns, we check the exp time to make sure
            # it is ok
            t_exp_max = self._t_exp_max
            if t_pattern > self._t_exp_max:
                feedback_content = (
                    f'The pattern takes {t_pattern:.2f} to finish, which '
                    f'exceeds the required maximum value of {t_exp_max}.'
                    )
                return [
                    False, True, feedback_content, 'invalid'
                    ]
            # then we just report the exposure time
            feedback_content = (
                f'Time to finish the pattern: {t_pattern:.2f}.')
            return [True, False, feedback_content, 'valid']

        app.clientside_callback(
            """
            function(ref_frame_value, preset_data) {
                if (preset_data === null) {
                    return null
                }
                data = {...preset_data}
                data['ref_frame'] = ref_frame_value
                return data
            }
            """,
            output=Output(self.info_store.id, 'data'),
            inputs=[
                Input(mapping_ref_frame_select.id, 'value'),
                Input(mapping_preset_data_store.id, 'data')
                ])

        @app.callback(
            Output(mapping_preset_tooltip.id, 'children'),
            [
                Input(mapping_preset_select.id, 'value')
                ]
            )
        def make_preset_tooltip(preset_name):
            if preset_name is None:
                raise PreventUpdate
            preset = self._presets.get(type='mapping', key=preset_name)
            header_text = preset.description
            if header_text is None:
                type = preset.get_data_item('type').value
                header_text = f'mapping type: {type}'
            content_text = preset.description_long
            if content_text is not None:
                content_text = html.P(content_text)
            return dbc.Popover([
                    dbc.PopoverBody(header_text),
                    dbc.PopoverBody(content_text)
                ],
                target=mapping_preset_select.id,
                trigger='hover'
                )

    def make_mapping_preset_form_callbacks(
            self, app,
            parent_id, preset_select_id, datastore_id):
        # here we create dynamic layout for given selected preset
        # and define pattern matching layout to collect values
        # we turn off auto_index because we'll use the unique key
        # to identify each field
        pmid = PatternMatchingId(
            container_id=parent_id, preset_name='', key='', auto_index=False)

        # the callback to make preset form
        @app.callback(
            Output(parent_id, 'children'),
            [
                Input(preset_select_id, 'value'),
                ],
            # prevent_initial_call=True
            )
        def make_mapping_preset_layout(preset_name):
            if preset_name is None:
                return None
            self.logger.debug(
                f"generate form for mapping preset {preset_name}")
            preset = self._presets.get(type='mapping', key=preset_name)
            container = NullComponent(id=parent_id)
            form_container = container.child(dbc.Form).child(
                        dbc.Row, className='gx-2 gy-2')
            # get default tooltip for mapping preset
            mapping_type = preset.get_data_item('type').value
            tooltips = self._mapping_config_tooltips[mapping_type]
            tooltips.update(self._mapping_config_tooltips['common'])
            for entry in preset.data:
                # make pattern matching id for all fields
                vmin = entry.value_min
                if entry.component_type == 'number':
                    if vmin is None:
                        vmin = '-∞'
                    vmax = entry.value_max
                    if vmax is None:
                        vmax = '+∞'
                    placeholder = f'{entry.value} [{vmin}, {vmax}]'
                else:
                    placeholder = f'{entry.value}'
                input_props = {
                    'type': entry.component_type,
                    'id': pmid(preset_name=preset_name, key=entry.key),
                    'value': entry.value,
                    'min': entry.value_min,
                    'max': entry.value_max,
                    'placeholder': placeholder,
                    # 'style': {
                    #     'max-width': '5rem'
                    #     }
                    }
                component_kw = {
                    'label_text': entry.label or entry.key,
                    'suffix_text': None if not entry.unit else entry.unit,
                    'size': 'sm',
                    'className': 'w-100',
                    'input_props': input_props,
                    }
                rupdate(component_kw, entry.component_kw)
                # TODO use custom BS5 to set more sensible breakpoints.
                entry_input = form_container.child(
                    dbc.Col, xxl=12, width=12).child(
                        LabeledInput(**component_kw)).input
                # tooltip
                tooltip_text = entry.description
                if not tooltip_text:
                    tooltip_text = tooltips[entry.key]
                tooltip_text += f' Preset default: {placeholder}'
                if entry.unit is not None:
                    tooltip_text += f' {entry.unit}'
                form_container.child(
                    dbc.Tooltip, tooltip_text,
                    target=entry_input.id)

            return container.layout

        # the callback to collect form data
        # TODO may be make this as clientside
        @app.callback(
            Output(datastore_id, 'data'),
            [
                Input(pmid(
                    container_id=parent_id,
                    preset_name=dash.ALL,
                    key=dash.ALL), 'value'),
                State(pmid(
                    container_id=parent_id,
                    preset_name=dash.ALL,
                    key=dash.ALL), 'invalid'),
                ]
            )
        def collect_data(input_values, invalid_values):
            logger = get_logger()
            logger.debug(f'input_values: {input_values}')
            logger.debug(f'invalid_values: {invalid_values}')
            if not input_values or any((i is None for i in input_values)) \
                    or any(invalid_values):
                # invalid form
                return None
            # get keys from callback context
            result = dict()
            inputs = dash.callback_context.inputs_list[0]
            # get the preset data and the fields dict
            preset_name = inputs[0]['id']['preset_name']
            preset = self._presets.get('mapping', preset_name)

            # here we extract the mapping config dict from preset data
            for input_ in inputs:
                key = input_['id']['key']
                item = preset.get_data_item(key)
                value = input_['value']
                value_text = f'{value} {item.unit or ""}'.strip()
                result[key] = value_text
            return result


class ObsPlannerTargetSelect(ComponentTemplate):

    class Meta:
        component_cls = dbc.Container

    logger = get_logger()

    def __init__(self, site, target_alt_min=20 << u.deg, **kwargs):
        kwargs.setdefault('fluid', True)
        super().__init__(**kwargs)
        self._site = site
        self._target_alt_min = target_alt_min
        container = self
        self._target_info_store = container.child(dcc.Store)
        self._info_store = container.child(dcc.Store)

    @property
    def target_info_store(self):
        return self._target_info_store

    @property
    def info_store(self):
        return self._info_store

    def setup_layout(self, app):
        container = self
        controls_form_container = container.child(dbc.Form).child(
                    dbc.Row, className='gx-2 gy-2')
        target_name_container = controls_form_container.child(html.Div)
        target_name_input = target_name_container.child(
            dbc.Input,
            size='sm',
            type='text',
            placeholder=(
                        'M1, NGC 1277, 17.7d -2.1d, 10h09m08s +20d19m18s'),
            value='M1',
            # autofocus=True,
            debounce=True,
            )
        target_name_feedback = target_name_container.child(
            dbc.FormFeedback, type='valid')
        target_name_container.child(
            dbc.Tooltip,
            """
Enter source name or coordinates. Names are sent to a name lookup server
for resolving the coordinates. Coordinates should be like "17.7d -2.1d"
or "10h09m08s +20d19m18s".
            """,
            target=target_name_input.id)


        # date picker
        date_picker_container = controls_form_container.child(html.Div)
        date_picker_group = date_picker_container.child(
            LabeledInput(
                label_text='Obs Date',
                className='w-auto',
                size='sm',
                input_props={
                    # these are the dbc.Input kwargs
                    'type': 'text',
                    'min': '1990-01-01',
                    'max': '2099-12-31',
                    'placeholder': 'yyyy-mm-dd',
                    # 'style': {
                    #     'flex': '0 1 7rem'
                    #     },
                    'debounce': True,
                    'value': '2022-01-01',
                    'pattern': r'[1-2]\d\d\d-[0-1]\d-[0-3]\d',
                    },
                ))
        date_picker_input = date_picker_group.input
        date_picker_container.child(
            dbc.Tooltip,
            f"""
The date of observation. The visibility report is done by checking if
the target elevation is > {self._target_alt_min} during the night time.
            """,
            target=date_picker_input.id)
        date_picker_feedback = date_picker_group.feedback

        # time picker
        time_picker_container = controls_form_container.child(html.Div)
        time_picker_group = time_picker_container.child(
            LabeledInput(
                label_text='Obs Start Time (UT)',
                className='w-auto',
                size='sm',
                input_props={
                    # these are the dbc.Input kwargs
                    'type': 'text',
                    'placeholder': 'HH:MM:SS',
                    # 'style': {
                    #     'flex': '0 1 7rem'
                    #     },
                    'debounce': True,
                    'value': '06:00:00',
                    'pattern': r'[0-1]\d:[0-5]\d:[0-5]\d',
                    },
                ))
        time_picker_input = time_picker_group.input
        time_picker_container.child(
            dbc.Tooltip,
            f"""
The time of observation. The target has to be at
elevation > {self._target_alt_min} during the night.
            """,
            target=time_picker_input.id)
        time_picker_feedback = time_picker_group.feedback
        check_button_container = container.child(
            html.Div,
            className=(
                'd-flex justify-content-end mt-2'
                )
            )
        check_button = check_button_container.child(
            dbc.Button, 'Plot Alt. vs Time',
            color='primary', size='sm',
            )
        check_result_modal = check_button_container.child(
            dbc.Modal, is_open=False, centered=False)

        self._site.make_info_display(container)

        super().setup_layout(app)

        @app.callback(
            [
                Output(self.target_info_store.id, 'data'),
                Output(target_name_input.id, 'valid'),
                Output(target_name_input.id, 'invalid'),
                Output(target_name_feedback.id, 'children'),
                Output(target_name_feedback.id, 'type'),
                ],
            [
                Input(target_name_input.id, 'value'),
             ]
            )
        def resolve_target(name):
            logger = get_logger()
            logger.info(f"resolve target name {name}")
            if not name:
                return (
                    None,
                    False, True,
                    'Enter the name or coordinate of target.',
                    'invalid'
                    )
            try:
                coord = parse_coordinates(name)
                coord_text = (
                    f'{coord.ra.degree}d '
                    f'{coord.dec.degree}d (J2000)'
                    )
                return (
                    {
                        'ra_deg': coord.ra.degree,
                        'dec_deg': coord.dec.degree,
                        'name': name,
                     },
                    True, False,
                    f'Target coordinate resolved: {coord_text}.',
                    'valid'
                    )
            except Exception as e:
                logger.debug(f'error parsing {name}', exc_info=True)
                return (
                    None,
                    False, True,
                    f'Unable to resolve target: {e}',
                    'invalid'
                    )

        obs_constraints = [
            AltitudeConstraint(self._target_alt_min, 91*u.deg),
            AtNightConstraint()
            ]
        observer = self._site.observer

        @app.callback(
            [
                Output(date_picker_input.id, 'valid'),
                Output(date_picker_input.id, 'invalid'),
                Output(date_picker_feedback.id, 'children'),
                Output(date_picker_feedback.id, 'type'),
                ],
            [
                Input(date_picker_input.id, 'value'),
                Input(self.target_info_store.id, 'data')
             ]
            )
        def validate_date(date_value, data):
            logger = get_logger()
            if data is None:
                return [False, True, "", 'invalid']
            try:
                t0 = Time(date_value)
            except ValueError:
                return (
                    False, True, 'Invalid Date. Use yyyy-mm-dd.', 'invalid')
            target_coord = SkyCoord(
                ra=data['ra_deg'] << u.deg, dec=data['dec_deg'] << u.deg)
            target = FixedTarget(name=data['name'], coord=target_coord)
            time_grid = t0 + (np.arange(0, 24, 0.5) << u.h)
            summary = observability_table(
                    obs_constraints, observer, [target], times=time_grid)
            logger.info(f'Visibility of targets on day of {t0}\n{summary}')
            ever_observable = summary['ever observable'][0]
            if ever_observable:
                target_uptime = (
                    summary['fraction of time observable'][0] * 24) << u.h
                t_mt = observer.target_meridian_transit_time(
                    t0, target, n_grid_points=48)
                feedback_content = (
                    f"Total up-time: {target_uptime:.1f}. "
                    f"Highest at {t_mt.datetime.strftime('UT %H:%M:%S')}.")
                return (
                    ever_observable,
                    not ever_observable,
                    feedback_content, 'valid'
                    )
            feedback_content = (
                'Target is not up at night. Pick another date.')
            return (
                ever_observable,
                not ever_observable,
                feedback_content,
                'invalid'
                )

        @app.callback(
            [
                Output(time_picker_input.id, 'valid'),
                Output(time_picker_input.id, 'invalid'),
                Output(time_picker_feedback.id, 'children'),
                Output(time_picker_feedback.id, 'type'),
                ],
            [
                Input(time_picker_input.id, 'value'),
                Input(date_picker_input.id, 'value'),
                Input(self.target_info_store.id, 'data'),
                ]
            )
        def validate_time(time_value, date_value, data):
            if data is None:
                return (False, True, '', 'invalid')
            # verify time value only.
            try:
                _ = Time(f'2000-01-01 {time_value}')
            except ValueError:
                return (
                    False, True, 'Invalid time. Use HH:MM:SS.', 'invalid')
            # verify target availability
            t0 = Time(f'{date_value} {time_value}')
            if not observer.is_night(t0):
                sunrise_time_str = observer.sun_rise_time(
                    t0, which="previous").iso
                sunset_time_str = observer.sun_set_time(
                    t0, which="next").iso
                feedback_content = (
                    f'The time entered is not at night. Sunrise: '
                    f'{sunrise_time_str}. '
                    f'Sunset: {sunset_time_str}'
                    )
                return (False, True, feedback_content, 'invalid')
            target_coord_icrs = SkyCoord(
                ra=data['ra_deg'] << u.deg, dec=data['dec_deg'] << u.deg
                )
            altaz_frame = observer.altaz(time=t0)
            target_coord_altaz = target_coord_icrs.transform_to(altaz_frame)
            target_az = target_coord_altaz.az
            target_alt = target_coord_altaz.alt
            alt_min = self._target_alt_min
            if target_alt < self._target_alt_min:
                feedback_content = (
                        f'Target at Az = {target_az.degree:.4f}d '
                        f'Alt ={target_alt.degree:.4f}d '
                        f'is too low (< {alt_min}) to observer. '
                        f'Pick another time.'
                    )
                return (False, True, feedback_content, 'invalid')
            feedback_content = (
                f'Target Az = {target_az.degree:.4f}d '
                f'Alt = {target_alt.degree:.4f}d.'
                )
            return (
                True, False,
                feedback_content,
                'valid'
                )

        @app.callback(
            [
                Output(check_result_modal.id, 'children'),
                Output(check_result_modal.id, 'is_open'),
                ],
            [
                Input(check_button.id, 'n_clicks'),
                State(date_picker_input.id, 'value'),
                State(time_picker_input.id, 'value'),
                State(self.target_info_store.id, 'data'),
                ],
            prevent_initial_call=True,
            )
        def check_visibility(n_clicks, date_value, time_value, target_data):
            # composing a dummy exec config object so we can call
            # the plotter
            def make_output(content):
                return [dbc.ModalBody(content), True]

            if target_data is None:
                return make_output('Invalid target format.')
            t_exp = 0 << u.min
            try:
                t0 = Time(f'{date_value}')
            except ValueError:
                return make_output("Invalid date format.")
            try:
                t0 = Time(f'{date_value} {time_value}')
                # this will show the target position
                t_exp = 2 << u.min
            except ValueError:
                pass
            plotter = ObsPlannerMappingPlotter(site=self._site)
            target_coord = SkyCoord(
                f"{target_data['ra_deg']} {target_data['dec_deg']}",
                unit=u.deg, frame='icrs')
            exec_config = Namespace(
                obs_params=Namespace(t_exp=t_exp),
                mapping=Namespace(
                    t0=t0,
                    target_coord=target_coord
                    )
                )
            fig = plotter._plot_visibility(exec_config)
            return make_output(dcc.Graph(figure=fig))

        app.clientside_callback(
            """
            function(
                date_value, date_valid,
                time_value, time_valid,
                target_data, target_valid) {
                if (date_valid && time_valid && target_valid) {
                    data = {...target_data}
                    data['date'] = date_value
                    data['time'] = time_value
                    return data
                }
                return null
            }
            """,
            output=Output(self.info_store.id, 'data'),
            inputs=[
                Input(date_picker_input.id, 'value'),
                Input(date_picker_input.id, 'valid'),
                Input(time_picker_input.id, 'value'),
                Input(time_picker_input.id, 'valid'),
                Input(self.target_info_store.id, 'data'),
                Input(target_name_input.id, 'valid'),
                ])


class ObsPlannerMappingPlotter(ComponentTemplate):

    class Meta:
        component_cls = dbc.Container

    logger = get_logger()

    def __init__(self, site, target_alt_min=20 << u.deg, **kwargs):
        kwargs.setdefault('fluid', True)
        super().__init__(**kwargs)
        self._site = site
        self._target_alt_min = target_alt_min

        container = self
        self._graphs = [
            c.child(dcc.Graph, figure=self._make_empty_figure())
            for c in container.colgrid(1, 4,).ravel()
            ]

    def make_mapping_plot_callbacks(self, app, exec_info_store_id):
        # update graph with figures in exec_info_store
        app.clientside_callback(
            _j2env.from_string("""
            function(exec_data) {
                if (exec_data === null) {
                    return Array(4).fill({{empty_fig}});
                }
                figs = exec_data['mapping_figs']
                return [
                    figs["visibility"],
                    figs["offset"],
                    figs["altaz"],
                    figs["icrs"],
                    ]
            }
            """).render(empty_fig=json.dumps(self._make_empty_figure())),
            [
                Output(graph.id, 'figure')
                for graph in self._graphs
                ],
            [
                Input(exec_info_store_id, 'data')
                ]
            )

    @timeit
    def make_figures(self, exec_config, traj_data):

        visibility_fig = self._plot_visibility(exec_config)
        mapping_figs = self._plot_mapping_pattern(
            exec_config, traj_data)
        figs = dict(**mapping_figs, visibility=visibility_fig)
        return figs

    @timeit
    def make_skyview_params(self, exec_config, traj_data):
        # return the dict that setup the skyview.
        mapping_config = exec_config.mapping
        bs_traj_data = traj_data['site']
        instru_traj_data = traj_data['instru']
        bs_sky_bbox_icrs = SkyBoundingBox.from_lonlat(
            bs_traj_data['ra'],
            bs_traj_data['dec'],
            )
        fov_sky_bbox = bs_sky_bbox_icrs
        if instru_traj_data is not None:
            # figure out instru overlay layout bbox 
            instru_sky_bbox_icrs = SkyBoundingBox.from_lonlat(
                instru_traj_data['ra'],
                instru_traj_data['dec'],
                )
            fov_sky_bbox = fov_sky_bbox.pad_with(
                instru_sky_bbox_icrs.width,
                instru_sky_bbox_icrs.height,
                )
        fov = max(
                fov_sky_bbox.width, fov_sky_bbox.height).to_value(u.deg)
        # set the view fov larger
        fov = fov / 0.618
        layers = list()
        # the mapping pattern layer
        layers.append(
            {
                'type': "overlay",
                "data": [{
                    "data": list(zip(
                        bs_traj_data['ra'].degree,
                        bs_traj_data['dec'].degree)),
                    "type": "polyline",
                    "color": "red",
                    "lineWidth": 1,
                    }],
                "options": {
                    'name': f"Mapping Trajectory",
                    "color": "red",
                    "show": True,
                    }
                },
            )
        if instru_traj_data is not None:
            layers.extend(instru_traj_data["skyview_layers"])
        params = dict({
            "target": exec_config.mapping.target,
            "fov": fov,
            "layers": layers,
            "options": {
                "showLayerBox": True
                }
            })
        return params

    @staticmethod
    def _make_day_grid(day_start):
        day_grid = day_start + (np.arange(0, 24 * 60 + 1) << u.min)
        return day_grid

    @functools.lru_cache
    def _get_target_coords_altaz_for_day(self, target_coord_str, day_start):
        day_grid = self._make_day_grid(day_start)
        observer = self._site.observer
        return SkyCoord(target_coord_str).transform_to(
            observer.altaz(time=day_grid))

    fig_layout_default = {
        'xaxis': dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='black',
            linewidth=4,
            ticks='outside',
            ),
        'yaxis': dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='black',
            linewidth=4,
            ticks='outside',
            ),
        'plot_bgcolor': 'white',
        'margin': dict(
            autoexpand=True,
            l=0,
            r=10,
            b=0,
            t=10,
            ),
        'modebar': {
            'orientation': 'v',
            },
        }

    def _make_empty_figure(self):
        return {
            "layout": {
                "xaxis": {
                    "visible": False
                    },
                "yaxis": {
                    "visible": False
                    },
                # "annotations": [
                #     {
                #         "text": "No matching data found",
                #         "xref": "paper",
                #         "yref": "paper",
                #         "showarrow": False,
                #         "font": {
                #             "size": 28
                #             }
                #         }
                #     ]
                }
            }

    @timeit
    def _plot_visibility(self, exec_config):
        logger = get_logger()
        mapping_config = exec_config.mapping
        obs_params = exec_config.obs_params
        # highlight the obstime and t_exp
        t_exp = obs_params.t_exp
        if t_exp is None:
            t_exp = mapping_config.get_offset_model().t_pattern

        # make a plot of the uptimes for given mapping_config
        observer = self._site.observer
        t0 = mapping_config.t0
        t1 = t0 + t_exp
        day_start = Time(int(t0.mjd), format='mjd')
        day_grid = self._make_day_grid(day_start)
        # day_end = day_start + (24 << u.h)

        target_coord = mapping_config.target_coord
        target_coords_altaz_for_day = self._get_target_coords_altaz_for_day(
            target_coord.to_string('hmsdms'), day_start)
        # target_name = mapping_config.target
        t_sun_rise = observer.sun_rise_time(day_start, which='next')
        t_sun_set = observer.sun_set_time(day_start, which='next')

        # since day_grid is sorted we can use bisect to locate index
        # in the time grid.
        # this is index right before sunrise
        i_sun_rise = bisect.bisect_left(day_grid, t_sun_rise) - 1
        # this is index right after sunset
        i_sun_set = bisect.bisect_right(day_grid, t_sun_set)
        # index for exposure time period
        i_t0 = bisect.bisect_left(day_grid, t0) - 1
        i_t1 = bisect.bisect_right(day_grid, t1)

        logger.debug(
            f'sun_rise: {t_sun_rise.iso}\n'
            f'sun_set: {t_sun_set.iso}\n'
            f'{i_sun_rise=} {i_sun_set=} '
            f'{i_t0=} {i_t1=}')
        # split the data into three segments at break points b0 and b1
        if i_sun_rise < i_sun_set:
            # middle is daytime
            b0 = i_sun_rise
            b1 = i_sun_set
            seg_is_daytime = [False, True, False]
        else:
            # middle is nighttime
            b0 = i_sun_set
            b1 = i_sun_rise
            seg_is_daytime = [True, False, True]
        seg_slices = [
            slice(None, b0),
            slice(b0, b1 + 1),
            slice(b1 + 1, None)
            ]

        trace_kw_daytime = {
            'line': {
                'color': 'orange'
                },
            'name': 'Daytime',
            'legendgroup': 'daytime'
            }
        trace_kw_nighttime = {
            'line': {
                'color': 'blue'
                },
            'name': 'Night',
            'legendgroup': 'nighttime'
            }

        fig = make_subplots(1, 1, fig_layout=self.fig_layout_default)

        trace_kw = {
                'type': 'scattergl',
                'mode': 'lines',
                }
        seg_showlegend = [True, True, False]
        # sometimes the first segment is empty so we put the legend on the
        # third one
        if len(day_grid[seg_slices[0]]) == 0:
            seg_showlegend = [False, True, True]
        # make seg_trace kwargs and create trace for each segment
        for s, is_daytime, showlegend in zip(
                seg_slices, seg_is_daytime, seg_showlegend):
            if is_daytime:
                trace_kw_s = dict(trace_kw, **trace_kw_daytime)
            else:
                trace_kw_s = dict(trace_kw, **trace_kw_nighttime)
            # create and add trace
            fig.add_trace(
                dict(trace_kw_s, **{
                    'x': day_grid[s].to_datetime(),
                    'y': target_coords_altaz_for_day[s].alt.degree,
                    'showlegend': showlegend
                    }))
        # obs period
        fig.add_trace(
            dict(trace_kw, **{
                'x': day_grid[i_t0:i_t1].to_datetime(),
                'y': target_coords_altaz_for_day[i_t0:i_t1].alt.degree,
                'mode': 'markers',
                'marker': {
                    'color': 'red',
                    'size': 8
                    },
                'name': "Target"
                })
            )
        # shaded region for too low elevation
        fig.add_hrect(
            y0=-90,
            y1=self._target_alt_min.to_value(u.deg),
            line_width=1, fillcolor="gray", opacity=0.2)

        # update some layout
        fig.update_xaxes(
            title_text="Time [UT]",
            automargin=True)
        fig.update_yaxes(
            title_text="Target Altitude [deg]",
            automargin=True,
            range=[-10, 90])
        fig.update_layout(
            yaxis_autorange=False,
            legend=dict(
                orientation="h",
                x=0.5,
                y=1.02,
                yanchor="bottom",
                xanchor="center",
                bgcolor="#dfdfdf",
                )
            )
        return fig

    @timeit
    def _plot_mapping_pattern(self, exec_config, traj_data):
        figs = [
            make_subplots(1, 1, fig_layout=self.fig_layout_default)
            for _ in range(3)]

        trace_kw = {
                'type': 'scattergl',
                'mode': 'lines',
                'line': {
                    'color': 'red'
                    },
                'showlegend': False,
                # 'marker': {
                #     'color': 'black',
                #     'size': 2
                #     }
                }

        bs_traj_data = traj_data['site']
        instru_traj_data = traj_data['instru']

        # offset
        fig = offset_fig = figs[0]
        fig.add_trace(dict(trace_kw, **{
            'x': bs_traj_data['dlon'].to_value(u.arcmin),
            'y': bs_traj_data['dlat'].to_value(u.arcmin),
            }))
        ref_frame_name = exec_config.mapping.ref_frame.name

        if ref_frame_name == 'icrs':
            fig.update_xaxes(
                title_text="Delta-Source RA [arcmin]")
            fig.update_yaxes(
                title_text="Delta-Source Dec [arcmin]")
        elif ref_frame_name == 'altaz':
            fig.update_xaxes(
                title_text="Delta-Source Az [arcmin]")
            fig.update_yaxes(
                title_text="Delta-Source Alt [arcmin]")
        else:
            fig.update_xaxes(
                title_text="Delta-Source [arcmin]")
            fig.update_yaxes(
                title_text="Delta-Source [arcmin]")
        if instru_traj_data is not None:
            # add instru traj_data
            overlay_traces = instru_traj_data['overlay_traces']
            for t in overlay_traces['offset']:
                fig.add_trace(dict(trace_kw, **t))

        # altaz
        fig = altaz_fig = figs[1]
        fig.add_trace(dict(trace_kw, **{
            'x': bs_traj_data['az'].to_value(u.deg),
            'y': bs_traj_data['alt'].to_value(u.deg),
            }))
        fig.add_trace(dict(trace_kw, **{
            'x': bs_traj_data['target_az'].to_value(u.deg),
            'y': bs_traj_data['target_alt'].to_value(u.deg),
            'line': {
                'color': 'blue'
                }
            }))
        fig.update_xaxes(
            title_text="Azimuth [deg]")
        fig.update_yaxes(
            title_text="Altitude [deg]")

        # icrs
        fig = icrs_fig = figs[2]
        fig.add_trace(dict(trace_kw, **{
            'x': bs_traj_data['ra'].to_value(u.deg),
            'y': bs_traj_data['dec'].to_value(u.deg),
            }))
        fig.update_xaxes(
            title_text="RA [deg]")
        fig.update_yaxes(
            title_text="Dec [deg]")

        # all
        for fig in figs:
            fig.update_xaxes(
                automargin=True,
                autorange='reversed')
            fig.update_yaxes(
                automargin=True)
            fig.update_xaxes(
                    row=1,
                    col=1,
                    scaleanchor='y1',
                    scaleratio=1.,
                    )
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    x=0.5,
                    y=1.02,
                    yanchor="bottom",
                    xanchor="center",
                    bgcolor="#dfdfdf",
                    )
                )
        return {
            'offset': offset_fig,
            'altaz': altaz_fig,
            'icrs': icrs_fig
            }
