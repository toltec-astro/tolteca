#!/usr/bin/env python

from dash_component_template import ComponentTemplate, NullComponent
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash
import dash_bootstrap_components as dbc
import dash_aladin_lite as dal
from dash_extensions.javascript import assign

from dasha.web.templates.common import (
        CollapseContent,
        LabeledChecklist,
        LabeledInput,
        LiveUpdateSection
        )
from dasha.web.templates.utils import PatternMatchingId, make_subplots
from dasha.web import exit_stack

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

from dataclasses import dataclass, field
import numpy as np
import functools
import bisect
from typing import Union
from io import StringIO
from schema import Or
import jinja2
import json

from tollan.utils.log import get_logger, timeit
from tollan.utils.fmt import pformat_yaml
from tollan.utils import rupdate
from tollan.utils.dataclass_schema import add_schema
from tollan.utils.namespace import Namespace

from ....utils import yaml_load, yaml_dump
from ....simu.utils import SkyBoundingBox
from ....simu import (
    mapping_registry,
    SimulatorRuntime, ObsParamsConfig)
from ....simu.mapping.utils import resolve_sky_coords_frame

from .base import ObsSite, ObsInstru
from .ocs3 import Ocs3API


_j2env = jinja2.Environment()
"""A jinja2 environment for generating clientside callbacks."""


class LiveViewer(ComponentTemplate):
    """An Live data view for telescope/instrument status."""

    class Meta:
        component_cls = dbc.Container

    logger = get_logger()

    def __init__(
            self,
            site_name='lmt',
            instru_name=None,
            pointing_catalog_path=None,
            js9_config_path=None,
            toltec_ocs3_url='socket://localhost:61559',
            lmt_ocs3_url='socket://localhost:61558',
            title_text='Live Viewer',
            **kwargs):
        kwargs.setdefault('fluid', True)
        super().__init__(**kwargs)
        obssite = self._site = ObsSite.from_name(site_name)
        if obssite.name == 'lmt':
            obssite.init_ocs3(lmt_ocs3_url)
        instru = self._instru = (
            None if instru_name is None
            else ObsInstru.from_name(instru_name))
        if instru is not None and instru.name == 'toltec':
            instru.init_ocs3(toltec_ocs3_url)
            instru._site = obssite
        self._pointing_catalog_path = pointing_catalog_path
        self._js9_config_path = js9_config_path
        self._title_text = title_text

    @classmethod
    def _create_ocs3_api(cls, ocs3_url):
        api = Ocs3API(url=ocs3_url)

        # make sure we clean up any resources when exit
        def _on_dasha_exit(*args, **kwargs):
            api.__exit__(*args, **kwargs)
            
        exit_stack.push(_on_dasha_exit)
        return api

    def setup_layout(self, app):

        if self._js9_config_path is not None:
            # this is required to locate the js9 helper
            # in case js9 is used
            # djs9.setup_js9(app, config_path=self._js9_config_path)
            # TODO add the above back when we actually need js9
            pass

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
        controls_panel, viewers_panel = body.colgrid(1, 2, width_ratios=[1, 3])
        controls_panel.style = {
                    'width': '375px'
                    }
        controls_panel.parent.style = {
            'flex-wrap': 'nowrap'
            }
        # make the plotting area auto fill the available space.
        viewers_panel.style = {
            'flex-grow': '1',
            'flex-shrink': '1',
            }
        # Left panel, these are containers for the input controls
        obssite_container, obsinstru_container = \
            controls_panel.colgrid(2, 1, gy=3)

        # site config panel
        obssite_title = f'Site: {self._site.display_name}'
        obssite_card = obssite_container.child(self.Card(
            title_text=obssite_title))
        obssite_container = obssite_card.body_container
        site_panel = self._site.make_controls(obssite_container)
        site_info_store = site_panel.info_store

        if self._instru is not None:
            # instru config panel
            obsinstru_title = f'Instrument: {self._instru.display_name}'
            obsinstru_card = obsinstru_container.child(self.Card(
                title_text=obsinstru_title))
            obsinstru_container = obsinstru_card.body_container
            instru_panel = self._instru.make_controls(
                obsinstru_container)
            instru_info_store = instru_panel.info_store
        else:
            # a dummy instru_info_store with nothing
            instru_info_store = obsinstru_container.child(dcc.Store)

        # Right panel, for plotting
        # dal_container, viewer_controls_container, \
        #     site_viewer_container, instru_viewer_container = \
        #     viewers_panel.colgrid(4, 1, gy=3)
        site_viewer_container, viewer_controls_container, \
            instru_viewer_container = \
                viewers_panel.colgrid(3, 1, gy=3)
        dal_container, site_viewer_container = site_viewer_container.colgrid(1, 2, gx=3, width_ratios=[3, 1])

        site_viewer =self._site.make_viewer(
            site_viewer_container, className='px-0')

        if self._instru is not None:
            instru_viewer_controls = self._instru.make_viewer_controls(
                viewer_controls_container, className='px-0 d-flex')
            instru_viewer = self._instru.make_viewer(
                instru_viewer_container, className='px-0')
        else:
            pass

        skyview_height = '40vh' if self._instru is None else '60vh'
        
        def _register_clientside_function(id, name, body):
            _inline_clientside_template = """
var clientside = window.dash_clientside = window.dash_clientside || {{}};
var ns = clientside["{namespace}"] = clientside["{namespace}"] || {{}};
ns["{function_name}"] = {clientside_function};
"""
            namespace = f"_dasha_clientside_func_{id}"
            app._inline_scripts.append(_inline_clientside_template.format(
                    namespace=namespace.replace('"', '\\"'),
                    function_name=name.replace('"', '\\"'),
                    clientside_function=body,
                ))
            return {
                'variable': f'dash_clientside.{namespace}.{name}'
            }
            
        site_data = self._site.get_ocs3_data()
        skyview = dal_container.child(
            dal.DashAladinLite,
            survey='P/DSS2/color',
            target=(
                f'{site_data["boresight"]["ra_deg"]} '
                f'{site_data["boresight"]["dec_deg"]}'),
            fov=(10 << u.deg).to_value(u.deg),
            style={"width": "100%", "height": skyview_height},
            options={
                "showLayerBox": True,
                "showSimbadPointerControl": True,
                "showReticle": True,
                # "showCooGrid": True
                },
            custom_scripts = { 
                "animateTo": _register_clientside_function(
                    dal_container.id,
                    'animateTo',
                    """
                    function(aladinlite, data, props) {
                        const {ra, dec} = data;
                        aladinlite.aladin.animateToRaDec(ra, dec);
                    }"""),
                }
            )

        super().setup_layout(app)
        
        site_panel.make_callbacks(
            app, timer_inputs=obssite_card.header.timer.inputs,
            )

        site_viewer.make_callbacks(
            app, site_info_store_id=site_info_store.id
        )
        if self._instru is not None:
            instru_panel.make_callbacks(
                app, timer_inputs=obsinstru_card.header.timer.inputs,
                )
            instru_viewer.make_callbacks(
                app, instru_info_store_id=instru_info_store.id)
   
        # connect tel pos to the alv
        # app.clientside_callback(
        #     """
        #     function(site_info) {
        #         if (!site_info) {
        #             return window.dash_clientside.no_update;
        #         }
        #         var ra = site_info.boresight.ra_deg;
        #         var dec = site_info.boresight.dec_deg;
        #         return ra + ' ' + dec
        #     }
        #     """,
        #     Output(skyview.id, "target"),
        #     Input(site_info_store.id, 'data')
        # )
        # @app.callback(
        #     Output(skyview.id, 'custom_script_calls'),
        #      [Input(site_info_store.id, 'data')], prevent_initial_call=True)
        # def goto_target(data):
        #     return {"animateTo": {
        #             "ra": data['boresight']['ra_deg'],
        #             "dec": data['boresight']['dec_deg'],
        #             }}

        # update the sky map layers and target
        app.clientside_callback(
            '''
            function(site_info, instru_info, traj_info) {
                if (!site_info && !instru_info) {
                    return Array(2).fill(window.dash_clientside.no_update);
                }
                var ra = site_info.boresight.ra_deg;
                var dec = site_info.boresight.dec_deg;
                var target = ra + ' ' + dec
                var layers = [];
                if (site_info && site_info.skyview_layers) {
                    layers = layers.concat(site_info.skyview_layers)
                }
                if (instru_info && instru_info.skyview_layers) {
                    layers = layers.concat(instru_info.skyview_layers)
                }
                if (!layers) {
                    layers = window.dash_clientside.no_update;
                }
                if (traj_info) {
                    layers = layers.concat(traj_info.skyview_layers);
                }
                var custom_script_call = {"animateTo": {
                        "ra": ra,
                        "dec": dec
                        }
                                      }
                return [custom_script_call, layers];
            }
            ''',
            [
                Output(skyview.id, 'custom_script_calls'),
                Output(skyview.id, 'layers'),
                ],
            [
                Input(site_info_store.id, 'data'),
                Input(instru_info_store.id, 'data'),
                Input(site_viewer.traj_info_store.id, 'data'),
                ]
            )

    class Card(ComponentTemplate):
        class Meta:
            component_cls = dbc.Card

        def __init__(self, title_text, **kwargs):
            super().__init__(**kwargs)
            container = self
            self.header = container.child(
                LiveUpdateSection(
                    title_component=html.H6(
                        title_text),
                    interval_options=[2000, 5000, 10000],
                    interval_option_value=5000,
                    className='card-header'
                    ))
            self.body_container = container.child(dbc.CardBody)
