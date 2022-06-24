#!/usr/bin/env python


from dash_component_template import ComponentTemplate
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import dash

from dasha.web.templates.common import (
        CollapseContent,
        LabeledChecklist,
        LabeledInput,
        )
import json
import cachetools.func
import numpy as np
import threading

from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.time import Time

from .base import ObsInstru
from .ocs3 import Ocs3ConsumerMixin
from ....simu.toltec.toltec_info import toltec_info


class Toltec(Ocs3ConsumerMixin, ObsInstru, name='toltec'):
    """An `ObsInstru` for TolTEC."""

    info = toltec_info
    display_name = info['name_long']
    
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
            self._info_display = container.child(CollapseContent(
                button_text='Details ...',
                )).content.child(html.Pre)
            self._ocs3_details = container.child(CollapseContent(
                button_text='OCS3 Details ...',
                )).content.child(html.Pre)

        @property
        def info_store(self):
            return self._info_store

        def make_callbacks(self, app, timer_inputs):
           
            @app.callback(
                Output(self.info_store.id, 'data'),
                timer_inputs
            )
            def make_info(n_calls):
                data = self._instru.get_ocs3_data()
                site_data = self._instru._site.get_ocs3_data()
                telpos_icrs = SkyCoord(
                    ra=site_data['sky']['attrs']['RaAct'],
                    dec=site_data['sky']['attrs']['DecAct'],
                    unit=(u.hour, u.deg)
                    )

                if data is None:
                    return dash.no_update
                det = data['detectors']
                tb = data['toltec_backend']
                
                skyview_layers = list()
                
                fov_size = 4 << u.arcmin
                
                skyview_layers.append({
                    "type": "overlay",
                    "data": [
                        {
                            "type": "circle",
                            "ra": telpos_icrs.ra.degree,
                            "dec":  telpos_icrs.dec.degree,
                            "radius": fov_size.to_value(u.deg) / 2,
                            "color": 'red'
                            },
                        ],
                    'options': {
                        "show": True,
                        "color": 'red',
                        "lineWidth": 1,
                        "name": "TolTEC FOV"
                        }
                    },
                )

                data = {
                    'data': data,
                    'skyview_layers': skyview_layers,
                }
                return data

            app.clientside_callback(
            '''
            function(info_data) {
                // console.log(info_data)
                var data = {...info_data};
                delete data.skyview_layers;
                return JSON.stringify(
                    data,
                    null,
                    2
                    );
            }
            ''',
            Output(self._info_display.id, 'children'),
            [
                Input(self.info_store.id, 'data'),
                ]
            )

            @app.callback(
                Output(self._ocs3_details.id, 'children'),
                timer_inputs
            )
            def update_ocs3_details(n_calls):
                data = self._instru.get_ocs3_data()
                return json.dumps(data, indent=2)

    _ocs3_lock = threading.Lock()
           
    @cachetools.func.ttl_cache(maxsize=1, ttl=1)
    def get_ocs3_data(self):
        api = self.ocs3_api
        dispatch_objs = {
            'toltec_backend': 'ToltecBackend',
            'detectors': 'ToltecDetectors',
            }
        query_str = ';'.join(dispatch_objs.values()) + ';'
        with self._ocs3_lock:
            data = api.query(query_str)
        if data is None:
            return None
        data = {k: data[v] for k, v in dispatch_objs.items()}
        return data
    
    class ViewerPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Container

        def __init__(self, instru, **kwargs):
            kwargs.setdefault("fluid", True)
            super().__init__(**kwargs)
            self._instru = instru
            container = self
            
            container.child(html.P("Some TolTEC view"))

    class ViewerControlPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Container

        def __init__(self, instru, **kwargs):
            kwargs.setdefault("fluid", True)
            super().__init__(**kwargs)
            container = self
            container.child(html.P("Some TolTEC viewer controls"))