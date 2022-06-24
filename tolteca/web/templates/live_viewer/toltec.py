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
from dasha.web.templates.utils import make_subplots

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
                button_text='ICS Details ...',
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
                if data is None:
                    return dash.no_update
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

            if False:
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

        site_data = self._site.get_ocs3_data()
        det_data = np.array(data['detectors']['attrs']['Values'])
        tb = data['toltec_backend']
        nkids = tb['attrs']['NumKids']
        # trim the det data to nkids
        det_data = [
            d[:nkids[i]]
            for i, d in enumerate(
            det_data)]
        
        skyview_layers = list()
        
        fov_size = 4 << u.arcmin
        
        skyview_layers.append({
            "type": "overlay",
            "data": [
                {
                    "type": "circle",
                    "ra": site_data['boresight']['ra_deg'],
                    "dec": site_data['boresight']['dec_deg'],
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
            'det_data': det_data,
            'ut': site_data['time']['ut'],
            'x': np.random.random(1)[0],
            'skyview_layers': skyview_layers,
        }
        return data

    
    class ViewerPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Container

        def __init__(self, instru, **kwargs):
            kwargs.setdefault("fluid", True)
            super().__init__(**kwargs)
            self._instru = instru
            container = self
            
            def _make_timestream_figure():
                fig = make_subplots(
                    1, 1, fig_layout=self.fig_layout_default)   
                trace_kw = {
                    'type': 'scattergl',
                    'mode': 'markers+lines',
                    'marker': {
                        # 'color': 'red',
                        'size': 8
                        },
                    'x': [],
                    'y': [],
                    'showlegend': True,
                }
                for i in range(13):
                    fig.add_trace(dict(trace_kw, name=f'Network {i}'))
                fig.update_yaxes(
                    title_text="Log S21 [ADU]",
                    automargin=True,
                    range=[1, 10])
                return fig

            self._timestream_graph = container.child(
                dcc.Graph, figure=_make_timestream_figure(),
                animate=True)
            
            container.child(html.P("Some TolTEC view"))
           
        def make_callbacks(self, app, instru_info_store_id):
            # @app.callback(
            #     Output(self._timestream_graph.id, 'extendData'),
            #     [Input(instru_info_store_id, 'data')]
            # )
            # def update_data(info_data):
            #     ut = info_data['ut']
            #     det_data = info_data['det_data']
            #     x = []
            #     y = []
            #     t = []
            #     j = 0
            #     for i in range(13):
            #         x.append([ut])
            #         y.append([np.log10(det_data[i][j])])
            #         t.append(i)
            #     return dict(x=x, y=y), t, 100

            app.clientside_callback(
                """
                function(info_data) {
                    var ut = info_data['ut']
                    var det_data = info_data['det_data']
                    x = []
                    y = []
                    t = []
                    j = 0
                    for (const i of Array(13).keys()) {
                        x.push([ut])
                        y.push([Math.log10(det_data[i][j])])
                        t.push(i)
                    }
                    return [{'x': x, 'y': y}, t, 100]
                }
                """,
                Output(self._timestream_graph.id, 'extendData'),
                [Input(instru_info_store_id, 'data')]
            )
            
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
            
    class ViewerControlPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Container

        def __init__(self, instru, **kwargs):
            kwargs.setdefault("fluid", True)
            super().__init__(**kwargs)
            container = self
            container.child(html.P("Some TolTEC viewer controls"))
