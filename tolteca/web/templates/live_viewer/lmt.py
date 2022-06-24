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

from .base import ObsSite
from .ocs3 import Ocs3ConsumerMixin
from ....simu.lmt import lmt_info


class Lmt(Ocs3ConsumerMixin, ObsSite, name='lmt'):
    """An `ObsSite` for LMT."""

    info = lmt_info
    display_name = info['name_long']
    observer = info['observer']
    
    class ControlPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Form

        def __init__(self, site, **kwargs):
            super().__init__(**kwargs)
            self._site = site
            container = self
            self._info_store = container.child(dcc.Store, data={
                'name': site.name
                })
            self._info_display = container.child(html.Pre)
            self._ocs3_details = container.child(CollapseContent(
                button_text='Details ...',
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
                data = self._site.get_ocs3_data()
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
                    data = self._site.get_ocs3_data()
                    return json.dumps(data, indent=2)

    _ocs3_lock = threading.Lock()
           
    @cachetools.func.ttl_cache(maxsize=1, ttl=1)
    def get_ocs3_data(self):
        api = self.ocs3_api
        dispatch_objs = {
            'telescope': 'Telescope',
            'sky': 'Sky',
            'source': 'Source',
            'time_place': 'TimePlace'
            }
        query_str = ';'.join(dispatch_objs.values()) + ';'
        with self._ocs3_lock:
            data = api.query(query_str)
        if data is None:
            return None
        data = {k: data[v] for k, v in dispatch_objs.items()}
        
        # proocess the data
        tel = data['telescope']
        sky = data['sky']
        tp = data['time_place']
        src = data['source']
        
        telpos_icrs = SkyCoord(
            ra=sky['attrs']['RaAct'],
            dec=sky['attrs']['DecAct'],
            unit=(u.hour, u.deg)
            )
        ut = Time(tp['attrs']['Systime'], format='unix')
        loc = EarthLocation.from_geodetic(
            lon=tp["attrs"]["ObsLongitude"],
            lat=tp["attrs"]["ObsLatitude"],
            height=tp['attrs']['ObsElevation'] << u.km)
    
        skyview_layers = list()
        
        view_port_size = 10 << u.arcmin
        altaz_mark_size = 2 << u.arcmin
        
        def _draw_frame_indicator(size):
            a = 0.15
            data = np.array([
                [-1, 0], [0, 0], [1, 0],
                [1-a, a], [1, 0], [1-a, -a], [1, 0],
                [0, 0], [0, -1], [0, 1],
                [-a, 1-a], [0, 1], [a, 1-a]
                ], dtype='d')
            return data * size / 2
        
        altaz_frame = self.observer.altaz(time=ut)
        telpos_altaz = telpos_icrs.transform_to(altaz_frame)
        fi = _draw_frame_indicator(
            size=altaz_mark_size) - view_port_size / 2 / 2 ** 0.5
        altaz_frame_indicator = SkyCoord(
            fi[:, 0], fi[:, 1], frame=telpos_altaz.skyoffset_frame()
            ).transform_to(
                altaz_frame).transform_to(telpos_icrs.frame)
        altaz_frame_indicator_data = np.array([
            altaz_frame_indicator.ra.degree,
            altaz_frame_indicator.dec.degree,
        ]).T.tolist()
        skyview_layers.append({
            "type": "overlay",
            "data": [
                {
                    "type": "circle",
                    "ra": telpos_icrs.ra.degree,
                    "dec": telpos_icrs.dec.degree,
                    "radius": view_port_size.to_value(u.deg) / 2,
                    "color": 'yellow'
                    },
                {
                    "type": "polyline",
                    "data": altaz_frame_indicator_data,
                    "color": 'yellow',
                    'lineWidth': 3
                    },
                ],
            'options': {
                "show": True,
                "color": 'yellow',
                "lineWidth": 1,
                "name": "LMT View Port"
                }
            },
        )
        data = {
            # 'data': data,
            'boresight': {
                'ra_deg': telpos_icrs.ra.degree,
                'dec_deg': telpos_icrs.dec.degree,
                'az_deg': tel['attrs']['AzActPos'],
                'alt_deg': tel['attrs']['ElActPos'],
                'l_deg': tel['attrs']['LAct'],
                'b_deg': tel['attrs']['BAct'],
                'par_angle_deg': sky['attrs']['ActParAng'],
                'gal_angle_deg': sky['attrs']['GalAng'],
            } ,
            'location': {
                'lon_str': loc.lon.to_string(u.deg),
                'lat_str': loc.lat.to_string(u.deg),
                'lon_deg': loc.lon.degree,
                'lat_deg': loc.lat.degree,
                'height_m': loc.height.to_value(u.m),
            },
            'time': {
                'ut': ut.to_datetime(),
                'mjd': ut.mjd,
                'isot': ut.isot,
                'lst': tp['attrs']['LST']
            },
            'skyview_layers': skyview_layers,
        }
        return data
    
    class ViewerPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Container

        def __init__(self, site, **kwargs):
            kwargs.setdefault("fluid", True)
            super().__init__(**kwargs)
            self._site = site
            container = self
            self._traj_info_store = container.child(dcc.Store)
            
            def _make_traj_figure():
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
                fig.add_trace(dict(trace_kw, name=f'Telescope Trajectory'))
                fig.update_xaxes(
                    title_text="RA [deg]",
                    automargin=True,
                )
                fig.update_yaxes(
                    title_text="Dec [deg]",
                    automargin=True,
                )
                fig.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ))
                return fig

            self._traj_graph = container.child(
                dcc.Graph, figure=_make_traj_figure(),
                animate=True)
           
        @property
        def traj_info_store(self):
            return self._traj_info_store

        def make_callbacks(self, app, site_info_store_id):
            app.clientside_callback(
                """
                function(info_data) {
                    var pos = info_data['boresight']
                    x = []
                    y = []
                    t = []
                    x.push([pos['ra_deg']])
                    y.push([pos['dec_deg']])
                    t.push(0)
                    return [{'x': x, 'y': y}, t, 100]
                }
                """,
                Output(self._traj_graph.id, 'extendData'),
                [Input(site_info_store_id, 'data')]
            )

            app.clientside_callback(
                """
                function(site_info, fig) {
                    data = fig.data[0]
                    // console.log(data)
                    // build skyview layers for the traj
                    var ra = data.x;
                    var dec = data.y;
                    traj_data = ra.map(function(e, i) {
                          return [e, dec[i]];
                        });
                    var layer = {
                        "type": "overlay",
                        "data": [
                            {
                                "type": "polyline",
                                "data": traj_data,
                                "color": 'cyan',
                                'lineWidth': 3
                                },
                            ],
                        'options': {
                            "show": true,
                            "color": 'cyan',
                            "lineWidth": 1,
                            "name": "Telescope Trajectory"
                            }
                        };
                    // console.log(layer)
                    return {
                        'skyview_layers': [layer]
                        }
                    }
                """,
                Output(self._traj_info_store.id, 'data'),
                [Input(site_info_store_id, 'data'),
                    State(self._traj_graph.id, 'figure')]
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
 