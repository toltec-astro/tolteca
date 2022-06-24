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
                
                altaz_frame = self._site.observer.altaz(time=ut)
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
                        'mjd': ut.mjd,
                        'isot': ut.isot,
                        'lst': tp['attrs']['LST']
                    },
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
        return data