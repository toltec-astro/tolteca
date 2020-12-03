#! /usr/bin/env python


from dasha.web.templates import ComponentTemplate
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from tollan.utils.log import get_logger
from plotly.subplots import make_subplots as _make_subplots
import numpy as np
from dasha.web.templates.collapsecontent import CollapseContent
from tollan.utils.fmt import pformat_yaml
import cachetools.func
import functools
import dash
# import dash_defer_js_import as dji
from .. import fs_toltec_hk_rootpath
import astropy.units as u
from tollan.utils.nc import NcNodeMapper
from pathlib import Path
import pytz
import datetime


def make_subplots(nrows, ncols, fig_layout=None, **kwargs):
    _fig_layout = {
            'uirevision': True,
            'xaxis_autorange': True,
            'yaxis_autorange': True,
            'showlegend': True,
            }
    if fig_layout is not None:
        _fig_layout.update(fig_layout)
    fig = _make_subplots(nrows, ncols, **kwargs)
    fig.update_layout(**_fig_layout)
    return fig


def make_labeled_drp(form, label, **kwargs):
    igrp = form.child(dbc.InputGroup, size='sm', className='pr-2')
    igrp.child(dbc.InputGroupAddon(label, addon_type='prepend'))
    return igrp.child(dbc.Select, **kwargs)


class HkDataViewer(ComponentTemplate):
    _component_cls = dbc.Container

    fluid = True
    logger = get_logger()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_layout(self, app):

        container = self

        timer, loading, error = self._setup_live_update_header(
                app, container, 'Dichroic Temp Monitor', 3000)

        body = container.child(dbc.Row).child(dbc.Col)
        details_container = body.child(
                CollapseContent(button_text='Details ...')).content

        controls_container, graph_container = container.child(
                dbc.Row).child(dbc.Col).grid(2, 1)

        controls_form = controls_container.child(dbc.Form, inline=True)

        datalen_drp = make_labeled_drp(
                controls_form, 'Show data of last',
                options=[
                    {
                        'label': f'{n}',
                        'value': n,
                        }
                    for n in ['15 min', '30 min', '1 hr', '12 hr', '1 d', '5 d']],
                value='15 min',
                )

        tz_drp = make_labeled_drp(
                controls_form, 'TZ',
                options=[
                    {
                        'label': tz,
                        'value': tz,
                        }
                    for tz in ['UTC', 'US/Eastern', 'America/Mexico_City']
                    ],
                value='US/Eastern'
                )

        def get_therm_channel_labels(nm):
            strlen = nm.getdim(
                    'Header.ToltecThermetry.ChanLabel_slen')
            return list(map(
                lambda x: x.decode().strip(), nm.getvar(
                    'Header.ToltecThermetry.ChanLabel')[:].view(
                    f'S{strlen}').ravel()))

        def get_hkdata_filepath():
            n = 'thermetry'
            r = Path(fs_toltec_hk_rootpath).expanduser()
            for p in [
                    r.joinpath(n).joinpath(f'{n}.nc'),
                    r.joinpath(f'{n}.nc'),
                    ]:
                if p.exists():
                    return p
            else:
                return None

        @functools.lru_cache(maxsize=32)
        def _get_hkdata(filepath):
            return NcNodeMapper(source=filepath)

        @cachetools.func.ttl_cache(maxsize=1, ttl=1)
        def get_hkdata():
            p = get_hkdata_filepath()
            if p is None:
                return None
            nc = _get_hkdata(p.resolve().as_posix())
            nc.sync()
            return nc

        graph = graph_container.child(dcc.Graph)

        @app.callback(
                [
                    Output(graph.id, 'figure'),
                    Output(loading.id, 'children'),
                    Output(error.id, 'children'),
                    ],
                [
                    Input(timer.id, 'n_intervals'),
                    Input(datalen_drp.id, 'value'),
                    Input(tz_drp.id, 'value'),
                    ]
                )
        def update_graph(n_intervals, datalen_value, tz_value):

            def make_data(datalen_value):
                nc = get_hkdata()
                if nc is None:
                    return None
                # figure out sample rate
                time_var = 'Data.ToltecThermetry.Time1'
                dt = nc.getvar(time_var)[-2:]
                if len(dt) == 2:
                    dt = (np.diff(dt)[0]) << u.s
                else:
                    dt = 5 << u.s
                # calc the slice from datalen
                datalen_value, datalen_unit = datalen_value.split()
                datalen = datalen_value << u.Unit(datalen_unit)
                n_samples = int(
                        (datalen / dt).to_value(u.dimensionless_unscaled))
                if n_samples < 1:
                    n_samples = 1

                return {
                        'nc': nc,
                        'dt': dt,
                        'slice': slice(-n_samples, None),
                        }

            data = make_data(datalen_value)
            if data is None:
                raise dash.exceptions.PreventUpdate

            fig = make_subplots(
                    2, 1, shared_xaxes=True,
                    vertical_spacing=0.05,
                    )
            fig.update_xaxes(row=2, col=1, title=f'Time ({tz_value})')
            fig.update_yaxes(
                    row=1, col=1, title='T (K)',
                    )
            fig.update_yaxes(
                    row=2, col=1, title='dT (K)',
                    )

            def make_trace_data(d, x, y):
                nc = d['nc']
                slice_ = d['slice']
                x = nc.getvar(x)[slice_]
                y = nc.getvar(y)[slice_]
                m = x > 0
                x = x[m]
                y = y[m]
                x_orig = x
                x = np.asarray(x_orig, dtype='datetime64[s]')
                x = x + tz_offset(tz_value, 'UTC')
                return {
                    'x': x,
                    'y': y,
                    'x_orig': x_orig
                    }

            def make_trace_kwargs(d, x, y, name):
                kwargs = {
                        'type': 'scattergl',
                        'mode': 'lines+markers',
                        'name': name
                        }
                td = make_trace_data(d, x, y)
                kwargs.update({
                    'x': td['x'],
                    'y': td['y'],
                    })
                return kwargs

            # get all therm labels
            labels = get_therm_channel_labels(data['nc'])
            for c, name, color in [
                    (1, 'DF1', '#00aaff'),
                    (9, 'DF2', '#00aa88'),
                    (2, 'Bolo', '#ff4400'),
                    ]:
                trace = make_trace_kwargs(
                        d=data,
                        x=f"Data.ToltecThermetry.Time{c}",
                        y=f"Data.ToltecThermetry.Temperature{c}",
                        name=f"{name} ({labels[c - 1]})",
                        )
                trace['marker_color'] = color
                trace['marker_line_color'] = color
                fig.append_trace(trace, row=1, col=1)
            for (c0, c1), name, color in [
                    ((1, 2), 'DF1-Bolo', '#00aaff'),
                    ((9, 2), 'DF2-Bolo', '#00aa88'),
                    ]:
                d0 = make_trace_data(
                        d=data,
                        x=f"Data.ToltecThermetry.Time{c0}",
                        y=f"Data.ToltecThermetry.Temperature{c0}",
                        )
                d1 = make_trace_data(
                        d=data,
                        x=f"Data.ToltecThermetry.Time{c1}",
                        y=f"Data.ToltecThermetry.Temperature{c1}",
                        )
                # subtrace the traces
                x = d1['x']
                y = np.interp(
                        d1['x_orig'], d0['x_orig'], d0['y']) - d1['y']
                fig.append_trace({
                    'x': x,
                    'y': y,
                    'name': name,
                    'type': 'scattergl',
                    'mode': 'lines+markers',
                    'marker_color': color,
                    'marker_line_color': color,
                    }, row=2, col=1)

            return fig, "", ""

        @app.callback(
                Output(details_container.id, 'children'),
                [
                    Input(timer.id, 'n_intervals')
                    ],
                )
        def update_details(n_intervals):
            data = get_hkdata()
            return html.Pre(pformat_yaml(data))

        super().setup_layout(app)

    def _setup_live_update_header(self, app, container, title, interval):
        header = container.child(dbc.Row, className='mb-2').child(
                dbc.Col, className='d-flex align-items-center')
        header.child(html.H3, title, className='mr-4 my-0')
        timer = header.child(dcc.Interval, interval=interval)
        loading = header.child(dcc.Loading)
        error = container.child(dbc.Row).child(dbc.Col)
        return timer, loading, error


def tz_off_from_ut(tz):
    if tz == 'UTC':
        return np.timedelta64(0, 'h')
    tz_now = datetime.datetime.now(pytz.timezone(tz))
    offset_hours = int(tz_now.utcoffset().total_seconds() / 3600.)
    return np.timedelta64(offset_hours, 'h')


def tz_offset(tz1, tz2):
    return tz_off_from_ut(tz1) - tz_off_from_ut(tz2)
