#! /usr/bin/env python

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output
from tolteca.utils.log import get_logger
from .. import get_current_dash_app
from .ncscope import NcScope
from cached_property import cached_property
from functools import lru_cache
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path


app = get_current_dash_app()
logger = get_logger()
ctx = 'thermometry-graph'

title_text = 'Thermometry'
title_icon = 'fas fa-thermometer-half'

UPDATE_INTERVAL = 5000  # ms


src = {
    'label': 'thermometry',
    'title': 'Thermometry',
    'runtime_link': '/Users/ma/Codes/toltec/kids/test_data/thermetry.nc'
    }


class Thermetry(NcScope):

    logger = get_logger()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def v_time(self, i):
        return self.nc.variables[f'Data.ToltecThermetry.Time{i + 1}']

    def v_temp(self, i):
        return self.nc.variables[f'Data.ToltecThermetry.Temperature{i + 1}']

    def v_resis(self, i):
        return self.nc.variables[f'Data.ToltecThermetry.Resistance{i + 1}']

    def n_times(self):
        return self.nc.dimensions['times'].size

    @cached_property
    def n_channels(self):
        return self.nc.dimensions[
                'Header.ToltecThermetry.ChanLabel_xlen'].size

    @cached_property
    def channel_labels(self):
        strlen = self.nc.dimensions[
                'Header.ToltecThermetry.ChanLabel_slen'].size
        return list(map(
            lambda x: x.decode().strip(), self.nc.variables[
                'Header.ToltecThermetry.ChanLabel'][:].view(
                f'S{strlen}').ravel()))

    @classmethod
    @lru_cache(maxsize=128)
    def from_filepath(cls, filepath):
        return cls(source=filepath)

    @classmethod
    def from_link(cls, link):
        return cls.from_filepath(Path(link).resolve())


fig_layout = dict(
    uirevision=True,
    yaxis={
        'type': 'log',
        'autorange': True,
        'title': 'Temperature (K)'
        },
    xaxis={
        'title': 'UT'
        },
    )


def get_layout(**kwargs):
    controls = html.Div([
            dbc.Row([
                daq.BooleanSwitch(
                    id=f'{ctx}-control-toggle-collate',
                    label={
                        'label': 'Collate',
                        'style': {
                            'margin': '0px 5px',
                            },
                        },
                    labelPosition='left',
                    on=True,
                    style={
                        'margin': '0px 5px',
                        }
                    ),
                ]),
            ], className='px-2')
    graph_view = html.Div([
        dcc.Interval(
            id=f'{ctx}-update-timer',
            interval=UPDATE_INTERVAL),
        dcc.Graph(
            id=f'{ctx}',
            figure=get_figure(collate=True),
            # animate=True,
            )
        ])
    return html.Div([
        dbc.Row([dbc.Col(html.H1(src['title'])), ]),
        dbc.Row([dbc.Col(controls), ]),
        dbc.Row([dbc.Col(graph_view), ]),
        ])


def get_traces():
    tm = Thermetry.from_link(src['runtime_link'])
    n_times = 100
    result = []
    for i in range(tm.n_channels):
        result.append({
            'x': np.asarray(tm.v_time(i)[-n_times:], dtype='datetime64[s]'),
            'y': tm.v_temp(i)[-n_times:],
            'name': tm.channel_labels[i],
            'mode': 'lines+markers',
            'type': 'scatter'
        })
    time_latest = np.max([t['x'][-1] for t in result])
    for t in result:
        mask = np.where(
                (t['x'] >= (time_latest - np.timedelta64(24, 'h'))) &
                (t['y'] > 0.))[0]
        t['x'] = t['x'][mask]
        t['y'] = t['y'][mask]
    return result


def get_figure(collate=False):
    traces = get_traces()
    if collate:
        n_panels = 1
        fig_height = 900
        fig_kwargs = dict()
    else:
        n_panels = len(traces)
        fig_height = 300 * n_panels
        fig_kwargs = dict(subplot_titles=[t['name'] for t in traces])

    fig = make_subplots(
            rows=n_panels, cols=1, **fig_kwargs)

    fig.update_layout(
            height=fig_height,
            **fig_layout)
    for i, t in enumerate(traces):
        if collate:
            row = 1
        else:
            row = i + 1
        col = 1
        fig.append_trace(t, row, col)
    return fig


@app.callback([
        Output(f'{ctx}', 'figure')
        ], [
        Input(f'{ctx}-update-timer', 'n_intervals'),
        Input(f'{ctx}-control-toggle-collate', 'on')
        ], [
        ])
def entry_update(n_intervals, collate):
    print(f"called with {n_intervals} {collate}")
    return get_figure(collate=collate),
