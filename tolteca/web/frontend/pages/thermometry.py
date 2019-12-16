#! /usr/bin/env python

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ClientsideFunction
from tolteca.utils.log import timeit, get_logger
from ...backend import dataframe_from_db, cache
from .. import get_current_dash_app
from ..common import TableViewComponent
import dash
import plotly
from pathlib import Path
from .ncscope import NcScope
import astropy.units as u
from cached_property import cached_property
from tolteca.utils.nc import NcNodeMapper
from functools import lru_cache
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np


app = get_current_dash_app()
logger = get_logger()

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


def get_layout(**kwargs):
    graph_view = html.Div([
        dcc.Interval(
            id='thermometry-graph-update-timer',
            interval=UPDATE_INTERVAL),
        dcc.Graph(id='thermometry-graph')
        ])
    return html.Div([
        dbc.Row([dbc.Col(html.H1(src['title'])), ]),
        dbc.Row([dbc.Col(html.Div(id='entry-updated')), ]),
        dbc.Row([dbc.Col(graph_view), ]),
        ])


@timeit
@app.callback([
        Output('entry-updated', 'children'),
        Output('thermometry-graph', 'figure')
        ], [
        Input('thermometry-graph-update-timer', 'n_intervals')], [
        ])
def entry_update(n_intervals):
    tm = Thermetry.from_link(src['runtime_link'])

    print(tm.n_channels)
    # tm.get_data(timeslice=[-12 * u.hours, None])

    fig = make_subplots(
        rows=tm.n_channels, cols=1,
        subplot_titles=tm.channel_labels)
    # fig['layout']['margin'] = {
    #     'l': 30, 'r': 10, 'b': 30, 't': 10
    # }
    # fig['layout']['height'] = 1000
    fig.update_layout(
            height=400 * tm.n_channels, width=800, title_text="Thermometry")

    # fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
    n_times = 100
    for i in range(0, tm.n_channels):
        fig.append_trace({
            'x': np.asarray(tm.v_time(i)[-n_times:], dtype='datetime64[s]'),
            'y': tm.v_temp(i)[-n_times:],
            'name': tm.channel_labels[i],
            'mode': 'lines+markers',
            'type': 'scatter'
        }, i + 1, 1)
    return html.Div("Updated"), fig
