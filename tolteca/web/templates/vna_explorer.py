#! /usr/bin/env python

# ToDo:
# 1) make zoom plot prettier
# 2) [done] move all the header material to a utility in .templates.common
# 3) speed up plot rendering

from dasha.web.templates import ComponentTemplate
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import colorsys
import dash
from tollan.utils.log import timeit, get_logger
from dasha.web.extensions.cache import cache
from dash.exceptions import PreventUpdate
from dasha.web.extensions.db import dataframe_from_db
from tolteca.datamodels.toltec import BasicObsData
import functools
import json
import cachetools.func
from pathlib import Path
from itertools import cycle

from dasha.web.templates.common import LiveUpdateSection, CollapseContent
from .common import HeaderWithToltecLogo
from .common.simple_basic_obs_select import KidsDataSelect


class VnaExplorer(ComponentTemplate):

    # sets up the global Div that the site lives under
    _component_cls = html.Div
    logger = get_logger()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # This section of code only runs once when the server is started.
    def setup_layout(self, app):
        container = self
        header_section, hr_container, body = container.grid(3, 1)
        hr_container.child(html.Hr())
        header_container = header_section.child(HeaderWithToltecLogo(
            logo_colwidth=4
            )).header_container

        title_container, controls_container = header_container.grid(2, 1)
        header = title_container.child(
                LiveUpdateSection(
                    title_component=html.H3("VNA Sweep Explorer"),
                    interval_options=[2000, 5000],
                    interval_option_value=2000
                    ))
        obsInput = controls_container.child(
                KidsDataSelect(multi=None)
                )
        obsInput.setup_live_update_section(
                app, header, query_kwargs={'obs_type': 'VNA'})

        # the above finishes settings of the kids data select header section
        # now is the graphs

        topPlots = body.child(dbc.Row)
        # add a table to show the number of resonances found
        s6 = {'min-width': '600px'}
        # resonanceTable = body.child(dbc.Row).child(dcc.Graph, style=s)
        resonanceTable = topPlots.child(dbc.Col).child(dcc.Graph, style=s6)

        # this is the big vna plot at the top
        s = {'min-width': '1200px'}
        vnaPlot = body.child(dbc.Row).child(dbc.Col).child(dcc.Graph, style=s)
        zoomPlot = topPlots.child(dbc.Col).child(dcc.Graph, style=s6)

        # before we register the callbacks
        super().setup_layout(app)

        # register the callbacks
        self._registerCallbacks(app, obsInput,
                                vnaPlot, zoomPlot,
                                resonanceTable)

    # register the callbacks
    def _registerCallbacks(self, app, obsInput,
                           vnaPlot, zoomPlot, resonanceTable):

        # Generate VNA plot
        @app.callback(
            [
                Output(vnaPlot.id, "figure"),
                Output(resonanceTable.id, "figure")
            ],
            obsInput.inputs,
        )
        def makeVnaPlot(obs_input):
            # obs_input is the basic_obs_select_value dict
            # seen in the "details..."
            # when the nwid multi is set in the select,
            # this is a list of dict
            # otherwise is a dict, which is our case here.
            if obs_input is None:
                raise PreventUpdate
            data = fetchVnaData(**obs_input)
            if data is None:
                raise PreventUpdate
            vnafig = getVnaPlot(data)
            rtfig = getResonanceTable(data)
            return [vnafig, rtfig]

        # Generate Zoomed-in plot
        @app.callback(
            [
                Output(zoomPlot.id, "figure"),
            ],
            [
                Input(vnaPlot.id, "clickData"),
            ] + obsInput.inputs
        )
        def makeZoomPlot(clickData, obs_input):
            if obs_input is None:
                raise PreventUpdate
            data = fetchVnaData(**obs_input)
            if data is None:
                raise PreventUpdate
            data = fetchVnaData(**obs_input)
            zoomfig = getZoomPlot(data, clickData)
            return [zoomfig]


# Read data from netcdf files
@timeit
@cache.memoize(timeout=60 * 60 * 60)
def _fetchVnaData(filepath):

    def makeS21(Is, Qs):
        nsweeps, ntones = Is.shape
        S21 = []
        for i in np.arange(ntones):
            s = np.sqrt(Is[:, i]**2 + Qs[:, i]**2)
            S21.append(20.*np.log10(s))
        return np.array(S21).transpose()

    with timeit(f"read in data from {filepath}"):
        vna = BasicObsData.read(filepath)

    return {
        'network': vna.meta['roachid'],
        'obsnum': vna.meta['obsnum'],
        'subobsnum': vna.meta['subobsnum'],
        'scannum': vna.meta['scannum'],
        'd21': vna.unified.D21,
        'd21_fs': vna.unified.frequency,
        'fs': vna.frequency,
        'S21': makeS21(vna.S21.real.value, vna.S21.imag.value),
        'candidates': vna.unified.meta['candidates'],
        'LoCenterFreq': vna.meta['flo_center'],
        'InputAtten': vna.meta['atten_in'],
        'OutputAtten': vna.meta['atten_out'],
         }


# This line makes the cache invalid when the server reloads
cache.delete_memoized(_fetchVnaData)


def fetchVnaData(**kwargs):
    raw_obs_processed_url = kwargs['raw_obs_processed']
    if raw_obs_processed_url is None:
        # no processed data
        return None
    data = _fetchVnaData(raw_obs_processed_url)
    return data


def get_color_pairs():
    colorsDark = px.colors.qualitative.Dark24

    def scale_lightness(color_hex, scale):
        rgb = np.array(px.colors.hex_to_rgb(color_hex)) / 255.
        # convert rgb to hls
        h, l, s = colorsys.rgb_to_hls(*rgb)
        # manipulate h, l, s values and return as rgb
        return "#{0:02x}{1:02x}{2:02x}".format(
                *(np.array(
                    colorsys.hls_to_rgb(h, min(1, l * scale), s = s)
                    ) * 255.).astype(int))
    # colorsLight = [scale_lightness(c, 2) for c in colorsDark]
    colorsLight = [
        "#{0:02x}{1:02x}{2:02x}".format(
            *(
                np.array(
                    px.colors.find_intermediate_color(
                        np.array(px.colors.hex_to_rgb(c)) / 255.,
                        (1, 1, 1), 0.5)) * 255.).astype(int)
            )
        for c in colorsDark]
    return colorsDark, colorsLight


def getZoomPlot(data, clickData):
    fig = go.Figure()

    # dict_keys(['curveNumber', 'pointNumber', 'pointIndex', 'x', 'y'])
    # there are three possibilities: the S21 plot was clicked, the D21
    # plot was clicked, or no plot was clicked.
    if(clickData is None):
        curve = 0
    else:
        curve = clickData['points'][0]['curveNumber']
    if(curve < 1000):
        # S21 plot was clicked, plot the points from the nearest
        # sweeps
        x = data['fs'][curve, :]*1.e-6
        y = data['S21'][curve, :]
    elif(curve == 1000):
        i0 = clickData['points'][0]['pointNumber']
        rg = np.arange(max(0, i0-200), min(len(data['d21_fs']), i0+200))
        x = data['d21_fs'][rg]*1.e-6
        y = data['d21'][rg]

    fig.add_trace(
        go.Scattergl(x=x,
                     y=y,
                     mode='lines',
        )
    )

    fig.update_yaxes(automargin=True)
    if(curve < 1000):
        fig.update_yaxes(title_text="abs(S21) [dB]")
    else:
        fig.update_yaxes(title_text="D21")
    fig.update_xaxes(title_text="Frequency [MHz]")
    fig.update_layout(
        height=200,
        showlegend=False,
        autosize=True,
        margin=dict(
            autoexpand=True,
            l=10,
            r=10,
            t=10,
        ),
        )
    return fig


# plotly code to generate vna plot
@timeit
def getVnaPlot(data):
    fig = make_subplots(rows=2, cols=1,
                        row_heights=[0.3,0.7],
                        shared_xaxes=True,
                        vertical_spacing=0.02)
    xaxis, yaxis = getXYAxisLayouts()

    # the S21 trace
    colorsDark, colorsLight = get_color_pairs()
    colors = []
    for i in np.arange(len(colorsDark)):
        colors.append(colorsDark[i])
        colors.append(colorsLight[i])
    colorsCycle = cycle(colors)

    ntones, nsweeps = data['fs'].shape
    with timeit("generate the S21 trace figure"):
        for j in np.arange(ntones):
            c = next(colorsCycle)
            fig.append_trace(
                    go.Scattergl(
                        x=data['fs'][j, ::2]*1.e-6,
                        y=data['S21'][j, ::2],
                        mode='markers',
                        marker=dict(color=c,
                                    size=3),
                ),
                row=2, col=1,
            )

    # the D21 trace
    with timeit("generate the D21 trace figure"):
        fig.append_trace(
                go.Scattergl(x=data['d21_fs'][::2]*1.e-6,
                    y=data['d21'][::2],
                         mode='markers',
                         marker=dict(color='blue',
                                     size=3),
            ),
            row=1, col=1,
        )

        # the candidate resonators
        c = data['candidates'].value*1.e-6
        res = []
        for j in np.arange(len(c)):
            res.append(
                dict(
                    type='line',
                    yref='paper', y0=0, y1=1,
                    xref='x', x0=c[j], x1=c[j],
                    line=dict(
                        color="Red",
                        width=1,
                        dash="dot",
                    )
                )
            )
#         fig.update_layout(shapes=res)

    with timeit("update y-axis and layout"):
        fig.update_yaxes(automargin=True)
        fig.update_yaxes(title_text="abs(S21) [dB]", row=2, col=1)
        fig.update_yaxes(title_text="D21", row=1, col=1)
        fig.update_xaxes(title_text="Frequency [MHz]", row=2, col=1)
        fig.update_layout(
            # uirevision=True,
            showlegend=False,
            height=600,
            # xaxis=xaxis,
            # yaxis=yaxis,
            autosize=True,
            margin=dict(
                autoexpand=True,
                l=10,
                r=10,
                t=0,
            ),
            plot_bgcolor='white'
        )
    return fig


# plotly table for displaying the loadings
def getResonanceTable(data):
    nets = []
    nResonators = []
    nets.append('Network {}'.format(data['network']))
    nResonators.append('{0:3.0f}'.format(len(data['candidates'])))
    fig = go.Figure(
        data=[
            go.Table(header=dict(values=nets),
                     cells=dict(values=nResonators))
        ]
    )
    fig.update_layout(
        title={'text': "# of Found Resonators", },
        height=100,
        autosize=False,
        margin=dict(
            l=0,
            r=0,
            b=1,
            t=30,
        ),)
    return fig


# common figure axis definitions
def getXYAxisLayouts():
    xaxis = dict(
        title_text="Frequency [MHz]",
        titlefont=dict(size=20),
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='black',
        linewidth=4,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=18,
            color='rgb(82, 82, 82)',
        ),
    )

    yaxis = dict(
        title_text="abs(S21) [dB]",
        titlefont=dict(size=20),
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='black',
        linewidth=4,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=18,
            color='rgb(82, 82, 82)',
        ),
    )
    return xaxis, yaxis
