#! /usr/bin/env python

# ToDo:
# 1) make zoom plot prettier
# 2) move all the header material to a utility in .templates.common
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


class VnaExplorer(ComponentTemplate):

    # sets up the global Div that the site lives under
    _component_cls = html.Div
    logger = get_logger()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_live_update_header(self, app, container, title, interval):
        header = container.child(dbc.Row, className='mb-2').child(
                dbc.Col, className='d-flex align-items-center')
        header.child(html.H3, title, className='mr-4 my-0')
        timer = header.child(dcc.Interval, interval=interval)
        loading = header.child(dcc.Loading)
        error = container.child(dbc.Row).child(dbc.Col)
        return timer, loading, error

    # This section of code only runs once when the server is started.
    def setup_layout(self, app):
        container = self

        # throw in a logo and title bar to make the site look nice
        logo = "http://toltec.astro.umass.edu/images/toltec_logo.png"

        # setup main body component (row and col)
        body = container.child(dbc.Row).child(dbc.Col)

        # the header holds the controls and the logo
        header = body.child(dbc.Row)
        controls = header.child(dbc.Col, width=12, md=8)
        logoBox = header.child(dbc.Col, width=4, className='d-none d-md-block')
        logoBox.child(dbc.Row,
                      html.Img(src=logo, height="150px"),
                      no_gutters=True,
                      justify="end")

        # the live update header to control timing
        timer, loading, error = self._setup_live_update_header(
                app, controls, 'VNA Sweep Explorer', 2500)

        # Set up an InputGroup to format the obsnum input
        # InputGroups have two children, dbc.Input and dbc.InputGroupAddon
        obsIG = controls.child(dbc.Row).child(dbc.Col, className='d-flex').child(dbc.InputGroup, size='sm', className='mb-3 w-auto mt-3')
        obsIG.child(dbc.InputGroupAddon("Select ObsNum", addon_type="prepend"))
        obsInput = obsIG.child(dbc.Select, options=[])

        # set up a network selection section
        plot_networks_checklist_section = controls.child(dbc.Row).child(dbc.Col)
        plot_networks_checklist_section.child(dbc.Label, 'Select network(s) to show in the plots:')
        plot_networks_checklist_container = plot_networks_checklist_section.child(dbc.Row, className='mx-0')
        checklist_presets_container = plot_networks_checklist_container.child(dbc.Col)
        checklist_presets = checklist_presets_container.child(dbc.Checklist, persistence=False, labelClassName='pr-1', inline=True)
        checklist_presets.options = [
                {'label': 'All', 'value': 'all'},
                {'label': '1.1mm', 'value': '1.1 mm Array'},
                {'label': '1.4mm', 'value': '1.4 mm Array'},
                {'label': '2.0mm', 'value': '2.0 mm Array'},
                ]

        checklist_presets.value = []
        checklist_networks_container = plot_networks_checklist_container.child(dbc.Col)
        # make three button groups
        netSelect = checklist_networks_container.child(dbc.RadioItems,
                                                       persistence=False,
                                                       labelClassName='pr-1',
                                                       inline=True)

        checklist_networks_options = [
                {'label': f'N{i}', 'value': i}
                for i in range(13)]
        array_names = ['1.1 mm Array', '1.4 mm Array', '2.0 mm Array']
        preset_networks_map = dict()
        preset_networks_map['1.1 mm Array'] = set(o['value'] for o in checklist_networks_options[0:7])
        preset_networks_map['1.4 mm Array'] = set(o['value'] for o in checklist_networks_options[7:10])
        preset_networks_map['2.0 mm Array'] = set(o['value'] for o in checklist_networks_options[10:13])
        preset_networks_map['all'] = functools.reduce(set.union, (preset_networks_map[k] for k in array_names))

        # a callback to update obsnum select options
        @app.callback(
                [
                    Output(obsInput.id, 'options'),
                    Output(loading.id, 'children'),
                    Output(error.id, 'children'),
                    ],
                [
                    Input(timer.id, 'n_intervals')
                    ]
                )
        def update_files_info(n_intervals):
            error_content = dbc.Alert('Unable to get data file list', color='danger')
            query_str = "select id,ObsNum,SubObsNum,ScanNum,GROUP_CONCAT(RoachIndex order by RoachIndex) as Roaches,TIMESTAMP(Date, Time) as DateTime from toltec_r1 where ObsType=2 group by ObsNum,SubObsNum,ScanNum order by id desc limit 100"
            try:
                df = dataframe_from_db(query_str, 'toltec', parse_dates=['Date', 'Time'])
                # self.logger.debug(f'{df}')
            except Exception as e:
                self.logger.debug(f"error querry db: {e}", exc_info=True)
                return dash.no_update, "", error_content
            if len(df) == 0:
                return dash.no_update, "", error_content

            def make_key(info):
                # return '{} {}-{}-{} {}'.format(
                #         info.DateTime,
                #         info.ObsNum,
                #         info.SubObsNum,
                #         info.ScanNum,
                #         info.Roaches,
                #         )
                return info.ObsNum

            def make_value(info):
                return json.dumps({
                            'obsnum': info.ObsNum,
                            'subobsnum': info.SubObsNum,
                            'scannum': info.ScanNum,
                        })

            options = []
            for info in df.itertuples():
                options.append({
                    'label': make_key(info),
                    'value': make_value(info)
                    })
            return options, "", ""

        # a callback to update the check state
        @app.callback(
                [
                    Output(netSelect.id, "options"),
                    Output(netSelect.id, "value"),
                    ],
                [
                    Input(checklist_presets.id, "value"),
                    Input(obsInput.id, "value"),
                ]
            )
        def on_preset_change(preset_values, info):
            self.logger.debug(f'info: {info}')
            # this is all the nws
            nw_values = set()
            for pv in preset_values:
                nw_values = nw_values.union(preset_networks_map[pv])
            options = [o for o in checklist_networks_options if o['value'] in nw_values]
            values = list(nw_values)
            # if we have info from the obsinput,
            # we can update the options with disabled state
            if info is None:
                return options, values
            files = query_filepaths(**json.loads(info))
            for option in options:
                v = option['value']
                option['disabled'] = (files is None) or (v not in files)
                if option['disabled']:
                    values.remove(v)

            # clear all checkboxes to avoid loading more than one
            # network at a time
            values = []

            return options, values

        # not a big hr fan but let's see how it looks
        body.child(dbc.Row).child(dbc.Col).child(html.Hr())

        # everything above this comment is generic and should be moved
        # to the .templates.common directory so that other apps can
        # use it
        #

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
        self._registerCallbacks(app, obsInput, netSelect,
                                vnaPlot, zoomPlot,
                                resonanceTable)

    # register the callbacks
    def _registerCallbacks(self, app, obsInput, netSelect,
                           vnaPlot, zoomPlot, resonanceTable):

        # Generate VNA plot
        @app.callback(
            [
                Output(vnaPlot.id, "figure"),
                Output(resonanceTable.id, "figure")
            ],
            [
               Input(netSelect.id, "value"),
               Input(obsInput.id, "value"),
            ]
        )
        def makeVnaPlot(selectedNets, selectedObsnum):
            if (selectedNets is None) or (selectedObsnum is None):
                raise PreventUpdate
            if isinstance(selectedNets, list):
                if len(selectedNets) == 0:
                    raise PreventUpdate
            data = fetchVnaData(selectedNets, **json.loads(selectedObsnum))
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
                Input(netSelect.id, "value"),
                Input(obsInput.id, "value"),
            ]
        )
        def makeZoomPlot(clickData, selectedNets, selectedObsnum):
            if (selectedNets is None) or (selectedObsnum is None):
                raise PreventUpdate
            if isinstance(selectedNets, list):
                if len(selectedNets) == 0:
                    raise PreventUpdate
            data = fetchVnaData(selectedNets, **json.loads(selectedObsnum))
            zoomfig = getZoomPlot(data, clickData, selectedNets)
            return [zoomfig]


# Read data from netcdf files
@timeit
@cache.memoize(timeout=60 * 60 * 60)
def _fetchVnaData(net, obsnum, subobsnum, scannum, filepath):

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


def fetchVnaData(net, obsnum, subobsnum, scannum):
    filepaths = query_filepaths(obsnum, subobsnum, scannum)
    data = _fetchVnaData(net, obsnum, subobsnum,
                         scannum, filepaths[net])
    return data


@timeit
@cachetools.func.ttl_cache(ttl=1)
def query_filepaths(obsnum, subobsnum, scannum):
    # here we check the data file exist if some obs is selected
    query_str = (
            f"select "
            f"id,RoachIndex,FileName "
            f"from toltec_r1 where ObsNum={obsnum} and SubObsNum={subobsnum} and ScanNum={scannum} "
            f"order by RoachIndex")
    try:
        df = dataframe_from_db(query_str, 'toltec')
    except Exception as e:
        return None
    files = dict()
    # populate the list of files
    rpath = Path('/data/data_toltec/reduced/')
    for nw, filepath in zip(df['RoachIndex'], df['FileName']):
        filepath = rpath.joinpath(
                Path(filepath).name).as_posix().replace(
                        'vnasweep.nc', 'vnasweep_processed.nc')
        if Path(filepath).exists():
            files[nw] = filepath
    return files


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


def getZoomPlot(data, clickData, selectedNet):
    fig = go.Figure()

    # if no network is selected, return an empty figure
    if(not selectedNet):
        return fig

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
            fig.add_trace(
                go.Scattergl(x=data['fs'][j, :]*1.e-6,
                             y=data['S21'][j, :],
                             mode='markers',
                             marker=dict(color=c,
                                         size=3),
                ),
                row=2, col=1,
            )

    # the D21 trace
    with timeit("generate the D21 trace figure"):
        fig.add_trace(
            go.Scattergl(x=data['d21_fs']*1.e-6,
                         y=data['d21'],
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
