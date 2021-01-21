#! /usr/bin/env python

from dasha.web.templates import ComponentTemplate
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
from glob import glob
import numpy as np
import netCDF4
import colorsys
import dash
import os
import bottleneck as bn
from tollan.utils.log import timeit, get_logger
from dasha.web.extensions.cache import cache
from dash.exceptions import PreventUpdate
from dasha.web.extensions.db import dataframe_from_db
import scipy.interpolate as interpolate
import functools
import json
import cachetools.func
from pathlib import Path
from .common import HeaderWithToltecLogo
from .common.simple_basic_obs_select import KidsDataSelect
from dasha.web.templates.common import LiveUpdateSection


class NoiseExplorer(ComponentTemplate):

    # sets up the global Div that the site lives under
    _component_cls = html.Div
    logger = get_logger()

    def __init__(self, *args, **kwargs):
        # the pros won't like this...
        self.selectedObsList = []
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
        header_section, hr_container, body = container.grid(3, 1)
        hr_container.child(html.Hr())
        header_container = header_section.child(HeaderWithToltecLogo(
            logo_colwidth=4
            )).header_container

        title_container, controls_container = header_container.grid(2, 1)
        header = title_container.child(
                LiveUpdateSection(
                    title_component=html.H3("Noise Explorer"),
                    interval_options=[2000, 5000],
                    interval_option_value=2000
                    ))
        obsInput = controls_container.child(dbc.Row).child(dbc.Col).child(
                KidsDataSelect(multi=['nwid', 'array'])
                )
        obsInput.setup_live_update_section(
            app, header, query_kwargs={'obs_type': 'Timestream',
                                       'n_obs': 100})


        # # Set up an InputGroup to format the obsnum input
        # controls = controls_container
        # # InputGroups have two children, dbc.Input and dbc.InputGroupAddon
        # obsIG = controls.child(dbc.Row).child(dbc.Col, className='d-flex').child(dbc.InputGroup,
        #                                   size='sm',
        #                                   className='mb-3 w-auto mt-3')
        # obsIG.child(dbc.InputGroupAddon("Select ObsNum", addon_type="prepend"))
        # obsInput = obsIG.child(dbc.Select, options=[])

        # set up a network selection section
        plot_networks_checklist_section = controls_container.child(
                dbc.Row).child(dbc.Col)
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
        netSelect = checklist_networks_container.child(dbc.Checklist, persistence=False, labelClassName='pr-1', inline=True)

        checklist_networks_options = [
                {'label': f'N{i}', 'value': i}
                for i in range(13)]
        array_names = ['1.1 mm Array', '1.4 mm Array', '2.0 mm Array']
        preset_networks_map = dict()
        preset_networks_map['1.1 mm Array'] = set(o['value'] for o in checklist_networks_options[0:7])
        preset_networks_map['1.4 mm Array'] = set(o['value'] for o in checklist_networks_options[7:10])
        preset_networks_map['2.0 mm Array'] = set(o['value'] for o in checklist_networks_options[10:13])
        preset_networks_map['all'] = functools.reduce(set.union, (preset_networks_map[k] for k in array_names))

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
            return options, values

        # not a big hr fan but let's see how it looks
        body.child(dbc.Row).child(dbc.Col).child(html.Hr())

        # this is the big psd plot at the top
        pPlot = body.child(dbc.Row).child(dbc.Col).child(dcc.Graph, style={'min-width': '1200px'})

        # and we need a switch to toggle log_x_axis
        opt = [{"label": "log x-axis", "value": 0}]
        logx = body.child(dbc.Row).child(dbc.Col).child(dbc.Checklist,
                                                        options=opt)

        # add a table to show the detector loadings
        loadingTable = body.child(dbc.Row).child(dcc.Graph, style={'min-width': '1200px'})

        # these next plots will go on the same row
        plotRow = body.child(dbc.Row)
        # a histogram of the median psds of each detector
        hPlot = plotRow.child(dbc.Col).child(dcc.Graph, style={'min-width': '600px'})
        # a plot of detector resonant freq. vs median psd value
        fPlot = plotRow.child(dbc.Col).child(dcc.Graph, style={'min-width': '600px'})

        # before we register the callbacks
        super().setup_layout(app)

        # register the callbacks
        self._registerCallbacks(app, obsInput, netSelect,
                                pPlot, hPlot, fPlot, logx,
                                loadingTable)

    # refreshes the file list
    def _updateFileList(self):
        # get list of files
        indexfiles = []
        srch = '/data/data_toltec/reduced/*timestream_processed.nc'
        indexfiles = glob(srch)

        # parse the file names to get the obsnums and networks
        obsnums, networks = parseFileList(indexfiles)
        self.obsnums = obsnums[::-1]
        self.networks = networks[::-1]

        # list of dictionaries for the label and value
        obsList = []
        for indx in np.arange(len(obsnums)-1, -1, -1):
            label = " {0:}".format(obsnums[indx])
            obsList.append({'label': label,
                            'value': obsnums[indx]})
        return obsnums, networks, obsList

    # register the callbacks
    def _registerCallbacks(self, app, obsInput, netSelect,
                           pPlot, hPlot, fPlot, logx, loadingTable):

        # Generate Median PSD plot vs frequency
        @app.callback(
            [
                Output(pPlot.id, "figure"),
                Output(hPlot.id, "figure"),
                Output(fPlot.id, "figure"),
                Output(loadingTable.id, "figure")
            ],
            [
               Input(netSelect.id, "value"),
               Input(obsInput.id, "value"),
               Input(logx.id, "value")
            ]
        )
        def makeMedianPsdPlot(selectedNets, selectedObsnum, logx):
            if (selectedNets is None) or (selectedObsnum is None):
                raise PreventUpdate
            data = fetchPsdData(selectedNets, **json.loads(selectedObsnum))
            if logx is None:
                logx = []
            if(len(logx) > 0):
                lx = 1
            else:
                lx = 0
            pfig = getPsdPlot(data, logx=lx)
            hfig = getDetMedHist(data)
            ffig = getPvsFPlot(data)
            ltfig = getLoadingTable(data)
            return [pfig, hfig, ffig, ltfig]


# Read data from netcdf files
@timeit
@cache.memoize(timeout=60 * 60 * 60)
def _fetchPsdData(net, obsnum, subobsnum, scannum, filepath):

    def make_stat(arr, axis=None):
        med = bn.nanmedian(arr, axis=axis)
        mad = bn.nanmedian((arr - np.swapaxes(med[None, :], 0, axis)))
        p25 = bn.nanmedian(arr[arr < med])
        p75 = bn.nanmedian(arr[arr > med])
        return med, mad, p25, p75

    with timeit(f"read in data from {filepath}"):
        nc = netCDF4.Dataset(filepath)
        fs = nc.variables['Header.Kids.tones'][:].data
        fpsd = nc.variables['Header.Kids.PsdFreq'][:].data
        xpsd = nc.variables['Data.Kids.xspsd'][:].data.transpose()
        rpsd = nc.variables['Data.Kids.rspsd'][:].data.transpose()
        nc.close()

    # construct the medians of the psds of the network
    medxPsd, madxPsd, stdxPsdLow, stdxPsdHigh = make_stat(xpsd, axis=0)
    medrPsd, madrPsd, stdrPsdLow, stdrPsdHigh = make_stat(rpsd, axis=0)

    # also calculate median of each detector's psd
    medxDet = bn.nanmedian(xpsd, axis=1)
    medrDet = bn.nanmedian(rpsd, axis=1)
    detFreqMHz = fs * 1e-6

    # check for NaNs, zeros or negatives
    if np.isnan(medxPsd).any():
        print("\nNaNs detected in medxPsd.")
    if np.isnan(medrPsd).any():
        print("\nNaNs detected in medrPsd.")

    # estimate the background loading on the network
    Tload = estimateTload(net, np.median(medxPsd), np.median(medrPsd))

    return {
         'network': net,
         'obsnum': obsnum,
         'subobsnum': subobsnum,
         'scannum': scannum,
         'fpsd': fpsd,
         'Tload': Tload,
         'medxPsd': medxPsd,
         'medrPsd': medrPsd,
         'madxPsd': madxPsd,
         'madrPsd': madrPsd,
         'stdxPsdHigh': stdxPsdHigh,
         'stdrPsdHigh': stdrPsdHigh,
         'stdxPsdLow': stdxPsdLow,
         'stdrPsdLow': stdrPsdLow,
         'detFreqMHz': detFreqMHz,
         'medxDet': np.log10(medxDet),
         'medrDet': np.log10(medrDet),
         }


# This line makes the cache invalid when the server reloads
cache.delete_memoized(_fetchPsdData)


def fetchPsdData(nets, obsnum, subobsnum, scannum):
    data = []
    filepaths = query_filepaths(obsnum, subobsnum, scannum)
    for net in nets:
        data.append(_fetchPsdData(net, obsnum, subobsnum, scannum, filepaths[net]))
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
                        'timestream.nc', 'timestream_processed.nc')
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


# plotly code to generate histogram plot
def getDetMedHist(data):
    fig = go.Figure()
    xaxis, yaxis = getXYAxisLayouts()
    xaxis['title_text'] = "LOG10(Median Detector PSD [1/Hz])"
    yaxis['title_text'] = "Detector Count"
    fig.update_layout(
        uirevision=True,
        showlegend=False,
        xaxis=xaxis,
        yaxis=yaxis,
        autosize=True,
        margin=dict(
            autoexpand=False,
            l=100,
            r=100,
            t=0,
        ),
        plot_bgcolor='white'
    )
    if(len(data) == 0):
        return fig

    colorsDark, colorsLight = get_color_pairs()

    for i in np.arange(len(data)):
        fig.add_trace(
            go.Histogram(
                x=data[i]['medxDet'],
                bingroup=1,
                marker_color=colorsDark[i],
            ),
        )
        fig.add_trace(
            go.Histogram(
                x=data[i]['medrDet'],
                bingroup=1,
                marker_color=colorsLight[i],
            ),
        )

    fig.update_xaxes(range=[-18, -15.5])
    fig.update_layout(barmode='overlay',)
    fig.update_traces(opacity=0.75)
    return fig


# plotly code to generate psd plot
def getPsdPlot(data, logx=0):
    lxs = "linear"
    if(logx):
        lxs = "log"

    fig = go.Figure()
    xaxis, yaxis = getXYAxisLayouts()
    fig.update_layout(
        uirevision=True,
        showlegend=True,
        yaxis_type="log",
        xaxis_type=lxs,
        xaxis=xaxis,
        yaxis=yaxis,
        autosize=False,
        margin=dict(
            autoexpand=False,
            l=250,
            r=300,
            t=50,
        ),
        plot_bgcolor='white'
    )
    fig.update_yaxes(range=[-17.1, -16])
    if(len(data) == 0):
        return fig

    colorsDark, colorsLight = get_color_pairs()
    maxf = 0.1
    for i in np.arange(len(data)):
        maxf = max(maxf, data[i]['fpsd'].max())
        fig.add_trace(
            go.Scattergl(x=data[i]['fpsd'],
                         y=data[i]['medxPsd'],
                         mode='lines',
                         name="Network {} - x".format(data[i]['network']),
                         line=dict(color=colorsDark[i], width=4),
            ),
        )
        fig.add_trace(
            go.Scattergl(x=data[i]['fpsd'],
                         y=data[i]['medrPsd'],
                         mode='lines',
                         name="Network {} - r".format(data[i]['network']),
                         line=dict(color=colorsLight[i]),
            ),
        )
    fig.update_xaxes(range=[0.1, maxf])

    # add horizontal line for blip at LMT
    blipLMT, text = getBlipLMT(data[i]['network'])
    fig.add_shape(
        type="line",
        x0=0,
        y0=blipLMT,
        x1=243,
        y1=blipLMT,
        line=dict(
            color="LightSeaGreen",
            width=4,
            dash="dashdot",
        ),
    )
    fig.add_trace(go.Scatter(
        showlegend=False,
        x=[30],
        y=[blipLMT*1.1],
        text=text,
        mode="text",
    ))

    fig.update_yaxes(automargin=True)
    return fig


# plotly code to generate med psd vs tone freq.
def getPvsFPlot(data):
    fig = go.Figure()
    xaxis, yaxis = getXYAxisLayouts()
    xaxis['title_text'] = "Detector Tone Freq. [MHz]"
    yaxis['title_text'] = "Median Detector PSD"
    fig.update_layout(
        uirevision=True,
        showlegend=True,
        yaxis_type="log",
        xaxis=xaxis,
        yaxis=yaxis,
        autosize=True,
        margin=dict(
            autoexpand=False,
            l=100,
            r=175,
            t=0,
        ),
        plot_bgcolor='white'
    )
    # fig.update_yaxes(range=[8.e-18, 2.e-16])
    if(len(data) == 0):
        return fig

    colorsDark, colorsLight = get_color_pairs()
    for i in np.arange(len(data)):
        fig.add_trace(
            go.Scattergl(x=data[i]['detFreqMHz'],
                         y=data[i]['medxPsd'],
                         mode='markers',
                         name="Network {} - x".format(data[i]['network']),
                         line=dict(color=colorsDark[i], width=4),
            ),
        )
        fig.add_trace(
            go.Scattergl(x=data[i]['detFreqMHz'],
                         y=data[i]['medrPsd'],
                         mode='markers',
                         name="Network {} - r".format(data[i]['network']),
                         line=dict(color=colorsLight[i]),
            ),
        )

    fig.update_yaxes(automargin=True)
    return fig


# plotly table for displaying the loadings
def getLoadingTable(data):
    nets = []
    loadings = []
    for i in np.arange(len(data)):
        nets.append('Network {}'.format(data[i]['network']))
        loadings.append('{0:3.1f} K'.format(data[i]['Tload']))
    fig = go.Figure(
        data=[
            go.Table(header=dict(values=nets),
                     cells=dict(values=loadings))
        ]
    )
    fig.update_layout(title={
        'text': "Estimated Loadings", },
        height=250, autosize=False,)
    return fig



# parser to get obsnums and networks from file names
def parseFileList(indexfiles):
    obsnums = []
    networks = []

    for f in indexfiles:
        f = os.path.basename(f)
        f = f.replace('toltec', '')
        u = f.find('_')
        n = f[:u]
        o = f[u+1:u+7]
        obsnums.append(int(o))
        networks.append(int(n))

    # go through sorted and unique obsnum list and find the networks
    networks = np.array(networks)
    obsnums = np.array(obsnums)
    oUnique = np.unique(obsnums)
    nw = []
    for o in oUnique:
        w = np.where(obsnums == o)
        nw.append([np.sort(networks[w[0]])])
    return oUnique.tolist(), nw


# common figure axis definitions
def getXYAxisLayouts():
    xaxis = dict(
        title_text="Frequency [Hz]",
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
        title_text="Median PSD [1/Hz]",
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


# returns blip at LMT
# 50, 72, 95 aW/rt(Hz) for 2.0, 1.4, 1.1mm respectively
def getBlipLMT(network):
    blipLMT = 95.e-18
    responsivity = 6.9e7
    text = "Estimated BLIP at LMT"
    if ((network >= 7) & (network <= 10)):
        blipLMT = 72.e-18
        responsivity *= 1.5
    if (network >= 11):
        blipLMT = 50.e-18
        responsivity *= 2.0
    blipLMT = blipLMT*responsivity
    blipLMT = blipLMT**2
    return blipLMT, text


# estimates photon noise
def estimateTload(network, medxPSD, medrPSD):
    Trange = [2., 5., 9., 15., 20.]
    if network <= 6:
        pn = [4.69e-18, 9.59e-18, 1.74e-17, 3.18e-17, 4.62e-17]
    elif (network >= 7) & (network <= 10):
        pn = [7.53e-18, 1.58e-17, 2.96e-17, 5.6e-17, 8.32e-17]
    else:
        pn = [6.98e-18, 1.54e-17, 3.03e-17, 6.02e-17, 9.21e-17]
    fn = interpolate.interp1d(pn, Trange, kind='quadratic', bounds_error=False, fill_value=np.nan)
    medianPhotonNoise = medxPSD-medrPSD
    Tload = fn(medianPhotonNoise)
    return Tload
