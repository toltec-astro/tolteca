#! /usr/bin/env python

from dasha.web.templates import ComponentTemplate
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
# from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from glob import glob
import numpy as np
import astropy.units as u
import colorsys
# import dash
import os
import bottleneck as bn
from tollan.utils.log import timeit, get_logger
from dasha.web.extensions.cache import cache
from dash.exceptions import PreventUpdate
import scipy.interpolate as interpolate
# import functools
from .common import HeaderWithToltecLogo
from .common.simple_basic_obs_select import KidsDataSelect
from dasha.web.templates.common import LiveUpdateSection
from tolteca.datamodels.toltec import BasicObsData


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
                    interval_option_value=5000
                    ))
        obsInput = controls_container.child(dbc.Row).child(dbc.Col).child(
                KidsDataSelect(multi=['nwid', 'array'])
                )
        obsInput.setup_live_update_section(
            app, header, query_kwargs={'obs_type': 'Timestream',
                                       'n_obs': 100})

        # this is the big psd plot at the top
        pRow = body.child(dbc.Row)
        pCol = pRow.child(dbc.Col, width=9)
        pPlot = pCol.child(
                dcc.Graph, style={'min-width': '800px'})

        # a histogram of the median psds of each detector
        hPlot = pRow.child(dbc.Col, width=3).child(
                dcc.Graph, style={'min-width': '400px'})

        # and we need a switch to toggle log_x_axis
        opt = [{"label": "log x-axis", "value": 0}]
        logx = body.child(dbc.Row).child(dbc.Col).child(dbc.Checklist,
                                                        options=opt)

        # add a table to show the detector loadings
        loadingTable = body.child(dbc.Row).child(
                dcc.Graph, style={'min-width': '1200px'})

        # these next plots will go on the same row
        plotRow = body.child(dbc.Row)

        # a plot of detector resonant freq. vs median psd value
        fPlot = plotRow.child(dbc.Col).child(
                dcc.Graph, style={'min-width': '600px'})

        # the individual detector plots
        iCtrlRow = body.child(dbc.Row)
        IQopt = [{"label": "Plot I/Q instead of x/r", "value": 0}]
        IQchoice = iCtrlRow.child(dbc.Col).child(dbc.Checklist,
                                                 options=IQopt)

        iPR = body.child(dbc.Row)
        iPlotCol = iPR.child(dbc.Col, width=9)
        iTabCol = iPR.child(dbc.Col, width=3)
        ipsdPlot = iPlotCol.child(dcc.Graph)

        ixPlot = iPlotCol.child(dcc.Graph)

        irPlot = iPlotCol.child(dcc.Graph)

        tunePlot = iTabCol.child(dcc.Graph)

        resTable = iTabCol.child(dcc.Graph)
        
        # before we register the callbacks
        super().setup_layout(app)

        # register the callbacks
        self._registerCallbacks(app, obsInput,
                                pPlot, hPlot, fPlot, logx,
                                loadingTable, ipsdPlot, ixPlot, irPlot,
                                tunePlot, IQchoice, resTable)

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
    def _registerCallbacks(self, app, obsInput,
                           pPlot, hPlot, fPlot, logx,
                           loadingTable,
                           ipsdPlot, ixPlot, irPlot,
                           tunePlot, IQchoice, resTable):

        # Generate Median PSD plot vs frequency
        @app.callback(
            [
                Output(pPlot.id, "figure"),
                Output(hPlot.id, "figure"),
                Output(fPlot.id, "figure"),
                Output(loadingTable.id, "figure")
            ],
            obsInput.inputs + [
               Input(logx.id, "value")
            ]
        )
        def makeMedianPsdPlot(obs_input, logx):
            # obs_input is the basic_obs_select_value dict
            # seen in the "details..."
            # when the nwid multi is set in the select,
            # this is a list of dict,
            # which is our case here.
            if obs_input is None:
                raise PreventUpdate
            data = fetchPsdData(obs_input)
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

        # Generate individual detector plots based on what is
        # clicked in the medPsd vs freq plot
        @app.callback(
            [
                Output(ipsdPlot.id, "figure"),
                Output(ixPlot.id, "figure"),
                Output(irPlot.id, "figure"),
                Output(tunePlot.id, "figure"),
                Output(resTable.id, "figure"),
            ],
            [
                Input(fPlot.id, "clickData"),
                Input(IQchoice.id, "value"),
            ] + obsInput.inputs
        )
        def makeIndividualPlots(clickData, IQchoice, obs_input):
            # obs_input is the basic_obs_select_value dict
            # seen in the "details..."
            # when the nwid multi is set in the select,
            # this is a list of dict,
            # which is our case here.
            if obs_input is None:
                raise PreventUpdate
            if clickData is None:
                raise PreventUpdate
            if IQchoice is None:
                IQchoice = []
            if(len(IQchoice) > 0):
                IQc = 1
            else:
                IQc = 0
            ires = clickData['points'][0]['pointNumber']
            data = fetchIndvData(obs_input, ires)
            if(len(data)>0):
                ipsdfig, ixfig, irfig, tfig = getiPlots(data, IQc)
                rtable = getTableFig(data)
            else:
                raise PreventUpdate
            return [ipsdfig, ixfig, irfig, tfig, rtable]


# Read data from netcdf files
@timeit
@cache.memoize(timeout=60 * 60 * 60)
def _fetchPsdData(filepath):

    with timeit(f"read in solved timestream from {filepath}"):
        sts = BasicObsData.read(filepath)

    def make_stat(arr, axis=None):
        med = bn.nanmedian(arr, axis=axis)
        mad = bn.nanmedian((arr - np.swapaxes(med[None, :], 0, axis)))
        p25 = bn.nanmedian(arr[arr < med])
        p75 = bn.nanmedian(arr[arr > med])
        return med, mad, p25, p75

    with timeit(f"read in data from {filepath}"):
        fs = sts.meta['tone_axis_data']['f_center']
        fpsd = sts.meta['f_psd']
        xpsd = sts.meta['x_psd']
        rpsd = sts.meta['r_psd']

    # construct the medians of the psds of the network
    medxPsd, madxPsd, stdxPsdLow, stdxPsdHigh = make_stat(xpsd, axis=0)
    medrPsd, madrPsd, stdrPsdLow, stdrPsdHigh = make_stat(rpsd, axis=0)

    # also calculate median of each detector's psd
    medxDet = bn.nanmedian(xpsd, axis=1)
    medrDet = bn.nanmedian(rpsd, axis=1)
    detFreqMHz = fs.to_value(u.MHz)

    # check for NaNs, zeros or negatives
    if np.isnan(medxPsd).any():
        print("\nNaNs detected in medxPsd.")
    if np.isnan(medrPsd).any():
        print("\nNaNs detected in medrPsd.")

    # estimate the background loading on the network
    net = sts.meta['nwid']
    Tload = estimateTload(net, np.median(medxPsd), np.median(medrPsd))

    return {
         'network': net,
         'obsnum': sts.meta['obsnum'],
         'subobsnum': sts.meta['subobsnum'],
         'scannum': sts.meta['scannum'],
         'fpsd': fpsd,
         'xpsd': xpsd,
         'rpsd': rpsd,
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
         'x': sts.x,
         'r': sts.r,
         }


# This line makes the cache invalid when the server reloads
cache.delete_memoized(_fetchPsdData)


def fetchPsdData(obs_input_list):
    logger = get_logger()
    data = []
    for obs_input in obs_input_list.values():
        raw_obs_processed_url = obs_input['raw_obs_processed']
        if raw_obs_processed_url is None:
            continue
        try:
            data.append(_fetchPsdData(raw_obs_processed_url))
        except Exception:
            logger.debug(
                f"unable to load file {raw_obs_processed_url}", exc_info=True)
            cache.delete_memoized(_fetchPsdData, raw_obs_processed_url)
            continue
    return data


# Read data from netcdf files
@cache.memoize(timeout=60 * 60 * 60)
def _fetchIndvData(filepath, ires, cal_obs, raw_obs):

    # this is the timestream data
    with BasicObsData.open(filepath) as fo:
        kd = fo.tone_loc[ires:ires+1].read()
        detFreqMHz = (kd[0].tones.to_value(u.Hz)+kd.meta['flo_center'])*1.e-6

    # processed obs don't have the raw I/Q data but we might want it
    with BasicObsData.open(raw_obs) as ro:
        raw = ro.tone_loc[ires:ires+1].read()

    # and this is the Tune data
    tune = _fetchTuneData(cal_obs, ires)

    # pack them together
    d = {
        'ires': ires,
        'network': kd.meta['roachid'],
        'obsnum': kd.meta['obsnum'],
        'subobsnum': kd.meta['subobsnum'],
        'scannum': kd.meta['scannum'],
        'fpsd': kd.meta['f_psd'],
        'xpsd': kd.meta['x_psd'][0],
        'rpsd': kd.meta['r_psd'][0],
        'Ipsd': kd.meta['I_psd'][0],
        'Qpsd': kd.meta['Q_psd'][0],
        'detFreqMHz': detFreqMHz,
        'x': kd[0].x.to_value(),
        'r': kd[0].r.to_value(),
        'I': raw[0].I.to_value(),
        'Q': raw[0].Q.to_value(),
        'tune': tune,
    }
    return d


def fetchIndvData(obs_input_list, ires):
    logger = get_logger()
    data = []
    for obs_input in obs_input_list.values():
        raw_obs_processed_url = obs_input['raw_obs_processed']
        raw_obs = obs_input['raw_obs']
        cal_obs = obs_input['cal_obs_processed']
        if raw_obs_processed_url is None:
            continue
        try:
            data.append(_fetchIndvData(raw_obs_processed_url, ires,
                                       cal_obs, raw_obs))
        except Exception:
            logger.debug(
                f"unable to load file {raw_obs_processed_url}",
                exc_info=True)
            cache.delete_memoized(_fetchIndvData,
                                  raw_obs_processed_url, ires,
                                  cal_obs, raw_obs)
            continue
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
                    colorsys.hls_to_rgb(h, min(1, l * scale), s=s)
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
            l=80,
            r=50,
            t=50,
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

    fig.update_xaxes(range=[-17.3, -15.5])
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
            l=100,
            r=150,
            t=50,
        ),
        plot_bgcolor='white'
    )
    fig.update_yaxes(range=[-17.3, -15.3])
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
            t=25,
        ),
        plot_bgcolor='white'
    )
    fig.update_yaxes(range=[-17.5, -15.3])
    if(len(data) == 0):
        return fig

    colorsDark, colorsLight = get_color_pairs()
    for i in np.arange(len(data)):
        fig.add_trace(
            go.Scattergl(x=data[i]['detFreqMHz'],
                         y=10**(data[i]['medxDet']),
                         mode='markers',
                         name="Network {} - x".format(data[i]['network']),
                         line=dict(color=colorsDark[i], width=4),
                         ),
        )
        fig.add_trace(
            go.Scattergl(x=data[i]['detFreqMHz'],
                         y=10**(data[i]['medrDet']),
                         mode='markers',
                         name="Network {} - r".format(data[i]['network']),
                         line=dict(color=colorsLight[i]),
                         ),
        )

    fig.update_yaxes(automargin=True)
    return fig


# the individual plots
def getiPlots(data, IQchoice):
    # the individual psd plot
    fig = go.Figure()
    xaxis, yaxis = getXYAxisLayouts()
    if(IQchoice):
        yaxis['title_text'] = "PSD [adu<sup>-2</sup>]"
    else:
        yaxis['title_text'] = 'PSD [1/Hz]'

    # set common margins
    margin = dict(autoexpand=False, l=100, r=150, t=25)
    fig.update_layout(
        uirevision=True,
        showlegend=True,
        yaxis_type="log",
        xaxis=xaxis,
        yaxis=yaxis,
        autosize=False,
        margin=margin,
        plot_bgcolor='white'
    )
    if(IQchoice == 0):
        fig.update_yaxes(range=[-17.3, -15.3])

    colorsDark, colorsLight = get_color_pairs()
    maxf = 0.1
    for i in np.arange(len(data)):

        # if I/Q is requested then calculate that psd
        if(IQchoice):
            y = data[i]['Ipsd']
        else:
            y = data[i]['xpsd']
        maxf = max(maxf, data[i]['fpsd'].max())
        fig.add_trace(
            go.Scattergl(x=data[i]['fpsd'],
                         y=y,
                         mode='lines',
                         name="Network {} - x".format(data[i]['network']),
                         line=dict(color=colorsDark[i], width=4),
                         ),
        )

        if(IQchoice):
            y = data[i]['Qpsd']
        else:
            y = data[i]['rpsd']
        fig.add_trace(
            go.Scattergl(x=data[i]['fpsd'],
                         y=y,
                         mode='lines',
                         name="Network {} - r".format(data[i]['network']),
                         line=dict(color=colorsLight[i]),
                         ),
        )
    fig.update_xaxes(range=[0.1, maxf])

    # if(IQchoice == 0):
    #     # add horizontal line for blip at LMT
    #     blipLMT, text = getBlipLMT(data[0]['network'])
    #     fig.add_shape(
    #         type="line",
    #         x0=0,
    #         y0=blipLMT,
    #         x1=243,
    #         y1=blipLMT,
    #         line=dict(
    #             color="LightSeaGreen",
    #             width=4,
    #             dash="dashdot",
    #         ),
    #     )
    #     fig.add_trace(go.Scatter(
    #         showlegend=False,
    #         x=[30],
    #         y=[blipLMT*1.1],
    #         text=text,
    #         mode="text",
    #     ))

    # the x/I-timestream figure
    xfig = go.Figure()
    xaxis, yaxis = getXYAxisLayouts()
    xaxis['title_text'] = 'time [s]'
    if(IQchoice):
        yaxis['title_text'] = 'I values timestream'
    else:
        yaxis['title_text'] = 'x values timestream'
    xfig.update_layout(
        uirevision=True,
        showlegend=True,
        xaxis=xaxis,
        yaxis=yaxis,
        autosize=False,
        margin=margin,
        plot_bgcolor='white'
    )

    # the r/Q-timestream figure
    rfig = go.Figure()
    if(IQchoice):
        yaxis['title_text'] = 'Q values timestream'
    else:
        yaxis['title_text'] = 'r values timestream'
    rfig.update_layout(
        uirevision=True,
        showlegend=True,
        xaxis=xaxis,
        yaxis=yaxis,
        autosize=False,
        margin=margin,
        plot_bgcolor='white'
    )

    maxf = data[0]['fpsd'].max()
    print(maxf)
    for i in np.arange(len(data)):
        maxf = max(maxf, data[i]['fpsd'].max())

        if(IQchoice):
            ydata = data[i]['I']
        else:
            ydata = data[i]['x']
        npts = len(ydata)
        xdata = np.linspace(0., npts/(1.*maxf*2.), npts)
        xfig.add_trace(
            go.Scattergl(x=xdata,
                         y=ydata,
                         mode='lines',
                         name="Network {} - x".format(data[i]['network']),
                         line=dict(color=colorsDark[i], width=4),
                         ),
        )

        if(IQchoice):
            ydata = data[i]['Q']
        else:
            ydata = data[i]['r']
        rfig.add_trace(
            go.Scattergl(x=xdata,
                         y=ydata,
                         mode='lines',
                         name="Network {} - r".format(data[i]['network']),
                         line=dict(color=colorsLight[i], width=4),
                         ),
        )

    fig.update_yaxes(automargin=True)
    xfig.update_yaxes(automargin=True)
    rfig.update_yaxes(automargin=True)
    tfig = getTunePlot(data)

    return [fig, xfig, rfig, tfig]


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
    fn = interpolate.interp1d(
            pn, Trange,
            kind='quadratic', bounds_error=False, fill_value=np.nan)
    medianPhotonNoise = medxPSD-medrPSD
    Tload = fn(medianPhotonNoise)
    return Tload


# Read data from disk or redis database
@timeit
@cache.memoize(timeout=60 * 60 * 60)
def _fetchTuneData(filepath, ires):

    with BasicObsData.open(filepath) as fo:
        tune = fo.tone_loc[ires:ires+1].read()

    with timeit(f"read in model data from {filepath}"):
        model = BasicObsData.read(filepath.replace('_processed.nc', '.txt'))
        model = model.get_model(ires)

    # generate a set of frequencies to evaluate the model
    f = tune.frequency
    nkids = f.shape[0]
    nsamps = f.shape[1]
    fm = f.copy()
    bw = 40000.*u.Hz
    for i in np.arange(nkids):
        mdf = f[i, :].mean()
        fm[i, :] = np.linspace(mdf-bw, mdf+bw, nsamps)

    # this is idiotic but I'm working on getting answers
    f0 = f.copy()
    fr = np.empty((nkids, 3), dtype='float') * u.Hz
    for i in np.arange(nkids):
        f0[i, :] = np.array([model.f0[i]]*nsamps) * u.Hz
        isort = np.argsort(np.abs(f[i, :].value-model.fr[i]))
        fr[i, :] = np.array([model.fr[i],
                             f[i, isort[0]].value,
                             f[i, isort[1]].value]) * u.Hz

    # derotate both the data and the model sweeps
    S21_orig = tune.S21.value.copy()
    S21 = model.derotate(tune.S21.value, tune.frequency)
    S21_model = model.derotate(model(fm).value, fm)
    S21_resid = S21_orig - model(f).value

    # find derotated S21 values at the resonance and tone frequencies
    S210 = model.derotate(model(f0).value, f0)
    S21r = model.derotate(model(fr).value, fr)
    S21_f0 = S210[:, 0]
    S21_fr = S21r[:, 0]

    # get angle that sweeps two points on either side of fr
    r = S21r[:, 0].real/2.
    phi_fr = (np.arctan2(np.abs(S21r[:, 1].imag), S21r[:, 1].real-r) +
              np.arctan2(np.abs(S21r[:, 2].imag), S21r[:, 2].real-r))

    phi_f0 = np.arctan2(S21_f0.imag, S21_f0.real-r)

    return {
        'network': tune.meta['roachid'],
        'obsnum': tune.meta['obsnum'],
        'subobsnum': tune.meta['subobsnum'],
        'scannum': tune.meta['scannum'],
        'LoCenterFreq': tune.meta['flo_center'],
        'InputAtten': tune.meta['atten_in'],
        'OutputAtten': tune.meta['atten_out'],
        'S21_orig': S21_orig,
        'S21_resid': S21_resid,
        'S21': S21,
        'f': tune.frequency,
        'S21_model': S21_model,
        'f_model': fm,
        'fr': model.fr,
        'f0': model.f0,
        'Qr': model.Qr,
        'S21_f0': S21_f0,
        'S21_fr': S21_fr,
        'phi_fr': phi_fr,
        'phi_f0': phi_f0,
         }


# plotly code to generate tune plot
def getTunePlot(data):
    fig = go.Figure()
    xaxis, yaxis = getXYAxisLayouts()
    xaxis['title_text'] = 'I'
    yaxis['title_text'] = 'Q'
    fig.update_yaxes(automargin=True)
    fig.update_layout(
        # uirevision=True,
        showlegend=False,
        # width=300,
        # height=300,
        xaxis=xaxis,
        yaxis=yaxis,
        autosize=True,
        margin=dict(
            autoexpand=True,
            l=20,
            r=20,
            t=25,
        ),
        plot_bgcolor='white'
    )

    # colors
    colorsDark, colorsLight = get_color_pairs()

    nplots = len(data)
    idx = 0

    for i in np.arange(nplots):
        s21 = data[i]['tune']['S21'][idx, :].value
        fig.add_trace(
            go.Scattergl(x=s21.real,
                         y=s21.imag,
                         mode='markers',
                         marker=dict(color=colorsDark[0],
                                     size=4),
            ),
        )

        s21 = data[i]['tune']['S21_model'][idx, :].value
        s21_f0 = data[i]['tune']['S21_f0'][idx].value
        s21_fr = data[i]['tune']['S21_fr'][idx].value
        fig.add_trace(
            go.Scattergl(x=s21.real,
                         y=s21.imag,
                         mode='lines',
                         line=dict(color=colorsLight[0]),
            ),
        )

        fig.add_trace(
            go.Scattergl(x=[s21_f0.real]*2,
                         y=[s21_f0.imag]*2,
                         mode='markers',
                         marker=dict(color='black',
                                     size=8),
            ),
        )

        fig.add_trace(
            go.Scattergl(x=[s21_fr.real]*2,
                         y=[s21_fr.imag]*2,
                         mode='markers',
                         marker=dict(color='red',
                                     size=8),
            ),
        )

    return fig


def getTableFig(data):
    # and finally, produce the table of kid values
    # start with just the data[0] data
    params = ['Net', 'Fr', 'F0', 'Qr', 'Phi0']
    curve = 0
    nstr = ''
    frstr = ''
    f0str = ''
    qrstr = ''
    phistr = ''
    for d in data:
        nstr += '{0:}, '.format(d['tune']['network'])
        frstr += '{0:6.3f}, '.format(d['tune']['fr'][curve]*1.e-6)
        f0str += '{0:6.3f}, '.format(d['tune']['f0'][curve]*1.e-6)
        qrstr += '{0:6.1f}, '.format(d['tune']['Qr'][curve])
        phistr += '{0:3.1f}, '.format(d['tune']['phi_f0'][curve].to_value())
    nstr = nstr[:-2]
    frstr = frstr[:-2] + ' MHz'
    f0str = f0str[:-2] + ' MHz'
    qrstr = qrstr[:-2]
    phistr = phistr[:-2] + ' rad'
    vals = [nstr, frstr, f0str, qrstr, phistr]
    values = [params, vals]
    tablefig = go.Figure(data=[go.Table(
        columnorder=[1, 2],
        columnwidth=[50, 100],
        header=dict(
            values=[['<b>Parameter</b>'],
                    ['<b>Value</b>']],
            line_color='darkslategray',
            fill_color='royalblue',
            align=['left', 'center'],
            font=dict(color='white', size=12),
            height=25
        ),
        cells=dict(
            values=values,
            line_color='darkslategray',
            fill=dict(color=['paleturquoise', 'white']),
            align=['left', 'center'],
            font_size=12,
            height=25)
    )
    ])
    tablefig.update_layout(
            # height=250,
            # width=400,
            autosize=True,
            margin=dict(
                autoexpand=True,
                l=10,
                r=10,
            #    t=30,
            ),
        )
    return tablefig
