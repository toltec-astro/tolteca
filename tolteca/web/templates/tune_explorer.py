#! /usr/bin/env python

# ToDo:
# 1) move all the header material to a utility in .templates.common
# 2) add plots of residuals of I and Q
# 3) add vertical lines to indiv. kid plots for tune value
# 4) add switch to convert between radial plot and histogram of phi values

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
from tollan.utils.log import timeit, get_logger
from dasha.web.extensions.cache import cache
from dash.exceptions import PreventUpdate
from dasha.web.extensions.db import dataframe_from_db
from tolteca.datamodels.toltec import BasicObsData
import json
import cachetools.func
from pathlib import Path
from itertools import cycle
from dasha.web.templates.common import LiveUpdateSection, CollapseContent
from .common import HeaderWithToltecLogo
from .common.simple_basic_obs_select import KidsDataSelect
from astropy import units as u


class TuneExplorer(ComponentTemplate):

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
                    title_component=html.H3("Tune Explorer"),
                    interval_options=[2000, 5000],
                    interval_option_value=5000
                    ))
        obsInput = controls_container.child(
                KidsDataSelect(multi=None)
                )
        obsInput.setup_live_update_section(
            app, header, query_kwargs={'obs_type': 'TUNE',
                                       'n_obs': 100})

        # add a kidSet plot to select groups of detectors by frequency
        s6 = {'min-width': '600px'}
        kidSetPlot = body.child(dbc.Row).child(dbc.Col).child(dcc.Graph,
                                                              style=s6)

        # a container for the full set of plots
        bigBox = body.child(dbc.Row)

        # this is the big tune grid
        tunePlot = bigBox.child(dbc.Col, width=6).child(dcc.Graph)

        # add a column of plots showing the selected resonance
        kidCol = bigBox.child(dbc.Col, width=3)
        S21Plot = kidCol.child(dbc.Row).child(dcc.Graph)
        IPlot = kidCol.child(dbc.Row).child(dcc.Graph)
        QPlot = kidCol.child(dbc.Row).child(dcc.Graph)

        # add another column of plots with the array average values
        arrayCol = bigBox.child(dbc.Col, width=3)
        tablePlot = arrayCol.child(dbc.Row).child(dcc.Graph)
        QrPlot = arrayCol.child(dbc.Row).child(dcc.Graph)
        PhiPlot = arrayCol.child(dbc.Row).child(dcc.Graph)

        # before we register the callbacks
        super().setup_layout(app)

        # register the callbacks
        self._registerCallbacks(app, obsInput,
                                tunePlot,
                                S21Plot, IPlot, QPlot,
                                kidSetPlot,
                                tablePlot, QrPlot, PhiPlot)

    # register the callbacks
    def _registerCallbacks(self, app, obsInput,
                           tunePlot,
                           S21Plot, IPlot, QPlot,
                           kidSetPlot,
                           tablePlot, QrPlot, PhiPlot):

        # Generate the kidSet plot at the top and arrayPlot on side
        @app.callback(
            [
                Output(kidSetPlot.id, "figure"),
                Output(QrPlot.id, "figure"),
                Output(PhiPlot.id, "figure")
            ],
            obsInput.inputs,
        )
        def makeKidSetPlot(obs_input):
            # obs_input is the basic_obs_select_value dict
            # seen in the "details..."
            # when the nwid multi is set in the select,
            # this is a list of dict
            # otherwise is a dict, which is our case here.
            if obs_input is None:
                raise PreventUpdate
            data = fetchTuneData(**obs_input)
            if data is None:
                raise PreventUpdate
            rtfig = getKidSetGrid(data)
            qrfig, phifig = getArrayAverageFig(data)
            return [rtfig, qrfig, phifig]

        # Generate TUNE plot
        @app.callback(
            [
                Output(tunePlot.id, "figure"),
            ],
            [
                Input(kidSetPlot.id, "clickData"),
            ] + obsInput.inputs
        )
        def makeTunePlot(kidSetClickData, obs_input):
            if obs_input is None:
                raise PreventUpdate
            data = fetchTuneData(**obs_input)
            if data is None:
                raise PreventUpdate
            if(kidSetClickData is None):
                kidSet = 0
            else:
                kidSet = kidSetClickData['points'][0]['curveNumber']
            tunefig = getTunePlot(data, kidSet)
            return [tunefig]

        # Generate Kid plot
        @app.callback(
            [
                Output(S21Plot.id, "figure"),
                Output(IPlot.id, "figure"),
                Output(QPlot.id, "figure"),
                Output(tablePlot.id, "figure")
            ],
            [
                Input(tunePlot.id, "clickData"),
                Input(kidSetPlot.id, "clickData"),
            ] + obsInput.inputs
        )
        def makeKidPlot(tuneClickData, kidSetClickData, obs_input):
            if obs_input is None:
                raise PreventUpdate
            data = fetchTuneData(**obs_input)
            if data is None:
                raise PreventUpdate
            if(kidSetClickData is None):
                kidSet = 0
            else:
                kidSet = kidSetClickData['points'][0]['curveNumber']
            s21fig, ifig, qfig, tablefig = getKidPlot(data, tuneClickData,
                                                      kidSet)
            return [s21fig, ifig, qfig, tablefig]


# Read data from disk or redis database
@timeit
@cache.memoize(timeout=60 * 60 * 60)
def _fetchTuneData(filepath):

    with timeit(f"read in processed data from {filepath}"):
        tune = BasicObsData.read(filepath)

    with timeit(f"read in model data from {filepath}"):
        model = BasicObsData.read(filepath.replace('_processed.nc', '.txt'))
        model = model.model

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
        'SenseAtten': tune.meta['atten_sense'],
        'DriveAtten': tune.meta['atten_drive'],
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


# This line makes the cache invalid when the server reloads
cache.delete_memoized(_fetchTuneData)


def fetchTuneData(**kwargs):
    raw_obs_processed_url = kwargs['raw_obs_processed']
    if raw_obs_processed_url is None:
        # no processed data
        return None
    data = _fetchTuneData(raw_obs_processed_url)
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
                        'tune.nc', 'tune_processed.nc')
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


def getArrayAverageFig(data):
    if(len(data) == 0):
        fig = go.figure()
        return [fig, fig]

    colorsDark, colorsLight = get_color_pairs()
    xaxis, yaxis = getXYAxisLayouts()

    # the histogram of fitted Qr values
    qrfig = go.Figure()
    qrfig.add_trace(
        go.Histogram(
            x=data['Qr'],
            bingroup=1,
            marker_color=colorsDark[7],
        ),
    )
    qrfig.update_layout(xaxis=xaxis, yaxis=yaxis)

    # the histogram of phi_f0 values
    linear = 0
    if(linear):
        phifig = go.Figure()
        phifig.add_trace(
            go.Histogram(
                x=data['phi_f0'],
                marker_color=colorsDark[1],
                xbins=dict(
                    start=-np.pi/2.,
                    end=np.pi/2.,
                    size=0.1,),
            ),
        )
        # vertical lines to show "in-tune" definition
        x = np.median(data['phi_fr'].value)*2.
        phifig.update_layout(
            shapes=[
                dict(type="line", xref="x", yref="paper",
                     x0=x, y0=0, x1=x, y1=200, line_width=1,
                     line_color=colorsLight[2]),
                dict(type="line", xref="x", yref='paper',
                     x0=-x, y0=0, x1=-x, y1=200, line_width=1,
                     line_color=colorsLight[2])
            ]
        )
        phifig.update_layout(xaxis=xaxis, yaxis=yaxis)
    else:
        phifig = go.Figure()
        h, b = np.histogram(data['phi_f0'], bins=30,
                            range=[-np.pi/2, np.pi/2.])
        bsize = np.pi/30.
        x = np.median(data['phi_fr'].value)*2. * 180./np.pi
        phifig.add_trace(
            go.Barpolar(
                r=[h.max()],
                theta=[0.],
                width=x*2,
                marker_color=colorsLight[2],
                ))
        phifig.add_trace(
            go.Barpolar(
                r=h,
                theta=(b[0:-1].value+bsize/2.)*180./np.pi,
                width=18./np.pi,
                marker_color=colorsDark[8],
                )
            )

    for fig in [qrfig, phifig]:
        fig.update_yaxes(automargin=True)
        fig.update_layout(
            #xaxis=xaxis,
            #yaxis=yaxis,
            showlegend=False,
            # width=800,
            height=250,
            width=400,
            autosize=True,
            margin=dict(
                autoexpand=True,
                l=10,
                r=10,
                t=20,
            ),
            plot_bgcolor='white'
        )

    # final touches on axis labels
    qrfig.update_yaxes(title="#")
    qrfig.update_xaxes(title="Fitted Qr")
    phifig.update_yaxes(title="#")
    phifig.update_xaxes(title="Phi [radians]",
                        range=[-np.pi/2., np.pi/2.])

    return [qrfig, phifig]


def getKidPlot(data, tuneClickData, kidSet):
    # what curve was clicked.  Each graph has 4 curves so divide by 4
    # to get the corresponding index in data.
    if(tuneClickData is None):
        curve = 0
    else:
        curve = int(np.floor(tuneClickData['points'][0]['curveNumber']/4))
        curve = curve + 25*kidSet

    s21fig = go.Figure()
    ifig = go.Figure()
    qfig = go.Figure()
    f = data['f'][curve, :].value*1.e-6
    I = data['S21_orig'][curve, :].real
    Q = data['S21_orig'][curve, :].imag
    s21mag = np.sqrt(I**2+Q**2)
    s21mag = 20.*np.log10(s21mag)

    ifig.add_trace(
        go.Scattergl(x=f,
                     y=I,
                     mode='lines',),
    )
    ifig.update_yaxes(title="I [ADCU]")

    qfig.add_trace(
        go.Scattergl(x=f,
                     y=Q,
                     mode='lines',),
    )
    qfig.update_yaxes(title="Q [ADCU]")

    s21fig.add_trace(
        go.Scattergl(x=f,
                     y=s21mag,
                     mode='lines',),
    )
    s21fig.update_yaxes(title="abs(S21) [dB]")

    xaxis, yaxis = getXYAxisLayouts()
    for fig in [s21fig, ifig, qfig]:
        fig.update_yaxes(automargin=True)
        fig.update_layout(
            xaxis=xaxis,
            yaxis=yaxis,
            showlegend=False,
            height=250,
            width=400,
            autosize=True,
            margin=dict(
                autoexpand=True,
                l=10,
                r=10,
                t=20,
            ),
            plot_bgcolor='white'
        )

    # and finally, produce the table of kid values
    params = ['Fr', 'F0', 'Qr', 'Phi0']
    vals = ['{0:6.3f} MHz'.format(data['fr'][curve]*1.e-6),
            '{0:6.3f} MHz'.format(data['f0'][curve]*1.e-6),
            '{0:6.1f}'.format(data['Qr'][curve]),
            '{0:3.1f}'.format(data['phi_f0'][curve]),
    ]
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
            height=250,
            width=400,
            autosize=True,
            margin=dict(
                autoexpand=True,
                l=20,
                r=40,
                t=30,
            ),
        )

    return s21fig, ifig, qfig, tablefig


# plotly code to generate tune plot
def getTunePlot(data, kidSet):
    fig = make_subplots(rows=5, cols=5,
                        vertical_spacing=0.02)
    xaxis, yaxis = getXYAxisLayouts()

    # colors
    colorsDark, colorsLight = get_color_pairs()
    colors = zip(colorsDark, colorsLight)
    colorsCycle = cycle(colors)

    for i in np.arange(5):
        for j in np.arange(5):
            idx = 5*i+j + 25*kidSet
            s21 = data['S21'][idx, :].value
            c = next(colorsCycle)
            fig.add_trace(
                go.Scattergl(x=s21.real,
                             y=s21.imag,
                             mode='markers',
                             marker=dict(color=c[0],
                                         size=4),
                ),
                row=i+1, col=j+1,
            )
            s21 = data['S21_model'][idx, :].value
            s21_f0 = data['S21_f0'][idx].value
            s21_fr = data['S21_fr'][idx].value
            fig.add_trace(
                go.Scattergl(x=s21.real,
                             y=s21.imag,
                             mode='lines',
                             line=dict(color=c[1]),
                ),
                row=i+1, col=j+1,
            )
            fig.add_trace(
                go.Scattergl(x=[s21_f0.real]*2,
                             y=[s21_f0.imag]*2,
                             mode='markers',
                             marker=dict(color='black',
                                         size=8),
                ),
                row=i+1, col=j+1,
            )
            fig.add_trace(
                go.Scattergl(x=[s21_fr.real]*2,
                             y=[s21_fr.imag]*2,
                             mode='markers',
                             marker=dict(color='red',
                                         size=8),
                ),
                row=i+1, col=j+1,
            )


    with timeit("update y-axis and layout"):
        fig.update_yaxes(automargin=True)
        fig.update_layout(
            # uirevision=True,
            showlegend=False,
            width=800,
            height=800,
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


# plotly table for displaying the number of Kids
def getKidSetGrid(data):
    network = data['network']
    nResonators, nf = data['S21'].shape
    ngrids = int(np.floor(nResonators/25))

    # before we dive in, how many resonators are sufficiently tuned?
    tunedCount = 0
    for i in np.arange(nResonators):
        if((data['phi_f0'][i] <= data['phi_fr'][i]*2) &
           (data['phi_f0'][i] >= -data['phi_fr'][i]*2)):
            tunedCount += 1

    # colors
    colorsDark, colorsLight = get_color_pairs()
    colorsCycle = cycle(colorsDark)

    # build a set of ngrids curves to signify each grid
    # the boundaries of the curves are given in frequencies
    fig = go.Figure()
    npts = 10
    for i in np.arange(ngrids):
        c = next(colorsCycle)
        si = i*25
        ei = si+24
        sf = data['f'][si, 0].value*1.e-6
        ef = data['f'][ei, -1].value*1.e-6
        if(ef < sf):
            ef = data['f'][si,-1].value*1.e-6
        fig.add_trace(
            go.Scattergl(x=np.linspace(sf, ef, npts),
                         y=[1.]*npts,
                         mode='lines',
                         line=dict(color=c,
                                   width=10))
        )
    fig.update_xaxes(title_text="",
                     showline=True,
                     showgrid=True,
                     showticklabels=True,
                     position=0.,
                     ticksuffix=" MHz",
                     )
    fig.update_yaxes(showticklabels=False,
                     showgrid=False,
                     showline=False)
    tt = "Network {0:}, {1:} Resonators: ({2:3.1f}% in tune)"
    fig.update_layout(
        title={'text': tt.format(network,
                                 nResonators,
                                 100.*tunedCount/nResonators)},
        height=100,
        autosize=True,
        showlegend=False,
        margin=dict(
            l=0,
            r=0,
            b=30,
            t=30,
        ),
    )
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
