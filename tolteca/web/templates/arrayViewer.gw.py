from plotly.subplots import make_subplots
from tollan.utils.log import timeit, get_logger
from dasha.web.templates import ComponentTemplate
from dasha.web.templates.collapsecontent import CollapseContent
from dasha.web.extensions.cache import cache
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import numpy as np
import dash
from tolteca.web.templates.dataprod.fts import FTS
from tolteca.web.templates.dataprod.efficiency import Efficiency
# from dasha.web.templates.common import LiveUpdateSection, CollapseContent
# from .common.simple_basic_obs_select import KidsDataSelect
from .common import HeaderWithToltecLogo
from astropy.table import Table


class arrayDisplay(ComponentTemplate):

    # sets up the global Div that the site lives under
    _component_cls = dbc.Container
    fluid = True
    logger = get_logger()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # This section of code only runs once when the server is started.
    def setup_layout(self, app):
        container = self
        header_section, hr_container, body = container.grid(3, 1)
        header_container = header_section.child(HeaderWithToltecLogo(
            logo_colwidth=4
            )).header_container
        title_container, controls_container = header_container.grid(2, 1)
        # header = title_container.child(
        #         LiveUpdateSection(
        #             title_component=html.H3("TolTEC Array Display"),
        #             interval_options=[2000, 5000],
        #             interval_option_value=5000
        #             ))
        hr_container.child(html.Hr())

        # ---------------------------
        # Controls
        # ---------------------------
        dPath = '/home/toltec/wilson/cooldown_210429/'
        allOpt = {
            'No Choice': ['None'],
            'FTS': [{'label': 'Combined', 'value': dPath+'fts/1530X'},
                    {'label': '15309', 'value': dPath+'fts/15309'},
                    {'label': '15300', 'value': dPath+'fts/15300'},
                    {'label': '14764', 'value': dPath+'fts/14764'}],
            'Efficiency': [{'label': 'Combined', 'value': dPath+'efficiency/153XX/'},
                           {'label': '15314', 'value': dPath+'efficiency/15314/'},
                           {'label': '14911', 'value': dPath+'efficiency/14911/'}]
            }

        controlBox = controls_container.child(dbc.Row).child(dbc.Col, width=5)
        settingsRow = controlBox.child(dbc.Row)
        choiceDrop = settingsRow.child(dbc.Col).child(
            dcc.Dropdown,
            options=[{'label': k, 'value': k} for k in allOpt.keys()],
            placeholder="Select Diagnostic",
            value='No Choice',
            searchable=False,
            style=dict(
                width='100%',
                verticalAlign="middle"
            ))
        subDropCol = settingsRow.child(dbc.Col, style={'display': 'none'})
        subDrop = subDropCol.child(dcc.Dropdown,
                                   placeholder="Select Dataset",)

        polOri = controlBox.child(dbc.Row).child(dbc.Col).child(
            dbc.Button, "Pol orientation 0", outline=True, size="sm",
            color="primary", className="mr-1")

        settings = {'choiceDrop': choiceDrop,
                    'subDropCol': subDropCol,
                    'subDrop': subDrop,
                    'polOri': polOri}

        # a container for the rest of the page
        bigBox = body

        # ---------------------------
        # Plots
        # ---------------------------
        arrayRow = bigBox.child(dbc.Row, width=12)
        col1p1 = arrayRow.child(dbc.Col, width=4)
        a1p1Plot = col1p1.child(dbc.Row).child(dcc.Graph)
        col1p4 = arrayRow.child(dbc.Col, width=4)
        a1p4Plot = col1p4.child(dbc.Row).child(dcc.Graph)
        col2p0 = arrayRow.child(dbc.Col, width=4)
        a2p0Plot = col2p0.child(dbc.Row).child(dcc.Graph)
        arrayPlots = {'a1p1Plot': a1p1Plot,
                      'a1p4Plot': a1p4Plot,
                      'a2p0Plot': a2p0Plot}

        netRow = bigBox.child(dbc.Row).child(dbc.Col, width=12)
        netCtrl = netRow.child(dbc.Row)
        netCheck = netCtrl.child(
            dbc.Checklist,
            options=[
                {'label': '{}'.format(i), 'value': i} for i in np.arange(13)],
            value=[],
            inline=True,
        )
        arrRadio = netCtrl.child(
            dbc.RadioItems,
            options=[
                {'label': '1.1mm', 'value': 'a1100'},
                {'label': '1.4mm', 'value': 'a1400'},
                {'label': '2.0mm', 'value': 'a2000'}],
            value=[],
            inline=True,
        )
        netPlot = netRow.child(dbc.Row).child(dcc.Graph)
        netBox = {'netPlot': netPlot,
                  'netCheck': netCheck,
                  'arrRadio': arrRadio, }

        # store the state of the button
        polStore = bigBox.child(dcc.Store)

        # store the data pulled from the netcdf files
        dataStore = bigBox.child(dcc.Store)

        # store the info for the clicked detector
        detStore = bigBox.child(dcc.Store)

        detailRow = bigBox.child(dbc.Row, className='mt-3').child(dbc.Col)
        detailCard = detailRow.child(dbc.Card)

        # before we register the callbacks
        super().setup_layout(app)

        # register the callbacks
        self._registerCallbacks(app, settings, arrayPlots,
                                netBox, dataStore, detailCard, detStore,
                                allOpt, polStore)

    def _registerCallbacks(self, app, settings, arrayPlots,
                           netBox, dataStore, detailCard, detStore,
                           allOpt, polStore):
        print("Registering Callbacks")

        # ---------------------------
        # Primary dropdown
        # ---------------------------
        @app.callback(
            [
                Output(settings['subDrop'].id, 'value'),
                Output(settings['subDrop'].id, 'options'),
                Output(settings['subDropCol'].id, 'style'),
            ],
            [
                Input(settings['choiceDrop'].id, 'value'),
            ],
        )
        def primaryDropdown(choice):
            options = [{'label': 'foo', 'value': 'bar'}]
            if (len(allOpt[choice]) == 1):
                data = None
                style = {'display': 'none'}
            elif len(allOpt[choice]) > 1:
                data = None
                style = {'display': 'block'}
                options = [opt for opt in allOpt[choice]]
            return [data, options, style]

        # ---------------------------
        # Secondary Dropdown
        # ---------------------------
        @app.callback(
            [
                Output(dataStore.id, "data"),
            ],
            [
                Input(settings['subDrop'].id, 'value'),
            ],
            [
                State(settings['choiceDrop'].id, 'value'),
            ],
            prevent_initial_call=True
        )
        def secondaryDropdown(path, choice):
            data = {'data': fetchArrayData(choice, path=path)}
            return [data]

        # ---------------------------
        # Network Selection
        # ---------------------------
        @app.callback(
            [
                Output(netBox['netCheck'].id, "value"),
            ],
            [
                Input(netBox['arrRadio'].id, "value"),
            ],
        )
        def netManage(a):
            if a == 'a1100':
                return [[0, 1, 2, 3, 4, 5, 6]]
            elif a == 'a1400':
                return [[7, 8, 9, 10]]
            elif a == 'a2000':
                return [[11, 12]]
            else:
                raise PreventUpdate

        # ---------------------------
        # Polarization Orientation Button
        # ---------------------------
        @app.callback(
            [
                Output(polStore.id, "data"),
                Output(settings['polOri'].id, "children"),
            ],
            [
                Input(settings['polOri'].id, "n_clicks"),
            ],
            State(settings['polOri'].id, "children")
        )
        def buttonUpdate(n_clicks, orientation):
            if n_clicks is None:
                raise PreventUpdate
            if n_clicks % 2:
                ori = 1
            else:
                ori = 0
            name = 'Pol Orientation {}'.format(ori)
            return [ori, name]

        # ---------------------------
        # Array Plots
        # ---------------------------
        @app.callback(
            [
                Output(arrayPlots['a1p1Plot'].id, "figure"),
                Output(arrayPlots['a1p4Plot'].id, "figure"),
                Output(arrayPlots['a2p0Plot'].id, "figure"),
            ],
            [
                Input(dataStore.id, "data"),
                Input(polStore.id, "data"),
            ],
        )
        def makeArrayPlots(store, orientation):
            if orientation is None:
                orientation = 0
            if store is None:
                raise PreventUpdate
            apt = getDefaultAPT(orientation)
            data = store['data']
            imatch = simpleMatch(data, apt)
            figs = makeArrayFig(data, apt, imatch)
            return [figs['a1100'], figs['a1400'], figs['a2000']]

        # ---------------------------
        # Network Plots
        # ---------------------------
        @app.callback(
            [
                Output(netBox['netPlot'].id, "figure"),
            ],
            [
                Input(dataStore.id, "data"),
                Input(netBox['netCheck'].id, "value"),
            ],
        )
        def makeNetworkPlots(store, nets):
            if store is None:
                raise PreventUpdate
            data = store['data']
            fig = makeNetworkFig(data, nets)
            return [fig]

        # ---------------------------
        # Detector Selection
        # ---------------------------
        @app.callback(
            Output(detStore.id, "data"),
            [
                Input(arrayPlots['a1p1Plot'].id, "clickData"),
                Input(arrayPlots['a1p4Plot'].id, "clickData"),
                Input(arrayPlots['a2p0Plot'].id, "clickData"),
            ])
        def clickForDet(a1p1Click, a1p4Click, a2p0Click):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
            # determine array
            array_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if(array_id == arrayPlots['a1p1Plot'].id):
                array = 'a1100'
                click = a1p1Click
            elif(array_id == arrayPlots['a1p4Plot'].id):
                array = 'a1400'
                click = a1p4Click
            else:
                array = 'a2000'
                click = a2p0Click
            # determine network and detector
            # for some reason, the very first click after the page
            # loads produces both a click event and somehow gets here
            # with click=None.
            if(click is not None):
                network = click['points'][0]['curveNumber']
                detID = click['points'][0]['pointIndex']
                out = {'array': array,
                       'network': network,
                       'detector': detID}
            else:
                raise PreventUpdate
            return out


# Based on the selection made in the pull-down box, fetch the data
# needed to populate the array plots.
@cache.memoize(timeout=60 * 60 * 60)
def fetchArrayData(choice, path=''):
    data = None
    if(choice == 'None'):
        return None
    elif(choice == 'FTS'):
        if(path is None):
            return None
        ftsPath = path
        fts = FTS(ftsPath)
        fts.getArrayData()
        fts.getNetworkAverageValues()
        fts.getArrayAverageValues()
        data = {'type': 'fts',
                'path': ftsPath,
                'nets': fts.nets,
                'goodNets': fts.goodNets,
                'a1100': fts.a1100,
                'a1400': fts.a1400,
                'a2000': fts.a2000}
    elif(choice == 'Efficiency'):
        if(path is None):
            return None
        efficiencyPath = path
        efficiency = Efficiency(efficiencyPath)
        efficiency.getArrayData()
        efficiency.getNetworkAverageValues()
        efficiency.getArrayAverageValues()
        data = {'type': 'efficiency',
                'path': efficiencyPath,
                'nets': efficiency.nets,
                'goodNets': efficiency.goodNets,
                'a1100': efficiency.a1100,
                'a1400': efficiency.a1400,
                'a2000': efficiency.a2000}
    return data


cache.delete_memoized(fetchArrayData)


# get the default array property table
@cache.memoize(timeout=60 * 60 * 60)
def getDefaultAPT(orientation):
    aptFile = '/home/toltec/toltec_astro/toltec_calib/prod/toltec_wyattmap/data/array_prop.ecsv'
    apt = Table.read(aptFile, format='ascii.ecsv')
    apt = apt[apt['ori'] == orientation]
    return apt


cache.delete_memoized(getDefaultAPT)


@cache.memoize(timeout=60 * 60 * 60)
def simpleMatch(data, apt):
    if data is None:
        return None
    imatch = {}
    for n in data['goodNets']:
        nw = int(n.replace('N', ''))
        anw = apt[apt['nw'] == nw]
        ilist = []
        for a in anw:
            fin = a['f_in']
            fd = data['nets'][n]['fres']
            i = (np.abs(fd-fin)).argmin()
            ilist.append(i)
            imatch[n] = ilist
    return imatch


cache.delete_memoized(simpleMatch)


# Fetch the network level figure
def makeNetworkFig(data, nets):
    if (data is None) | (len(nets) == 0):
        return getEmptyFig(600, 400)
    ns = ['N{}'.format(n) for n in nets]
    type = data['type']
    path = data['path']
    if type == 'fts':
        fts = FTS(path)
        return fts.getPlotlyNetworkAvg(data, ns, fhigh=500)
    if type == 'efficiency':
        efficiency = Efficiency(path)
        return efficiency.getPlotlyNetworkAvg(data, ns)


# Generate the array figures
def makeArrayFig(data, apt, imatch):
    anames = ['a1100', 'a1400', 'a2000']
    acolors = ['#0b0930', '#1a1a64', '#2c2a89', '#453AA4',
               '#5c49c6', '#7b6fde', '#2a75b3', '#4b0908',
               '#6a0c0b', '#aa0505', '#b97d10', '#f2d259',
               '#f8e891']

    asize = [7, 9, 11]
    figs = {}
    for an, c, sz in zip(anames, acolors, asize):
        kids = apt[apt['array_name'] == an]
        nets = list(np.unique(kids['nw']))
        tfig = go.Figure()
        xaxis, yaxis = getXYAxisLayouts()
        xaxis['title'] = an

        if data is None:
            for nw in nets:
                c = acolors[nw]
                marker = dict(color=c, size=sz, showscale=True,
                              line=dict(width=1, color=c),
                              cmin=0, cmax=10,
                              colorbar=dict(
                                  nticks=0,
                                  tickmode='auto'))
                anw = kids[kids['nw'] == nw]
                tfig.add_trace(
                    go.Scattergl(x=anw['x_t'],
                                 y=anw['y_t'],
                                 mode='markers',
                                 marker=marker))
        else:
            cmin = 1e9
            cmax = 1e-9
            for nw in nets:
                n = 'N{}'.format(nw)
                if(n in data['goodNets']):
                    cmin = min(np.array(data['nets'][n]['signal']).min(), cmin)
                    cmax = max(np.array(data['nets'][n]['signal']).max(), cmax)
            marker = dict(size=sz,
                          cmin=cmin,
                          cmax=cmax,
                          colorscale='thermal',
                          showscale=True,
                          line=dict(width=1,
                                    color=c),
                          colorbar=dict(
                              nticks=5,
                              tickmode='auto'))
            # do this one network at a time
            for nw in nets:
                n = 'N{}'.format(nw)
                anw = kids[kids['nw'] == nw]
                if(n in data['goodNets']):
                    cp = np.array(data['nets'][n]['signal'])[imatch[n][:]]
                else:
                    cp = ['white']*len(anw['f_in'])
                marker['color'] = cp
                tfig.add_trace(
                    go.Scattergl(x=anw['x_t'],
                                 y=anw['y_t'],
                                 mode='markers',
                                 marker=marker))
        tfig.update_layout(
            uirevision=True,
            showlegend=False,
            width=550,
            height=550,
            xaxis=xaxis,
            yaxis=yaxis,
            autosize=True,
            margin=dict(
                autoexpand=True,
                l=10,
                r=10,
                t=10,
            ),
            plot_bgcolor='white'
        )
        figs[an] = tfig
    return figs


def getEmptyFig(width, height):
    xaxis, yaxis = getXYAxisLayouts()
    fig = go.Figure()
    fig.update_layout(
        uirevision=True,
        showlegend=False,
        width=width,
        height=height,
        xaxis=xaxis,
        yaxis=yaxis,
        autosize=True,
        margin=dict(
            autoexpand=True,
            l=10,
            r=10,
            t=30,
        ),
        plot_bgcolor='white'
    )
    return fig


# common figure axis definitions
def getXYAxisLayouts():
    xaxis = dict(
        titlefont=dict(size=20),
        showline=False,
        showgrid=False,
        showticklabels=False,
        linecolor='white',
        linewidth=0,
        ticks='',
    )

    yaxis = dict(
        titlefont=dict(size=20),
        showline=False,
        showgrid=False,
        showticklabels=False,
        linecolor='white',
        linewidth=0,
        ticks='',
    )
    return xaxis, yaxis
