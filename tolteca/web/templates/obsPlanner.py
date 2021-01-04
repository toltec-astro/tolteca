from tollan.utils.log import timeit, get_logger
from dasha.web.templates import ComponentTemplate
from dasha.web.extensions.cache import cache
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import cachetools.func
from pathlib import Path
from itertools import cycle
from .common import HeaderWithToltecLogo
import plotly.graph_objs as go
import dash
import json
import yaml
from io import StringIO

from .common.toltec_sensitivity.Detector import Detector

from astropy.wcs.utils import celestial_frame_to_wcs
from astropy.convolution import Gaussian2DKernel
from astroquery.utils import parse_coordinates
from astropy.convolution import convolve_fft
from tolteca.simu import SimulatorRuntime
from astropy.coordinates import SkyCoord
from astroquery.skyview import SkyView
from astropy import units as u
from astropy.time import Time
from astropy.wcs import WCS
from datetime import date
import numpy as np


class obsPlanner(ComponentTemplate):

    # sets up the global Div that the site lives under
    _component_cls = dbc.Container
    logger = get_logger()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # This section of code only runs once when the server is started.
    def setup_layout(self, app):
        container = self
        header_section, hr_container, body = container.grid(3, 1)
        hr_container.child(html.H3, "TolTEC Observation Planner",
                           style={'color': 'blue'})
        hr_container.child(html.Hr())

        # a container for the rest of the page
        bigBox = body.child(dbc.Row)

        # control column
        controlBox = bigBox.child(dbc.Col, width=3)

        # ---------------------------
        # Settings Card
        # ---------------------------
        settingsRow = controlBox.child(dbc.Row)
        settingsCard = settingsRow.child(dbc.Col).child(dbc.Card)
        t_header = settingsCard.child(dbc.CardHeader)
        t_header.child(html.H5, "Global Settings", className='mt-3')
        t_body = settingsCard.child(dbc.CardBody)

        bandRow = t_body.child(dbc.Row, justify='begin')
        bandRow.child(html.Label("TolTEC Band: "))
        bandIn = bandRow.child(
            dcc.RadioItems, options=[
                {'label': '1.1', 'value': 1.1},
                {'label': '1.4', 'value': 1.4},
                {'label': '2.0', 'value': 2.0},
            ],
            value=1.1,
            labelStyle={'display': 'inline-block'},
            inputStyle={"margin-right": "10px",
                        "margin-left": "10px"},
        )

        atmQRow = t_body.child(dbc.Row, justify='begin')
        atmQRow.child(html.Label("Atmos. quartile: "))
        atmQIn = atmQRow.child(
            dcc.RadioItems, options=[
                {'label': '25%', 'value': 25},
                {'label': '50%', 'value': 50},
                {'label': '75%', 'value': 75},
            ],
            value=50,
            labelStyle={'display': 'inline-block'},
            inputStyle={"margin-right": "5px",
                        "margin-left": "20px"},)

        telRow = t_body.child(dbc.Row, justify='begin')
        telRow.child(html.Label("Telescope RMS [um]: "))
        telRMSIn = telRow.child(dcc.Input, value=76.,
                                min=50., max=110.,
                                debounce=True, type='number',
                                style={'width': '25%',
                                       'margin-right': '20px',
                                       'margin-left': '20px'})

        unitsRow = t_body.child(dbc.Row, justify='begin')
        unitsRow.child(html.Label("Coverage Units: "))
        unitsIn = unitsRow.child(
            dcc.RadioItems, options=[
                {'label': 's/pixel', 'value': "time"},
                {'label': 'mJy/beam', 'value': "sens"},
            ],
            value="sens",
            labelStyle={'display': 'inline-block'},
            inputStyle={"margin-right": "10px",
                        "margin-left": "10px"},
        )

        showArrRow = t_body.child(dbc.Row, justify='begin')
        showArray = showArrRow.child(
            dcc.Checklist,
            options=[{'label': 'Show array', 'value': 'array'}],
            value=[],
            inputStyle={"marginRight": "20px", "marginLeft": "5px"})

        overlayRow = t_body.child(dbc.Row, justify='begin')
        overlayDrop = overlayRow.child(
            dcc.Dropdown,
            options=[
                {'label': 'No Overlay', 'value': 'None'},
                {'label': 'DSS', 'value': 'DSS'},
                {'label': 'Planck 217 I', 'value': 'Planck 217 I'},
                {'label': 'Planck 353 I', 'value': 'Planck 353 I'},
                {'label': 'WISE 12', 'value': 'WISE 12'},
            ],
            placeholder="Select Overlay Image",
            searchable=False,
            value='None',
            style=dict(
                width='100%',
                verticalAlign="middle"
            ))

        settings = {'bandIn': bandIn,
                    'atmQIn': atmQIn,
                    'telRMSIn': telRMSIn,
                    'unitsIn': unitsIn,
                    'showArray': showArray,
                    'overlayDrop': overlayDrop}

        # ---------------------------
        # Source Card
        # ---------------------------
        sourceBox = controlBox.child(dbc.Row, className='mt-3').child(dbc.Col)
        sourceCard = sourceBox.child(dbc.Card, color='danger', inverse=False,
                                     outline=True)
        c_header = sourceCard.child(dbc.CardHeader)
        c_body = sourceCard.child(dbc.CardBody)
        c_header.child(html.H5, "Target Choice", className='mb-2')

        targNameRow = c_body.child(dbc.Row, justify='end')
        targNameRow.child(html.Label("Search Target Name: "))
        targName = targNameRow.child(dcc.Input, value="", debounce=True,
                                     type='text',
                                     style=dict(width='45%',))

        targRaRow = c_body.child(dbc.Row, justify='end')
        targRaRow.child(html.Label("Target Ra [deg]: "))
        targRa = targRaRow.child(dcc.Input, debounce=True,
                                 value=150.08620833, type='number',
                                 style=dict(width='45%',))

        targDecRow = c_body.child(dbc.Row, justify='end')
        targDecRow.child(html.Label("Target Dec [deg]: "))
        targDec = targDecRow.child(dcc.Input, debounce=True,
                                   value=2.58899167, type='number',
                                   style=dict(width='45%',))

        obsTimeRow = c_body.child(dbc.Row, justify='end')
        obsTimeRow.child(html.Label("Obs Start Time (UT): "))
        obsTime = obsTimeRow.child(dcc.Input, debounce=True,
                                   value="01:30:00", type='text',
                                   style=dict(width='45%',))

        obsDateRow = c_body.child(dbc.Row, justify='end')
        obsDateRow.child(html.Label("Obs Date: "))
        obsDate = obsDateRow.child(dcc.DatePickerSingle,
                                   min_date_allowed=date(2000, 11, 19),
                                   max_date_allowed=date(2030, 12, 31),
                                   initial_visible_month=date(2021, 4, 4),
                                   date=date(2021, 4, 4),
                                   style=dict(width='45%',))

        targetAlertRow = c_body.child(dbc.Row)
        targetAlert = targetAlertRow.child(
            dbc.Alert,
            "Source elevation too high or too low for obsDate and obsTime.",
            is_open=False,
            color='danger',
            duration=8000)

        targetPopUp = c_body.child(dbc.Row).child(
            dcc.ConfirmDialog,
            message='Set obsTime and obsDate so that 20 < elevation < 80 deg.')

        target = {'targName': targName,
                  'targRa': targRa,
                  'targDec': targDec,
                  'obsTime': obsTime,
                  'obsDate': obsDate,
                  'targetAlert': targetAlert,
                  'targetPopUp': targetPopUp}

        # ---------------------------
        # Mapping Tabs
        # ---------------------------
        mappingBox = controlBox.child(dbc.Row, className='mt-3').child(dbc.Col).child(dbc.Tabs)
        lissBox = mappingBox.child(dbc.Tab, label="Lissajous")
        lissCard = lissBox.child(dbc.Card)
        l_header = lissCard.child(dbc.CardHeader)
        l_body = lissCard.child(dbc.CardBody)
        l_header.child(html.H5, "Lissajous Controls", className='mb-2')

        lisRotInRow = l_body.child(dbc.Row, justify='end')
        lisRotInRow.child(html.Label("Rot [deg]: "))
        lisRotIn = lisRotInRow.child(dcc.Input, value=0.,
                                     min=0., max=180.,
                                     debounce=True, type='number',
                                     style={'width': '25%',
                                            'margin-right': '20px'})

        lisxLenInRow = l_body.child(dbc.Row, justify='end')
        lisxLenInRow.child(html.Label("x_length [arcmin]: "))
        lisxLenIn = lisxLenInRow.child(dcc.Input, value=0.5,
                                       min=0.001, max=10.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

        lisyLenInRow = l_body.child(dbc.Row, justify='end')
        lisyLenInRow.child(html.Label("y_length [arcmin]: "))
        lisyLenIn = lisyLenInRow.child(dcc.Input, value=0.5,
                                       min=0.001, max=10.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

        lisxOmegaInRow = l_body.child(dbc.Row, justify='end')
        lisxOmegaInRow.child(html.Label("x_omega: "))
        lisxOmegaIn = lisxOmegaInRow.child(dcc.Input, value=9.2,
                                           min=1., max=20.,
                                           debounce=True, type='number',
                                           style={'width': '25%',
                                                  'margin-right': '20px'})

        lisyOmegaInRow = l_body.child(dbc.Row, justify='end')
        lisyOmegaInRow.child(html.Label("y_omega: "))
        lisyOmegaIn = lisyOmegaInRow.child(dcc.Input, value=8,
                                           min=1., max=20.,
                                           debounce=True, type='number',
                                           style={'width': '25%',
                                                  'margin-right': '20px'})

        lisDeltaInRow = l_body.child(dbc.Row, justify='end')
        lisDeltaInRow.child(html.Label("delta [deg]: "))
        lisDeltaIn = lisDeltaInRow.child(dcc.Input, value=45.,
                                         min=0.0, max=90.,
                                         debounce=True, type='number',
                                         style={'width': '25%',
                                                'margin-right': '20px'})

        listExpInRow = l_body.child(dbc.Row, justify='end')
        listExpInRow.child(html.Label("t_exp [s]: "))
        listExpIn = listExpInRow.child(dcc.Input, value=120.,
                                       min=1., max=1800.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

        refFrameLissRow = l_body.child(dbc.Row, justify='begin')
        refFrameLissRow.child(html.Label("Tel Frame: "))
        refFrameLiss = refFrameLissRow.child(
            dcc.RadioItems, options=[
                {'label': 'Az/El', 'value': 'altaz'},
                {'label': 'Ra/Dec', 'value': 'icrs'},
            ],
            value='altaz',
            labelStyle={'display': 'inline-block'},
            inputStyle={"margin-right": "5px",
                        "margin-left": "20px"},
        )

        lissWriteRow = l_body.child(dbc.Row, justify='end')
        lissWrite = lissWriteRow.child(
            dbc.Button, "Execute Pattern", color="danger", size='sm',
            style={'width': '45%', "margin-right": '10px'})

        rasBox = mappingBox.child(dbc.Tab, label="Raster")
        rasCard = rasBox.child(dbc.Card)
        r_header = rasCard.child(dbc.CardHeader)
        r_body = rasCard.child(dbc.CardBody)
        r_header.child(html.H5, "Raster Controls", className='mb-2')

        rasRotInRow = r_body.child(dbc.Row, justify='end')
        rasRotInRow.child(html.Label("Rot [deg]: "))
        rasRotIn = rasRotInRow.child(dcc.Input, value=0.,
                                     min=0., max=90.,
                                     debounce=True, type='number',
                                     style={'width': '25%',
                                            'margin-right': '20px'})

        rasLenInRow = r_body.child(dbc.Row, justify='end')
        rasLenInRow.child(html.Label("length [arcmin]: "))
        rasLenIn = rasLenInRow.child(dcc.Input, value=15.,
                                     min=0.0001, max=30.,
                                     debounce=True, type='number',
                                     style={'width': '25%',
                                            'margin-right': '20px'})

        rasStepInRow = r_body.child(dbc.Row, justify='end')
        rasStepInRow.child(html.Label("step [arcmin]: "))
        rasStepIn = rasStepInRow.child(dcc.Input, value=0.5,
                                       min=0.1, max=4.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

        rasnScansInRow = r_body.child(dbc.Row, justify='end')
        rasnScansInRow.child(html.Label("nScans: "))
        rasnScansIn = rasnScansInRow.child(dcc.Input, value=15,
                                           min=1, max=30,
                                           debounce=True, type='number',
                                           style={'width': '25%',
                                                  'margin-right': '20px'})

        rasSpeedInRow = r_body.child(dbc.Row, justify='end')
        rasSpeedInRow.child(html.Label("speed [arcsec/s]: "))
        rasSpeedIn = rasSpeedInRow.child(dcc.Input, value=50.,
                                         min=0.0001, max=500,
                                         debounce=True, type='number',
                                         style={'width': '25%',
                                                'margin-right': '20px'})

        rastTurnInRow = r_body.child(dbc.Row, justify='end')
        rastTurnInRow.child(html.Label("t_turnaround [s]: "))
        rastTurnIn = rastTurnInRow.child(dcc.Input, value=5.,
                                         min=0.1, max=10.,
                                         debounce=True, type='number',
                                         style={'width': '25%',
                                                'margin-right': '20px'})

        rastExpInRow = r_body.child(dbc.Row, justify='end')
        rastExpInRow.child(html.Label("t_exp [s]: "))
        rastExpIn = rastExpInRow.child(dcc.Input, value=5.75,
                                       min=2, max=60.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

        refFrameRastRow = r_body.child(dbc.Row, justify='begin')
        refFrameRastRow.child(html.Label("Tel Frame: "))
        refFrameRast = refFrameRastRow.child(
            dcc.RadioItems, options=[
                {'label': 'Az/El', 'value': 'altaz'},
                {'label': 'Ra/Dec', 'value': 'icrs'},
            ],
            value='altaz',
            labelStyle={'display': 'inline-block'},
            inputStyle={"margin-right": "5px",
                        "margin-left": "20px"},
        )

        rasWriteRow = r_body.child(dbc.Row, justify='end')
        rasWrite = rasWriteRow.child(
            dbc.Button, "Execute Pattern", color="danger", size='sm',
            style={'width': '45%', "margin-right": '10px'})

        output_state_container = controlBox.child(dbc.Row).child(dbc.Col).child(dbc.Card)
        lis_output_state = output_state_container.child(html.Div)
        ras_output_state = output_state_container.child(html.Div)

        mapping = {'lisRotIn': lisRotIn,
                   'lisxLenIn': lisxLenIn,
                   'lisyLenIn': lisyLenIn,
                   'lisxOmegaIn': lisxOmegaIn,
                   'lisyOmegaIn': lisyOmegaIn,
                   'lisDeltaIn': lisDeltaIn,
                   'listExpIn': listExpIn,
                   'refFrameLiss': refFrameLiss,
                   'lissWrite': lissWrite,
                   'rasRotIn': rasRotIn,
                   'rasLenIn': rasLenIn,
                   'rasStepIn': rasStepIn,
                   'rasnScansIn': rasnScansIn,
                   'rasSpeedIn': rasSpeedIn,
                   'rastTurnIn': rastTurnIn,
                   'rastExpIn': rastExpIn,
                   'refFrameRast': refFrameRast,
                   'rasWrite': rasWrite,
                   'lis_output_state': lis_output_state,
                   'ras_output_state': ras_output_state}

        # This is just a hidden trigger to execute the mapmaking code
        exBox = controlBox.child(dbc.Row, justify='end')
        Execute = exBox.child(html.Div)

        # plots
        plotsBox = bigBox.child(dbc.Col, width=9)
        horizRow = plotsBox.child(dbc.Row)
        horizCard = horizRow.child(dbc.Col, width=12).child(
            dbc.Card)
            # style={"width": "110rem", "marginTop": 0, "marginBottom": 0})
        h_header = horizCard.child(dbc.CardHeader)
        h_header.child(html.H5, "Horizon Coordinates", className='mb-2')
        h_body = horizCard.child(dbc.CardBody)
        hbr = h_body.child(dbc.Row)
        dazelPlot = hbr.child(dbc.Col).child(dcc.Graph)
        tazelPlot = hbr.child(dbc.Col).child(dcc.Graph)
        uptimePlot = hbr.child(dbc.Col).child(dcc.Graph)

        celesRow = plotsBox.child(dbc.Row, className='mt-3')

        outTable = celesRow.child(dbc.Col, width=2)
        oTCard = outTable.child(dbc.Card)
        oTheader = oTCard.child(dbc.CardHeader)
        oTheader.child(html.H5, "Summary Quantities", className='mb-2')
        oTbody = oTCard.child(dbc.CardBody)
        oTable = oTbody.child(dbc.Row).child(
            dbc.Table, bordered=True, striped=True, hover=True)

        celesCard = celesRow.child(dbc.Col, width=10).child(
            dbc.Card)
        e_header = celesCard.child(dbc.CardHeader)
        e_header.child(html.H5, "Celestial Coordinates", className='mb-2')
        e_body = celesCard.child(dbc.CardBody)
        cbr = e_body.child(dbc.Row)
        # ticrsPlot = cbr.child(dbc.Col).child(dcc.Graph)
        # cicrsPlot = cbr.child(dbc.Col).child(dcc.Graph)
        ticrsPlot = cbr.child(dbc.Col).child(dcc.Loading, type='spinner').child(dcc.Graph)
                                             # style={"float": "right"})
        cicrsPlot = cbr.child(dbc.Col).child(dcc.Loading, type='spinner').child(dcc.Graph)
                                             # style={"float": "left"})
        # this holds the dynamic settings of the sim rt config
        updated_context = self.child(dcc.Store, data=None)
        # before we register the callbacks
        super().setup_layout(app)

        # register the callbacks
        self._registerCallbacks(app, settings, target, mapping,
                                Execute, dazelPlot, tazelPlot, uptimePlot,
                                oTable, ticrsPlot, cicrsPlot,
                                updated_context)

    def _registerCallbacks(self, app, settings, target, mapping,
                           Execute, dazelPlot, tazelPlot, uptimePlot,
                           oTable, ticrsPlot, cicrsPlot, updated_context):
        print("Registering Callbacks")

        # update target ra and dec based on name search
        @app.callback(
            [
                Output(target['targRa'].id, "value"),
                Output(target['targDec'].id, "value"),
            ],
            [
                Input(target['targName'].id, "value"),
            ],
            prevent_initial_call=True
        )
        def getTargCoordsByName(targName):
            if(targName == "-"):
                raise PreventUpdate
            try:
                target = parse_coordinates(targName)
            except:
                print("Target {} not recognized.".format(targName))
                ra = 0.
                dec = 0.
                return [ra, dec]

            ra = target.ra.to_value(u.deg)
            dec = target.dec.to_value(u.deg)
            return [ra, dec]

        # update the Lissajous parameter set
        @app.callback(
            [
                Output(mapping['lis_output_state'].id, "children"),
            ],
            [
                Input(mapping['lissWrite'].id, "n_clicks"),
            ],
            [
                State(mapping['lisRotIn'].id, "value"),
                State(mapping['lisxLenIn'].id, "value"),
                State(mapping['lisyLenIn'].id, "value"),
                State(mapping['lisxOmegaIn'].id, "value"),
                State(mapping['lisyOmegaIn'].id, "value"),
                State(mapping['lisDeltaIn'].id, "value"),
                State(mapping['listExpIn'].id, "value"),
                State(target['targRa'].id, "value"),
                State(target['targDec'].id, "value"),
                State(target['obsTime'].id, "value"),
                State(target['obsDate'].id, "date"),
                State(mapping['refFrameLiss'].id, "value"),
            ],
            prevent_initial_call=True
        )
        def updateLissDict(n, r, xl, yl, xo, yo, delta, tExp, tra, tdec,
                           obsTime, obsDate, refFrame):
            # format date and time of start of observation
            date_object = date.fromisoformat(obsDate)
            date_string = date_object.strftime('%Y-%m-%d')
            t0 = date_string+'T'+obsTime
            d = {
                'rot': (r*u.deg).to_value(u.rad),
                'x_length': (xl*u.arcmin).to_value(u.rad),
                'y_length': (yl*u.arcmin).to_value(u.rad),
                'x_omega': xo,
                'y_omega': yo,
                'delta': (delta*u.deg).to_value(u.rad),
                't_exp': tExp,
                'length': 0.,
                'step': 0.,
                'nScans': 0,
                'speed': 0.,
                't_turnaround': 0.,
                'target_ra': tra,
                'target_dec': tdec,
                't0': t0,
                'ref_frame': refFrame,
            }
            return [json.dumps(d), ]

        # update the Raster parameter set
        @app.callback(
            [
                Output(mapping['ras_output_state'].id, "children"),
            ],
            [
                Input(mapping['rasWrite'].id, "n_clicks"),
            ],
            [
                State(mapping['rasRotIn'].id, "value"),
                State(mapping['rasLenIn'].id, "value"),
                State(mapping['rasStepIn'].id, "value"),
                State(mapping['rasnScansIn'].id, "value"),
                State(mapping['rasSpeedIn'].id, "value"),
                State(mapping['rastTurnIn'].id, "value"),
                State(mapping['rastExpIn'].id, "value"),
                State(target['targRa'].id, "value"),
                State(target['targDec'].id, "value"),
                State(target['obsTime'].id, "value"),
                State(target['obsDate'].id, "date"),
                State(mapping['refFrameRast'].id, "value"),
            ],
            prevent_initial_call=True
        )
        def updateRasterDict(n, r, l, s, ns, speed, turn, tExp, tra, tdec,
                             obsTime, obsDate, refFrame):
            # format date and time of start of observation
            date_object = date.fromisoformat(obsDate)
            date_string = date_object.strftime('%Y-%m-%d')
            t0 = date_string+'T'+obsTime
            d = {
                'rot': (r*u.deg).to_value(u.rad),
                'length': (l*u.arcmin).to_value(u.rad),
                'step': (s*u.arcmin).to_value(u.rad),
                'nScans': ns,
                'speed': (speed*u.arcsec).to_value(u.rad),
                't_turnaround': turn,
                't_exp': (tExp*u.min).to_value(u.s),
                'x_length': 0.,
                'y_length': 0.,
                'x_omega': 0.,
                'y_omega': 0.,
                'delta': 0.,
                'target_ra': tra,
                'target_dec': tdec,
                't0': t0,
                'ref_frame': refFrame,
            }
            return [json.dumps(d), ]

        @app.callback(
            [
                Output(updated_context.id, "data"),
                Output(Execute.id, "children"),
            ],
            [
                Input(mapping['lis_output_state'].id, "children"),
                Input(mapping['ras_output_state'].id, "children"),
            ],
            [
                State(settings['bandIn'].id, "value"),
            ],
            prevent_initial_call=True
        )
        def updateContext(lis_output, ras_output, band):
            print(lis_output)
            print(ras_output)
            ctx = dash.callback_context
            print(ctx.triggered[0]['prop_id'])
            if (ctx.triggered[0]['prop_id'] == f"{mapping['lis_output_state'].id}.children"):
                d = json.loads(lis_output)
                c = writeSimuContext(d, band, lissajous=1)
                print("lissajous context written")
            else:
                d = json.loads(ras_output)
                c = writeSimuContext(d, band, lissajous=0)
                print("raster context written")
            print(f'current_context: {c}')
            return [c, ""]

        # Generate the horizon plots
        @app.callback(
            [
                Output(dazelPlot.id, "figure"),
                Output(tazelPlot.id, "figure"),
                Output(uptimePlot.id, "figure"),
                Output(target['targetAlert'].id, "is_open"),
                Output(target['targetPopUp'].id, "displayed"),
            ],
            [
                Input(Execute.id, "children"),
            ],
            [
                State(settings['showArray'].id, "value"),
                State(updated_context.id, 'data'),
                State(settings['bandIn'].id, "value"),
            ],
            prevent_initial_call=True
        )
        def makeHorizonPlots(n, s2s, updated_context, band):
            sim = fetchSim(updated_context)
            if sim is None:
                print('no valid sim')
                raise PreventUpdate
            obs = generateMappings(sim, band)
            ufig = makeUptimesFig(obs)
            mAlt = obs['target_altaz'].alt.mean()
            if(mAlt.to_value(u.deg) < 20):
                tfig = getEmptyFig(400, 400)
                cfig = getEmptyFig(400, 400)
                elnotok = True
                tpu = True
            else:
                showArray = s2s.count('array')
                tfig, cfig = getHorizonPlots(obs, showArray=showArray)
                elnotok = False
                tpu = False
            return [tfig, cfig, ufig, elnotok, tpu]

        # Generate the celestial plots
        @app.callback(
            [
                Output(oTable.id, "children"),
                Output(ticrsPlot.id, "figure"),
                Output(cicrsPlot.id, "figure"),
            ],
            [
                Input(Execute.id, "children"),
            ],
            [
                State(settings['showArray'].id, "value"),
                State(settings['overlayDrop'].id, "value"),
                State(updated_context.id, 'data'),
                State(settings['bandIn'].id, "value"),
                State(settings['atmQIn'].id, "value"),
                State(settings['telRMSIn'].id, "value"),
                State(settings['unitsIn'].id, "value"),
            ],
            prevent_initial_call=True
        )
        def makeCelestialPlots(n, s2s, overlay, updated_context,
                               band, atmQ, telRMS, units):
            sim = fetchSim(updated_context)
            if sim is None:
                raise PreventUpdate
            obs = generateMappings(sim, band)
            mAlt = obs['target_altaz'].alt.mean()
            if(mAlt.to_value(u.deg) < 20):
                tfig = getEmptyFig(500, 500)
                cfig = getEmptyFig(500, 500)
            else:
                d = Detector(band, elevation=mAlt.to_value(u.deg),
                             atmQuartile=atmQ,
                             telescopeRMSmicrons=telRMS)
                showArray = s2s.count('array')
                tableData, tfig, cfig = getCelestialPlots(
                    sim, obs, d, units,
                    overlay=overlay,
                    showArray=showArray,
                )
            return [tableData, tfig, cfig]


sim_template_config = SimulatorRuntime('/home/toltec/toltec_astro/run/base_simu/').config
# here we get rid of the simu tree to avoid unexpected overwrites
sim_template_config.pop("simu")


def fetchSim(config):
    sim = SimulatorRuntime.from_config(sim_template_config, config)
    print(sim.config)
    return sim


def target_to_frame(target, simulator, frame, t_absolute):
    return target.transform_to(simulator.resolve_sky_map_ref_frame(
        ref_frame=frame, time_obs=t_absolute))


def obs_coords_to_frame(obs_coords, simulator, frame, t_absolute):
    return obs_coords.transform_to(simulator.resolve_sky_map_ref_frame(
        ref_frame=frame, time_obs=t_absolute))


# Generates obs_coords in the three primary reference frames.
def generateMappings(sim, band):
    m_obs = sim.get_mapping_model()
    p_obs = sim.get_obs_params()
    target = m_obs.target
    simulator = sim.get_instrument_simulator()

    # construct the relative and absolute times
    t_total = m_obs.get_total_time()
    t = np.arange(0, t_total.to_value(u.s),
                  1./p_obs['f_smp'].to_value(u.Hz)) * u.s
    t0 = m_obs.t0
    t_absolute = t0+t

    # Get the reference frame of the map trajectory. Remember that
    # m_obs.ref_frame is taken from the reference frame of the target
    ref_frame = simulator.resolve_sky_map_ref_frame(
        ref_frame=m_obs.ref_frame,
        time_obs=t_absolute)

    # Get the target coordinates in the mapping reference frame
    target_in_ref_frame = target.transform_to(ref_frame)

    # if(m_obs.ref_frame.name == 'altaz'):
    #     target_altaz = target_in_ref_frame
    # else:
    #     target_altaz = target.transform_to('altaz')

    target_icrs = target_to_frame(target, simulator, 'icrs', t_absolute)
    target_altaz = target_to_frame(target, simulator, 'altaz', t_absolute)
    target_galac = target_to_frame(target, simulator, 'galactic', t_absolute)

    # make a special evaluation of the target over the full day of the obs
    # round to the day
    t1 = Time(int(t0.mjd), format='mjd')
    t_day = t1 + np.arange(0, 24*60)*u.min
    target_day = target_to_frame(target, simulator, 'altaz', t_day)

    # and now generate the observation coordinates in different frames
    # Note that obs_coords has time, location, coordinates, and frame
    # and so can be used to generate coordinates in any frame as shown
    # below.
    obs_coords = m_obs.evaluate_at(target_in_ref_frame, t)
    obs_coords_icrs = obs_coords.transform_to('icrs')
    obs_coords_altaz = obs_coords_to_frame(obs_coords, simulator,
                                           'altaz', t_absolute)
    obs_coords_galactic = obs_coords.transform_to('galactic')

    # the array
    apt = simulator.table
    if(band == 1.1):
        aname = 'a1100'
    elif(band == 1.4):
        aname = 'a1400'
    else:
        aname = 'a2000'
    kidArray = apt[apt['array_name'] == aname]

    obs = {
        't': t,
        't_absolute': t_absolute,
        't0': t0,
        't_total': t_total,
        'obs_coords': obs_coords,
        'obs_coords_icrs': obs_coords_icrs,
        'obs_coords_altaz': obs_coords_altaz,
        'obs_coords_galactic': obs_coords_galactic,
        'target': target,
        'target_icrs': target_icrs,
        'target_altaz': target_altaz,
        'target_galac': target_galac,
        'target_day': target_day,
        't_day': t_day,
        'kidArray': kidArray,
        'band': band,
        }
    print(obs)
    return obs


def fetchConvolved(wcs, bimage, aimage, pixSize, t_total, fwhmArcsec):
    # here is the convolved image
    c = convolve_fft(bimage, aimage, normalize_kernel=False)

    # and also smooth by a beam-sized gaussian
    sigma = (fwhmArcsec/2.355)/pixSize.to_value(u.arcsec)
    g = Gaussian2DKernel(sigma, sigma)
    cimage = convolve_fft(c, g, normalize_kernel=False)

    # After all of this, cimage is in the units of integration time
    # (in seconds) per pixel.  You can check that cimage.sum() =
    # aimage.sum()*t_total.

    s = cimage.shape
    cra = wcs.pixel_to_world_values(np.arange(0, s[0]), 0)[0]
    cdec = wcs.pixel_to_world_values(0, np.arange(0, s[1]))[1]
    return cimage, cra, cdec


@cache.memoize()
def fetchOverlay(overlay, target, cra, cdec):
    w = -(cra[-1]-cra[0])*u.deg*np.cos(np.deg2rad(target.dec))
    h = (cdec[-1]-cdec[0])*u.deg
    s = SkyView.get_images(position=target, survey=overlay,
                           width=h, height=w,
                           grid=True, gridlabels=True)
    owcs = WCS(s[0][0].header)
    overlayImage = s[0][0].data
    s = overlayImage.shape
    ora = owcs.pixel_to_world_values(np.arange(0, s[0]), 0)[0]
    odec = owcs.pixel_to_world_values(0, np.arange(0, s[1]))[1]
    return overlayImage, ora, odec


def makeContour(cimage, cra, cdec):
    return go.Contour(
        z=cimage,
        x=cra,
        y=cdec,
        contours=dict(
            start=cimage.max()/5.,
            end=cimage.max(),
            size=cimage.max()/10.,
        ),
        opacity=0.5,
    )


def getHorizonPlots(obs, showArray=0):
    # setup
    # obs_coords_altaz are absolute coordinates in alt-az
    # offsets are the delta-source alt-az coordinates
    offaz = ((obs['obs_coords_altaz'].az -
              obs['target_altaz'].az) *
             np.cos(np.median(obs['target_altaz'].alt)))
    offalt = (obs['obs_coords_altaz'].alt -
              obs['target_altaz'].alt)
    offsets = (offaz, offalt)
    kidArray = obs['kidArray']

    # generate trajectory figure in delta-source coordinates
    tfig = go.Figure()
    xaxis, yaxis = getXYAxisLayouts()
    if(showArray):
        tfig.add_trace(
            go.Scattergl(x=kidArray['x_t'].to_value(u.arcmin),
                         y=kidArray['y_t'].to_value(u.arcmin),
                         mode='markers',
                         marker=dict(color='#642c3c',
                                     size=8),
                         ),
            )
    tfig.add_trace(
        go.Scattergl(x=offsets[0].to_value(u.arcmin),
                     y=offsets[1].to_value(u.arcmin),
                     mode='lines+markers',
                     marker=dict(color='black',
                                 size=2),
                     line=dict(color='red')),
    )
    tfig.update_layout(
        uirevision=True,
        showlegend=False,
        width=400,
        height=400,
        xaxis=xaxis,
        yaxis=yaxis,
        autosize=True,
        margin=dict(
            autoexpand=True,
            l=10,
            r=10,
            t=0,
        ),
        plot_bgcolor='white'
    )
    tfig.update_xaxes(title_text="Delta-Source Az [arcmin]",
                      automargin=True)
    tfig.update_yaxes(title_text="Delta-Source Alt [arcmin]",
                      automargin=True,
                      scaleanchor="x", scaleratio=1)

    # generate trajectory figure in absolute Alt-Az coordinates
    cfig = go.Figure()
    xaxis, yaxis = getXYAxisLayouts()
    cfig.add_trace(
        go.Scattergl(x=obs['obs_coords_altaz'].az.to_value(u.deg),
                     y=obs['obs_coords_altaz'].alt.to_value(u.deg),
                     mode='lines',
                     line=dict(color='red')),
    )
    cfig.add_trace(
        go.Scattergl(x=obs['target_altaz'].az.to_value(u.deg),
                     y=obs['target_altaz'].alt.to_value(u.deg),
                     mode='lines',
                     line=dict(color='purple')),
    )
    cfig.update_layout(
        uirevision=True,
        showlegend=False,
        width=400,
        height=400,
        xaxis=xaxis,
        yaxis=yaxis,
        autosize=True,
        margin=dict(
            autoexpand=True,
            l=10,
            r=10,
            t=0,
        ),
        plot_bgcolor='white'
    )
    cfig.update_xaxes(title_text="Azimuth [deg]",
                      automargin=True)
    cfig.update_yaxes(title_text="Elevation [deg]",
                      automargin=True)

    return tfig, cfig


def makeUptimesFig(obs):
    # make a plot of the uptimes for this day for this target
    daytime = []
    obstime = []
    targel = []
    for i in np.arange(len(obs['t_day'])):
        daytime.append(np.datetime64(obs['t_day'][i].to_datetime()))
    for i in np.arange(0, len(obs['t_absolute']), 20):
        obstime.append(np.datetime64(obs['t_absolute'][i].to_datetime()))
        targel.append(obs['target_altaz'].alt[i].to_value(u.deg))
    upfig = go.Figure()
    xaxis, yaxis = getXYAxisLayouts()
    upfig.add_trace(
        go.Scattergl(
            x=daytime,
            y=obs['target_day'].alt.to_value(u.deg),
            mode='lines',
            line=dict(color='green'),
        )
    )
    upfig.add_trace(
        go.Scattergl(x=obstime,
                     y=targel,
                     mode='lines',
                     line=dict(color='red',
                               width=10),
        )
    )
    upfig.add_trace(
        go.Scatter(x=daytime,
                   y=[20]*len(daytime),
                   mode='lines',
                   line=dict(color='gray'),
        )
    )
    upfig.add_trace(
        go.Scatter(x=daytime,
                   y=[-10]*len(daytime),
                   mode='lines',
                   line=dict(color='gray'),
                   fill='tonexty'
        )
    )
    upfig.update_layout(
        uirevision=True,
        showlegend=False,
        width=400,
        height=400,
        xaxis=xaxis,
        yaxis=yaxis,
        autosize=True,
        margin=dict(
            autoexpand=True,
            l=10,
            r=10,
            t=0,
        ),
        plot_bgcolor='white'
    )
    upfig.update_xaxes(title_text="Time",
                       automargin=True)
    upfig.update_yaxes(title_text="Target Elevation [deg]",
                       automargin=True,
                       range=[-10, 90])
    return upfig


def getCelestialPlots(sim, obs, d, units,
                      overlay='None', showArray=0,
                      sensDegred=np.sqrt(2.)):
    simulator = sim.get_instrument_simulator()
    p_obs = sim.get_obs_params()
    sampleTime = 1./p_obs['f_smp'].to_value(u.Hz)
    obs_ra = obs['obs_coords_icrs'].ra
    obs_dec = obs['obs_coords_icrs'].dec
    npts = len(obs_ra)
    kidArray = obs['kidArray']

    # calculating the positions of every detector in the array at all
    # times is very expensive.  Since TolTEC has a circular array,
    # let's ignore the rotation for this calculation unless it's
    # really desirable to do it properly
    projectArray = 0
    if(projectArray):
        print("Sit back ... this will take a while.")
        m_proj = simulator.get_sky_projection_model(
            ref_coord=obs['obs_coords_icrs'],
            time_obs=obs['t_absolute'])
        x_t = np.tile(kidArray['x_t'], (npts, 1))
        y_t = np.tile(kidArray['y_t'], (npts, 1))
        a_ra, a_dec = m_proj(x_t, y_t, frame='icrs')
    else:
        print("Ignoring array rotation during observations.")
        m_proj = simulator.get_sky_projection_model(
            ref_coord=obs['target_icrs'],
            time_obs=Time(obs['t0'])+np.median(obs['t']))
        a_ra, a_dec = m_proj(kidArray['x_t'], kidArray['y_t'], frame='icrs')

    # create a wcs for the observations and a pixel array for the data
    pixSize = 1.*u.arcsec
    nPixDec = np.ceil((obs_dec.degree.max() -
                       obs_dec.degree.min() +
                       (4.*u.arcmin).to_value(u.degree)) /
                      pixSize.to_value(u.deg))
    nPixRa = np.ceil((obs_ra.degree.max() -
                      obs_ra.degree.min() +
                      (4.*u.arcmin).to_value(u.degree))
                     / pixSize.to_value(u.deg))

    # correct for RA wrap if needed
    if((obs_ra.degree.max() > 270.) & (obs_ra.degree.min() < 90.)):
        nPixRa = np.ceil((obs_ra.degree.min() -
                          (360.-obs_ra.degree.max()) +
                          (4.*u.arcmin).to_value(u.degree))
                         / pixSize.to_value(u.deg))

    # base the wcs on these values
    wcs_input_dict = {
        'CTYPE1': 'RA---TAN',
        'CUNIT1': 'deg',
        'CDELT1': -pixSize.to_value(u.deg),
        'CRPIX1': nPixRa/2.,
        'CRVAL1': obs['target_icrs'].ra.degree,
        'NAXIS1': nPixRa,
        'CTYPE2': 'DEC--TAN',
        'CUNIT2': 'deg',
        'CDELT2': pixSize.to_value(u.deg),
        'CRPIX2': nPixDec/2.,
        'CRVAL2': obs['target_icrs'].dec.degree,
        'NAXIS2': nPixDec
    }
    wcs = WCS(wcs_input_dict)
    pixels_boresight = wcs.world_to_pixel_values(obs_ra, obs_dec)
    pixels_array = wcs.world_to_pixel_values(a_ra, a_dec)

    prabins = np.arange(wcs.array_shape[0])
    pdecbins = np.arange(wcs.array_shape[1])

    arabins = np.arange(np.floor(pixels_array[1].min()),
                        np.ceil(pixels_array[1].max()+1))
    adecbins = np.arange(np.floor(pixels_array[0].min()),
                         np.ceil(pixels_array[0].max()+1))

    # boresight image is made from a histogram
    bimage, _, _ = np.histogram2d(pixels_boresight[1],
                                  pixels_boresight[0],
                                  bins=[prabins, pdecbins])
    bimage *= sampleTime

    # so is the array image
    aimage, _, _ = np.histogram2d(pixels_array[1],
                                  pixels_array[0],
                                  bins=[arabins, adecbins])

    # the convolved image
    cimage, cra, cdec = fetchConvolved(wcs, bimage, aimage, pixSize,
                                       obs['t_total'], d.fwhmArcsec)

    # convert to a sensitivity in mJy rms if requested
    w = np.where(cimage > 0.1*cimage.max())
    mapArea = len(w[0])*(pixSize.to_value(u.deg))**2
    sigma = (d.fwhmArcsec/2.355)/pixSize.to_value(u.arcsec)
    simage = cimage.copy()*0.
    simage[w] = sensDegred*d.nefd/np.sqrt(cimage[w]*2.*np.pi*sigma**2)
    if(units == 'sens'):
        plotTitle = 'Estimated Depth per Beam-sized Area [mJy]'
        cimage = simage
    else:
        plotTitle = 'Integration time per {} pixel'.format(pixSize)

    # construct Table data
    tableData = None
    name = 'TolTEC ({}mm)'.format(obs['band'])
    obsDur = "{0:4.1f}".format(obs['t_total'])
    mAlt = "{0:3.1f} deg".format(
        obs['target_altaz'].alt.mean().to_value(u.deg))
    dsens = "{0:2.3f} mJy rt(s)".format(d.nefd*sensDegred)
    marea = "{0:2.4f} deg^2".format(mapArea)
    medSens = "{0:2.3f} mJy rms".format(np.median(simage[w]))
    bod = []
    bod.append(html.Tr([html.Td(name, colSpan="2")],
                       className='table-success'))
    bod.append(html.Tr([html.Td("Total time"), html.Td(obsDur)]))
    bod.append(html.Tr([html.Td("Mean Alt"), html.Td(mAlt)]))
    bod.append(html.Tr([html.Td("Detector sens."), html.Td(dsens)]))
    bod.append(html.Tr([html.Td("Map Area"), html.Td(marea)]))
    bod.append(html.Tr([html.Td("Median sens."), html.Td(medSens)]))
    tableData = [html.Tbody(bod)]

    # the overlay image
    if(overlay != 'None'):
        overlayImage, ora, odec = fetchOverlay(overlay, obs['target_icrs'],
                                               cra, cdec)

    # the coverage image
    xaxis, yaxis = getXYAxisLayouts()
    cfig = go.Figure()

    if(overlay != 'None'):
        cfig.add_trace(
            go.Heatmap(
                z=overlayImage,
                x=ora,
                y=odec,
                opacity=1.0,
                showscale=False,
            ),
        )

    if(1):
        cfig.add_trace(
            go.Heatmap(
                z=cimage,
                x=cra,
                y=cdec,
                opacity=0.85,
                ),
            )
    else:
        cfig.add_trace(
            makeContour(cimage, cra, cdec),
        )

    cfig.update_layout(
        title=plotTitle,
        uirevision=True,
        showlegend=False,
        width=500,
        height=500,
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
    cfig.update_xaxes(title_text="Ra [Deg]",
                      automargin=True)
    cfig.update_yaxes(title_text="Dec [Deg]",
                      scaleanchor="x", scaleratio=1,
                      automargin=True)

    # generate trajectory figure
    tfig = go.Figure()
    if(overlay != 'None'):
        tfig.add_trace(
            go.Heatmap(
                z=overlayImage,
                x=ora,
                y=odec,
                opacity=0.75,
                showscale=False,
            ),
        )

    if(showArray):
        tfig.add_trace(
            go.Scattergl(
                x=a_ra.to_value(u.deg),
                y=a_dec.to_value(u.deg),
                mode='markers',
                marker=dict(color='orange',
                            size=8),
            ),
        )

    tfig.add_trace(
        go.Scattergl(x=obs_ra.to_value(u.deg),
                     y=obs_dec.to_value(u.deg),
                     mode='lines',
                     line=dict(color='red')),
    )
    tfig.update_layout(
        uirevision=True,
        showlegend=False,
        width=500,
        height=500,
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
    tfig.update_xaxes(title_text="Ra [deg]",
                      automargin=True)
    tfig.update_yaxes(title_text="Dec [deg]",
                      scaleanchor="x", scaleratio=1,
                      automargin=True)

    return tableData, tfig, cfig


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


# create config to update to the sim rt
def writeSimuContext(d, band, lissajous=0):
    if(band == 1.1):
        bandName = '\"a1100\"'
    elif(band == 1.4):
        bandName = '\"a1400\"'
    else:
        bandName = '\"a2000\"'

    oF = StringIO()
    oF.write("# vim: et ts=2 sts=2 sw=2\n")
    oF.write("---\n")
    oF.write("\n")
    oF.write("_60_simu:\n")
    oF.write("  example_mapping_tel_nc: &example_mapping_tel_nc\n")
    oF.write("    type: lmt_tcs\n")
    oF.write("    filepath: ./tel_toltec_2020-10-01_103371_00_0000.nc\n")
    oF.write("  example_mapping_model_raster: &example_mapping_model_raster\n")
    oF.write("    type: tolteca.simu:SkyRasterScanModel\n")
    oF.write("    rot: {} rad\n".format(d['rot']))
    oF.write("    length: {} rad\n".format(d['length']))
    oF.write("    space: {} rad\n".format(d['step']))
    oF.write("    n_scans: {}\n".format(d['nScans']))
    oF.write("    speed: {} rad/s\n".format(d['speed']))
    oF.write("    t_turnover: {} s\n".format(d['t_turnaround']))
    oF.write("    target: {0:}d {1:}d\n".format(d['target_ra'], d['target_dec']))
    oF.write("    ref_frame: {}\n".format(d['ref_frame']))
    oF.write("    t0: {}\n".format(d['t0']))
    oF.write("    # lst0: ...\n")
    oF.write("  example_mapping_model_lissajous: &example_mapping_model_lissajous\n")
    oF.write("    type: tolteca.simu:SkyLissajousModel\n")
    oF.write("    rot: {} rad\n".format(d['rot']))
    oF.write("    x_length: {} rad\n".format(d['x_length']))
    oF.write("    y_length: {} rad\n".format(d['y_length']))
    oF.write("    x_omega: {} rad/s\n".format(d['x_omega']))
    oF.write("    y_omega: {} rad/s\n".format(d['y_omega']))
    oF.write("    delta: {} rad\n".format(d['delta']))
    oF.write("    target: {0:}d {1:}d\n".format(d['target_ra'], d['target_dec']))
    oF.write("    ref_frame: {}\n".format(d['ref_frame']))
    oF.write("    t0: {}\n".format(d['t0']))
    oF.write("\n")
    oF.write("simu:\n")
    oF.write("  # this is the actual simulator\n")
    oF.write("  jobkey: example_simu\n")
    oF.write("  # plot: true\n")
    oF.write("  instrument:\n")
    oF.write("    name: toltec\n")
    oF.write("    calobj: cal/calobj_default/index.yaml\n")
    oF.write("    select: 'array_name == {}'\n".format(bandName))
    oF.write("    # select: 'pg == 1'\n")
    oF.write("  obs_params:\n")
    oF.write("    f_smp: 12.2 Hz  # the sample frequency\n")
    oF.write("    t_exp: {} s\n".format(d['t_exp']))
    oF.write("  sources:\n")
    oF.write("    - type: point_source_catalog\n")
    oF.write("      filepath: inputs/example_input.asc\n")
    if(lissajous):
        oF.write("  mapping: *example_mapping_model_lissajous\n")
    else:
        oF.write("  mapping: *example_mapping_model_raster\n")
    return yaml.safe_load(oF.getvalue())
