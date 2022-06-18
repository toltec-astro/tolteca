from tollan.utils.log import timeit, get_logger
from dasha.web.templates import ComponentTemplate
from dasha.web.extensions.cache import cache
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_extensions import Download
from dash_extensions.snippets import send_bytes
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import dash
import json
import yaml
from io import StringIO

from astropy.convolution import Gaussian2DKernel
from astroquery.utils import parse_coordinates
from astropy.convolution import convolve_fft
from tolteca.simu import SimulatorRuntime
from astroquery.skyview import SkyView
from astropy import units as u
from astropy.time import Time
from astropy.wcs import WCS
from astropy.io import fits
from datetime import date
import numpy as np

from pathlib import Path
from .. import env_registry, env_prefix
import sys

from .common import HeaderWithToltecLogo
from .common.simuControls import getSettingsCard
from .common.simuControls import getSourceCard

TOLTEC_SENSITIVITY_MODULE_PATH_ENV = (
        f"{env_prefix}_CUSTOM_TOLTEC_SENSITIVITY_MODULE_PATH")
env_registry.register(
        TOLTEC_SENSITIVITY_MODULE_PATH_ENV,
        "The path to locate the toltec_sensitivity module",
        "toltec_sensitivity")
_toltec_sensitivity_module_path = env_registry.get(
        TOLTEC_SENSITIVITY_MODULE_PATH_ENV)

TOLTECA_SIMU_TEMPLATE_PATH_ENV = (
        f"{env_prefix}_CUSTOM_SIMU_TEMPLATE_PATH")
env_registry.register(
        TOLTECA_SIMU_TEMPLATE_PATH_ENV,
        "The path to locate the tolteca.simu template workdir",
        "base_simu")
_tolteca_simu_template_path = env_registry.get(
        TOLTECA_SIMU_TEMPLATE_PATH_ENV)

sys.path.insert(
        0,
        Path(_toltec_sensitivity_module_path).expanduser().parent.as_posix())
from toltec_sensitivity import Detector


class sensitivityCalculator(ComponentTemplate):

    # sets up the global Div that the site lives under
    _component_cls = dbc.Container
    fluid = True
    logger = get_logger()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # This section of code only runs once when the server is started.
    def setup_layout(self, app):
        container = self
        header_container = container.child(HeaderWithToltecLogo(
            logo_colwidth=2
            )).header_container
        header_section, hr_container, body = header_container.grid(3, 1)
        hr_container.child(html.H2,
                           "TolTEC Simple Sensitivity Calculator",
                           className='my-2')
        hr_container.child(html.Hr())

        # a container for the rest of the page
        bigBox = body.child(dbc.Row)

        # ---------------------------
        # Settings Card
        # ---------------------------
        settingsRow = body.child(dbc.Row)
        settingsCard = settingsRow.child(dbc.Col).child(
            dbc.Card, color="primary", outline=True)
        t_header = settingsCard.child(dbc.CardHeader)
        t_header.child(html.H5, "Global Settings", className='mb-2')

        sc = settingsCard.child(dbc.CardBody).child(dbc.Row)
        t_body = sc.child(dbc.Col)
        req = sc.child(dbc.Col)
        req.child(dbc.Row).child(html.H4("Settings for LMT2021-S2 Proposals"))
        req.child(dbc.Row).child(html.P("Atmos. quartile: 50%"))
        req.child(dbc.Row).child(html.P("Telescope RMS: 100um"))
        req.child(dbc.Row).child(html.P("Degradation Factor: 3"))

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
        telRMSIn = telRow.child(dcc.Input, value=100.,
                                min=50., max=110.,
                                debounce=True, type='number',
                                style={'width': '25%',
                                       'margin-right': '20px',
                                       'margin-left': '20px'})

        elRow = t_body.child(dbc.Row, justify='begin')
        elRow.child(html.Label("Mean Source Elevation [deg]: "))
        meanElIn = elRow.child(dcc.Input, value=50.,
                               min=20., max=70.,
                               debounce=True, type='number',
                               style={'width': '25%',
                                      'margin-right': '20px',
                                      'margin-left': '20px'})

        degRow = t_body.child(dbc.Row, justify='begin')
        degRow.child(html.Label("Degradation factor: "))
        degIn = degRow.child(
            dcc.RadioItems, options=[
                {'label': '1 (optimal)', 'value': 1},
                {'label': '3 (optimistic)', 'value': 3},
                {'label': '7 (AzTEC level)', 'value': 7},
            ],
            value=3,
            labelStyle={'display': 'inline-block'},
            inputStyle={"margin-right": "5px",
                        "margin-left": "20px"},)

        polRow = t_body.child(dbc.Row, justify='begin')
        polRow.child(html.Label("Total Intensity or Polarization: "))
        polIn = polRow.child(
            dcc.RadioItems, options=[
                {'label': 'Intensity (I)', 'value': 0},
                {'label': 'Polarization (Q,U)', 'value': 1},
            ],
            value=0,
            labelStyle={'display': 'inline-block'},
            inputStyle={"margin-right": "10px",
                        "margin-left": "10px"},
        )

        settings = {'polIn': polIn,
                    'atmQIn': atmQIn,
                    'telRMSIn': telRMSIn,
                    'meanElIn': meanElIn,
                    'degIn': degIn}

        #
        # Empty row
        #
        emptyRow = body.child(dbc.Row)
        emptyRow.child(dbc.Col).child(html.H5, '')

        # ---------------------------
        # RMS for Map Area Card
        # ---------------------------
        calcRow = body.child(dbc.Row)
        rmsRow = calcRow.child(dbc.Col).child(dbc.Row)
        rmsCard = rmsRow.child(dbc.Col).child(
            dbc.Card, color="success", outline=True)
        r_header = rmsCard.child(dbc.CardHeader)
        r_header.child(html.H5, "RMS Given Map Size and Time",
                       className='mb-2')
        r_body = rmsCard.child(dbc.CardBody)

        xSizeRow = r_body.child(dbc.Row, justify='begin')
        xSizeRow.child(html.Label("Map x-size [arcmin]: "))
        xSizeIn = xSizeRow.child(dcc.Input, value=20.,
                                 min=4., max=600.,
                                 debounce=True, type='number',
                                 style={'width': '15%',
                                        'margin-right': '20px',
                                        'margin-left': '20px'})

        ySizeRow = r_body.child(dbc.Row, justify='begin')
        ySizeRow.child(html.Label("Map y-size [arcmin]: "))
        ySizeIn = ySizeRow.child(dcc.Input, value=20.,
                                 min=4., max=600.,
                                 debounce=True, type='number',
                                 style={'width': '15%',
                                        'margin-right': '20px',
                                        'margin-left': '20px'})

        intTimeRow = r_body.child(dbc.Row, justify='begin')
        intTimeRow.child(html.Label("Integration Time [mins]: "))
        intTimeIn = intTimeRow.child(dcc.Input, value=30.,
                                     min=1., max=6000.,
                                     debounce=True, type='number',
                                     style={'width': '15%',
                                            'margin-right': '20px',
                                            'margin-left': '20px'})
        r_body.child(html.Hr())

        sensOutRow = r_body.child(dbc.Row, justify='begin')
        sensOutRow.child(html.Label(''))
        so = r_body.child(dbc.Col, justify='center')
        sensOut = {
            'title': sensOutRow.child(html.H5, ''),
            'a1100': so.child(dbc.Row, justify='center').child(html.H5, ''),
            'a1400': so.child(dbc.Row, justify='center').child(html.H5, ''),
            'a2000': so.child(dbc.Row, justify='center').child(html.H5, '')}

        sensCalc = {
            'xSizeIn': xSizeIn,
            'ySizeIn': ySizeIn,
            'intTimeIn': intTimeIn,
            'sensOut': sensOut}

        # ---------------------------
        # Time for Map Area and Depth Card
        # ---------------------------
        timeRow = calcRow.child(dbc.Col).child(dbc.Row)
        timeCard = timeRow.child(dbc.Col).child(
            dbc.Card, color="warning", outline=True)
        s_header = timeCard.child(dbc.CardHeader)
        s_header.child(html.H5, "TIME Given Map Size and RMS", className='mb-2')
        s_body = timeCard.child(dbc.CardBody)

        xsSizeRow = s_body.child(dbc.Row, justify='begin')
        xsSizeRow.child(html.Label("Map x-size [arcmin]: "))
        xsSizeIn = xsSizeRow.child(dcc.Input, value=20.,
                                   min=4., max=600.,
                                   debounce=True, type='number',
                                   style={'width': '15%',
                                          'margin-right': '20px',
                                          'margin-left': '20px'})

        ysSizeRow = s_body.child(dbc.Row, justify='begin')
        ysSizeRow.child(html.Label("Map y-size [arcmin]: "))
        ysSizeIn = ysSizeRow.child(dcc.Input, value=20.,
                                   min=4., max=600.,
                                   debounce=True, type='number',
                                   style={'width': '15%',
                                          'margin-right': '20px',
                                          'margin-left': '20px'})

        rmssRow = s_body.child(dbc.Row, justify='begin')
        rmssRow.child(html.Label("Desired Map RMS [mJy]: "))
        rmssIn = rmssRow.child(dcc.Input, value=1.0,
                               min=0.01, max=100.,
                               debounce=True, type='number',
                               style={'width': '15%',
                                      'margin-right': '20px',
                                      'margin-left': '20px'})
        s_body.child(html.Hr())

        timeOutRow = s_body.child(dbc.Row, justify='begin')
        timeOutRow.child(html.Label(''))
        rb = s_body.child(dbc.Col, justify='center')
        timeOut = {
            'title': timeOutRow.child(html.H5, ''),
            'a1100': rb.child(dbc.Row, justify='center').child(html.H5, ''),
            'a1400': rb.child(dbc.Row, justify='center').child(html.H5, ''),
            'a2000': rb.child(dbc.Row, justify='center').child(html.H5, '')}

        timeCalc = {
            'xSizeIn': xsSizeIn,
            'ySizeIn': ysSizeIn,
            'rmsIn': rmssIn,
            'timeOut': timeOut}

        # ---------------------------
        # Jumbotron with the info
        # ---------------------------
        jRow = body.child(dbc.Row)
        jcontent = [
            html.H4("Notes on this calculator", className="display-5"),
            html.P(
                "This calculator is an implementation of the mapping speed calculation described in Bryan et al. 2018.  It uses models of the detector noise, the atmosphere, the atmosphere fluctuations, and the telescope surface - all of which are subject to change as we commission the instrument.",
                className="lead",
            ),
            html.Hr(className="my-2"),
            html.P(
                "At present there is no overhead calculation made.  We will be adding a model for the overheads in the coming month.  For a more sophisticated calculation of observing time and depths that requires carefully designing the full observation, please use the link below."
            ),
            html.P(html.A("Fancy ObsPlanner", href='http://toltecdr.astro.umass.edu/obs_planner', target="_blank"), className="lead"),
            html.Hr(className="my-2"),
            html.P("For polarimetry measurements, the relevant quantity is the 1-sigma uncertainty (error bar) in the polarized intensity. Here the polarized intensity is defined as the total intensity multiplied by the polarization fraction. Because of the need to separately measure Stokes Q and Stokes U, the integration time required to reach a given polarized intensity error bar is double that required to reach the same total intensity error bar. We apply an additional penalty due to imperfect polarization modulation (e.g., non-ideal behavior in the half-wave plate). This penalty is a factor of 10% in sensitivity (~20% in observing time)."),
            html.P("See:"),
            html.P("A Primer on Far-Infrared Polarimetry, Hildebrand, R.H., et al. 2000, Publications of the Astronomical Society of the Pacific, 112, 1215."),
            html.P("Appendix B.1 of Planck Intermediate Results XIX (https://arxiv.org/pdf/1405.0871.pdf)"),
        ]
        jtron = jRow.child(dbc.Col, width=12).child(dbc.Jumbotron, jcontent)

        # before we register the callbacks
        super().setup_layout(app)

        # register the callbacks
        self._registerCallbacks(app, settings, sensCalc, timeCalc)

    def _registerCallbacks(self, app, settings, sensCalc, timeCalc):
        print("Registering Callbacks")

        # calculate the rms given a map size and integration time
        @app.callback(
            [
                Output(sensCalc['sensOut']['title'].id, "children"),
                Output(sensCalc['sensOut']['a1100'].id, "children"),
                Output(sensCalc['sensOut']['a1400'].id, "children"),
                Output(sensCalc['sensOut']['a2000'].id, "children"),
            ],
            [
                Input(settings['polIn'].id, "value"),
                Input(settings['atmQIn'].id, "value"),
                Input(settings['telRMSIn'].id, "value"),
                Input(settings['meanElIn'].id, "value"),
                Input(settings['degIn'].id, "value"),
                Input(sensCalc['xSizeIn'].id, "value"),
                Input(sensCalc['ySizeIn'].id, "value"),
                Input(sensCalc['intTimeIn'].id, "value")
            ],
        )
        def calcSensitivity(pol, atm, telrms, meanEl, degradation,
                            xSize, ySize, intTime):
            depth = calculateMapRMS(pol, atm, telrms, meanEl,
                                    degradation, xSize, ySize, intTime)
            if(pol):
                tt = "Estimated Polarized Emission Map RMS: "
            else:
                tt = "Estimated Total Intensity Map RMS: "
            a1100 = "1.1mm: {0:2.3f} mJy".format(depth[0])
            a1400 = "1.4mm: {0:2.3f} mJy".format(depth[1])
            a2000 = "2.0mm: {0:2.3f} mJy".format(depth[2])

            print(tt)
            print(a1100)
            
            return [tt, a1100, a1400, a2000]

        # calculate the time required to map a field to a given depth
        @app.callback(
            [
                Output(timeCalc['timeOut']['title'].id, "children"),
                Output(timeCalc['timeOut']['a1100'].id, "children"),
                Output(timeCalc['timeOut']['a1400'].id, "children"),
                Output(timeCalc['timeOut']['a2000'].id, "children"),
            ],
            [
                Input(settings['polIn'].id, "value"),
                Input(settings['atmQIn'].id, "value"),
                Input(settings['telRMSIn'].id, "value"),
                Input(settings['meanElIn'].id, "value"),
                Input(settings['degIn'].id, "value"),
                Input(timeCalc['xSizeIn'].id, "value"),
                Input(timeCalc['ySizeIn'].id, "value"),
                Input(timeCalc['rmsIn'].id, "value")
            ],
        )
        def calcTime(pol, atm, telrms, meanEl, degradation,
                     xSize, ySize, rms):
            time = calculateMapTime(pol, atm, telrms, meanEl,
                                    degradation, xSize, ySize, rms)
            if(pol):
                tt = "Estimated Polarized Emission Mapping Time: "
            else:
                tt = "Estimated Total Intensity Mapping Time: "
            a1100 = "1.1mm: {0:2.1f} mins".format(time[0])
            a1400 = "1.4mm: {0:2.1f} mins".format(time[1])
            a2000 = "2.0mm: {0:2.1f} mins".format(time[2])
            return [tt, a1100, a1400, a2000]


def calculateMapRMS(pol, atm, telrms, meanEl,
                    degradation, xSize, ySize, intTime):
    rms = []
    b = [1.1, 1.4, 2.0]
    for band in b:
        d = Detector(band,
                     elevation=meanEl,
                     atmQuartile=atm,
                     telescopeRMSmicrons=telrms)
        mappingSpeed = d.mappingSpeed/degradation
        mapArea = ((xSize*u.arcmin).to_value(u.deg) *
                   (ySize*u.arcmin).to_value(u.deg))
        time = (intTime*u.minute).to_value(u.hour)
        depth = np.sqrt(mapArea/(time*mappingSpeed))
        if(pol):
            depth *= (np.sqrt(2.)+0.1)
        rms.append(depth)
    return rms


def calculateMapTime(pol, atm, telrms, meanEl,
                     degradation, xSize, ySize, rms):
    t = []
    b = [1.1, 1.4, 2.0]
    for band in b:
        d = Detector(band,
                     elevation=meanEl,
                     atmQuartile=atm,
                     telescopeRMSmicrons=telrms)
        mappingSpeed = d.mappingSpeed/degradation
        mapArea = ((xSize*u.arcmin).to_value(u.deg) *
                   (ySize*u.arcmin).to_value(u.deg))

        time = (mapArea/mappingSpeed/(rms**2))*60.
        if(pol):
            time *= (np.sqrt(2.)+0.1)**2
        t.append(time)
    return t
