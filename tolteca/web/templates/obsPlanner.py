from tollan.utils.log import timeit, get_logger
from dash_component_template import ComponentTemplate
# from dasha.web.extensions.cache import cache
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

from astropy.coordinates import EarthLocation
from astropy.convolution import Gaussian2DKernel
from astroquery.utils import parse_coordinates
from astropy.convolution import convolve_fft
from tolteca.simu import SimulatorRuntime
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroplan import Observer
from astropy.time import Time
from astropy.wcs import WCS
from astropy.io import fits
from datetime import date
import numpy as np

from pathlib import Path
from .. import env_mapper
import sys

from .common.simuControls import getSettingsCard
from .common.simuControls import getSourceCard
from .common.simuControls import getLissajousControls
from .common.simuControls import getDoubleLissajousControls
from .common.simuControls import getRasterControls
from .common.simuControls import getRastajousControls
from .common.smaPointingSourceFinder import SMAPointingCatalog
# from .common.smaPointingSourceFinder import findPointingSource
# from .common.smaPointingSourceFinder import generateOTScript

env_registry = env_mapper.registry
TOLTEC_SENSITIVITY_MODULE_PATH_ENV = (
        f"{env_prefix}_CUSTOM_TOLTEC_SENSITIVITY_MODULE_PATH")
env_registry.register(
        TOLTEC_SENSITIVITY_MODULE_PATH_ENV,
        "The path to locate the toltec_sensitivity module",
        "toltec_sensitivity")
_toltec_sensitivity_module_path = env_registry.get(
        TOLTEC_SENSITIVITY_MODULE_PATH_ENV)

SMA_POINTING_CATALOG_PATH_ENV = (
        f"{env_prefix}_CUSTOM_SMA_POINTING_CATALOG_PATH")
env_registry.register(
        SMA_POINTING_CATALOG_PATH_ENV,
        "The path to locate the SMA pointing catalog",
        "sma_pointing_sources.ecsv")
_sma_pointing_catalog_path = env_registry.get(
        SMA_POINTING_CATALOG_PATH_ENV)
_sma_pointing_catalog = SMAPointingCatalog(filepath=_sma_pointing_catalog_path)

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


class obsPlanner(ComponentTemplate):

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
        hr_container.child(html.H2,
                           "TolTEC Observation Planner",
                           className='my-2')
        hr_container.child(html.Hr())

        # a container for the rest of the page
        bigBox = body.child(dbc.Row)

        # control column
        controlBox = bigBox.child(dbc.Col, width=3)

        # ---------------------------
        # Settings Card
        # ---------------------------
        settings = getSettingsCard(controlBox)

        # ---------------------------
        # Source Card
        # ---------------------------
        target = getSourceCard(controlBox)

        # ---------------------------
        # Mapping Tabs
        # ---------------------------
        mappingBox = controlBox.child(
            dbc.Row, className='mt-3').child(dbc.Col).child(dbc.Tabs)

        # Single Lissajous Parameter Set
        lisControls = getLissajousControls(mappingBox)

        # Double Lissajous Parameter Set
        doubleLisControls = getDoubleLissajousControls(mappingBox)

        # Raster Pattern Parameter Set
        rasterControls = getRasterControls(mappingBox)

        # Rastajous Pattern Parameter Set
        rastajousControls = getRastajousControls(mappingBox)

        # Combine the mapping controls into a single dictionary
        mapping = {}
        mapping.update(lisControls)
        mapping.update(doubleLisControls)
        mapping.update(rasterControls)
        mapping.update(rastajousControls)

        # This is just a hidden trigger to execute the mapmaking code
        exBox = controlBox.child(dbc.Row, justify='end')
        Execute = exBox.child(html.Div)

        # The text output controls
        butRow = controlBox.child(dbc.Row)
        yamlWrite = butRow.child(dbc.Col, width=4).child(
            dbc.Button, "Download YAML", color="info", outline=True, size='sm')
        mcWrite = butRow.child(dbc.Col, width=4).child(
            dbc.Button, "Download M&C script", color="info", outline=True, size='sm')
        outRow = controlBox.child(dbc.Row)
        yamlOut = outRow.child(dbc.Col, width=1).child(dcc.Download)
        mcOut = outRow.child(dbc.Col, width=1).child(dcc.Download)
        mcPointOut = outRow.child(dbc.Col, width=1).child(dcc.Download)
        outButtons = {'yaml': yamlWrite,
                      'mc': mcWrite,
                      'yamlOut': yamlOut,
                      'mcOut': mcOut,
                      'mcPoint': mcPointOut}
        
        # plots
        plotsBox = bigBox.child(dbc.Col, width=9)
        horizRow = plotsBox.child(dbc.Row)
        horizCard = horizRow.child(dbc.Col, width=12).child(
            dbc.Card)
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
        ticrsPlot = cbr.child(dbc.Col).child(
            dcc.Loading, type='spinner').child(dcc.Graph)
        cicrsPlot = cbr.child(dbc.Col).child(
            dcc.Loading, type='spinner').child(dcc.Graph)
        cdr = e_body.child(dbc.Row)
        fitsWrite = cdr.child(dbc.Col, width=2).child(
            dbc.Button, "Download FITS", color="success", size='sm')
        fitsOut = cdr.child(dbc.Col, width=1, align='end').child(
            dbc.Row, align='start').child(
                dcc.Loading, type='spinner').child(Download)
        # fitsOut = cdr.child(Download)

        # this holds the dynamic settings of the sim rt config
        updated_context = self.child(dcc.Store, data=None)

        # before we register the callbacks
        super().setup_layout(app)

        # register the callbacks
        self._registerCallbacks(app, settings, target, mapping,
                                Execute, dazelPlot, tazelPlot, uptimePlot,
                                oTable, ticrsPlot, cicrsPlot,
                                updated_context, fitsOut, fitsWrite,
                                outButtons)

    def _registerCallbacks(self, app, settings, target, mapping,
                           Execute, dazelPlot, tazelPlot, uptimePlot,
                           oTable, ticrsPlot, cicrsPlot, updated_context,
                           fitsOut, fitsWrite, outButtons):
        print("Registering Callbacks")

        # download a yaml file with the simulation parameters
        @app.callback(
            [
                Output(outButtons['yamlOut'].id, "data"),
            ],
            [
                Input(outButtons['yaml'].id, "n_clicks")
            ],
            [
                State(updated_context.id, 'data'),
            ],
            prevent_initial_call=True
        )
        def makeYamlOutput(n, updated_context):
            if updated_context is None:
                print('run the sim first')
                raise PreventUpdate
            sim = fetchSim(updated_context)
            if sim is None:
                print('no valid sim')
                raise PreventUpdate
            cfg = {'simu': sim.config['simu']}
            ss = StringIO()
            sim.write_config_to_yaml(cfg, ss)
            return [dict(content=ss.getvalue(), filename="sim.txt")]

        # download an M&C script with the simulation parameters
        @app.callback(
            [
                Output(outButtons['mcOut'].id, "data"),
            ],
            [
                Input(outButtons['mc'].id, "n_clicks")
            ],
            [
                State(updated_context.id, 'data'),
            ],
            prevent_initial_call=True
        )
        def makeMCOutput(n, updated_context):
            if updated_context is None:
                print('run the sim first')
                raise PreventUpdate
            sim = fetchSim(updated_context)
            if sim is None:
                print('no valid sim')
                raise PreventUpdate
            ot_content = sim.export(format='lmtot')
            return [dict(content=ot_content, filename="mc_target.script.txt")]

        # download an M&C script with the pointing parameters
        @app.callback(
            [
                Output(outButtons['mcPoint'].id, "data"),
            ],
            [
                Input(outButtons['mc'].id, "n_clicks")
            ],
            [
                State(updated_context.id, 'data'),
                State(target['pointing']['store'].id, "data"),
            ],
            prevent_initial_call=True
        )
        def makeMCPointOutput(n, updated_context, ps):
            if updated_context is None:
                print('run the sim first')
                raise PreventUpdate
            print()
            print(ps)
            print()
            ot_content = _sma_pointing_catalog.generateOTScript(ps)
            return [dict(content=ot_content, filename="mc_pointing.script.txt")]
        
        # download a fits file of the coverage map
        @app.callback(
            [
                Output(fitsOut.id, "data"),
            ],
            [
                Input(fitsWrite.id, "n_clicks")
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
        def makeDownloadFits(n, s2s, overlay, updated_context,
                             band, atmQ, telRMS, units):
            sim = fetchSim(updated_context)
            if sim is None:
                raise PreventUpdate
            obs = generateMappings(sim, band)
            mAlt = obs['target_altaz'].alt.mean()
            if(mAlt.to_value(u.deg) < 20):
                raise PreventUpdate
            else:
                d = Detector(band, elevation=mAlt.to_value(u.deg),
                             atmQuartile=atmQ,
                             telescopeRMSmicrons=telRMS)
                showArray = s2s.count('array')
                tableData, tfig, cfig, hdul = getCelestialPlots(
                    sim, obs, d, units,
                    overlay=overlay,
                    showArray=showArray,
                )
            return [send_bytes(hdul.writeto, "obs_planner_cov.fits")]

        # update pointing source based on target ra and dec
        @app.callback(
            [
                Output(target['pointing']['store'].id, "data"),
                Output(target['pointing']['div'].id, "children"),
            ],
            [
                Input(target['targRa'].id, "value"),
                Input(target['targDec'].id, "value"),
            ],
        )
        def getPointingSource(ra, dec):
            print(ra)
            print(dec)
            target_coord = SkyCoord(
                ra = ra << u.deg,
                dec = dec << u.deg,
                frame = 'icrs')
            ps = _sma_pointing_catalog.findPointingSource(target_coord)
            rat = ps['ra']
            c1 = rat.find(':')
            ratt = rat[0:c1]+'h'+rat[c1+1:]
            ratt = ratt.replace(":", "m")
            pstxt = ps['name']+':  '+ratt + ps['dec']
            psb = ps['freq'].strip('][').split(', ')[0].replace("'","")
            psf = ps['flux'].strip('][').split(', ')[0].replace("'","")
            pstxt = ps['name']+':  SMA: '+psb+', '+psf+'Jy'
            print(pstxt)
            return [ps, pstxt]

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

            print()
            print(targName)
            print(target.ra)
            print(target.dec)
            print()
            
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
                'rot': (r*u.deg).to_string(),
                'x_length': (xl*u.arcmin).to_string(),
                'y_length': (yl*u.arcmin).to_string(),
                'x_omega': xo,
                'y_omega': yo,
                'delta': (delta*u.deg).to_string(),
                't_exp': tExp,
                'target_ra': tra,
                'target_dec': tdec,
                't0': t0,
                'ref_frame': refFrame,
            }
            return [json.dumps(d), ]

        # update the Double Lissajous parameter set
        @app.callback(
            [
                Output(mapping['dlis_output_state'].id, "children"),
            ],
            [
                Input(mapping['dlissWrite'].id, "n_clicks"),
            ],
            [
                State(mapping['dlisRotIn'].id, "value"),
                State(mapping['dlisDeltaIn'].id, "value"),
                State(mapping['dlisxLen0In'].id, "value"),
                State(mapping['dlisyLen0In'].id, "value"),
                State(mapping['dlisxOmega0In'].id, "value"),
                State(mapping['dlisyOmega0In'].id, "value"),
                State(mapping['dlisDelta0In'].id, "value"),
                State(mapping['dlisxLen1In'].id, "value"),
                State(mapping['dlisyLen1In'].id, "value"),
                State(mapping['dlisxOmega1In'].id, "value"),
                State(mapping['dlisyOmega1In'].id, "value"),
                State(mapping['dlisDelta1In'].id, "value"),
                State(mapping['listExpIn'].id, "value"),
                State(target['targRa'].id, "value"),
                State(target['targDec'].id, "value"),
                State(target['obsTime'].id, "value"),
                State(target['obsDate'].id, "date"),
                State(mapping['refFrameLiss'].id, "value"),
            ],
            prevent_initial_call=True
        )
        def updateDoubleLissDict(n, r, delta,
                                 xl0, yl0, xo0, yo0, delta0,
                                 xl1, yl1, xo1, yo1, delta1,
                                 tExp, tra, tdec,
                                 obsTime, obsDate, refFrame):
            # format date and time of start of observation
            date_object = date.fromisoformat(obsDate)
            date_string = date_object.strftime('%Y-%m-%d')
            t0 = date_string+'T'+obsTime
            d = {
                'd_rot': (r*u.deg).to_string(),
                'd_delta': (delta*u.deg).to_string(),
                'd_x_length_0': (xl0*u.arcmin).to_string(),
                'd_y_length_0': (yl0*u.arcmin).to_string(),
                'd_x_omega_0': xo0,
                'd_y_omega_0': yo0,
                'd_delta_0': (delta0*u.deg).to_string(),
                'd_x_length_1': (xl1*u.arcmin).to_string(),
                'd_y_length_1': (yl1*u.arcmin).to_string(),
                'd_x_omega_1': xo1,
                'd_y_omega_1': yo1,
                'd_delta_1': (delta1*u.deg).to_string(),
                't_exp': tExp,
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
                State(target['targRa'].id, "value"),
                State(target['targDec'].id, "value"),
                State(target['obsTime'].id, "value"),
                State(target['obsDate'].id, "date"),
                State(mapping['refFrameRast'].id, "value"),
            ],
            prevent_initial_call=True
        )
        def updateRasterDict(n, r, l, s, ns, speed, turn, tra, tdec,
                             obsTime, obsDate, refFrame):
            # format date and time of start of observation
            date_object = date.fromisoformat(obsDate)
            date_string = date_object.strftime('%Y-%m-%d')
            t0 = date_string+'T'+obsTime
            d = {
                'rot': (r*u.deg).to_string(),
                'length': (l*u.arcmin).to_string(),
                'step': (s*u.arcmin).to_string(),
                'nScans': ns,
                'speed': (speed*u.arcsec/u.s).to_string(),
                't_turnaround': turn,
                't_exp': "1 ct",
                'target_ra': tra,
                'target_dec': tdec,
                't0': t0,
                'ref_frame': refFrame,
            }
            return [json.dumps(d), ]

        # update the Rastajous parameter set
        @app.callback(
            [
                Output(mapping['rj_output_state'].id, "children"),
            ],
            [
                Input(mapping['rjWrite'].id, "n_clicks"),
            ],
            [
                State(mapping['rjRotIn'].id, "value"),
                State(mapping['rjLenIn'].id, "value"),
                State(mapping['rjStepIn'].id, "value"),
                State(mapping['rjnScansIn'].id, "value"),
                State(mapping['rjSpeedIn'].id, "value"),
                State(mapping['rjtTurnIn'].id, "value"),
                State(mapping['rjDeltaIn'].id, "value"),
                State(mapping['rjxLen0In'].id, "value"),
                State(mapping['rjyLen0In'].id, "value"),
                State(mapping['rjxOmega0In'].id, "value"),
                State(mapping['rjyOmega0In'].id, "value"),
                State(mapping['rjDelta0In'].id, "value"),
                State(mapping['rjxLen1In'].id, "value"),
                State(mapping['rjyLen1In'].id, "value"),
                State(mapping['rjxOmega1In'].id, "value"),
                State(mapping['rjyOmega1In'].id, "value"),
                State(mapping['rjDelta1In'].id, "value"),
                State(target['targRa'].id, "value"),
                State(target['targDec'].id, "value"),
                State(target['obsTime'].id, "value"),
                State(target['obsDate'].id, "date"),
                State(mapping['rjRefFrame'].id, "value"),
            ],
            prevent_initial_call=True
        )
        def updateRastajousDict(n, r, l, s, ns, speed, turn,
                                delta,
                                xl0, yl0, xo0, yo0, delta0,
                                xl1, yl1, xo1, yo1, delta1,
                                tra, tdec,
                                obsTime, obsDate, refFrame):
            # format date and time of start of observation
            date_object = date.fromisoformat(obsDate)
            date_string = date_object.strftime('%Y-%m-%d')
            t0 = date_string+'T'+obsTime
            d = {
                'rot': (r*u.deg).to_string(),
                'length': (l*u.arcmin).to_string(),
                'step': (s*u.arcmin).to_string(),
                'nScans': ns,
                'speed': (speed*u.arcsec/u.s).to_string(),
                't_turnaround': turn,
                'd_delta': (delta*u.deg).to_string(),
                'd_x_length_0': (xl0*u.arcmin).to_string(),
                'd_y_length_0': (yl0*u.arcmin).to_string(),
                'd_x_omega_0': xo0,
                'd_y_omega_0': yo0,
                'd_delta_0': (delta0*u.deg).to_string(),
                'd_x_length_1': (xl1*u.arcmin).to_string(),
                'd_y_length_1': (yl1*u.arcmin).to_string(),
                'd_x_omega_1': xo1,
                'd_y_omega_1': yo1,
                'd_delta_1': (delta1*u.deg).to_string(),
                't_exp': "1 ct",
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
                Input(mapping['dlis_output_state'].id, "children"),
                Input(mapping['ras_output_state'].id, "children"),
                Input(mapping['rj_output_state'].id, "children"),
            ],
            [
                State(settings['bandIn'].id, "value"),
                State(settings['atmQIn'].id, "value"),
            ],
            prevent_initial_call=True
        )
        def updateContext(lis_output, dlis_output, ras_output, rj_output, band, atmQ):
            print(lis_output)
            print(dlis_output)
            print(ras_output)
            ctx = dash.callback_context
            print(ctx.triggered[0]['prop_id'])
            if (ctx.triggered[0]['prop_id'] == f"{mapping['lis_output_state'].id}.children"):
                d = json.loads(lis_output)
                c = writeSimuContext(d, band, mapType='lissajous', atmQ=atmQ)
                print("lissajous context written")
            elif (ctx.triggered[0]['prop_id'] == f"{mapping['dlis_output_state'].id}.children"):
                d = json.loads(dlis_output)
                c = writeSimuContext(d, band, mapType='doubleLissajous', atmQ=atmQ)
                print("double lissajous context written")
            elif (ctx.triggered[0]['prop_id'] == f"{mapping['rj_output_state'].id}.children"):
                d = json.loads(rj_output)
                c = writeSimuContext(d, band, mapType='rastajous', atmQ=atmQ)
                print("rastajous context written")
            else:
                d = json.loads(ras_output)
                c = writeSimuContext(d, band, mapType='raster', atmQ=atmQ)
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
                tableData = None
                tfig = getEmptyFig(500, 500)
                cfig = getEmptyFig(500, 500)
            else:
                d = Detector(band, elevation=mAlt.to_value(u.deg),
                             atmQuartile=atmQ,
                             telescopeRMSmicrons=telRMS)
                showArray = s2s.count('array')
                tableData, tfig, cfig, hdul = getCelestialPlots(
                    sim, obs, d, units,
                    overlay=overlay,
                    showArray=showArray,
                )
            return [tableData, tfig, cfig]


sim_template_config = SimulatorRuntime(
        Path(_tolteca_simu_template_path).expanduser()).config
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
    t_pattern = m_obs.get_total_time()
    if (m_obs.pattern == 'raster') | (m_obs.pattern == 'rastajous'):
        t_exp = t_pattern
    else:
        t_exp = p_obs['t_exp']
    t = np.arange(0, t_exp.to_value(u.s),
                  1./p_obs['f_smp_mapping'].to_value(u.Hz)) * u.s
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
        't_pattern': t_pattern,
        't_exp': t_exp,
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


@timeit
def fetchConvolved(wcs, bimage, aimage, pixSize, t_exp, fwhmArcsec):
    # here is the convolved image
    with timeit("First convolution: "):
        c = convolve_fft(
            bimage, aimage, normalize_kernel=False, allow_huge=True)

    # and also smooth by a beam-sized gaussian
    sigma = (fwhmArcsec/2.355)/pixSize.to_value(u.arcsec)
    g = Gaussian2DKernel(sigma, sigma)
    with timeit("Second convolution: "):
        cimage = convolve_fft(c, g, normalize_kernel=False)

    # After all of this, cimage is in the units of integration time
    # (in seconds) per pixel.  You can check that cimage.sum() =
    # aimage.sum()*t_total.

    s = cimage.shape
    cra = wcs.pixel_to_world_values(np.arange(0, s[0]), 0)[0]
    cdec = wcs.pixel_to_world_values(0, np.arange(0, s[1]))[1]
    return cimage, cra, cdec


# @cache.memoize()
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
    return overlayImage, ora, odec, owcs


@timeit
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


@timeit
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


@timeit
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


    # Figure out the sun rise and set times at the telescope
    LMT = EarthLocation.from_geodetic(-97.31481605209875,
                                      18.98578175043638, 4500.)
    lmt = Observer(location=LMT, name="LMT", timezone="US/Central")
    sun_rise = lmt.sun_rise_time(obs['t_day'][0], which='next')
    sun_set = lmt.sun_set_time(obs['t_day'][0], which='next')

    # sun rise and set indices
    def nearest(items, pivot):
        return min(items, key=lambda x: abs(x - pivot))
    obt = np.array(obs['t_day'].mjd)
    sr = list(obt).index(nearest(obt, sun_rise.mjd))
    ss = list(obt).index(nearest(obt, sun_set.mjd))
    
    c = ['blue']*len(daytime)
    if (sun_rise.mjd < sun_set.mjd):
        c[sr:ss] = ['orange']*(ss-sr+1)
    else:
        c[:ss] = ['orange']*(ss+1)
        c[sr:-1] = ['orange']*len(c[sr:-1])
     
    upfig = go.Figure()
    xaxis, yaxis = getXYAxisLayouts()
    upfig.add_trace(
        go.Scattergl(
            x=daytime,
            y=obs['target_day'].alt.to_value(u.deg),
            mode='markers',
            marker=dict(color=c,
                        size=4),
            showlegend=False,
        )
    )
    upfig.add_trace(
        go.Scattergl(
            x=daytime[0:1],
            y=obs['target_day'][0:1].alt.to_value(u.deg),
            mode='lines',
            line=dict(color='orange'),
            name='Daytime',
            visible='legendonly',
        )
    )
    upfig.add_trace(
        go.Scattergl(
            x=daytime[0:1],
            y=obs['target_day'][0:1].alt.to_value(u.deg),
            mode='lines',
            line=dict(color='blue'),
            name='Night',
            visible='legendonly',
        )
    )
    upfig.add_trace(
        go.Scattergl(x=obstime,
                     y=targel,
                     mode='lines',
                     line=dict(color='red',
                               width=10),
                     showlegend=False,
        )
    )
    upfig.add_trace(
        go.Scatter(x=daytime,
                   y=[20]*len(daytime),
                   mode='lines',
                   line=dict(color='gray'),
                   showlegend=False,
        )
    )
    upfig.add_trace(
        go.Scatter(x=daytime,
                   y=[-10]*len(daytime),
                   mode='lines',
                   line=dict(color='gray'),
                   fill='tonexty',
                   showlegend=False,
        )
    )
    upfig.update_layout(
        uirevision=True,
        showlegend=True,
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
    upfig.update_xaxes(title_text="Time [UT]",
                       automargin=True)
    upfig.update_yaxes(title_text="Target Elevation [deg]",
                       automargin=True,
                       range=[-10, 90])
    upfig.update_layout(
        legend=dict(
            orientation="h",
            x=1,
            y=1.02,
            yanchor="bottom",
            xanchor="right",
            bordercolor="Black",
            borderwidth=1,
        )
    )
    return upfig


@timeit
def getCelestialPlots(sim, obs, d, units,
                      overlay='None', showArray=0,
                      sensDegred=np.sqrt(2.)):
    simulator = sim.get_instrument_simulator()
    p_obs = sim.get_obs_params()
    sampleTime = 1./p_obs['f_smp_mapping'].to_value(u.Hz)
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

    # if map is too big, increase the pixel size
    if((nPixRa > 2000) | (nPixDec > 2000)):
        newPixSize = 3.*u.arcsec
        nPixDec = np.ceil(
            nPixDec*pixSize.to_value(u.degree)/newPixSize.to_value(u.degree))
        nPixRa = np.ceil(
           nPixRa*pixSize.to_value(u.degree)/newPixSize.to_value(u.degree))
        pixSize = newPixSize

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
        'CDELT1': pixSize.to_value(u.deg),
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

    prabins = np.arange(wcs.pixel_shape[0])
    pdecbins = np.arange(wcs.pixel_shape[1])

    arabins = np.arange(np.floor(pixels_array[0].min()),
                        np.ceil(pixels_array[0].max()+1))
    adecbins = np.arange(np.floor(pixels_array[1].min()),
                         np.ceil(pixels_array[1].max()+1))

    # boresight image is made from a histogram
    # note that x is vertical and y is horizontal in np.histogram2d
    bimage, _, _ = np.histogram2d(pixels_boresight[0],
                                  pixels_boresight[1],
                                  bins=[prabins, pdecbins])
    bimage = bimage.transpose()
    bimage *= sampleTime

    # so is the array image
    aimage, _, _ = np.histogram2d(pixels_array[0],
                                  pixels_array[1],
                                  bins=[arabins, adecbins])

    # the convolved image
    cimage, cra, cdec = fetchConvolved(wcs, bimage, aimage, pixSize,
                                       obs['t_exp'], d.fwhmArcsec)

    # update the wcs with the new image information
    wcs_dict = wcs_input_dict.copy()
    wcs_dict['NAXIS1'] = cimage.shape[0]
    wcs_dict['NAXIS2'] = cimage.shape[1]
    wcs = WCS(wcs_input_dict)

    # convert to a sensitivity in mJy rms if requested
    w = np.where(cimage > 0.1*cimage.max())
    mapArea = len(w[0])*(pixSize.to_value(u.deg))**2
    sigma = (d.fwhmArcsec/2.355)/pixSize.to_value(u.arcsec)
    simage = cimage.copy()*0.
    simage[w] = sensDegred*d.nefd/np.sqrt(cimage[w]*2.*np.pi*sigma**2)
    if(units == 'sens'):
        plotTitle = 'Estimated Depth per Beam-sized Area [mJy]'
        cimage = simage
        hunits = ('UNITS', 'mJy/beam', 'Sensitivity in mJy/beam')
    else:
        plotTitle = 'Integration time per {} pixel'.format(pixSize)
        hunits = ('UNITS',
                  's/{} arcsec pixel'.format(pixSize.to_value(u.arcsec)),
                  'Integration time per pixel')

    # construct Table data
    tableData = None
    name = 'TolTEC ({}mm)'.format(obs['band'])
    obsDur = "{0:4.1f}".format(obs['t_exp'])
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

    # write the image as a fits file
    h = wcs.to_header()
    h['DASHPAGE'] = ('obsPlanner.py', 'TolTEC ObsPlanner')
    h.append(hunits)
    h.append(('OBSDUR',
              '{0:2.3f}'.format(obs['t_exp'].to_value(u.s)),
              'Observation Duration in s'))
    h.append(('MEANALT',
              '{0:3.1f}'.format(
                  obs['target_altaz'].alt.mean().to_value(u.deg)),
              'deg.'))
    h.append(('RDETSENS', '{0:2.3f}'.format(d.nefd*sensDegred), 'mJy rt(s)'))
    phdu = fits.PrimaryHDU(cimage, h)
    hdul = fits.HDUList([phdu])

    # the overlay image
    if(overlay != 'None'):
        overlayImage, ora, odec, owcs = fetchOverlay(
            overlay, obs['target_icrs'],
            cra, cdec)
        range = [max(cra.max(), ora.max()),
                 max(cra.min(), ora.min())]
        # fits.append(fitsfile, overlayImage, owcs.to_header())
        # print("Appended overlay image to {}".format(fitsfile))
        ohdu = fits.ImageHDU(overlayImage, owcs.to_header(), name=overlay)
        hdul.append(ohdu)
    else:
        cm = 2.*(cra.max()-cra.mean())/np.cos(np.deg2rad(cdec.mean()))+cra.mean()
        range = [cm, cra.min()]

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
                      automargin=True, autorange='reversed')
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
    range = [obs_ra.max().to_value(u.deg), obs_ra.min().to_value(u.deg)]

    if(showArray):
        range = [max(obs_ra.max().to_value(u.deg),
                     a_ra.max().to_value(u.deg)),
                 min(obs_ra.min().to_value(u.deg),
                     a_ra.min().to_value(u.deg))]

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
                      automargin=True,
                      autorange='reversed')
    tfig.update_yaxes(title_text="Dec [deg]",
                      scaleanchor="x", scaleratio=1,
                      automargin=True)

    return tableData, tfig, cfig, hdul


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
def writeSimuContext(d, band, mapType='raster', atmQ=25):
    if(band == 1.1):
        bandName = '\"a1100\"'
    elif(band == 1.4):
        bandName = '\"a1400\"'
    else:
        bandName = '\"a2000\"'
    atm_model_name = f'am_q{atmQ:2d}'
    # target_coord = SkyCoord(
    #         ra=d['target_ra'] << u.deg,
    #         dec=d['target_dec'] << u.deg, frame='icrs')
    # target_str = (
    #         f'{target_coord.ra.to_string(unit=u.hourangle, sep="hms", precision=5, pad=True)}'
    #         f'{target_coord.dec.to_string(unit=u.deg, sep="dms", precision=5, pad=True, alwayssign=True)}'
    #         )
    target_str = "{0:}d {1:}d".format(d['target_ra'], d['target_dec'])

    oF = StringIO()
    oF.write("# vim: et ts=2 sts=2 sw=2\n")
    oF.write("---\n")
    oF.write("\n")
    oF.write("_60_simu:\n")
    oF.write("  example_mapping_tel_nc: &example_mapping_tel_nc\n")
    oF.write("    type: lmt_tcs\n")
    oF.write("    filepath: ./tel_toltec_2020-10-01_103371_00_0000.nc\n")
    if(mapType == 'raster'):
        oF.write("  example_mapping_model_raster: &example_mapping_model_raster\n")
        oF.write("    type: tolteca.simu:SkyRasterScanModel\n")
        oF.write("    rot: {}\n".format(u.Quantity(d['rot'])))
        oF.write("    length: {}\n".format(u.Quantity(d['length'])))
        oF.write("    space: {}\n".format(u.Quantity(d['step'])))
        oF.write("    n_scans: {}\n".format(d['nScans']))
        oF.write("    speed: {}\n".format(u.Quantity(d['speed'])))
        oF.write("    t_turnover: {} s\n".format(d['t_turnaround']))
        oF.write("    target: {}\n".format(target_str))
        oF.write("    ref_frame: {}\n".format(d['ref_frame']))
        oF.write("    t0: {}\n".format(d['t0']))
        oF.write("    # lst0: ...\n")
    elif(mapType == 'lissajous'):
        oF.write("  example_mapping_model_lissajous: &example_mapping_model_lissajous\n")
        oF.write("    type: tolteca.simu:SkyLissajousModel\n")
        oF.write("    rot: {}\n".format(u.Quantity(d['rot'])))
        oF.write("    x_length: {}\n".format(u.Quantity(d['x_length'])))
        oF.write("    y_length: {}\n".format(u.Quantity(d['y_length'])))
        oF.write("    x_omega: {} rad/s\n".format(d['x_omega']))
        oF.write("    y_omega: {} rad/s\n".format(d['y_omega']))
        oF.write("    delta: {}\n".format(u.Quantity(d['delta'])))
        oF.write("    target: {}\n".format(target_str))
        oF.write("    ref_frame: {}\n".format(d['ref_frame']))
        oF.write("    t0: {}\n".format(d['t0']))
    elif(mapType == 'doubleLissajous'):
        oF.write("  example_mapping_model_double_lissajous: &example_mapping_model_double_lissajous\n")
        oF.write("    type: tolteca.simu:SkyDoubleLissajousModel\n")
        oF.write("    rot: {}\n".format(u.Quantity(d['d_rot'])))
        oF.write("    delta: {}\n".format(u.Quantity(d['d_delta'])))
        oF.write("    x_length_0: {}\n".format(u.Quantity(d['d_x_length_0'])))
        oF.write("    y_length_0: {}\n".format(u.Quantity(d['d_y_length_0'])))
        oF.write("    x_omega_0: {} rad/s\n".format(d['d_x_omega_0']))
        oF.write("    y_omega_0: {} rad/s\n".format(d['d_y_omega_0']))
        oF.write("    delta_0: {}\n".format(u.Quantity(d['d_delta_0'])))
        oF.write("    x_length_1: {}\n".format(u.Quantity(d['d_x_length_1'])))
        oF.write("    y_length_1: {}\n".format(u.Quantity(d['d_y_length_1'])))
        oF.write("    x_omega_1: {} rad/s\n".format(d['d_x_omega_1']))
        oF.write("    y_omega_1: {} rad/s\n".format(d['d_y_omega_1']))
        oF.write("    delta_1: {}\n".format(u.Quantity(d['d_delta_1'])))
        # oF.write("    target: {0:}d {1:}d\n".format(d['target_ra'], d['target_dec']))
        oF.write("    target: {}\n".format(target_str))
        oF.write("    ref_frame: {}\n".format(d['ref_frame']))
        oF.write("    t0: {}\n".format(d['t0']))
    elif(mapType == 'rastajous'):
        oF.write("  example_mapping_model_rastajous: &example_mapping_model_rastajous\n")
        oF.write("    type: tolteca.simu:SkyRastajousModel\n")
        oF.write("    rot: {}\n".format(u.Quantity(d['rot'])))
        oF.write("    length: {}\n".format(u.Quantity(d['length'])))
        oF.write("    space: {}\n".format(u.Quantity(d['step'])))
        oF.write("    n_scans: {}\n".format(d['nScans']))
        oF.write("    speed: {}\n".format(u.Quantity(d['speed'])))
        oF.write("    t_turnover: {} s\n".format(d['t_turnaround']))
        oF.write("    delta: {}\n".format(u.Quantity(d['d_delta'])))
        oF.write("    x_length_0: {}\n".format(u.Quantity(d['d_x_length_0'])))
        oF.write("    y_length_0: {}\n".format(u.Quantity(d['d_y_length_0'])))
        oF.write("    x_omega_0: {} rad/s\n".format(d['d_x_omega_0']))
        oF.write("    y_omega_0: {} rad/s\n".format(d['d_y_omega_0']))
        oF.write("    delta_0: {}\n".format(u.Quantity(d['d_delta_0'])))
        oF.write("    x_length_1: {}\n".format(u.Quantity(d['d_x_length_1'])))
        oF.write("    y_length_1: {}\n".format(u.Quantity(d['d_y_length_1'])))
        oF.write("    x_omega_1: {} rad/s\n".format(d['d_x_omega_1']))
        oF.write("    y_omega_1: {} rad/s\n".format(d['d_y_omega_1']))
        oF.write("    delta_1: {}\n".format(u.Quantity(d['d_delta_1'])))
        oF.write("    target: {0:}d {1:}d\n".format(d['target_ra'], d['target_dec']))
        oF.write("    ref_frame: {}\n".format(d['ref_frame']))
        oF.write("    t0: {}\n".format(d['t0']))
    oF.write("\n")
    oF.write("simu:\n")
    # oF.write("  # this is the actual simulator\n")
    oF.write("  jobkey: example_simu\n")
    # oF.write("  # plot: true\n")
    oF.write("  instrument:\n")
    oF.write("    name: toltec\n")
    # oF.write("    calobj: cal/calobj_default/index.yaml\n")
    # oF.write("    select: 'array_name == {}'\n".format(bandName))
    # oF.write("    # select: 'pg == 1'\n")
    oF.write("  obs_params:\n")
    oF.write("    f_smp_data: 488 Hz  # the sample frequency for data\n")
    oF.write("    f_smp_mapping: 12.2 Hz  # the sample freq for mapping\n")
    if((mapType == 'rastajous') | (mapType == 'raster')):
        oF.write("    t_exp: 1 ct\n")
    else:
        oF.write("    t_exp: {} s\n".format(d['t_exp']))
    oF.write("  sources:\n")
    oF.write("    - type: point_source_catalog\n")
    oF.write("      filepath: inputs/example_input.asc\n")
    oF.write("    - type: toltec_array_loading\n")
    oF.write("      atm_model_name: {}\n".format(atm_model_name))
    if(mapType == 'lissajous'):
        oF.write("  mapping: *example_mapping_model_lissajous\n")
    elif(mapType == 'doubleLissajous'):
        oF.write("  mapping: *example_mapping_model_double_lissajous\n")
    elif(mapType == 'raster'):
        oF.write("  mapping: *example_mapping_model_raster\n")
    elif(mapType == 'rastajous'):
        oF.write("  mapping: *example_mapping_model_rastajous\n")
    return yaml.safe_load(oF.getvalue())
