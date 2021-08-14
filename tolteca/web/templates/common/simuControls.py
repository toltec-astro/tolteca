from dasha.web.templates.collapsecontent import CollapseContent
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from datetime import date


# These functions retun dasha components that are useful for setting
# up the simulator yaml file.  These have been extracted from the
# TolTEC Observation Planner (obsPlanner.py) so looking at them in
# action at that page will be very helpful.
# Each of these functions returns a dictionary of controls.

# The global settings card provides general settings for setting up
# the simulation and telescope.  This should be extended and
# generalized even further since it is somewhat TolTEC-specific at the
# moment.
# Inputs:
#   controlBox - a dasha work area
def getSettingsCard(controlBox):
    settingsRow = controlBox.child(dbc.Row)
    settingsCard = settingsRow.child(dbc.Col).child(dbc.Card)
    t_header = settingsCard.child(dbc.CardHeader)
    t_header.child(html.H5, "Global Settings", className='mb-2')
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
    return settings


# The source card contains information about the target source and the
# observing date and time.  Several of the controls are tightly tied
# together through the code in the callbacks, so refer to
# obsPlanner.py for the full set of code.
# Inputs:
#   controlBox - a dasha work area
def getSourceCard(controlBox):
    sourceBox = controlBox.child(dbc.Row, className='mt-3').child(dbc.Col)
    sourceCard = sourceBox.child(dbc.Card, color='danger', inverse=False,
                                 outline=True)
    c_header = sourceCard.child(dbc.CardHeader)
    c_body = sourceCard.child(dbc.CardBody)
    c_header.child(html.H5, "Target Choice", className='mb-2')

    targNameRow = c_body.child(dbc.Row, justify='end')
    targNameRow.child(html.Label("Target Name or Coord String: "))
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
    return target


def getLissajousControls(mappingBox):
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

    lis_output_state = l_body.child(dbc.Row).child(
        CollapseContent(
            button_text='Details ...')).content

    lisControls = {'lisRotIn': lisRotIn,
                   'lisxLenIn': lisxLenIn,
                   'lisyLenIn': lisyLenIn,
                   'lisxOmegaIn': lisxOmegaIn,
                   'lisyOmegaIn': lisyOmegaIn,
                   'lisDeltaIn': lisDeltaIn,
                   'listExpIn': listExpIn,
                   'refFrameLiss': refFrameLiss,
                   'lissWrite': lissWrite,
                   'lis_output_state': lis_output_state}
    return lisControls


def getDoubleLissajousControls(mappingBox):
    dlissBox = mappingBox.child(dbc.Tab, label="Double Lissajous")
    dlissCard = dlissBox.child(dbc.Card)
    dl_header = dlissCard.child(dbc.CardHeader)
    dl_body = dlissCard.child(dbc.CardBody)
    dl_header.child(html.H5, "Double Lissajous Controls", className='mb-2')

    dlisRotInRow = dl_body.child(dbc.Row, justify='end')
    dlisRotInRow.child(html.Label("Rot [deg]: "))
    dlisRotIn = dlisRotInRow.child(dcc.Input, value=0.,
                                   min=0., max=180.,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})

    dlisDeltaInRow = dl_body.child(dbc.Row, justify='end')
    dlisDeltaInRow.child(html.Label("delta [deg]: "))
    dlisDeltaIn = dlisDeltaInRow.child(dcc.Input, value=45.,
                                       min=0.0, max=90.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

    dlisxLen0InRow = dl_body.child(dbc.Row, justify='end')
    dlisxLen0InRow.child(html.Label("x_length_0 [arcmin]: "))
    dlisxLen0In = dlisxLen0InRow.child(dcc.Input, value=0.5,
                                       min=0.001, max=10.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

    dlisyLen0InRow = dl_body.child(dbc.Row, justify='end')
    dlisyLen0InRow.child(html.Label("y_length_0 [arcmin]: "))
    dlisyLen0In = dlisyLen0InRow.child(dcc.Input, value=0.5,
                                       min=0.001, max=10.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

    dlisxOmega0InRow = dl_body.child(dbc.Row, justify='end')
    dlisxOmega0InRow.child(html.Label("x_omega_0: "))
    dlisxOmega0In = dlisxOmega0InRow.child(dcc.Input, value=9.2,
                                           min=1., max=20.,
                                           debounce=True, type='number',
                                           style={'width': '25%',
                                                  'margin-right': '20px'})

    dlisyOmega0InRow = dl_body.child(dbc.Row, justify='end')
    dlisyOmega0InRow.child(html.Label("y_omega_0: "))
    dlisyOmega0In = dlisyOmega0InRow.child(dcc.Input, value=8,
                                           min=1., max=20.,
                                           debounce=True, type='number',
                                           style={'width': '25%',
                                                  'margin-right': '20px'})

    dlisDelta0InRow = dl_body.child(dbc.Row, justify='end')
    dlisDelta0InRow.child(html.Label("delta_0 [deg]: "))
    dlisDelta0In = dlisDelta0InRow.child(dcc.Input, value=45.,
                                         min=0.0, max=90.,
                                         debounce=True, type='number',
                                         style={'width': '25%',
                                                'margin-right': '20px'})

    dlisxLen1InRow = dl_body.child(dbc.Row, justify='end')
    dlisxLen1InRow.child(html.Label("x_length_1 [arcmin]: "))
    dlisxLen1In = dlisxLen1InRow.child(dcc.Input, value=0.5,
                                       min=0.001, max=10.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

    dlisyLen1InRow = dl_body.child(dbc.Row, justify='end')
    dlisyLen1InRow.child(html.Label("y_length_1 [arcmin]: "))
    dlisyLen1In = dlisyLen1InRow.child(dcc.Input, value=0.5,
                                       min=0.001, max=10.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

    dlisxOmega1InRow = dl_body.child(dbc.Row, justify='end')
    dlisxOmega1InRow.child(html.Label("x_omega_1: "))
    dlisxOmega1In = dlisxOmega1InRow.child(dcc.Input, value=9.2,
                                           min=1., max=20.,
                                           debounce=True, type='number',
                                           style={'width': '25%',
                                                  'margin-right': '20px'})

    dlisyOmega1InRow = dl_body.child(dbc.Row, justify='end')
    dlisyOmega1InRow.child(html.Label("y_omega_1: "))
    dlisyOmega1In = dlisyOmega1InRow.child(dcc.Input, value=8,
                                           min=1., max=20.,
                                           debounce=True, type='number',
                                           style={'width': '25%',
                                                  'margin-right': '20px'})

    dlisDelta1InRow = dl_body.child(dbc.Row, justify='end')
    dlisDelta1InRow.child(html.Label("delta_1 [deg]: "))
    dlisDelta1In = dlisDelta1InRow.child(dcc.Input, value=45.,
                                         min=0.0, max=90.,
                                         debounce=True, type='number',
                                         style={'width': '25%',
                                                'margin-right': '20px'})

    dlistExpInRow = dl_body.child(dbc.Row, justify='end')
    dlistExpInRow.child(html.Label("t_exp [s]: "))
    dlistExpIn = dlistExpInRow.child(dcc.Input, value=120.,
                                     min=1., max=1800.,
                                     debounce=True, type='number',
                                     style={'width': '25%',
                                            'margin-right': '20px'})

    drefFrameLissRow = dl_body.child(dbc.Row, justify='begin')
    drefFrameLissRow.child(html.Label("Tel Frame: "))
    drefFrameLiss = drefFrameLissRow.child(
        dcc.RadioItems, options=[
            {'label': 'Az/El', 'value': 'altaz'},
            {'label': 'Ra/Dec', 'value': 'icrs'},
        ],
        value='altaz',
        labelStyle={'display': 'inline-block'},
        inputStyle={"margin-right": "5px",
                    "margin-left": "20px"},
    )

    dlissWriteRow = dl_body.child(dbc.Row, justify='end')
    dlissWrite = dlissWriteRow.child(
        dbc.Button, "Execute Pattern", color="danger", size='sm',
        style={'width': '45%', "margin-right": '10px'})

    dlis_output_state = dl_body.child(dbc.Row).child(
        CollapseContent(
            button_text='Details ...')).content

    doubleLisControls = {'dlisRotIn': dlisRotIn,
                         'dlisDeltaIn': dlisDeltaIn,
                         'dlisxLen0In': dlisxLen0In,
                         'dlisyLen0In': dlisyLen0In,
                         'dlisxOmega0In': dlisxOmega0In,
                         'dlisyOmega0In': dlisyOmega0In,
                         'dlisDelta0In': dlisDelta0In,
                         'dlisxLen1In': dlisxLen1In,
                         'dlisyLen1In': dlisyLen1In,
                         'dlisxOmega1In': dlisxOmega1In,
                         'dlisyOmega1In': dlisyOmega1In,
                         'dlisDelta1In': dlisDelta1In,
                         'dlistExpIn': dlistExpIn,
                         'drefFrameLiss': drefFrameLiss,
                         'dlissWrite': dlissWrite,
                         'dlis_output_state': dlis_output_state}
    return doubleLisControls


def getRasterControls(mappingBox):
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

    ras_output_state = r_body.child(dbc.Row).child(
        CollapseContent(
            button_text='Details ...')).content

    rasterControls = {'rasRotIn': rasRotIn,
                      'rasLenIn': rasLenIn,
                      'rasStepIn': rasStepIn,
                      'rasnScansIn': rasnScansIn,
                      'rasSpeedIn': rasSpeedIn,
                      'rastTurnIn': rastTurnIn,
                      'refFrameRast': refFrameRast,
                      'rasWrite': rasWrite,
                      'ras_output_state': ras_output_state}
    return rasterControls


def getRastajousControls(mappingBox):
    rjBox = mappingBox.child(dbc.Tab, label="Rastajous")
    rjCard = rjBox.child(dbc.Card)
    rj_header = rjCard.child(dbc.CardHeader)
    rj_body = rjCard.child(dbc.CardBody)
    rj_header.child(html.H5, "Rastajous Controls", className='mb-2')

    rotInRow = rj_body.child(dbc.Row, justify='end')
    rotInRow.child(html.Label("Rot [deg]: "))
    rjRotIn = rotInRow.child(dcc.Input, value=0.,
                             min=0., max=90.,
                             debounce=True, type='number',
                             style={'width': '25%',
                                    'margin-right': '20px'})

    rjLenInRow = rj_body.child(dbc.Row, justify='end')
    rjLenInRow.child(html.Label("length [arcmin]: "))
    rjLenIn = rjLenInRow.child(dcc.Input, value=10.,
                               min=0.0001, max=30.,
                               debounce=True, type='number',
                               style={'width': '25%',
                                      'margin-right': '20px'})

    rjStepInRow = rj_body.child(dbc.Row, justify='end')
    rjStepInRow.child(html.Label("step [arcmin]: "))
    rjStepIn = rjStepInRow.child(dcc.Input, value=1.,
                                 min=0.1, max=4.,
                                 debounce=True, type='number',
                                 style={'width': '25%',
                                        'margin-right': '20px'})

    rjnScansInRow = rj_body.child(dbc.Row, justify='end')
    rjnScansInRow.child(html.Label("nScans: "))
    rjnScansIn = rjnScansInRow.child(dcc.Input, value=3,
                                     min=1, max=30,
                                     debounce=True, type='number',
                                     style={'width': '25%',
                                            'margin-right': '20px'})

    rjSpeedInRow = rj_body.child(dbc.Row, justify='end')
    rjSpeedInRow.child(html.Label("speed [arcsec/s]: "))
    rjSpeedIn = rjSpeedInRow.child(dcc.Input, value=5.,
                                   min=0.0001, max=500,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})

    rjtTurnInRow = rj_body.child(dbc.Row, justify='end')
    rjtTurnInRow.child(html.Label("t_turnaround [s]: "))
    rjtTurnIn = rjtTurnInRow.child(dcc.Input, value=0.1,
                                   min=0.1, max=10.,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})

    rjDeltaInRow = rj_body.child(dbc.Row, justify='end')
    rjDeltaInRow.child(html.Label("delta [deg]: "))
    rjDeltaIn = rjDeltaInRow.child(dcc.Input, value=45.,
                                   min=0.0, max=90.,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})

    rjxLen0InRow = rj_body.child(dbc.Row, justify='end')
    rjxLen0InRow.child(html.Label("x_length_0 [arcmin]: "))
    rjxLen0In = rjxLen0InRow.child(dcc.Input, value=2.,
                                   min=0.001, max=10.,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})

    rjyLen0InRow = rj_body.child(dbc.Row, justify='end')
    rjyLen0InRow.child(html.Label("y_length_0 [arcmin]: "))
    rjyLen0In = rjyLen0InRow.child(dcc.Input, value=2.,
                                   min=0.001, max=10.,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})

    rjxOmega0InRow = rj_body.child(dbc.Row, justify='end')
    rjxOmega0InRow.child(html.Label("x_omega_0: "))
    rjxOmega0In = rjxOmega0InRow.child(dcc.Input, value=9.2,
                                       min=1., max=20.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

    rjyOmega0InRow = rj_body.child(dbc.Row, justify='end')
    rjyOmega0InRow.child(html.Label("y_omega_0: "))
    rjyOmega0In = rjyOmega0InRow.child(dcc.Input, value=8,
                                       min=1., max=20.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

    rjDelta0InRow = rj_body.child(dbc.Row, justify='end')
    rjDelta0InRow.child(html.Label("delta_0 [deg]: "))
    rjDelta0In = rjDelta0InRow.child(dcc.Input, value=45.,
                                     min=0.0, max=90.,
                                     debounce=True, type='number',
                                     style={'width': '25%',
                                            'margin-right': '20px'})

    rjxLen1InRow = rj_body.child(dbc.Row, justify='end')
    rjxLen1InRow.child(html.Label("x_length_1 [arcmin]: "))
    rjxLen1In = rjxLen1InRow.child(dcc.Input, value=0.5,
                                   min=0.001, max=10.,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})

    rjyLen1InRow = rj_body.child(dbc.Row, justify='end')
    rjyLen1InRow.child(html.Label("y_length_1 [arcmin]: "))
    rjyLen1In = rjyLen1InRow.child(dcc.Input, value=0.5,
                                   min=0.001, max=10.,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})

    rjxOmega1InRow = rj_body.child(dbc.Row, justify='end')
    rjxOmega1InRow.child(html.Label("x_omega_1: "))
    rjxOmega1In = rjxOmega1InRow.child(dcc.Input, value=9.2/20.,
                                       min=1.e-4, max=20.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

    rjyOmega1InRow = rj_body.child(dbc.Row, justify='end')
    rjyOmega1InRow.child(html.Label("y_omega_1: "))
    rjyOmega1In = rjyOmega1InRow.child(dcc.Input, value=8/20.,
                                       min=1.e-4, max=20.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

    rjDelta1InRow = rj_body.child(dbc.Row, justify='end')
    rjDelta1InRow.child(html.Label("delta_1 [deg]: "))
    rjDelta1In = rjDelta1InRow.child(dcc.Input, value=45.,
                                     min=0.0, max=90.,
                                     debounce=True, type='number',
                                     style={'width': '25%',
                                            'margin-right': '20px'})

    refFrameRow = rj_body.child(dbc.Row, justify='begin')
    refFrameRow.child(html.Label("Tel Frame: "))
    rjRefFrame = refFrameRow.child(
        dcc.RadioItems, options=[
            {'label': 'Az/El', 'value': 'altaz'},
            {'label': 'Ra/Dec', 'value': 'icrs'},
        ],
        value='altaz',
        labelStyle={'display': 'inline-block'},
        inputStyle={"margin-right": "5px",
                    "margin-left": "20px"},
    )

    writeRow = rj_body.child(dbc.Row, justify='end')
    rjWrite = writeRow.child(
        dbc.Button, "Execute Pattern", color="danger", size='sm',
        style={'width': '45%', "margin-right": '10px'})

    rj_output_state = rj_body.child(dbc.Row).child(
        CollapseContent(
            button_text='Details ...')).content

    rastajousControls = {'rjRotIn': rjRotIn,
                         'rjLenIn': rjLenIn,
                         'rjStepIn': rjStepIn,
                         'rjnScansIn': rjnScansIn,
                         'rjSpeedIn': rjSpeedIn,
                         'rjtTurnIn': rjtTurnIn,
                         'rjDeltaIn': rjDeltaIn,
                         'rjxLen0In': rjxLen0In,
                         'rjyLen0In': rjyLen0In,
                         'rjxOmega0In': rjxOmega0In,
                         'rjyOmega0In': rjyOmega0In,
                         'rjDelta0In': rjDelta0In,
                         'rjxLen1In': rjxLen1In,
                         'rjyLen1In': rjyLen1In,
                         'rjxOmega1In': rjxOmega1In,
                         'rjyOmega1In': rjyOmega1In,
                         'rjDelta1In': rjDelta1In,
                         'rjRefFrame': rjRefFrame,
                         'rjWrite': rjWrite,
                         'rj_output_state': rj_output_state}
    return rastajousControls
