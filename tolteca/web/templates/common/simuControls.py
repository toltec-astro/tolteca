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

def makeTooltip(row, text):
    tooltipStyle = {
        'font-size': 15,
        'maxWidth': 300,
        'width': 300
    }
    tip = row.child(
        dbc.Tooltip,
        text, 
        target=row.id,
        style=tooltipStyle)
    return tip



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

    bandRow = t_body.child(html.Div, className='d-flex', justify='begin')
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

    atmQRow = t_body.child(html.Div, className='d-flex', justify='begin')
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

    telRow = t_body.child(html.Div, className='d-flex', justify='begin')
    telRow.child(html.Label("Telescope RMS [um]: "))
    telRMSIn = telRow.child(dcc.Input, value=76.,
                            min=50., max=110.,
                            debounce=True, type='number',
                            style={'width': '25%',
                                   'margin-right': '20px',
                                   'margin-left': '20px'})

    unitsRow = t_body.child(html.Div, className='d-flex', justify='begin')
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

    showArrRow = t_body.child(html.Div, className='d-flex', justify='begin')
    showArray = showArrRow.child(
        dcc.Checklist,
        options=[{'label': 'Show array', 'value': 'array'}],
        value=[],
        inputStyle={"marginRight": "20px", "marginLeft": "5px"})

    overlayRow = t_body.child(html.Div, className='d-flex', justify='begin')
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

    targNameRow = c_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    targNameRow.child(html.Label("Target Name or Coord String: "))
    targName = targNameRow.child(
        dcc.Input, value="", debounce=True,
        type='text',
        style=dict(width='45%',))
    makeTooltip(targNameRow, "Enter source coordinates in hh:mm:ss+dd:mm:ss " \
                "or enter a target name.  If the name server doesn't recognize " \
                "the name, the RA and Dec fields will both be set to zero.")

    targRaRow = c_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    targRaRow.child(html.Label("Target Ra [deg]: "))
    targRa = targRaRow.child(dcc.Input, debounce=True,
                             value=150.08620833, type='number',
                             style=dict(width='45%',))
    makeTooltip(targRaRow, "Target RA in degrees.  This will be set to zero " \
                "if a target name is entered above that is not found in the " \
                "target database.")
    
    targDecRow = c_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    targDecRow.child(html.Label("Target Dec [deg]: "))
    targDec = targDecRow.child(dcc.Input, debounce=True,
                               value=2.58899167, type='number',
                               style=dict(width='45%',))
    makeTooltip(targDecRow, "Target Dec in degrees.  This will be set to zero " \
                "if a target name is entered above that is not found in the " \
                "target database.")

    obsTimeRow = c_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    obsTimeRow.child(html.Label("Obs Start Time (UT): "))
    obsTime = obsTimeRow.child(dcc.Input, debounce=True,
                               value="11:30:00", type='text',
                               style=dict(width='45%',))
    makeTooltip(obsTimeRow, "Select a UT time for " \
                "your observation in hh:mm:ss format.  Note that if you select " \
                "a date and time when the target is below the horizon, an " \
                "error will be thrown and you will need to reselect the time " \
                "based on the uptimes plot in the upper right.  The same plot " \
                "will show if your time is during daytime or nighttime on that " \
                "particular date.")
    
    obsDateRow = c_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    obsDateRow.child(html.Label("Obs Date: "))
    obsDate = obsDateRow.child(dcc.DatePickerSingle,
                               min_date_allowed=date(2000, 11, 19),
                               max_date_allowed=date(2030, 12, 31),
                               initial_visible_month=date(2021, 10, 25),
                               date=date(2021, 10, 25),
                               style=dict(width='45%',))
    makeTooltip(obsDateRow, "Select a date for your observation. " \
                "If you select a date and time when the target is below the " \
                "horizon, an error will be thrown and you will need to reselect " \
                "the time based on the uptimes plot in the upper right.")

    targetAlertRow = c_body.child(html.Div, className='d-flex justify-content-end')
    targetAlert = targetAlertRow.child(
        dbc.Alert,
        "Source elevation too high or too low for obsDate and obsTime.",
        is_open=False,
        color='danger',
        duration=8000)

    targetPopUp = c_body.child(html.Div, className='d-flex justify-content-end').child(
        dcc.ConfirmDialog,
        message='Set obsTime and obsDate so that 20 < elevation < 80 deg.')

    # The pointing source
    pointingRow = c_body.child(html.Div, className='d-flex justify-content-end')
    pointingRow.child(html.Label("Nearest Pointing Source:  ",
                                 title="A suggested pointing source from the SMA calibrator list."))
    pstore = pointingRow.child(dcc.Store)
    pdiv = pointingRow.child(html.Div)
    pointing = {'store' : pstore,
                'div' : pdiv}

    target = {'targName': targName,
              'targRa': targRa,
              'targDec': targDec,
              'obsTime': obsTime,
              'obsDate': obsDate,
              'targetAlert': targetAlert,
              'targetPopUp': targetPopUp,
              'pointing': pointing}
    return target


def getLissajousControls(mappingBox):
    lissBox = mappingBox.child(dbc.Tab, label="Lissajous")
    lissCard = lissBox.child(dbc.Card)
    l_header = lissCard.child(dbc.CardHeader).child(dbc.Row)
    l_body = lissCard.child(dbc.CardBody)
    l_header.child(dbc.Col).child(html.H5, "Lissajous Controls", className='mb-2')
    infoIcon = l_header.child(dbc.Col).child(html.Div, justify='end').child(
        html.Img, src="http://toltec.astro.umass.edu/images/info.png", height=22,
        title="This is the most basic lissajous pattern.  It is useful for photometry maps and other very compact maps.  Making the x and y lengths too large will result in very uneven coverage and so is discouraged.  Try the Double Lissajous in that case instead.")

    lisRotInRow = l_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    lisRotInRow.child(html.Label("Rot [deg]: "))
    lisRotIn = lisRotInRow.child(dcc.Input, value=0.,
                                 min=0., max=180.,
                                 debounce=True, type='number',
                                 style={'width': '25%',
                                        'margin-right': '20px'})
    makeTooltip(lisRotInRow, "Overall rotation of the pattern in degrees " \
                "with respect to the Az/Ra axis.")

    lisxLenInRow = l_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    lisxLenInRow.child(html.Label("x_length [arcmin]: "))
    lisxLenIn = lisxLenInRow.child(dcc.Input, value=0.5,
                                   min=0.001, max=10.,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})
    makeTooltip(lisxLenInRow, "Full width of the pattern along the " \
                "x-direction: 0.001 < x_length < 10 arcminutes.")

    lisyLenInRow = l_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    lisyLenInRow.child(html.Label("y_length [arcmin]: "))
    lisyLenIn = lisyLenInRow.child(dcc.Input, value=0.5,
                                   min=0.001, max=10.,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})
    makeTooltip(lisyLenInRow, "Full width of the pattern along the " \
                "y-direction: 0.001 < y_length < 10 arcminutes.")

    lisxOmegaInRow = l_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    lisxOmegaInRow.child(html.Label("x_omega: "))
    lisxOmegaIn = lisxOmegaInRow.child(dcc.Input, value=9.2,
                                       min=1., max=20.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})
    makeTooltip(lisxOmegaInRow, "Pattern angular frequency [unitless].  This " \
                "is normalized internally. The key value ends up being the " \
                "ratio of x_omega and y_omega.")

    lisyOmegaInRow = l_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    lisyOmegaInRow.child(html.Label("y_omega: "))
    lisyOmegaIn = lisyOmegaInRow.child(dcc.Input, value=8,
                                       min=1., max=20.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})
    makeTooltip(lisyOmegaInRow, "Pattern angular frequency [unitless].  This " \
                "is normalized internally. The key value ends up being the " \
                "ratio of x_omega and y_omega.")

    lisDeltaInRow = l_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    lisDeltaInRow.child(html.Label("delta [deg]: "))
    lisDeltaIn = lisDeltaInRow.child(dcc.Input, value=45.,
                                     min=0.0, max=90.,
                                     debounce=True, type='number',
                                     style={'width': '25%',
                                            'margin-right': '20px'})
    makeTooltip(lisDeltaInRow, "Phase difference in degrees.")

    listExpInRow = l_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    listExpInRow.child(html.Label("t_exp [s]: "))
    listExpIn = listExpInRow.child(dcc.Input, value=120.,
                                   min=1., max=1800.,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})
    makeTooltip(listExpInRow, "Exposure time of observation in seconds. 1<t_exp<1800 s.")

    refFrameLissRow = l_body.child(html.Div, className='d-flex justify-content-end', justify='begin')
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

    lissWriteRow = l_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    lissWrite = lissWriteRow.child(
        dbc.Button, "Execute Pattern", color="danger", size='sm',
        style={'width': '45%', "margin-right": '10px'})

    lis_output_state = l_body.child(html.Div, className='d-flex justify-content-end').child(
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
    dl_header = dlissCard.child(dbc.CardHeader).child(dbc.Row)
    dl_body = dlissCard.child(dbc.CardBody)
    dl_header.child(dbc.Col, width=8).child(html.H5, "Double Lissajous Controls", className='mb-2')
    infoIcon = dl_header.child(dbc.Col).child(html.Div, justify='end').child(
        html.Img, src="http://toltec.astro.umass.edu/images/info.png", height=22,
        title="This is the advanced lissajous pattern which simply sums to other lissajous patterns.  The general idea is to have a small, slow lissajous run underneath a larger lissajous pattern.  We haven't optimized the parameters for these yet but eventually we will and then we'll set them as the defaults here.")

    dlisRotInRow = dl_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    dlisRotInRow.child(html.Label("Rot [deg]: "))
    dlisRotIn = dlisRotInRow.child(dcc.Input, value=0.,
                                   min=0., max=180.,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})

    dlisDeltaInRow = dl_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    dlisDeltaInRow.child(html.Label("delta [deg]: "))
    dlisDeltaIn = dlisDeltaInRow.child(dcc.Input, value=45.,
                                       min=0.0, max=90.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

    dlisxLen0InRow = dl_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    dlisxLen0InRow.child(html.Label("x_length_0 [arcmin]: "))
    dlisxLen0In = dlisxLen0InRow.child(dcc.Input, value=0.5,
                                       min=0.001, max=10.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

    dlisyLen0InRow = dl_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    dlisyLen0InRow.child(html.Label("y_length_0 [arcmin]: "))
    dlisyLen0In = dlisyLen0InRow.child(dcc.Input, value=0.5,
                                       min=0.001, max=10.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

    dlisxOmega0InRow = dl_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    dlisxOmega0InRow.child(html.Label("x_omega_0: "))
    dlisxOmega0In = dlisxOmega0InRow.child(dcc.Input, value=9.2,
                                           min=1., max=20.,
                                           debounce=True, type='number',
                                           style={'width': '25%',
                                                  'margin-right': '20px'})

    dlisyOmega0InRow = dl_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    dlisyOmega0InRow.child(html.Label("y_omega_0: "))
    dlisyOmega0In = dlisyOmega0InRow.child(dcc.Input, value=8,
                                           min=1., max=20.,
                                           debounce=True, type='number',
                                           style={'width': '25%',
                                                  'margin-right': '20px'})

    dlisDelta0InRow = dl_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    dlisDelta0InRow.child(html.Label("delta_0 [deg]: "))
    dlisDelta0In = dlisDelta0InRow.child(dcc.Input, value=45.,
                                         min=0.0, max=90.,
                                         debounce=True, type='number',
                                         style={'width': '25%',
                                                'margin-right': '20px'})

    dlisxLen1InRow = dl_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    dlisxLen1InRow.child(html.Label("x_length_1 [arcmin]: "))
    dlisxLen1In = dlisxLen1InRow.child(dcc.Input, value=0.5,
                                       min=0.001, max=10.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

    dlisyLen1InRow = dl_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    dlisyLen1InRow.child(html.Label("y_length_1 [arcmin]: "))
    dlisyLen1In = dlisyLen1InRow.child(dcc.Input, value=0.5,
                                       min=0.001, max=10.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

    dlisxOmega1InRow = dl_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    dlisxOmega1InRow.child(html.Label("x_omega_1: "))
    dlisxOmega1In = dlisxOmega1InRow.child(dcc.Input, value=9.2,
                                           min=1., max=20.,
                                           debounce=True, type='number',
                                           style={'width': '25%',
                                                  'margin-right': '20px'})

    dlisyOmega1InRow = dl_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    dlisyOmega1InRow.child(html.Label("y_omega_1: "))
    dlisyOmega1In = dlisyOmega1InRow.child(dcc.Input, value=8,
                                           min=1., max=20.,
                                           debounce=True, type='number',
                                           style={'width': '25%',
                                                  'margin-right': '20px'})

    dlisDelta1InRow = dl_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    dlisDelta1InRow.child(html.Label("delta_1 [deg]: "))
    dlisDelta1In = dlisDelta1InRow.child(dcc.Input, value=45.,
                                         min=0.0, max=90.,
                                         debounce=True, type='number',
                                         style={'width': '25%',
                                                'margin-right': '20px'})

    dlistExpInRow = dl_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    dlistExpInRow.child(html.Label("t_exp [s]: "))
    dlistExpIn = dlistExpInRow.child(dcc.Input, value=120.,
                                     min=1., max=1800.,
                                     debounce=True, type='number',
                                     style={'width': '25%',
                                            'margin-right': '20px'})

    drefFrameLissRow = dl_body.child(html.Div, className='d-flex justify-content-end', justify='begin')
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

    dlissWriteRow = dl_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    dlissWrite = dlissWriteRow.child(
        dbc.Button, "Execute Pattern", color="danger", size='sm',
        style={'width': '45%', "margin-right": '10px'})

    dlis_output_state = dl_body.child(html.Div, className='d-flex justify-content-end').child(
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
    r_header = rasCard.child(dbc.CardHeader).child(dbc.Row)
    r_body = rasCard.child(dbc.CardBody)
    r_header.child(dbc.Col, width=8).child(html.H5, "Raster Controls", className='mb-2')
    infoIcon = r_header.child(dbc.Col).child(html.Div, justify='end').child(
        html.Img, src="http://toltec.astro.umass.edu/images/info.png", height=22,
        title="The raster pattern is our go-to pattern for making large maps " \
        "with high-speed scans.")
    
    rasRotInRow = r_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rasRotInRow.child(html.Label("Rot [deg]: "))
    rasRotIn = rasRotInRow.child(dcc.Input, value=0.,
                                 min=0., max=90.,
                                 debounce=True, type='number',
                                 style={'width': '25%',
                                        'margin-right': '20px'})
    makeTooltip(rasRotInRow, "Rotation angle with respect to Az or " \
                "Ra axis [degrees]. 0<Rot<90 deg.")

    rasLenInRow = r_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rasLenInRow.child(html.Label("length [arcmin]: "))
    rasLenIn = rasLenInRow.child(dcc.Input, value=15.,
                                 min=0.0001, max=30.,
                                 debounce=True, type='number',
                                 style={'width': '25%',
                                        'margin-right': '20px'})
    makeTooltip(rasLenInRow,
                "Scan length of pattern [arcminuts]. For memory reasons, this " \
                "tool restricts scan lengths to 30 arcmin.  Use tolteca.simu " \
                "for simuations requiring larger maps.")

    rasStepInRow = r_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rasStepInRow.child(html.Label("step [arcmin]: "))
    rasStepIn = rasStepInRow.child(dcc.Input, value=0.5,
                                   min=0.1, max=4.,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})
    makeTooltip(rasStepInRow,
                "Step size of pattern [arcminutes].  This is the step size " \
                "between rows in the pattern.  To avoid big discontinuities in " \
                "the pattern, this should be less than the field of view of " \
                "4 arcminutes.")

    rasnScansInRow = r_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rasnScansInRow.child(html.Label("nScans: "))
    rasnScansIn = rasnScansInRow.child(dcc.Input, value=15,
                                       min=1, max=30,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})
    makeTooltip(rasnScansInRow,
                "The number of rows in the pattern. For memory reasons, this " \
                "tool restricts the maximum number of rows to 30.  Use tolteca.simu " \
                "for simuations requiring larger maps.")

    rasSpeedInRow = r_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rasSpeedInRow.child(html.Label("speed [arcsec/s]: "))
    rasSpeedIn = rasSpeedInRow.child(dcc.Input, value=50.,
                                     min=0.0001, max=500,
                                     debounce=True, type='number',
                                     style={'width': '25%',
                                            'margin-right': '20px'})
    makeTooltip(rasSpeedInRow,
                "Scan speed across a row [arcsec/s].  This should be balanced with " \
                "the turn around time (typically 5s) to ensure good efficiency.  " \
                "Otherwise, go as fast as possible to decouple the atmospheric " \
                "signal from the astronomical signal.")

    rastTurnInRow = r_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rastTurnInRow.child(html.Label("t_turnaround [s]: "))
    rastTurnIn = rastTurnInRow.child(dcc.Input, value=5.,
                                     min=3, max=10.,
                                     debounce=True, type='number',
                                     style={'width': '25%',
                                            'margin-right': '20px'})
    makeTooltip(rastTurnInRow,
                "The turn-around time at the end of the row.  We don't have a " \
                "solid measure of this so we recommend 5s.")

    refFrameRastRow = r_body.child(html.Div, className='d-flex justify-content-end', justify='begin')
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

    rasWriteRow = r_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rasWrite = rasWriteRow.child(
        dbc.Button, "Execute Pattern", color="danger", size='sm',
        style={'width': '45%', "margin-right": '10px'})

    ras_output_state = r_body.child(html.Div, className='d-flex justify-content-end').child(
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
    rj_header = rjCard.child(dbc.CardHeader).child(dbc.Row)
    rj_body = rjCard.child(dbc.CardBody)
    rj_header.child(dbc.Col).child(html.H5, "Rastajous Controls", className='mb-2')
    infoIcon = rj_header.child(dbc.Col).child(html.Div, justify='end').child(
        html.Img, src="http://toltec.astro.umass.edu/images/info.png", height=22,
        title="A Rastajous is a combination of a Double Lissajous and a Raster map.  The idea is that the lissajous pattern will be slowly swept over a larger field, akin to painting with a very fat brush.  The key is to get the relative timing of the two patterns right.  The raster part of the map should be pretty slow so that the turns at the ends of the scans can be preserved.")
    
    rotInRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rotInRow.child(html.Label("Rot [deg]: "))
    rjRotIn = rotInRow.child(dcc.Input, value=0.,
                             min=0., max=90.,
                             debounce=True, type='number',
                             style={'width': '25%',
                                    'margin-right': '20px'})

    rjLenInRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rjLenInRow.child(html.Label("length [arcmin]: "))
    rjLenIn = rjLenInRow.child(dcc.Input, value=10.,
                               min=0.0001, max=30.,
                               debounce=True, type='number',
                               style={'width': '25%',
                                      'margin-right': '20px'})

    rjStepInRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rjStepInRow.child(html.Label("step [arcmin]: "))
    rjStepIn = rjStepInRow.child(dcc.Input, value=1.,
                                 min=0.1, max=4.,
                                 debounce=True, type='number',
                                 style={'width': '25%',
                                        'margin-right': '20px'})

    rjnScansInRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rjnScansInRow.child(html.Label("nScans: "))
    rjnScansIn = rjnScansInRow.child(dcc.Input, value=3,
                                     min=1, max=30,
                                     debounce=True, type='number',
                                     style={'width': '25%',
                                            'margin-right': '20px'})

    rjSpeedInRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rjSpeedInRow.child(html.Label("speed [arcsec/s]: "))
    rjSpeedIn = rjSpeedInRow.child(dcc.Input, value=5.,
                                   min=0.0001, max=500,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})

    rjtTurnInRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rjtTurnInRow.child(html.Label("t_turnaround [s]: "))
    rjtTurnIn = rjtTurnInRow.child(dcc.Input, value=0.1,
                                   min=0.1, max=10.,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})

    rjDeltaInRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rjDeltaInRow.child(html.Label("delta [deg]: "))
    rjDeltaIn = rjDeltaInRow.child(dcc.Input, value=45.,
                                   min=0.0, max=90.,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})

    rjxLen0InRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rjxLen0InRow.child(html.Label("x_length_0 [arcmin]: "))
    rjxLen0In = rjxLen0InRow.child(dcc.Input, value=2.,
                                   min=0.001, max=10.,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})

    rjyLen0InRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rjyLen0InRow.child(html.Label("y_length_0 [arcmin]: "))
    rjyLen0In = rjyLen0InRow.child(dcc.Input, value=2.,
                                   min=0.001, max=10.,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})

    rjxOmega0InRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rjxOmega0InRow.child(html.Label("x_omega_0: "))
    rjxOmega0In = rjxOmega0InRow.child(dcc.Input, value=9.2,
                                       min=1., max=20.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

    rjyOmega0InRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rjyOmega0InRow.child(html.Label("y_omega_0: "))
    rjyOmega0In = rjyOmega0InRow.child(dcc.Input, value=8,
                                       min=1., max=20.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

    rjDelta0InRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rjDelta0InRow.child(html.Label("delta_0 [deg]: "))
    rjDelta0In = rjDelta0InRow.child(dcc.Input, value=45.,
                                     min=0.0, max=90.,
                                     debounce=True, type='number',
                                     style={'width': '25%',
                                            'margin-right': '20px'})

    rjxLen1InRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rjxLen1InRow.child(html.Label("x_length_1 [arcmin]: "))
    rjxLen1In = rjxLen1InRow.child(dcc.Input, value=0.5,
                                   min=0.001, max=10.,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})

    rjyLen1InRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rjyLen1InRow.child(html.Label("y_length_1 [arcmin]: "))
    rjyLen1In = rjyLen1InRow.child(dcc.Input, value=0.5,
                                   min=0.001, max=10.,
                                   debounce=True, type='number',
                                   style={'width': '25%',
                                          'margin-right': '20px'})

    rjxOmega1InRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rjxOmega1InRow.child(html.Label("x_omega_1: "))
    rjxOmega1In = rjxOmega1InRow.child(dcc.Input, value=9.2/20.,
                                       min=1.e-4, max=20.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

    rjyOmega1InRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rjyOmega1InRow.child(html.Label("y_omega_1: "))
    rjyOmega1In = rjyOmega1InRow.child(dcc.Input, value=8/20.,
                                       min=1.e-4, max=20.,
                                       debounce=True, type='number',
                                       style={'width': '25%',
                                              'margin-right': '20px'})

    rjDelta1InRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rjDelta1InRow.child(html.Label("delta_1 [deg]: "))
    rjDelta1In = rjDelta1InRow.child(dcc.Input, value=45.,
                                     min=0.0, max=90.,
                                     debounce=True, type='number',
                                     style={'width': '25%',
                                            'margin-right': '20px'})

    refFrameRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='begin')
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

    writeRow = rj_body.child(html.Div, className='d-flex justify-content-end', justify='end')
    rjWrite = writeRow.child(
        dbc.Button, "Execute Pattern", color="danger", size='sm',
        style={'width': '45%', "margin-right": '10px'})

    rj_output_state = rj_body.child(html.Div, className='d-flex justify-content-end').child(
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
