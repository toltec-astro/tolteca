import numpy as np


from dasha.web.templates import ComponentTemplate
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Output, Input, State, MATCH, ALL
from dash import no_update

import dash_table

from tolteca.web.templates.beammap_sources.wyatt_classes import obs, ncdata
from pathlib import Path
from dasha.web.extensions.cache import cache
import functools

import pandas as pd

#Temp
import sys

'''
# Uses the wyatt_classes.py file for the class to hold the data.
sys.path.append('/Users/mmccrackan/dasha/dasha/examples/beammap_sources/')
from wyatt_classes import obs, ncdata
'''


f_tone_filepath = Path(__file__).parent.joinpath(
        'beammap_sources/10886_f_tone.npy')

df = pd.DataFrame(
    {
        "Parameter": ["S/N", "X Centroid [px]", "Y Centroid [px]",
                      "X FWHM [px]", "Y FWHM [px]", "Frequency [MHz]",
                      "Network", "Detector Number on Network"],
        "Value": ["N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"],
    }
)


@cache.memoize(timeout=60 * 60 * 60)
def get_ncobs(*args, **kwargs):
    return obs(*args, **kwargs)

class beammap(ComponentTemplate):
    _component_cls = dbc.Container
    fluid = True

    def _setup_tab(self, app, ncobs, array_indices, container, drp_container_options, nw_checklist):

        # Get parameter values, freq, and NW array from ncobs
        def from_ncobs(nw_chk_value, array_indices):
            # Dict for parameter values
            p = {}
            # List for color axis.  First populated with NWs
            z = []
            # Frequencies (stored separately in ncobs from parameters)
            f = []

            # Initialize each parameter name as key with empty list
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] = []

            # Fill up p, f, and z for all detectors in the given networks.
            for i in range(len(nw_chk_value)):
                if int(nw_chk_value[i]) in array_indices:
                    for j in range(len(ncobs.pnames)):
                        p[ncobs.pnames[j]].extend(ncobs.ncs[int(nw_chk_value[i])].p[ncobs.pnames[j]])
                    f.extend(ncobs.ncs[int(nw_chk_value[i])].f)

                    z.extend(np.ones(len(ncobs.ncs[int(nw_chk_value[i])].f))*int(nw_chk_value[i]))
            return p, z, f

        # Check set parameters, freq, and color arrays to requested axis
        def get_array_values(p, f, z, drp_val):
            data = []
            if drp_val != 'NW':
                if drp_val != 'f':
                    data = np.array(p[drp_val])
                else:
                    data = np.array(f)
            else:
                data = np.array(z)

            return data

        # Limit parameter, color, and freq to range from a histogram
        def limit_data(p, z, f, var, hn_CD):
            try:
                tmp = np.array(p[var])
                ai = np.where((tmp >= hn_CD['range']['x'][0]) & (tmp <= hn_CD['range']['x'][1]))[0]
            except:
                ai = list(range(len(p[var])))

            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] = np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            z = np.array(z)[ai]

            return p, z, f, ai

        # Get maxes and mins for an array or set to nrows, ncols
        def get_maxmin(axis, drp_value):
            if (str(drp_value) == 'x') or (str(drp_value) == 'y'):
                axismax = max(ncobs.nrows, ncobs.ncols)
                axismin = 0
            else:
                axismax = np.max(axis)
                axismin = np.min(axis)
            return axismin, axismax

        # TEMP
        bmap_figure = {
            'data': [{
                'z': np.zeros([ncobs.nrows, ncobs.ncols]),
                'type': 'heatmap',
                'mode': 'markers+text',
                'colorbar': {'title': 'N/A'},

            }],
            'layout': {}
            }

        xslice_figure = {
            'data': [{
                'z': np.zeros(ncobs.ncols),
                'type': 'scatter',
                'mode': 'markers+text',
                'colorbar': {'title': 'N/A'},
            }],
            'layout': {}
            }

        yslice_figure = {
            'data': [{
                'z': np.zeros(ncobs.nrows),
                'type': 'scatter',
                'mode': 'markers+text',
                'colorbar': {'title': 'N/A'},
            }],
            'layout': {}
            }

        # TEMP

        # Container is the tab container for each array
        # Column for table and beammap/slice plots
        table_col = container.child(dbc.Col, width=3)

        # Column for beammap & slice tabs
        bmp_tab_col = table_col.child(dbc.Row).child(dbc.Col).child(dbc.Tabs, className='mt-3')

        # Containers for each tab for the beammap, yslice, and xslice plots
        bmp_graph = bmp_tab_col.child(dbc.Tab,label='beammap').child(dcc.Graph, figure = bmap_figure)
        yslice_graph = bmp_tab_col.child(dbc.Tab,label='y slice').child(dcc.Graph, figure = yslice_figure)
        xslice_graph = bmp_tab_col.child(dbc.Tab,label='x slice').child(dcc.Graph, figure = xslice_figure)

        # Column containing the parameter, value table
        table_col_col = table_col.child(dbc.Row).child(dbc.Col, className='mx-2').child(dbc.Table, df, striped=True, bordered=True, hover=True)

        # Column container for the array plot and drop down menus
        p_container_section_col = container.child(dbc.Col, className='mt-3 border-left')

        # Drop down menu
        drp_container_col = p_container_section_col.child(dbc.Row, className='mt-1 mb-2').child(dbc.Col, className='d-flex')
        p_container_col = p_container_section_col.child(dbc.Row).child(dbc.Col)

        # Set up an Input Grup to control the axes of the array plot
        drp_container_group = drp_container_col.child(dbc.InputGroup, size='sm', className='mr-2 w-auto')
        drp_container_group.child(dbc.InputGroupAddon("X-Axis", addon_type="prepend"))

        # Add options to the Input gropu
        drp_x = drp_container_group.child(dbc.Select, options=drp_container_options, value='x')
        drp_y = drp_container_group.child(dbc.Select, options=drp_container_options, value='y')
        drp_clr = drp_container_group.child(dbc.Select, options=drp_container_options, value='NW')

        # Graph for array plot
        pl = p_container_col.child(dcc.Graph, figure={})

        # Row for histograms in new column next to array plot
        hist_tab_row = container.child(dbc.Col, width=4).child(dbc.Row)

        # Histogram figures (will combine into one)
        hist_graph_0 = hist_tab_row.child(dbc.Col, width=6, className='px-0').child(dcc.Graph,align="center")
        hist_graph_1 = hist_tab_row.child(dbc.Col, width=6, className='px-0').child(dcc.Graph,align="center")
        hist_graph_2 = hist_tab_row.child(dbc.Col, width=6, className='px-0').child(dcc.Graph,align="center")
        hist_graph_3 = hist_tab_row.child(dbc.Col, width=6, className='px-0').child(dcc.Graph,align="center")
        hist_graph_4 = hist_tab_row.child(dbc.Col, width=6, className='px-0').child(dcc.Graph,align="center")
        hist_graph_5 = hist_tab_row.child(dbc.Col, width=6, className='px-0').child(dcc.Graph,align="center")

        # Callback for array plot
        # Takes as input the dropdowns (for axes), the network checklist
        # (for which arrays/NWs) and each histogram (for limiting data)
        # Returns figure dict for pl
        @app.callback(
            Output(pl.id, component_property='figure'),
            [Input(drp_x.id, 'value'),
             Input(drp_y.id, 'value'),
             Input(drp_clr.id, 'value'),
             Input(nw_checklist.id, 'value'),
             Input(hist_graph_0.id, 'selectedData'),
             Input(hist_graph_1.id, 'selectedData'),
             Input(hist_graph_2.id, 'selectedData'),
             Input(hist_graph_3.id, 'selectedData'),
             Input(hist_graph_4.id, 'selectedData'),
             Input(hist_graph_5.id, 'selectedData')])
        def update_array(drp_x_value, drp_y_value, drp_clr_value, nw_chk_value,
                         h0_CD, h1_CD, h2_CD, h3_CD, h4_CD, h5_CD):

            # Fill up p, z, and f based on the selected arrays/NWs
            p, z, f = from_ncobs(nw_chk_value, array_indices)

            # Check if x, y, and z are NW, freq, or parameter
            x = get_array_values(p, f, z, drp_x_value)
            y = get_array_values(p, f, z, drp_y_value)
            z = get_array_values(p, f, z, drp_clr_value)

            # Don't update if any of the arrays are empty.  Prevents errors
            if (x == []) or (y == []) or (z == []):
                return no_update

            # Limit the shown data based on histograms
            for hn_CD in [h0_CD, h1_CD, h2_CD, h3_CD, h4_CD, h5_CD]:
                for pname in ncobs.pnames:
                    p, z, f, ai = limit_data(p, z, f, pname, hn_CD)

            # Get max/mins of the x and y axes to keep axes from changing alot
            xmin, xmax = get_maxmin(x, drp_x_value)
            ymin, ymax = get_maxmin(y, drp_y_value)

            # Define figure object
            array_figure = {
                'data': [{
                    'x': x,
                    'y': y,
                    'type': 'scatter',
                    'mode': 'markers+text',
                    'marker': {
                        'color': z,
                        'showscale': True
                    }
                }],
                'layout': {
                    'clickmode': 'event+select',
                    'autosize': False,
                    'automargin': False,
                    'xaxis': {'range': [xmin, xmax]},
                    'yaxis': {'range': [ymin, ymax]}
                    }
                }

            # Return figure object
            return array_figure

        fig_types = ['bmap', 'yslice', 'xslice']
        bslice_containers = [bmp_graph, yslice_graph, xslice_graph]
        h_containers = [hist_graph_0, hist_graph_1, hist_graph_2, hist_graph_3,
                        hist_graph_4, hist_graph_5]

        def update_bslice_plot(pl_SD, nw_chk_value,
                                   h0_CD, h1_CD, h2_CD, h3_CD, h4_CD, h5_CD, fig_type=None):
            try:
                pointNumber = pl_SD['points'][0]['pointNumber']
                print(pointNumber)
            except:
                return no_update

            # Fill up p, z, and f based on the selected arrays/NWs
            p, z, f = from_ncobs(nw_chk_value, array_indices)

            dets = []
            nws = []

            for i in range(len(nw_chk_value)):
                ci = int(nw_chk_value[i])
                if ci in array_indices:
                    nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                    dets.extend(range(ncobs.ncs[ci].ndets))

            # Limit the shown data based on histograms
            for hn_CD in [h0_CD, h1_CD, h2_CD, h3_CD, h4_CD, h5_CD]:
                for pname in ncobs.pnames:
                    p, z, f, ai = limit_data(p, z, f, pname, hn_CD)
                    dets = np.array(dets)[ai]
                    nws = np.array(nws)[ai]

            # Get the NW and detector number of the selected detector
            nw = nws[pointNumber]
            det = dets[pointNumber]

            # Get the map for the corresponding detector that was clicked
            z = ncobs.ncs[np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]].x_onoff[:,:,det]

            nwi = np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]

            print(nw, det)

            # To find which row and column to slice on, we find the x,y
            # centroid and round to the nearest element in the matrix.
            row = int(np.round(ncobs.ncs[nwi].p['x'][det]))
            col = int(np.round(ncobs.ncs[nwi].p['y'][det]))

            # Define figure object
            bslice_figure = {
                'data': [{
                    'x': 0,
                    'y': 0,
                    'z': z,
                    'type': 'heatmap',
                    'mode': 'markers+text',
                    'colorscale': 'Viridis',
                    'colorbar': {'title': 'S/N'},
                    'marker': {
                        'color': 'k',
                        'showscale': True
                    }
                }],
                'layout': {
                    'autosize': False,
                    'automargin': False,
                    #'xaxis': {'range': [xmin, xmax]},
                    #'yaxis': {'range': [ymin, ymax]}
                    }
                }

            if fig_type == 'bmap':
                bslice_figure['data'][0]['x'] = list(range(ncobs.nrows))
                bslice_figure['data'][0]['y'] = list(range(ncobs.ncols))
                bslice_figure['data'][0]['z'] = z
                bslice_figure['data'][0]['type'] = 'heatmap'
            elif fig_type == 'yslice':
                bslice_figure['data'][0]['x'] = list(range(ncobs.nrows))
                bslice_figure['data'][0]['y'] = z[:, row]
                bslice_figure['data'][0]['type'] = 'line'
                bslice_figure['data'][0]['mode'] = 'lines+text'
            elif fig_type == 'xslice':
                bslice_figure['data'][0]['x'] = list(range(ncobs.ncols))
                bslice_figure['data'][0]['y'] = z[col, :]
                bslice_figure['data'][0]['type'] = 'line'
                bslice_figure['data'][0]['mode'] = 'lines+text'

            return bslice_figure

        for fig_type_i in range(len(fig_types)):
            app.callback(
                Output(bslice_containers[fig_type_i].id, 'figure'),
                [Input(pl.id, 'selectedData'),
                 Input(nw_checklist.id, 'value'),
                 Input(h_containers[0].id, 'selectedData'),
                 Input(h_containers[1].id, 'selectedData'),
                 Input(h_containers[2].id, 'selectedData'),
                 Input(h_containers[3].id, 'selectedData'),
                 Input(h_containers[4].id, 'selectedData'),
                 Input(h_containers[5].id, 'selectedData')])(
                     functools.partial(update_bslice_plot,fig_type=fig_types[fig_type_i]))



    def setup_layout(self, app):

        array_names = ['a1100', 'a1400', 'a2000']
        array_indices = {}
        array_indices[array_names[0]] = [0, 1, 2, 3, 4, 5, 6]
        array_indices[array_names[1]] = [7, 8, 9, 10]
        array_indices[array_names[2]] = [11, 12]

        drp_container_options = [

            {'label': 'S/N', 'value': 'amps'},
            {'label': 'x', 'value': 'x'},
            {'label': 'y', 'value': 'y'},
            {'label': 'fwhmx', 'value': 'fwhmx'},
            {'label': 'fwhmy', 'value': 'fwhmy'},
            {'label': 'Freq', 'value': 'f'},
            {'label': 'NW', 'value': 'NW'}

        ]

        body = self.child(dbc.Row).child(dbc.Col)

        title = 'Wyatt Beammap 2020.05.06'
        title_row = body.child(dbc.Row)
        title_row.children = [html.H1(f'{title}')]

        plot_networks_checklist_section = body.child(dbc.Row).child(dbc.Col)
        plot_networks_checklist_section.child(dbc.Label, 'Select network(s) to show in the plots:')
        plot_networks_checklist_container = plot_networks_checklist_section.child(dbc.Row, className='mx-0')
        checklist_presets_container = plot_networks_checklist_container.child(dbc.Col, width=4)
        checklist_presets = checklist_presets_container.child(dbc.Checklist, persistence=True, labelClassName='pr-1', inline=True)
        checklist_presets.options = [
                {'label': 'All', 'value': 'all'},
                {'label': '1.1mm', 'value': 'a1100'},
                {'label': '1.4mm', 'value': 'a1400'},
                {'label': '2.0mm', 'value': 'a2000'},
                ]

        checklist_presets.value = []
        checklist_networks_container = plot_networks_checklist_container.child(dbc.Col)
        # make three button groups
        nw_checklist = checklist_networks_container.child(dbc.Checklist, persistence=True, labelClassName='pr-1', inline=True)

        checklist_networks_options = [
                {'label': 'N0', 'value': '0'},
                {'label': 'N1', 'value': '1'},
                {'label': 'N2', 'value': '2'},
                {'label': 'N3', 'value': '3'},
                {'label': 'N4', 'value': '4'},
                {'label': 'N5', 'value': '5'},
                {'label': 'N6', 'value': '6'},
                {'label': 'N7', 'value': '7'},
                {'label': 'N8', 'value': '8'},
                {'label': 'N9', 'value': '9'},
                {'label': 'N10', 'value': '-1'},
                {'label': 'N11', 'value': '10'},
                {'label': 'N12', 'value': '11'},
            ]

        preset_networks_map = dict()
        preset_networks_map['a1100'] = set(o['value'] for o in checklist_networks_options[0:7])
        preset_networks_map['a1400'] = set(o['value'] for o in checklist_networks_options[7:11])
        preset_networks_map['a2000'] = set(o['value'] for o in checklist_networks_options[11:13])
        preset_networks_map['all'] = functools.reduce(set.union, (preset_networks_map[k] for k in array_names))

        @app.callback(
                [
                    Output(nw_checklist.id, "options"),
                    Output(nw_checklist.id, "value"),
                    ],
                [
                    Input(checklist_presets.id, "value"),
                ]
            )
        def on_preset_change(preset_values):
            nw_values = set()
            for pv in preset_values:
                nw_values = nw_values.union(preset_networks_map[pv])
            # print(f'select preset {preset_values} network values {nw_values}')
            options = [o for o in checklist_networks_options if o['value'] in nw_values]
            return options, list(nw_values)

        # Hardcoded rows,cols, sampling frequency for now
        nrows = 21
        ncols = 25
        sf = 488.281/4
        # Hardcoded path to files
        #path = '/Users/mmccrackan/toltec/data/tests/wyatt/'
        path = '/home/mmccrackan/wyatt/'

        # Hardcoded obsnum (directory name containing nc files)
        #obsnum = 'coadd_20200506'
        obsnum = 'coadd'

        # Load all of the nc files into the ncobs object.  May break if
        # there are multiple nc files for each network.
        ncobs = get_ncobs(obsnum, nrows, ncols, path, sf, order='C', transpose = False)

        # Frequencies are acquired separately due to a potential bug in the
        # kids reduce code
        #f = np.load('/Users/mmccrackan/toltec/data/tests/wyatt/10886/10886_f_tone.npy',allow_pickle=True).item()
        f = np.load(file_tone_path,allow_pickle=True).item()
        for i in range(len(ncobs.nws)):
            try:
                ncobs.ncs[i].f = f[int(ncobs.nws[i])]
            except:
                print('cannot get frequencies for nws ' + str(ncobs.nws[i]))

        # Container that has a dbc.Tabs for all the arrays
        array_tab_col = body.child(dbc.Row).child(dbc.Col).child(dbc.Tabs, className='mt-3')

        # Loop through the different arrays and make the plots and callbacks
        for array_name in list(array_indices.keys()):
            tab_row = array_tab_col.child(dbc.Tab,label=f'{array_name}').child(dbc.Row)
            self._setup_tab(app, ncobs, array_indices[array_name], tab_row, drp_container_options, nw_checklist)



extensions = [
    {
        'module': 'dasha.web.extensions.dasha',
        'config': {
            'template': beammap,
            'title_text': ' Beammap Dasha Page',
            }
        },
    ]
