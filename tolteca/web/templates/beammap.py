#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import glob
from dasha.web.templates import ComponentTemplate
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Output, Input, State, MATCH, ALL
from dash import no_update, exceptions

import dash_table

from tolteca.web.templates.beammap_sources.wyatt_classes import obs, ncdata
from pathlib import Path
from dasha.web.extensions.cache import cache
import functools

import pandas as pd
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.io.misc.yaml import load as yaml_load

#Temp
import sys

# Uses the wyatt_classes.py file for the class to hold the data.
#sys.path.append('/Users/mmccrackan/dasha/dasha/examples/beammap_sources_2/')
#from wyatt_classes import obs, ncdata

f_tone_filepath = Path(__file__).parent.joinpath(
        'beammap_sources/10886_f_tone.npy')

df = pd.DataFrame(
    {
        "Parameter": ["S/N", "X Centroid [cm]", "Y Centroid [cm]",
                      "X FWHM [cm]", "Y FWHM [cm]", "Frequency [MHz]",
                      "Network", "Detector Number on Network"],
        "Value": ["N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]
    }
)
#dimensions of array plots
a_width=700
a_height=700

#dimensions of beammap plots
b_width=325
b_height=325

# Define model
@custom_model
def gaussian(x, amp=100., mu=10., sigma=3.):
    return (amp * np.exp(-0.5 * ((x - mu) / sigma)**2))

@cache.memoize(timeout=60 * 60 * 60)
def get_ncobs(*args, **kwargs):
    return obs(*args, **kwargs)

# Get parameter values, freq, and NW array from ncobs
def from_ncobs(nw_chk_value, array_indices, ncobs):
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
            if ncobs.ncs[int(nw_chk_value[i])] != -1:
                for j in range(len(ncobs.pnames)):
                    p[ncobs.pnames[j]].extend(ncobs.ncs[int(nw_chk_value[i])].p[ncobs.pnames[j]])
                f.extend(ncobs.ncs[int(nw_chk_value[i])].f)
    
                z.extend(np.ones(len(ncobs.ncs[int(nw_chk_value[i])].f))*int(nw_chk_value[i]))
    return p, z, f

# Check set parameters, freq, and color arrays to requested axis
def get_array_values(p, f, z, drp_val):
    data = []
    if drp_val != 'NW':
        if drp_val != 'freq':
            data = np.array(p[drp_val])
        else:
            data = np.array(f)
    else:
        data = np.array(z)

    return data

# Limit parameter, color, and freq to range from a histogram
def limit_data(p, z, f, var, hn_CD, ncobs):
    try:
        if var != 'freq':
            tmp = np.array(p[var])
        else:
            tmp = np.array(f)
        ai = np.where((tmp >= hn_CD['range']['x'][0]) & (tmp <= hn_CD['range']['x'][1]))[0]
    except:
        if var != 'freq':
            ai = list(range(len(p[var])))
        else:
            ai = list(range(len(f)))

    for i in range(len(ncobs.pnames)):
        p[ncobs.pnames[i]] = np.array(p[ncobs.pnames[i]])[ai]
    f = np.array(f)[ai]
    z = np.array(z)[ai]

    return p, z, f, ai

def get_limit_data_from_snr(p):
    # Fit model to data
    m_init = gaussian()
    fit = LevMarLSQFitter()

    h, be = np.histogram(p['amps'],100)
    x = be[1:]
    y = h
    m = fit(m_init, x, y)

    limit = m.mu.value + 2*m.sigma.value
    return limit

# Get maxes and mins for an array or set to nrows, ncols
def get_maxmin(axis, drp_value, ncobs):
    if (str(drp_value) == 'x') or (str(drp_value) == 'y'):
        axismax = 1.25*max(ncobs.nrows*2.5, ncobs.ncols*2.5)
        axismax = 1.25*25*2.5 #temp
        axismin = 0
    else:
        axismax = np.max(axis)
        axismin = np.min(axis)
    return axismin, axismax

class beammap(ComponentTemplate):
    _component_cls = dbc.Container
    fluid = True

    def _setup_tab(self, app, ncobs_map, container, drp_container_options, nw_checklist, drp_obsnum, array_indx=None):

        ''' Set up all the content'''
        # Container is the tab container for each array
        # Column for table and beammap/slice plots
        table_col = container.child(dbc.Col, width=3)

        # Column for beammap & slice tabs
        bmp_tab_col = table_col.child(dbc.Row, width=3).child(dbc.Col).child(dcc.Tabs, className='custom-tabs-container')#className='mt-3')

        # Containers for each tab for the beammap, yslice, and xslice plots
        bmp_graph = bmp_tab_col.child(dbc.Tab,label='beammap',className='custom-tabs').child(dcc.Graph,width=3, figure = {})
        yslice_graph = bmp_tab_col.child(dbc.Tab,label='y slice',className='custom-tabs').child(dcc.Graph, figure = {})
        xslice_graph = bmp_tab_col.child(dbc.Tab,label='x slice',className='custom-tabs').child(dcc.Graph, figure = {})

        # Column containing the parameter, value table
        tbl = dbc.Table.from_dataframe(df)
        table_col_col = table_col.child(dbc.Row).child(dbc.Col, className='mx-2').child(dbc.Table, tbl, striped=True, bordered=True, hover=True)

        # Column container for the array plot and drop down menus
        p_container_section_col = container.child(dbc.Col, className='mt-3 border-left')

        # Drop down menu
        drp_container_col = p_container_section_col.child(dbc.Row, className='mt-1 mb-2').child(dbc.Col, className='d-flex')
        p_container_col = p_container_section_col.child(dbc.Row, width=3).child(dbc.Col)

        # Set up an Input Grup to control the axes of the array plot
        drp_container_group = drp_container_col.child(dbc.InputGroup, size='sm', className='mr-2 w-auto')
        drp_container_group.child(dbc.InputGroupAddon("X-Axis", addon_type="prepend"))

        # Add options to the Input gropu
        drp_x = drp_container_group.child(dbc.Select, options=drp_container_options, value='x')

        drp_container_group = drp_container_col.child(dbc.InputGroup, size='sm', className='mr-2 w-auto')
        drp_container_group.child(dbc.InputGroupAddon("Y-Axis", addon_type="prepend"))
        drp_y = drp_container_group.child(dbc.Select, options=drp_container_options, value='y')

        drp_container_group = drp_container_col.child(dbc.InputGroup, size='sm', className='mr-2 w-auto')
        drp_container_group.child(dbc.InputGroupAddon("Color", addon_type="prepend"))

        drp_clr = drp_container_group.child(dbc.Select, options=drp_container_options, value='NW')

        # Graph for array plot
        pl = p_container_col.child(dcc.Graph, figure={})

        # Row for histograms in new column next to array plot
        hist_tab_row = container.child(dbc.Col).child(dbc.Row, width=3,className='mt-5 w-auto')

        # Histogram figures (will combine into one)
        hist_graph_0 = hist_tab_row.child(dbc.Col, width=6, className='px-0').child(dcc.Graph,align="center")
        hist_graph_1 = hist_tab_row.child(dbc.Col, width=6, className='px-0').child(dcc.Graph,align="center")
        hist_graph_2 = hist_tab_row.child(dbc.Col, width=6, className='px-0').child(dcc.Graph,align="center")
        hist_graph_3 = hist_tab_row.child(dbc.Col, width=6, className='px-0').child(dcc.Graph,align="center")
        hist_graph_4 = hist_tab_row.child(dbc.Col, width=6, className='px-0').child(dcc.Graph,align="center")
        hist_graph_5 = hist_tab_row.child(dbc.Col, width=6, className='px-0').child(dcc.Graph,align="center")

        fig_types = ['bmap', 'yslice', 'xslice']
        bslice_containers = [bmp_graph, yslice_graph, xslice_graph]
        h_containers = [hist_graph_0, hist_graph_1, hist_graph_2, hist_graph_3,
                        hist_graph_4, hist_graph_5]

        params = ['x','y','fwhmx','fwhmy','amps','snr']
        params.append('freq')
        params.remove('snr')
        label_names = ['X [cm]', 'Y [cm]', 'X FWHM [cm]', 'Y FWHM [cm]', 'S/N', 'Freq [Mhz]']

        labels = {}
        for param in range(len(params)):
            labels[params[param]] = label_names[param]


        def update_array(drp_x_value, drp_y_value, drp_clr_value, nw_chk_value,
                         h0_CD, h1_CD, h2_CD, h3_CD, h4_CD, h5_CD, obsnum, array_i=None):


            if obsnum != None:
                ncobs = ncobs_map[obsnum]
            else:
                return no_update

            # Fill up p, z, and f based on the selected arrays/NWs
            p, z, f = from_ncobs(nw_chk_value, array_i, ncobs)

            hn_CDs = [h0_CD, h1_CD, h2_CD, h3_CD, h4_CD, h5_CD]

            # Limit the shown data based on histograms
            for indx in range(len(h_containers)):
                p, z, f, ai = limit_data(p, z, f, params[indx],
                                         hn_CDs[indx], ncobs)

            # Check if x, y, and z are NW, freq, or parameter
            x = get_array_values(p, f, z, drp_x_value)
            y = get_array_values(p, f, z, drp_y_value)
            z = get_array_values(p, f, z, drp_clr_value)

            # Don't update if any of the arrays are empty.  Prevents errors
            if (x == []) or (y == []) or (z == []):
                return no_update

            # Get max/mins of the x and y axes to keep axes from changing alot
            xmin, xmax = get_maxmin(x, drp_x_value, ncobs)
            ymin, ymax = get_maxmin(y, drp_y_value, ncobs)
            
            if drp_clr_value == 'amps':
                if z.all() !=None:
                    limit = get_limit_data_from_snr(p)
                    if limit > 0:
                        z[z > limit] = limit
                else:
                    z = z

            if drp_x_value != 'NW':
                x_clr = labels[drp_x_value]
            else:
                x_clr = 'NW'

            if drp_y_value != 'NW':
                y_clr = labels[drp_y_value]
            else:
                y_clr = 'NW'

            if drp_clr_value != 'NW':
                z_clr = labels[drp_clr_value]
            else:
                z_clr = 'NW'

            if ncobs.obsnum == 'coaddBest':
                if array_i == [0,1,2,3,4,5,6]:
                    title = 'coaddJuly20 (unscaled)'
                elif array_i == [7,8,9,10]:
                    title = 'coaddAug0620 (unscaled)'
                else:
                    title = 'coaddAug0720 (scaled by 1.25)'
            else:
                title = ''


            # Define figure dict
            array_figure = {
                'data': [{
                    'x': x,
                    'y': y,
                    'type': 'scatter',
                    'mode': 'markers',
                    'marker': {
                        'color': z,
                        'line': {'color': 'k',
                                 'width': 1},
                        'showscale': True,
                        'colorbar': {'title': z_clr}
                    },
                    'selected': {
                        'marker': {'color': '#6ffc03',
                                   'symbol': 'X'},
                    }
                }],
                'layout': {
                    #'width': a_width,
                    #'height': a_height,
                    'clickmode': 'event+select',
                    'autosize': True,
                    'automargin': False,
                    'editable': True,
                    'animate': True,
                    'uirevision': True,
                    'title': title,
                    'xaxis': {'title': x_clr, 'range': [xmin, xmax]},
                    'yaxis': {'title': y_clr, 'range': [ymin, ymax], 'scaleanchor': "x", 'scaleratio': xmax/ymax}
                    }
                }
            # Return figure object
            return array_figure

        # Callback for array plot
        # Takes as input the dropdowns (for axes), the network checklist
        # (for which arrays/NWs) and each histogram (for limiting data)
        # Returns figure dict for pl
        app.callback(
            Output(pl.id, 'figure'),
            [Input(drp_x.id, 'value'),
             Input(drp_y.id, 'value'),
             Input(drp_clr.id, 'value'),
             Input(nw_checklist.id, 'value'),
             Input(hist_graph_0.id, 'selectedData'),
             Input(hist_graph_1.id, 'selectedData'),
             Input(hist_graph_2.id, 'selectedData'),
             Input(hist_graph_3.id, 'selectedData'),
             Input(hist_graph_4.id, 'selectedData'),
             Input(hist_graph_5.id, 'selectedData'),
             Input(drp_obsnum.id, 'value')])(
                 functools.partial(update_array, array_i=array_indx))

        def update_bslice_plot(pl_SD, nw_chk_value, h0_CD, h1_CD, h2_CD,
                               h3_CD, h4_CD, h5_CD, obsnum, fig_type=None,
                               array_i=None):

            try:
                pointNumber = pl_SD['points'][0]['pointNumber']
            except:
                return {}

            hn_CDs = [h0_CD, h1_CD, h2_CD, h3_CD, h4_CD, h5_CD]

            if obsnum != None:
                ncobs = ncobs_map[obsnum]
            else:
                return no_update

            # Fill up p, z, and f based on the selected arrays/NWs
            p, z, f = from_ncobs(nw_chk_value, array_i, ncobs)

            dets = []
            nws = []

            for i in range(len(nw_chk_value)):
                ci = int(nw_chk_value[i])
                if ci in array_i:
                    if ncobs.ncs[ci] != -1:
                        nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                        dets.extend(range(ncobs.ncs[ci].ndets))

            # Limit the shown data based on histograms
            #for hn_CD in [h0_CD, h1_CD, h2_CD, h3_CD, h4_CD, h5_CD]:
                #for pname in ncobs.pnames:
            for indx in range(len(h_containers)):
                    p, z, f, ai = limit_data(p, z, f, params[indx],
                                             hn_CDs[indx], ncobs)
                    dets = np.array(dets)[ai]
                    nws = np.array(nws)[ai]

            # Get the NW and detector number of the selected detector
            nw = nws[pointNumber]
            det = dets[pointNumber]

            # Get the map for the corresponding detector that was clicked
            z = ncobs.ncs[np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]].x_onoff[:,:,det]

            nwi = np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]

            # To find which row and column to slice on, we find the x,y
            # centroid and round to the nearest element in the matrix.
            #row = int(np.round(ncobs.ncs[nwi].p['x'][det]))
            #col = int(np.round(ncobs.ncs[nwi].p['y'][det]))

            row, col = np.where(z == np.max(z))
            row = row[0]
            col = col[0]

            # Define figure dict
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
                    'width': b_width,
                    'height': b_height,
                    'autosize': True,
                    'automargin': False,
                    'xaxis': {'title': 'x (pixels)'},
                    'yaxis': {'title': 'y (pixels)'}
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
                bslice_figure['data'][0]['y'] = z[row,:]
                bslice_figure['data'][0]['type'] = 'line'
                bslice_figure['data'][0]['mode'] = 'lines+text'
                bslice_figure['layout']['xaxis'] = {'title': 'x (pixels)'}
                bslice_figure['layout']['yaxis'] = {'title': 'S/N'}
            elif fig_type == 'xslice':
                bslice_figure['data'][0]['x'] = list(range(ncobs.ncols))
                bslice_figure['data'][0]['y'] = z[:,col]
                bslice_figure['data'][0]['type'] = 'line'
                bslice_figure['data'][0]['mode'] = 'lines+text'
                bslice_figure['layout']['xaxis'] = {'title': 'y (pixels)'}
                bslice_figure['layout']['yaxis'] = {'title': 'S/N'}

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
                 Input(h_containers[5].id, 'selectedData'),
                 Input(drp_obsnum.id, 'value')])(
                     functools.partial(update_bslice_plot, fig_type=fig_types[fig_type_i], array_i=array_indx))


        def update_table(pl_SD, nw_chk_value, h0_CD, h1_CD, h2_CD,
                         h3_CD, h4_CD, h5_CD, obsnum, array_i=None):
            try:
                pointNumber = pl_SD['points'][0]['pointNumber']
            except:
                return no_update

            if obsnum != None:
                ncobs = ncobs_map[obsnum]
            else:
                return no_update

            # Fill up p, z, and f based on the selected arrays/NWs
            p, z, f = from_ncobs(nw_chk_value, array_i, ncobs)

            dets = []
            nws = []
            hn_CDs = [h0_CD, h1_CD, h2_CD, h3_CD, h4_CD, h5_CD]

            for i in range(len(nw_chk_value)):
                ci = int(nw_chk_value[i])
                if ci in array_i:
                    if ncobs.ncs[ci] != -1:
                        nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                        dets.extend(range(ncobs.ncs[ci].ndets))

            # Limit the shown data based on histograms
            for indx in range(len(h_containers)):
                    p, z, f, ai = limit_data(p, z, f, params[indx],
                                             hn_CDs[indx], ncobs)
                    dets = np.array(dets)[ai]
                    nws = np.array(nws)[ai]

            # Get the NW and detector number of the selected detector
            nw = nws[pointNumber]
            det = dets[pointNumber]

            nwi = np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]

            df2 = pd.DataFrame(
           {
               "Parameter": ["S/N", "X Centroid [cm]", "Y Centroid [cm]",
                      "X FWHM [cm]", "Y FWHM [cm]", "Frequency [MHz]",
                      "Network", "Detector Number on Network"],
               "Value": ["N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]
            })

            df2['Value'][0] = '%.2f' % (ncobs.ncs[nwi].p['amps'][det])
            df2['Value'][1] = '%.2f' % (ncobs.ncs[nwi].p['x'][det])
            df2['Value'][2] = '%.2f' % (ncobs.ncs[nwi].p['y'][det])
            df2['Value'][3] = '%.2f' % (ncobs.ncs[nwi].p['fwhmx'][det])
            df2['Value'][4] = '%.2f' % (ncobs.ncs[nwi].p['fwhmy'][det])
            df2['Value'][5] = '%.2f' % (ncobs.ncs[nwi].f[det])
            df2['Value'][6] = int(nw)
            df2['Value'][7] = int(det)

            return dbc.Table.from_dataframe(df2)


        app.callback(
            Output(table_col_col.id, 'children'),
            [Input(pl.id, 'selectedData'),
             Input(nw_checklist.id, 'value'),
             Input(hist_graph_0.id, 'selectedData'),
             Input(hist_graph_1.id, 'selectedData'),
             Input(hist_graph_2.id, 'selectedData'),
             Input(hist_graph_3.id, 'selectedData'),
             Input(hist_graph_4.id, 'selectedData'),
             Input(hist_graph_5.id, 'selectedData'),
             Input(drp_obsnum.id, 'value')])(
                 functools.partial(update_table, array_i=array_indx))

        def update_hist(nw_chk_value, obsnum, param=None, array_i=None, label=None):

            if obsnum != None:
                ncobs = ncobs_map[obsnum]
            else:
                return no_update

            h = []
            for j in range(len(nw_chk_value)):
                if int(nw_chk_value[j]) in array_i:
                    if param in ncobs.pnames:
                        if ncobs.ncs[int(nw_chk_value[j])] !=-1:
                            h.extend(ncobs.ncs[int(nw_chk_value[j])].p[param])
                    elif param == 'freq':
                        if ncobs.ncs[int(nw_chk_value[j])] !=-1:
                            h.extend(ncobs.ncs[int(nw_chk_value[j])].f)

            hist_fig = {
                'data': [{
                'x': h,
                #'text': 0,
                'textposition': 'top',
                'customdata': 0,
                'type': 'histogram',
                'mode': 'markers+text',
                'marker': { 'color': '#D4AC0D', #set color equal to a variable
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],

                'layout': {
                'xaxis': {'title': label},
                'yaxis': {'title': 'N'},
                #'width': h_width,
                #'height': h_height,
                #'margin': {'l': 0, 'r': 0, 'b': 15, 't': 5},
                'dragmode': 'select',
                'hovermode': 'closest',
                'editable': True,
                'animate': True,
                'clickmode': 'event+select'
                    }
                }

            return hist_fig

        for indx in range(len(h_containers)):
            app.callback(
            Output(h_containers[indx].id, 'figure'),
            [Input(nw_checklist.id, 'value'),
             Input(drp_obsnum.id, 'value')])(
                 functools.partial(update_hist, param=params[indx],
                                   array_i=array_indx, label=labels[params[indx]]))

    def setup_layout(self, app):
        # Hardcoded rows,cols, sampling frequency for now
        nrows = 25
        ncols = 25
        sf = 488.281/4

        indexfiles = glob.glob('/home/mmccrackan/wyatt/indexfiles/*y*ml')
        #indexfiles.append('/Users/mmccrackan/toltec/config/wyatt/index.yaml')

        ncobs_map = {}
        for indx in range(len(indexfiles)):
            with open(indexfiles[indx], 'r') as fo:
                index = yaml_load(fo)
            obsnum = index['ObsNum']
            ncobs_map[indexfiles[indx]] = get_ncobs(obsnum=obsnum, nrows=nrows,
                                                    ncols=ncols, index=index,
                                                    sampfreq=sf, order='C',
                                                    transpose=False,scale=True)

            # Frequencies are acquired separately due to a potential bug in the
            # kids reduce code
            #f = np.load('/Users/mmccrackan/toltec/data/tests/wyatt/10886/10886_f_tone.npy',allow_pickle=True).item()
            '''f = np.load(f_tone_filepath, allow_pickle=True).item()
            for i in range(len(ncobs_map[indexfiles[indx]].nws)):
                try:
                    ncobs_map[indexfiles[indx]].ncs[i].f = f[int(ncobs_map[indexfiles[indx]].nws[i])]
                except:
                    print('cannot get frequencies for nw ' + str(ncobs_map[indexfiles[indx]].nws[i]))
            '''

        array_names = ['1.1 mm Array', '1.4 mm Array', '2.0 mm Array']
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
            {'label': 'Freq', 'value': 'freq'},
            {'label': 'NW', 'value': 'NW'}
        ]

        body = self.child(dbc.Row).child(dbc.Col)

        title = 'Wyatt Beammap Explorer'
        title_row = body.child(dbc.Row)
        title_row.children = [html.H1(f'{title}')]

        # Set up an Input Grup to select obsnum to plot
        drp_obsnum_container_group = body.child(dbc.Row, width=4).child(dbc.InputGroup, size='sm', className='mb-3 w-auto mt-3 ml-3')
        drp_obsnum_container_group.child(dbc.InputGroupAddon("ObsNums", addon_type="prepend"))

        drp_obsnum_options = []

        for indx in range(len(indexfiles)):
            with open(indexfiles[indx], 'r') as fo:
                index = yaml_load(fo)
            drp_obsnum_options.append({'label': index['ObsNum'], 'value': indexfiles[indx]})

        # Add options to the Input group
        drp_obsnum = drp_obsnum_container_group.child(dbc.Select, options=drp_obsnum_options)

        plot_networks_checklist_section = body.child(dbc.Row).child(dbc.Col)
        plot_networks_checklist_section.child(dbc.Label, 'Select network(s) to show in the plots:')
        plot_networks_checklist_container = plot_networks_checklist_section.child(dbc.Row, className='mx-0')
        checklist_presets_container = plot_networks_checklist_container.child(dbc.Col, width=4)
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
        nw_checklist = checklist_networks_container.child(dbc.Checklist, persistence=False, labelClassName='pr-1', inline=True)

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
                {'label': 'N10', 'value': '10'},
                {'label': 'N11', 'value': '11'},
                {'label': 'N12', 'value': '12'},
            ]

        preset_networks_map = dict()
        preset_networks_map['1.1 mm Array'] = set(o['value'] for o in checklist_networks_options[0:7])
        preset_networks_map['1.4 mm Array'] = set(o['value'] for o in checklist_networks_options[7:10])
        preset_networks_map['2.0 mm Array'] = set(o['value'] for o in checklist_networks_options[10:13])
        preset_networks_map['all'] = functools.reduce(set.union, (preset_networks_map[k] for k in array_names))

        def on_preset_change(preset_values):
           nw_values = set()
           for pv in preset_values:
               nw_values = nw_values.union(preset_networks_map[pv])
           # print(f'select preset {preset_values} network values {nw_values}')
           options = [o for o in checklist_networks_options if o['value'] in nw_values]
           return options, list(nw_values)

        app.callback(
                [
                    Output(nw_checklist.id, "options"),
                    Output(nw_checklist.id, "value"),
                    ],
                [
                    Input(checklist_presets.id, "value"),
                ]
            )(functools.partial(on_preset_change))

        # Container that has a dcc.Tabs for all the arrays
        array_tab_col = body.child(dbc.Row).child(dbc.Col).child(dcc.Tabs,className='custom-tabs-container')#className='mt-3')

        # Loop through the different arrays and make the plots and callbacks
        for array_name in list(array_indices.keys()):
            tab_row = array_tab_col.child(dbc.Tab,className='custom-tab',label=f'{array_name}').child(dbc.Row)
            self._setup_tab(app, ncobs_map, tab_row, drp_container_options, nw_checklist, drp_obsnum, array_indx=array_indices[array_name])

        link_row = body.child(dbc.Col)
        link_row.children = html.Div([html.A("Beammap Dasha Page Tutorial",
                                             href='https://umass.box.com/s/32f7i07ki9yq36o2yip253t692w49tln', target="_blank")], style={'textAlign': 'center'})

extensions = [
    {
        'module': 'dasha.web.extensions.dasha',
        'config': {
            'template': beammap,
            'title_text': ' Beammap Dasha Page',
            }
        },
    ]
