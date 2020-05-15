#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:52:00 2020

@author: mmccrackan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 20:09:18 2020

@author: mmccrackan
"""

from dasha.web.templates import ComponentTemplate
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
import dash_table

import base64
import datetime
import io

import numpy as np
import netCDF4
import glob
import json
import re
import sys

sys.path.insert(0, "/Users/mmccrackan/toltec/python/wyatt/")
from wyatt_classes import obs, ncdata

class beammap(ComponentTemplate):
    _component_cls = dbc.Container
    fluid = True

    def setup_layout(self, app):
        body = self.child(dbc.Row).child(dbc.Col)
        table_header = [html.Thead(html.Tr([html.Th("Parameter"), html.Th("Value")]))]

        #dimensions of array plots
        a_width=600
        a_height=600
        
        #dimensions of beammap plots
        b_width=400
        b_height=400
        
        #dimensions of histogram plots
        h_width = 500 
        h_height = 400

        button_container = body.child(dbc.Row)
        file_container = button_container.child(dbc.Col).child(html.Div, className='d-flex')
        
        file_container.child(dbc.Label("Files Found:",className='mr-2'))
        files = file_container.child(html.Div, 'N/A')
        
        path_input = button_container.child(dbc.Col).child(dcc.Input, placeholder="Enter File Path: ",
            type='string',
            value='/Users/mmccrackan/toltec/data/tests/wyatt/coadd_20200506/')
        
        
        nw_checklist = button_container.child(dbc.Col).child(dcc.Checklist,
        options=[
            {'label': 'network 0', 'value': '0'},
            {'label': 'network 1', 'value': '1'},
            {'label': 'network 2', 'value': '4'},
            {'label': 'network 3', 'value': '5'},
            {'label': 'network 4', 'value': '6'},
            {'label': 'network 5', 'value': '7'},
            {'label': 'network 6', 'value': '8'},
            {'label': 'network 7', 'value': '9'},
            {'label': 'network 8', 'value': '10'},
            {'label': 'network 9', 'value': '11'},
            {'label': 'network 10', 'value': '2'},
            {'label': 'network 11', 'value': '3'},

        ],
            value=[],
            labelStyle={'display': 'inline-block'}
        )  
                
        
        array_tab_container = body.child(dbc.Row).child(dcc.Tabs,vertical=True)
        a1100_container = array_tab_container.child(dcc.Tab,label='1.1mm').child(dbc.Row)
        
        p1100 = a1100_container.child(dbc.Col).child(dcc.Graph)
        drp_1100=a1100_container.child(dbc.Col).child(dcc.Dropdown, options=[
            {'label': 'S/N', 'value': 'amps'},
            {'label': 'x', 'value': 'x'},
            {'label': 'y', 'value': 'y'},
            {'label': 'fwhmx', 'value': 'fwhmx'},
            {'label': 'fwhmy', 'value': 'fwhmy'}
        ],
        value=['x','y'],
        multi=True
        )
        
        a1100_b_tab = a1100_container.child(dcc.Tabs)
        
        p1100_b = a1100_b_tab.child(dcc.Tab,label='beammap').child(dbc.Col).child(dcc.Graph)
        p1100_b2 = a1100_b_tab.child(dcc.Tab,label='y-slice').child(dbc.Col).child(dcc.Graph)
        p1100_b3 = a1100_b_tab.child(dcc.Tab,label='x-slice').child(dbc.Col).child(dcc.Graph)
        
        t1100 = a1100_container.child(dbc.Table, bordered=True,
dark=False, hover=True, responsive=True,striped=True,width=100)

        
        a1100_h_tab = a1100_container.child(dbc.Row)
        p1100_h0 = a1100_h_tab.child(dbc.Row).child(dbc.Col).child(dcc.Graph,align="center")
        p1100_h1 = a1100_h_tab.child(dbc.Col).child(dcc.Graph,align="center")
        p1100_h2 = a1100_h_tab.child(dbc.Col).child(dcc.Graph,align="center")
        p1100_h3 = a1100_h_tab.child(dbc.Col).child(dcc.Graph,align="center")
        p1100_h4 = a1100_h_tab.child(dbc.Col).child(dcc.Graph,align="center")

        nrows = 21
        ncols = 25
        sf = 488.281/4
        path = '/Users/mmccrackan/toltec/data/tests/wyatt/'
        
        obsnum = 'coadd_20200506'
        #obsnum = 10891
        ncobs = obs(obsnum,nrows,ncols,path,sf,order='C',transpose=False)
        
        f = np.load('/Users/mmccrackan/toltec/data/tests/wyatt/10886/10886_f_tone.npy',allow_pickle=True).item()
        
        for i in range(len(ncobs.nws)):
            try:
                ncobs.ncs[i].f = f[int(ncobs.nws[i])]
            except:
                print('cannot get frequencies for nws ' + str(ncobs.nws[i]))
                
        @app.callback(Output(files.id, 'children'),
              [Input(path_input.id,'value'),
              Input(nw_checklist.id,'value')])
        def update_output(path, value):
            file_list = glob.glob(path + '/*.nc')
            file_list_short = []
            beammap_files = []
            
            for i in range(len(file_list)):
                file_list_short.append(file_list[i][len(path):] + ', ')
                nw = re.findall(r'\d+', file_list[i])
                print('nw',nw)
                if nw[-1] in value:
                    beammap_files.append(file_list[i])
                            
            if file_list == []:
                return 'N/A'
            else:
                return np.sort(file_list_short)
            
        
        @app.callback(Output(nw_checklist.id, 'value'),
                      [Input(path_input.id,'value')])
        def clear_checklist(path):
            return []
        
        
        @app.callback(
            Output(p1100.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(drp_1100.id, 'value'),
             Input(nw_checklist.id, 'value')]
            ) 
        def update_a1100(obsnum,value,checklist):            
            x = []
            y = []
            z = []
            
            for i in range(len(checklist)):
                if int(checklist[i]) in [0,1,4,5,6,7,8]:
                    x.extend(ncobs.ncs[int(checklist[i])].p[value[0]])
                    y.extend(ncobs.ncs[int(checklist[i])].p[value[1]])
                    
                    if len(value) == 3:
                        z.extend(ncobs.ncs[int(checklist[i])].p[value[2]])
                    else:
                        z.extend(np.ones(len(ncobs.ncs[int(checklist[i])].p[value[0]]))*int(checklist[i]))
            
            figure={
                'data': [{
                    'x': x,
                    'y': y,
                    'textposition': 'top',
                    'customdata': 0,
                    'type': 'scatter',
                    'mode': 'markers+text',
                    'marker': {
                        'color': z,
                        'colorscale':'Viridis', # one of plotly colorscales
                        'showscale': True,
                        'size': 5 
                    },
                    'unselected': {
                        'marker': { 'opacity': 0.3 },
                        'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                    }
                }],
                'layout': {
                    'title': "1.1 mm Array",
                    'xaxis': {'title': str(value[0])},
                    'yaxis': {'title': str(value[1])},
                    'width': a_width, 
                    'height': a_height,
                    #'margin': {'l': 0, 'r': 0, 'b': 15, 't': 5},
                    'dragmode': 'select',
                    'hovermode': 'closest',
                    'editable': True,
                    'animate': True,
                    'clickmode': 'event+select'
                }}
            
            return figure
        
        
        @app.callback(
            Output(p1100_b.id, 'figure'),
            [Input(p1100.id,'selectedData'),
             Input(nw_checklist.id, 'value')])
        def update_b1100(clickData,checklist):
            #try:
                pointNumber = clickData['points'][0]['pointNumber']
                dets = []
                nws = []
                
                for i in range(len(checklist)):
                    ci = int(checklist[i])
                    if ci in [0,1,4,5,6,7,8]:
                        nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                        dets.extend(range(ncobs.ncs[ci].ndets))
                        
                nw = nws[pointNumber]
                det = dets[pointNumber]
                
                print(nw,det)
                print(np.where(np.array(ncobs.nws) == str(int(nw)))[0][0])
                
                z = ncobs.ncs[np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]].x_onoff[:,:,det]
    
                figure = {
                    'data': [{
                        'x': list(range(ncobs.nrows)),
                        'y': list(range(ncobs.ncols)),
                        'z': z,
                        'textposition': 'top',
                        'customdata': 0,
                        'type': 'heatmap',
                        'mode': 'markers+text',
                        'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
                        'colorscale':'tropic', # one of plotly colorscales
                        'unselected': {
                            'marker': { 'opacity': 0.3 },
                            'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                        }
                    }],
                    'layout': {
                        'width': b_width, 
                        'height': b_height,
                        'title': "1.1 mm Array",
                        'xaxis': {'title': 'x (pixels)'},
                        'yaxis': {'title': 'y (pixels)'},
                        'dragmode': 'select',
                        'hovermode': True,
                    }}
                
                return figure
            #except:
              #  pass
        
        
        @app.callback(
            Output(p1100_b2.id, 'figure'),
            [Input(p1100.id,'selectedData'),
             Input(nw_checklist.id, 'value')])
        def update_b1100_2(clickData,checklist):
            try:
                pointNumber = clickData['points'][0]['pointNumber']
                dets = []
                nws = []
                
                for i in range(len(checklist)):
                    ci = int(checklist[i])
                    if ci in [0,1,4,5,6,7,8]:
                        nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                        dets.extend(range(ncobs.ncs[ci].ndets))
                        
                nw = nws[pointNumber]
                det = dets[pointNumber]
                
                print(nw,det)
                nwi = np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]
                
                z = ncobs.ncs[np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]].x_onoff[:,:,det]
                
                row = int(np.round(ncobs.ncs[nwi].p['x'][det]))
                col = int(np.round(ncobs.ncs[nwi].p['y'][det]))
                
                figure = {
                    'data': [{
                         'x': list(range(ncobs.nrows)),
                         'y': z[:,row],
                        #'z': z,
                        'textposition': 'top',
                        'customdata': 0,
                        'type': 'line',
                        'mode': 'lines+text',
                        #'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
                        'colorscale':'tropic', # one of plotly colorscales
                        'unselected': {
                            'marker': { 'opacity': 0.3 },
                            'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                        }
                    }],
                    'layout': {
                        'width': b_width, 
                        'height': b_height,
                        'title': "1.1 mm Array",
                        'xaxis': {'title': 'y (pixels)'},
                        'yaxis': {'title': 'S/N'},
                        'dragmode': 'select',
                        'hovermode': True,
                    }}
                
                return figure
            except:
                pass
            
            
        @app.callback(
            Output(p1100_b3.id, 'figure'),
            [Input(p1100.id,'selectedData'),
             Input(nw_checklist.id, 'value')])
        def update_b1100_3(clickData,checklist):
            try:
                pointNumber = clickData['points'][0]['pointNumber']
                dets = []
                nws = []
                
                for i in range(len(checklist)):
                    ci = int(checklist[i])
                    if ci in [0,1,4,5,6,7,8]:
                        nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                        dets.extend(range(ncobs.ncs[ci].ndets))
                        
                nw = nws[pointNumber]
                det = dets[pointNumber]
                
                print(nw,det)
                nwi = np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]
                
                z = ncobs.ncs[np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]].x_onoff[:,:,det]
                
                row = int(np.round(ncobs.ncs[nwi].p['x'][det]))
                col = int(np.round(ncobs.ncs[nwi].p['y'][det]))
                
                figure = {
                    'data': [{
                        'x': list(range(ncobs.ncols)),
                        'y': z[col,:],
                        #'z': z,
                        'textposition': 'top',
                        'customdata': 0,
                        'type': 'line',
                        'mode': 'lines+text',
                        #'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
                        'colorscale':'tropic', # one of plotly colorscales
                        'unselected': {
                            'marker': { 'opacity': 0.3 },
                            'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                        }
                    }],
                    'layout': {
                        'width': b_width, 
                        'height': b_height,
                        'title': "1.1 mm Array",
                         'xaxis': {'title': 'x (pixels)'},
                         'yaxis': {'title': 'S/N'},
                        'dragmode': 'select',
                        'hovermode': True,
                    }}
                
                return figure
            except:
                pass
            
        @app.callback(
            Output(t1100.id, 'children'),
            [Input(p1100.id,'selectedData'),
             Input(nw_checklist.id, 'value')])
        def update_t1100(clickData,checklist):
            try:
                pointNumber = clickData['points'][0]['pointNumber']
                dets = []
                nws = []
                
                for i in range(len(checklist)):
                    ci = int(checklist[i])
                    if ci in [0,1,4,5,6,7,8]:
                        nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                        dets.extend(range(ncobs.ncs[ci].ndets))
                        
                nw = nws[pointNumber]
                det = dets[pointNumber]
                
                print(nw,det)
                nwi = np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]

                row0 = html.Tr([html.Td("S/N"), html.Td('%.3f' % (ncobs.ncs[nwi].p['amps'][det]))])
                row1 = html.Tr([html.Td("x"), html.Td('%.3f' % (ncobs.ncs[nwi].p['x'][det]))])
                row2 = html.Tr([html.Td("y"), html.Td('%.3f' % (ncobs.ncs[nwi].p['y'][det]))])
                row3 = html.Tr([html.Td("fwhm_x"), html.Td('%.3f' % (ncobs.ncs[nwi].p['fwhmx'][det]))])
                row4 = html.Tr([html.Td("fwhm_y"), html.Td('%.3f' % (ncobs.ncs[nwi].p['fwhmy'][det]))])
                row5 = html.Tr([html.Td("f"), html.Td('N/A')])
                row6 = html.Tr([html.Td("nw"), html.Td(int(nw))])
                row7 = html.Tr([html.Td("det"), html.Td(int(det))])
    
                table_body = [html.Tbody([row0,row1, row2, row3, row4,row5,row6,row7])]
                return table_header + table_body
            
            except:
                 return []
             
                
             
        @app.callback(
            Output(p1100_h0.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p1100_h0(obsnum,checklist):
            h = []            
            for i in range(len(checklist)):
                if int(checklist[i]) in [0,1,4,5,6,7,8]:
                    h.extend(ncobs.ncs[int(checklist[i])].p['amps'])
                    
            figure={
                'data': [{
                'x': h,
                #'text': 0,
                'textposition': 'top',
                'customdata': 0,
                'type': 'histogram',
                'mode': 'markers+text',
                'marker': { 'color': '#330C73', #set color equal to a variable
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                'title': "1.1 mm Array",
                'xaxis': {'title': 'S/N'},
                'yaxis': {'title': 'N'},
                'width': h_width, 
                'height': h_height,
                #'margin': {'l': 0, 'r': 0, 'b': 15, 't': 5},
                'dragmode': 'select',
                'hovermode': 'closest',
                'editable': True,
                'animate': True,
                'clickmode': 'event+select'
                # Display a rectangle to highlight the previously selected region
                }}
            
            return figure
        
        
        @app.callback(
            Output(p1100_h1.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p1100_h0(obsnum,checklist):
            h = []            
            for i in range(len(checklist)):
                if int(checklist[i]) in [0,1,4,5,6,7,8]:
                    h.extend(ncobs.ncs[int(checklist[i])].p['x'])
                    
            figure={
                'data': [{
                'x': h,
                #'text': 0,
                'textposition': 'top',
                'customdata': 0,
                'type': 'histogram',
                'mode': 'markers+text',
                'marker': { 'color': '#330C73', #set color equal to a variable
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                'title': "1.1 mm Array",
                'xaxis': {'title': 'x'},
                'yaxis': {'title': 'N'},
                'width': h_width, 
                'height': h_height,
                #'margin': {'l': 0, 'r': 0, 'b': 15, 't': 5},
                'dragmode': 'select',
                'hovermode': 'closest',
                'editable': True,
                'animate': True,
                'clickmode': 'event+select'
                # Display a rectangle to highlight the previously selected region
                }}
            
            return figure
        
        
        @app.callback(
            Output(p1100_h2.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p1100_h0(obsnum,checklist):
            h = []            
            for i in range(len(checklist)):
                if int(checklist[i]) in [0,1,4,5,6,7,8]:
                    h.extend(ncobs.ncs[int(checklist[i])].p['y'])
                    
            figure={
                'data': [{
                'x': h,
                #'text': 0,
                'textposition': 'top',
                'customdata': 0,
                'type': 'histogram',
                'mode': 'markers+text',
                'marker': { 'color': '#330C73', #set color equal to a variable
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                'title': "1.1 mm Array",
                'xaxis': {'title': 'y'},
                'yaxis': {'title': 'N'},
                'width': h_width, 
                'height': h_height,
                #'margin': {'l': 0, 'r': 0, 'b': 15, 't': 5},
                'dragmode': 'select',
                'hovermode': 'closest',
                'editable': True,
                'animate': True,
                'clickmode': 'event+select'
                # Display a rectangle to highlight the previously selected region
                }}
            
            return figure
        
        
        @app.callback(
            Output(p1100_h3.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p1100_h0(obsnum,checklist):
            h = []            
            for i in range(len(checklist)):
                if int(checklist[i]) in [0,1,4,5,6,7,8]:
                    h.extend(ncobs.ncs[int(checklist[i])].p['fwhmx'])
                    
            figure={
                'data': [{
                'x': h,
                #'text': 0,
                'textposition': 'top',
                'customdata': 0,
                'type': 'histogram',
                'mode': 'markers+text',
                'marker': { 'color': '#330C73', #set color equal to a variable
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                'title': "1.1 mm Array",
                'xaxis': {'title': 'fwhm_x'},
                'yaxis': {'title': 'N'},
                'width': h_width, 
                'height': h_height,
                #'margin': {'l': 0, 'r': 0, 'b': 15, 't': 5},
                'dragmode': 'select',
                'hovermode': 'closest',
                'editable': True,
                'animate': True,
                'clickmode': 'event+select'
                # Display a rectangle to highlight the previously selected region
                }}
            
            return figure
        
        
        @app.callback(
            Output(p1100_h4.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p1100_h0(obsnum,checklist):
            h = []            
            for i in range(len(checklist)):
                if int(checklist[i]) in [0,1,4,5,6,7,8]:
                    h.extend(ncobs.ncs[int(checklist[i])].p['fwhmy'])
                    
            figure={
                'data': [{
                'x': h,
                #'text': 0,
                'textposition': 'top',
                'customdata': 0,
                'type': 'histogram',
                'mode': 'markers+text',
                'marker': { 'color': '#330C73', #set color equal to a variable
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                'title': "1.1 mm Array",
                'xaxis': {'title': 'fwhm_y'},
                'yaxis': {'title': 'N'},
                'width': h_width, 
                'height': h_height,
                #'margin': {'l': 0, 'r': 0, 'b': 15, 't': 5},
                'dragmode': 'select',
                'hovermode': 'closest',
                'editable': True,
                'animate': True,
                'clickmode': 'event+select'
                # Display a rectangle to highlight the previously selected region
                }}
            
            return figure
        
        
        
        
              
extensions = [
    {
        'module': 'dasha.web.extensions.dasha',
        'config': {
            'template': beammap,
            'title_text': ' Beammap Dasha Page',
            }
        },
    ]
