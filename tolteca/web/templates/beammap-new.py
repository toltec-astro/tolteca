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

#Uses the wyatt_classes.py file for the class to hold the data.
sys.path.insert(0, "/Users/mmccrackan/toltec/python/wyatt/")
from wyatt_classes import obs, ncdata

class beammap(ComponentTemplate):
    _component_cls = dbc.Container
    fluid = True

    def setup_layout(self, app):
        body = self.child(dbc.Row).child(dbc.Col)
        #Header for the 3 tables for selected detectors.
        table_header = [html.Thead(html.Tr([html.Th("Parameter"), html.Th("Value")]))]
        row0 = html.Tr([html.Td("S/N"), html.Td('N/A')])
        row1 = html.Tr([html.Td("X Centroid [px]"), html.Td('N/A')])
        row2 = html.Tr([html.Td("Y Centroid [px]"), html.Td('N/A')])
        row3 = html.Tr([html.Td("X FWHM [px]"), html.Td('N/A')])
        row4 = html.Tr([html.Td("Y FWHM [px]"), html.Td('N/A')])
        row5 = html.Tr([html.Td("Frequency [MHz]"), html.Td('N/A')])
        row6 = html.Tr([html.Td("Network"), html.Td('N/A')])
        row7 = html.Tr([html.Td("Detector Number on Network"), html.Td('N/A')])

        table_body = [html.Tbody([row0,row1, row2, row3, row4,row5,row6,row7])]
        
        #Hex Colors for the networks in the array plot
        colors = ['#4A148C','#311B92','#1A237E','#01579B','#006064','#004D40',
                  '#1B5E20','#33691E','#827717','#F57F17','#FF6F00','#E65100',
                  '#3E2723']

        #dimensions of array plots
        a_width=750
        a_height=750
        
        #dimensions of beammap plots
        b_width=400
        b_height=400
        
        #dimensions of histogram plots
        h_width = 500 
        h_height = 400       
        
        #hardcoded rows,cols, sampling frequency for now
        nrows = 21
        ncols = 25
        sf = 488.281/4
        #hardcoded path to files
        path = '/Users/mmccrackan/toltec/data/tests/wyatt/'
        
        #hardcoded obsnum (directory name containing nc files)
        obsnum = 'coadd_20200506'
        
        #Load all of the nc files into the ncobs object.  May break if
        #there are multiple nc files for each network.
        ncobs = obs(obsnum,nrows,ncols,path,sf,order='C',transpose=False)
        
        #Frequencies are acquired separately due to a potential bug in the
        #kids reduce code
        f = np.load('/Users/mmccrackan/toltec/data/tests/wyatt/10886/10886_f_tone.npy',allow_pickle=True).item()
        
        for i in range(len(ncobs.nws)):
            try:
                ncobs.ncs[i].f = f[int(ncobs.nws[i])]
            except:
                print('cannot get frequencies for nws ' + str(ncobs.nws[i]))
        
        title = 'Wyatt Beammap 2020.05.06'
        title_container = body.child(dbc.Row)
        title_container.children = [html.H1(f'{title}')]
        
        checklist_container = body.child(dbc.Row)
        checklist_label = checklist_container.child(dbc.Col,style={'margin': 25}).child(dbc.Label,"Select Networks to Plot:")
        
        #body.child(dbc.Row).child(dbc.Label("Select Networks: ", className='mr-2'),style={'width':'100%', 'margin':25, 'textAlign': 'center'})        
        
        #Creates the checklist for the different networks.  Values correspond
        #to the ordering in the ncobs.nws list.
        nw_checklist = checklist_container.child(dbc.Col,style={'margin': 25}).child(dcc.Checklist,
        options=[
            {'label': 'N0', 'value': '0'},
            {'label': 'N1', 'value': '1'},
            {'label': 'N2', 'value': '4'},
            {'label': 'N3', 'value': '5'},
            {'label': 'N4', 'value': '6'},
            {'label': 'N5', 'value': '7'},
            {'label': 'N6', 'value': '8'},
            {'label': 'N7', 'value': '9'},
            {'label': 'N8', 'value': '10'},
            {'label': 'N9', 'value': '11'},
            {'label': 'N10', 'value': '-1'},
            {'label': 'N11', 'value': '2'},
            {'label': 'N12', 'value': '3'},
        ],
            value=[],
            labelStyle={'display': 'inline-block'},
            #style={'width':'100%', 'margin':0, 'textAlign': 'center'},
        )
        
        #This container is for the array plot.  Note it uses dcc.Tabs to create
        #a overall tab.
        array_tab_container = body.child(dbc.Row).child(dcc.Tabs,vertical=False)
        
        #Container for the various options at the top of the page
        button_container = body.child(dbc.Row)
        file_container = button_container.child(dbc.Col).child(html.Div, className='d-flex')
        
        file_container.child(dbc.Label("Files Found:",className='mr-2'))
        files = file_container.child(html.Div, 'N/A')
        
        path_input = button_container.child(dbc.Col).child(dcc.Input, placeholder="Enter File Path: ",
            type='string',
            value='/Users/mmccrackan/toltec/data/tests/wyatt/coadd_20200506/')
               
        #Make a tab for the 1.1 mm plots
        a1100_container = array_tab_container.child(dcc.Tab,label='1.1mm').child(dbc.Row)
        
        #A container for the 1.1 mm table
        t1100_container = a1100_container.child(dbc.Col,style={'margin':0})#.child(dbc.Jumbotron, [html.H1(children="Selected Detector")],style={"width": "25%"})
        
        #Make a tab for the 1.1 mm beammap and slice plots
        #a1100_b_tab = a1100_container.child(dbc.Col,style={'width':'40%', 'margin':0, 'textAlign': 'center'}).child(dcc.Tabs,vertical=True)
        
        a1100_b_tab = t1100_container.child(dbc.Tabs,style={'margin':0, 'textAlign': 'center'})
        
        #Tabs and graphs for the 1.1 mm beammap and slice plots
        #p1100_b = a1100_b_tab.child(dcc.Tab,label='beammap').child(dbc.Col).child(dcc.Graph)
        #p1100_b2 = a1100_b_tab.child(dcc.Tab,label='y-slice').child(dbc.Col).child(dcc.Graph)
        #p1100_b3 = a1100_b_tab.child(dcc.Tab,label='x-slice').child(dbc.Col).child(dcc.Graph)
        
        p1100_b = a1100_b_tab.child(dbc.Tab,label='beammap').child(dcc.Graph,figure = {
                'data': [{
                    #'x': list(range(ncobs.nrows)),
                    #'y': list(range(ncobs.ncols)),
                    'z': np.zeros([ncobs.nrows,ncobs.ncols]),
                    'textposition': 'top',
                    'customdata': 0,
                    'type': 'scatter',
                    'mode': 'markers+text',
                    'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
                    'colorscale':'Gray', # one of plotly colorscales
                    'colorbar': {'title': 'N/A'},
                    'unselected': {
                        'marker': { 'opacity': 0.3 },
                        'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                    }
                }],
                'layout': {
                    'width': b_width, 
                    'height': b_height,
                    #'title': "1.1 mm Array",
                    'xaxis': {'title': 'x (pixels)'},
                    'yaxis': {'title': 'y (pixels)'},
                    'dragmode': 'select',
                    'hovermode': True,
                }})
        p1100_b2 = a1100_b_tab.child(dbc.Tab,label='y slice').child(dcc.Graph, figure={'layout': {
                    'width': b_width, 
                    'height': b_height,
                    'xaxis': {'title': 'y (pixels)'},
                    'yaxis': {'title': 'S/N'},
                    'dragmode': 'select',
                    'hovermode': True,
                }})
        p1100_b3 = a1100_b_tab.child(dbc.Tab, label='x slice').child(dcc.Graph, figure={'layout': {
                    'width': b_width, 
                    'height': b_height,
                    #'title': "2.0 mm Array",
                    'xaxis': {'title': 'x (pixels)'},
                    'yaxis': {'title': 'S/N'},
                    'dragmode': 'select',
                    'hovermode': True,
                }})
        
        
        t1100 = t1100_container.child(dbc.Table,bordered=True, striped=True, hover=True,responsive=True,style={'width': '100%'})

        t1100.children = table_header + table_body

        #This creates a graph for the 1.1 mm array plot
        #p1100 = a1100_container.child(dbc.Col,style={'width':'50%', 'margin':0, 'textAlign': 'center'}).child(dcc.Graph)
        p1100_container = a1100_container.child(dbc.Col,style={'width': '100%', 'margin':0, 'textAlign': 'center'}).child(dbc.Card).child(dbc.CardBody,style={"marginRight": 75,"marginTop": 0, "marginBottom": 0,'width': '100%'}).child(dbc.CardHeader,html.H5("1.1mm Array"))
        

        #Inside of the 1.1 mm container, make a dropdown to control the
        #plotted axes for the array plot.  Separate for each array.   Defaults
        #to x, y
        drp_container = p1100_container.child(dbc.Row)
        drp_1100=drp_container.child(dbc.Col,style={'width':'10%', 'margin':0, 'textAlign': 'center'}).child(dcc.Dropdown, options=[
            {'label': 'S/N', 'value': 'amps'},
            {'label': 'x', 'value': 'x'},
            {'label': 'y', 'value': 'y'},
            {'label': 'fwhmx', 'value': 'fwhmx'},
            {'label': 'fwhmy', 'value': 'fwhmy'},
            {'label': 'Freq', 'value': 'f'}

        ],
        value='x',
        multi=False)
        
        drp_1100_2=drp_container.child(dbc.Col,style={'width':'10%', 'margin':0, 'textAlign': 'center'}).child(dcc.Dropdown, options=[
            {'label': 'S/N', 'value': 'amps'},
            {'label': 'x', 'value': 'x'},
            {'label': 'y', 'value': 'y'},
            {'label': 'fwhmx', 'value': 'fwhmx'},
            {'label': 'fwhmy', 'value': 'fwhmy'},
            {'label': 'Freq', 'value': 'f'}

        ],
        value='y',
        multi=False)
        
        drp_1100_3=drp_container.child(dbc.Col,style={'width':'10%', 'margin':0, 'textAlign': 'center'}).child(dcc.Dropdown, options=[
            {'label': 'NW', 'value': 'NW'},
            {'label': 'S/N', 'value': 'amps'},
            {'label': 'x', 'value': 'x'},
            {'label': 'y', 'value': 'y'},
            {'label': 'fwhmx', 'value': 'fwhmx'},
            {'label': 'fwhmy', 'value': 'fwhmy'},
            {'label': 'Freq', 'value': 'f'}
        ],
        value='NW',
        multi=False)
        
        p1100 = p1100_container.child(dcc.Graph, figure={
                'layout': {
                    #'title': "1.1 mm Array",
                    'autosize': False,
                    'automargin': False,
                    'xaxis': {'range': [0,max(ncobs.nrows,ncobs.ncols)]},
                    'yaxis': {'range': [0,max(ncobs.nrows,ncobs.ncols)]},
                    'width': a_width, 
                    'height': a_height,
                    'dragmode': 'select',
                    'hovermode': 'closest',
                    'editable': True,
                    'animate': True,
                    'clickmode': 'event+select',
                    'plot_bgcolor': "#F9F9F9",
                    'paper_bgcolor': "#F9F9F9",
                }})
        
        #Create a tab for the 1.1 mm histograms        
        a1100_h_tab = a1100_container.child(dbc.Row)
        
        #Set up each histogram for the 1.1 mm array.  Each is a new column.
        p1100_h0 = a1100_container.child(dbc.Row).child(dcc.Graph,align="center")
        p1100_h1 = a1100_container.child(dbc.Row).child(dcc.Graph,align="center")
        p1100_h2 = a1100_container.child(dbc.Row).child(dcc.Graph,align="center")
        p1100_h3 = a1100_container.child(dbc.Row).child(dcc.Graph,align="center")
        p1100_h4 = a1100_container.child(dbc.Row).child(dcc.Graph,align="center")
        p1100_h5 = a1100_container.child(dbc.Row).child(dcc.Graph,align="center")

        
        #Call back for displaying the list of files.  Searches the path for
        #nc files and returns them without the path.
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
                if nw[-1] in value:
                    beammap_files.append(file_list[i])
                            
            if file_list == []:
                return 'N/A'
            elif len(file_list) <= 12:
                return np.sort(file_list_short)
            else:
                return np.sort(file_list_short[:12],'...') 
        
        #Callback to clear the network checklist when the path is changed.
        #Prevents it from immediately plotting selected networks when the a
        #new set of files is selected.
        @app.callback(Output(nw_checklist.id, 'value'),
                      [Input(path_input.id,'value')])
        def clear_checklist(path):
            return []
        
        #This creates the array plot for the 1.1 mm array.  It takes the
        #dropdown and checklist as input.  From the checklist, it creates
        #lists of all selected parameters from the dropdown using only the
        #networks selected.  This then updates 'data' in the figure.
        @app.callback(
            Output(p1100.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(drp_1100.id, 'value'),
             Input(drp_1100_2.id, 'value'),
             Input(drp_1100_3.id, 'value'),
             Input(nw_checklist.id, 'value'),
             Input(p1100_h0.id,'selectedData'),
             Input(p1100_h1.id,'selectedData'),
             Input(p1100_h2.id,'selectedData'),
             Input(p1100_h3.id,'selectedData'),
             Input(p1100_h4.id,'selectedData'),
             Input(p1100_h5.id,'selectedData'),]) 
        def update_a1100(obsnum,value,value_2,value_3,checklist,clickData_h0,clickData_h1,clickData_h2,
                         clickData_h3,clickData_h4,clickData_h5):
            #Lists for the x,y, and color
            x = []
            y = []
            z = []
            
            p = {}
            f = []
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] = []
    
            #Fill up x,y, and z for all detectors in the given networks.
            for i in range(len(checklist)):
                if int(checklist[i]) in [0,1,4,5,6,7,8]:
                    for j in range(len(ncobs.pnames)):
                        p[ncobs.pnames[j]].extend(ncobs.ncs[int(checklist[i])].p[ncobs.pnames[j]])
                    f.extend(ncobs.ncs[int(checklist[i])].f)

                    #x.extend(ncobs.ncs[int(checklist[i])].p[value[0]])
                    #y.extend(ncobs.ncs[int(checklist[i])].p[value[1]])
                    
                    #if len(value)==2:
                    if value_3 == 'NW':
                        if value!='f':
                            z.extend(np.ones(len(ncobs.ncs[int(checklist[i])].p[value]))*int(checklist[i]))  
                        else:
                            z.extend(np.ones(len(ncobs.ncs[int(checklist[i])].f))*int(checklist[i]))
                         
                        ''' 
                           if value!='f':
                             char_arr = np.chararray(len(ncobs.ncs[int(checklist[i])].p[value]),unicode=True,itemsize=7)
                             char_arr[:] = colors[int(checklist[i])]
                        else:
                            char_arr = np.chararray(len(ncobs.ncs[int(checklist[i])].f),unicode=True,itemsize=7)
                            char_arr[:] = colors[int(checklist[i])]
                        z.extend(char_arr)
                        print(z)
                        '''
                        
            #if len(value) == 3:
            if value_3 !='NW':
                if value_3 !='f':
                    z = np.array(p[value_3])
                else:
                    z = np.array(f)
            #Each block from the try to the z = statement take the limits
            #from a histogram selection and limit the parameters to only
            #those points that fall into that selection.
            try:
                print('clickData',clickData_h0['range']['x'])
                amp_temp = np.array(p['amps'])
                ai = np.where((amp_temp>=clickData_h0['range']['x'][0]) & (amp_temp<=clickData_h0['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['amps'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            
            z = np.array(z)[ai]
            
            try:
                print('clickData',clickData_h1['range']['x'])
                amp_temp = np.array(p['x'])
                ai = np.where((amp_temp>=clickData_h1['range']['x'][0]) & (amp_temp<=clickData_h1['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['x'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            
            z = np.array(z)[ai]
            
            
            try:
                print('clickData',clickData_h2['range']['x'])
                amp_temp = np.array(p['y'])
                ai = np.where((amp_temp>=clickData_h2['range']['x'][0]) & (amp_temp<=clickData_h2['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['y'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            
            z = np.array(z)[ai]
            
        
            try:
                print('clickData',clickData_h3['range']['x'])
                amp_temp = np.array(p['fwhmx'])
                ai = np.where((amp_temp>=clickData_h3['range']['x'][0]) & (amp_temp<=clickData_h3['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmx'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            
            z = np.array(z)[ai]
            
            try:
                print('clickData',clickData_h4['range']['x'])
                amp_temp = np.array(p['fwhmy'])
                ai = np.where((amp_temp>=clickData_h4['range']['x'][0]) & (amp_temp<=clickData_h4['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmy'])))
                
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            
            z = np.array(z)[ai]
            
            
            try:
                print('clickData',clickData_h5['range']['x'])
                amp_temp = np.array(f)
                ai = np.where((amp_temp>=clickData_h5['range']['x'][0]) & (amp_temp<=clickData_h5['range']['x'][1]))[0]
            except:
                ai = list(range(len(f)))
                
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            
            z = np.array(z)[ai]
            
            if value!='f':
                x = np.array(p[value])
            else:
                x = np.array(f)
            if value_2!='f':
                y = np.array(p[value_2])
            else:
                y = np.array(f)
            
            if (str(value) == 'x') or (str(value) == 'y'):
                xmax = max(ncobs.nrows,ncobs.ncols)
                xmin = 0
            else:
                xmax = np.max(x)
                xmin = np.min(x)
                
            if (str(value_2) == 'x') or (str(value_2) == 'y'):
                ymax = max(ncobs.nrows,ncobs.ncols)
                ymin = 0
            else:
                ymax = np.max(y)
                ymin = np.min(y)
            
            if value=='amps':
                label='S/N'  
            elif value=='x':
                label='X Centroid [px]'
            elif value=='y':
                label='Y Centroid [px]'
            elif value=='fwhmx':
                label='X FWHM [px]'
            elif value=='fwhmy':
                label='Y FWHM [px]'
            elif value=='f':
                label='Freq [MHz]'
                
            if value_2=='amps':
                label_2='S/N'  
            elif value_2=='x':
                label_2='X Centroid [px]'
            elif value_2=='y':
                label_2='Y Centroid [px]'
            elif value_2=='fwhmx':
                label_2='X FWHM [px]'
            elif value_2=='fwhmy':
                label_2='Y FWHM [px]'
            elif value_2=='f':
                label_2='Freq [MHz]'
            
            if value_3=='amps':
                label_3='S/N'  
            elif value_3=='x':
                label_3='X Centroid [px]'
            elif value_3=='y':
                label_3='Y Centroid [px]'
            elif value_3=='fwhmx':
                label_3='X FWHM [px]'
            elif value_3=='fwhmy':
                label_3='Y FWHM [px]'
            elif value_3=='f':
                label_3='Freq [MHz]'
            elif value_3=='NW':
                label_3 = 'NW'
            
            #return dict with all plotting options
            figure={
                'data': [{
                    'x': x,
                    'y': y,
                    'textposition': 'top',
                    'customdata': 0,
                    'type': 'scatter',
                    'mode': 'markers',
                    'marker': {
                        'color': z,
                        #'colorscale':'Viridis', # one of plotly colorscales
                        'showscale': True,
                        'colorbar': {'title': label_3},

                        #'size': 5 
                    },
                    'unselected': {
                        'marker': { 'opacity': 1.0 },
                        'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                    },
                    'selected': {
                        'marker': {'color': 'red'},
                        'textfont': { 'color': 'rgba(1,0,0,0)' }
                    }
                }],
                'layout': {
                    #'title': "1.1 mm Array",
                    'autosize': False,
                    'automargin': False,
                    'xaxis': {'title': label,'range': [xmin,xmax]},
                    'yaxis': {'title': label_2,'range': [ymin,ymax]},
                    'width': a_width, 
                    'height': a_height,
                    'dragmode': 'select',
                    'hovermode': 'closest',
                    'editable': True,
                    'animate': True,
                    'clickmode': 'event+select',
                    'plot_bgcolor': "#F9F9F9",
                    'paper_bgcolor': "#F9F9F9",
                }}
            
            return figure
        
        
        #Creates the beammap plot for the 1.1 mm array.  It takes the
        #clickData from the 1.1 mm array plot as input along with the checklist
        #It then finds the corresponding detector by assembling network and
        #detector lists and plots that beammap.  The beammap matrix is assembled
        #at the start when ncobs is created.
        @app.callback(
            Output(p1100_b.id, 'figure'),
            [Input(p1100.id,'selectedData'),
             Input(nw_checklist.id, 'value'),
             Input(p1100_h0.id,'selectedData'),
             Input(p1100_h1.id,'selectedData'),
             Input(p1100_h2.id,'selectedData'),
             Input(p1100_h3.id,'selectedData'),
             Input(p1100_h4.id,'selectedData'),
             Input(p1100_h5.id,'selectedData'),])
        def update_b1100(clickData,checklist,clickData_h0,clickData_h1,clickData_h2,
                         clickData_h3,clickData_h4,clickData_h5):
            #try:
                #pointNumber is the number in a list of all plotted points of
                #the nearest data point to where was clicked
            try:
                pointNumber = clickData['points'][0]['pointNumber']
                print(pointNumber)
            except:
                pass
                
            p = {}
            f = []
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] = []
            
            #Fill up x,y, and z for all detectors in the given networks.
            for i in range(len(checklist)):
                if int(checklist[i]) in [0,1,4,5,6,7,8]:
                    for j in range(len(ncobs.pnames)):
                        p[ncobs.pnames[j]].extend(ncobs.ncs[int(checklist[i])].p[ncobs.pnames[j]])
                    f.extend(ncobs.ncs[int(checklist[i])].f)
                
                #Make lists of the networks and detector numbers so we can
                #find out what network pointNumber is in and what detector
                #number in that network it corresponds to.
                dets = []
                nws = []
                
                for i in range(len(checklist)):
                    ci = int(checklist[i])
                    if ci in [0,1,4,5,6,7,8]:
                        nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                        dets.extend(range(ncobs.ncs[ci].ndets))
             
                        
            print('nws',len(nws))

            try:
                print('clickData',clickData_h0['range']['x'])
                amp_temp = np.array(p['amps'])
                ai = np.where((amp_temp>=clickData_h0['range']['x'][0]) & (amp_temp<=clickData_h0['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['amps'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            try:
                print('clickData',clickData_h1['range']['x'])
                amp_temp = np.array(p['x'])
                ai = np.where((amp_temp>=clickData_h1['range']['x'][0]) & (amp_temp<=clickData_h1['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['x'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h2['range']['x'])
                amp_temp = np.array(p['y'])
                ai = np.where((amp_temp>=clickData_h2['range']['x'][0]) & (amp_temp<=clickData_h2['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['y'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

        
            try:
                print('clickData',clickData_h3['range']['x'])
                amp_temp = np.array(p['fwhmx'])
                ai = np.where((amp_temp>=clickData_h3['range']['x'][0]) & (amp_temp<=clickData_h3['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmx'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h4['range']['x'])
                amp_temp = np.array(p['fwhmy'])
                ai = np.where((amp_temp>=clickData_h4['range']['x'][0]) & (amp_temp<=clickData_h4['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmy'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            
            try:
                print('clickData',clickData_h5['range']['x'])
                amp_temp = np.array(f)
                ai = np.where((amp_temp>=clickData_h5['range']['x'][0]) & (amp_temp<=clickData_h5['range']['x'][1]))[0]
            except:
                ai = list(range(len(f)))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            nw = nws[pointNumber]
            det = dets[pointNumber]
                
            #Get the map for the corresponding detector that was clicked                
            z = ncobs.ncs[np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]].x_onoff[:,:,det]

            #return the figure dict
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
                    'colorscale':'Viridis', # one of plotly colorscales
                    'colorbar': {'title': 'S/N'},
                    'unselected': {
                        'marker': { 'opacity': 0.3 },
                        'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                    }
                }],
                'layout': {
                    'width': b_width, 
                    'height': b_height,
                    #'title': "1.1 mm Array",
                    'xaxis': {'title': 'x (pixels)'},
                    'yaxis': {'title': 'y (pixels)'},
                    'dragmode': 'select',
                    'hovermode': True,
                }}
            
            return figure
        
        
        #Creates the y slice plot for the 1.1 mm array.  It takes the
        #clickData from the 1.1 mm array plot as input along with the checklist
        #It then finds the corresponding detector by assembling network and
        #detector lists and plots that y slice.  The beammap matrix is assembled
        #at the start when ncobs is created.
        @app.callback(
            Output(p1100_b2.id, 'figure'),
            [Input(p1100.id,'selectedData'),
             Input(nw_checklist.id, 'value'),
             Input(p1100_h0.id,'selectedData'),
             Input(p1100_h1.id,'selectedData'),
             Input(p1100_h2.id,'selectedData'),
             Input(p1100_h3.id,'selectedData'),
             Input(p1100_h4.id,'selectedData'),
             Input(p1100_h5.id,'selectedData')])
        def update_b1100_2(clickData,checklist,clickData_h0,clickData_h1,clickData_h2,
                         clickData_h3,clickData_h4,clickData_h5):
                 #try:
                #pointNumber is the number in a list of all plotted points of
                #the nearest data point to where was clicked
            try:
                pointNumber = clickData['points'][0]['pointNumber']
                print(pointNumber)
            except:
                pass
                
            p = {}
            f = []
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] = []
            
            #Fill up x,y, and z for all detectors in the given networks.
            for i in range(len(checklist)):
                if int(checklist[i]) in [0,1,4,5,6,7,8]:
                    for j in range(len(ncobs.pnames)):
                        p[ncobs.pnames[j]].extend(ncobs.ncs[int(checklist[i])].p[ncobs.pnames[j]])
                    f.extend(ncobs.ncs[int(checklist[i])].f)
                
                #Make lists of the networks and detector numbers so we can
                #find out what network pointNumber is in and what detector
                #number in that network it corresponds to.
                dets = []
                nws = []
                
                for i in range(len(checklist)):
                    ci = int(checklist[i])
                    if ci in [0,1,4,5,6,7,8]:
                        nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                        dets.extend(range(ncobs.ncs[ci].ndets))
             
                        
            print('nws',len(nws))

            try:
                print('clickData',clickData_h0['range']['x'])
                amp_temp = np.array(p['amps'])
                ai = np.where((amp_temp>=clickData_h0['range']['x'][0]) & (amp_temp<=clickData_h0['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['amps'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            try:
                print('clickData',clickData_h1['range']['x'])
                amp_temp = np.array(p['x'])
                ai = np.where((amp_temp>=clickData_h1['range']['x'][0]) & (amp_temp<=clickData_h1['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['x'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h2['range']['x'])
                amp_temp = np.array(p['y'])
                ai = np.where((amp_temp>=clickData_h2['range']['x'][0]) & (amp_temp<=clickData_h2['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['y'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

        
            try:
                print('clickData',clickData_h3['range']['x'])
                amp_temp = np.array(p['fwhmx'])
                ai = np.where((amp_temp>=clickData_h3['range']['x'][0]) & (amp_temp<=clickData_h3['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmx'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h4['range']['x'])
                amp_temp = np.array(p['fwhmy'])
                ai = np.where((amp_temp>=clickData_h4['range']['x'][0]) & (amp_temp<=clickData_h4['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmy'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            
            try:
                print('clickData',clickData_h5['range']['x'])
                amp_temp = np.array(f)
                ai = np.where((amp_temp>=clickData_h5['range']['x'][0]) & (amp_temp<=clickData_h5['range']['x'][1]))[0]
            except:
                ai = list(range(len(f)))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            nw = nws[pointNumber]
            det = dets[pointNumber]
                
            #Get the map for the corresponding detector that was clicked                
            z = ncobs.ncs[np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]].x_onoff[:,:,det]
            
            nwi = np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]
            
            #To find which row and column to slice on, we find the x,y
            #centroid and round to the nearest element in the matrix.
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
                    'colorscale':'Viridis', # one of plotly colorscales
                    'unselected': {
                        'marker': { 'opacity': 0.3 },
                        'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                    }
                }],
                'layout': {
                    'width': b_width, 
                    'height': b_height,
                    #'title': "1.1 mm Array",
                    'xaxis': {'title': 'y (pixels)'},
                    'yaxis': {'title': 'S/N'},
                    'dragmode': 'select',
                    'hovermode': True,
                }}
            
            return figure
            
         
        #Creates the x slice plot for the 1.1 mm array.  It takes the
        #clickData from the 1.1 mm array plot as input along with the checklist
        #It then finds the corresponding detector by assembling network and
        #detector lists and plots that x slice.  The beammap matrix is assembled
        #at the start when ncobs is created.
        @app.callback(
            Output(p1100_b3.id, 'figure'),
            [Input(p1100.id,'selectedData'),
             Input(nw_checklist.id, 'value'),
             Input(p1100_h0.id,'selectedData'),
             Input(p1100_h1.id,'selectedData'),
             Input(p1100_h2.id,'selectedData'),
             Input(p1100_h3.id,'selectedData'),
             Input(p1100_h4.id,'selectedData'),
             Input(p1100_h5.id,'selectedData')])
        def update_b1100_3(clickData,checklist,clickData_h0,clickData_h1,clickData_h2,
                         clickData_h3,clickData_h4,clickData_h5):
                      #try:
                #pointNumber is the number in a list of all plotted points of
                #the nearest data point to where was clicked
            try:
                pointNumber = clickData['points'][0]['pointNumber']
                print(pointNumber)
            except:
                pass
                
            p = {}
            f = []
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] = []
            
            #Fill up x,y, and z for all detectors in the given networks.
            for i in range(len(checklist)):
                if int(checklist[i]) in [0,1,4,5,6,7,8]:
                    for j in range(len(ncobs.pnames)):
                        p[ncobs.pnames[j]].extend(ncobs.ncs[int(checklist[i])].p[ncobs.pnames[j]])
                    f.extend(ncobs.ncs[int(checklist[i])].f)
                
                #Make lists of the networks and detector numbers so we can
                #find out what network pointNumber is in and what detector
                #number in that network it corresponds to.
                dets = []
                nws = []
                
                for i in range(len(checklist)):
                    ci = int(checklist[i])
                    if ci in [0,1,4,5,6,7,8]:
                        nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                        dets.extend(range(ncobs.ncs[ci].ndets))
             
                        
            print('nws',len(nws))

            try:
                print('clickData',clickData_h0['range']['x'])
                amp_temp = np.array(p['amps'])
                ai = np.where((amp_temp>=clickData_h0['range']['x'][0]) & (amp_temp<=clickData_h0['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['amps'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            try:
                print('clickData',clickData_h1['range']['x'])
                amp_temp = np.array(p['x'])
                ai = np.where((amp_temp>=clickData_h1['range']['x'][0]) & (amp_temp<=clickData_h1['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['x'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h2['range']['x'])
                amp_temp = np.array(p['y'])
                ai = np.where((amp_temp>=clickData_h2['range']['x'][0]) & (amp_temp<=clickData_h2['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['y'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

        
            try:
                print('clickData',clickData_h3['range']['x'])
                amp_temp = np.array(p['fwhmx'])
                ai = np.where((amp_temp>=clickData_h3['range']['x'][0]) & (amp_temp<=clickData_h3['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmx'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h4['range']['x'])
                amp_temp = np.array(p['fwhmy'])
                ai = np.where((amp_temp>=clickData_h4['range']['x'][0]) & (amp_temp<=clickData_h4['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmy'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            
            try:
                print('clickData',clickData_h5['range']['x'])
                amp_temp = np.array(f)
                ai = np.where((amp_temp>=clickData_h5['range']['x'][0]) & (amp_temp<=clickData_h5['range']['x'][1]))[0]
            except:
                ai = list(range(len(f)))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            nw = nws[pointNumber]
            det = dets[pointNumber]
                
            #Get the map for the corresponding detector that was clicked                
            z = ncobs.ncs[np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]].x_onoff[:,:,det]
            
            nwi = np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]
                            
            #To find which row and column to slice on, we find the x,y
            #centroid and round to the nearest element in the matrix.
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
                    'colorscale':'Viridis', # one of plotly colorscales
                    'unselected': {
                        'marker': { 'opacity': 1.0 },
                        'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                    },
                    'selected': {'marker': {'color': 'red'}}
                }],
                'layout': {
                    'width': b_width, 
                    'height': b_height,
                    #'title': "1.1 mm Array",
                     'xaxis': {'title': 'x (pixels)'},
                     'yaxis': {'title': 'S/N'},
                    'dragmode': 'select',
                    'hovermode': True,
                }}
            
            return figure
            
        #This callback updates the table for a selected detector and is
        #triggered on the selection of new data in the 1.1 mm array plot.  It
        #pulls the parameters from the nocbs object.  The detector and
        #network are calculated in the same way as the beammap plot.
        @app.callback(
            Output(t1100.id, 'children'),
            [Input(p1100.id,'selectedData'),
             Input(nw_checklist.id, 'value'),
             Input(p1100_h0.id,'selectedData'),
             Input(p1100_h1.id,'selectedData'),
             Input(p1100_h2.id,'selectedData'),
             Input(p1100_h3.id,'selectedData'),
             Input(p1100_h4.id,'selectedData'),
             Input(p1100_h5.id,'selectedData')])
        def update_t1100(clickData,checklist,clickData_h0,clickData_h1,clickData_h2,
                         clickData_h3,clickData_h4,clickData_h5):
                      #try:
                #pointNumber is the number in a list of all plotted points of
                #the nearest data point to where was clicked
            try:
                pointNumber = clickData['points'][0]['pointNumber']
                print(pointNumber)
            except:
                pass
                
            p = {}
            f = []
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] = []
            
            #Fill up x,y, and z for all detectors in the given networks.
            for i in range(len(checklist)):
                if int(checklist[i]) in [0,1,4,5,6,7,8]:
                    for j in range(len(ncobs.pnames)):
                        p[ncobs.pnames[j]].extend(ncobs.ncs[int(checklist[i])].p[ncobs.pnames[j]])
                    f.extend(ncobs.ncs[int(checklist[i])].f)
                
                #Make lists of the networks and detector numbers so we can
                #find out what network pointNumber is in and what detector
                #number in that network it corresponds to.
                dets = []
                nws = []
                
                for i in range(len(checklist)):
                    ci = int(checklist[i])
                    if ci in [0,1,4,5,6,7,8]:
                        nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                        dets.extend(range(ncobs.ncs[ci].ndets))
             
                        
            print('nws',len(nws))

            try:
                print('clickData',clickData_h0['range']['x'])
                amp_temp = np.array(p['amps'])
                ai = np.where((amp_temp>=clickData_h0['range']['x'][0]) & (amp_temp<=clickData_h0['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['amps'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            try:
                print('clickData',clickData_h1['range']['x'])
                amp_temp = np.array(p['x'])
                ai = np.where((amp_temp>=clickData_h1['range']['x'][0]) & (amp_temp<=clickData_h1['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['x'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h2['range']['x'])
                amp_temp = np.array(p['y'])
                ai = np.where((amp_temp>=clickData_h2['range']['x'][0]) & (amp_temp<=clickData_h2['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['y'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

        
            try:
                print('clickData',clickData_h3['range']['x'])
                amp_temp = np.array(p['fwhmx'])
                ai = np.where((amp_temp>=clickData_h3['range']['x'][0]) & (amp_temp<=clickData_h3['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmx'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h4['range']['x'])
                amp_temp = np.array(p['fwhmy'])
                ai = np.where((amp_temp>=clickData_h4['range']['x'][0]) & (amp_temp<=clickData_h4['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmy'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            
            try:
                print('clickData',clickData_h5['range']['x'])
                amp_temp = np.array(f)
                ai = np.where((amp_temp>=clickData_h5['range']['x'][0]) & (amp_temp<=clickData_h5['range']['x'][1]))[0]
            except:
                ai = list(range(len(f)))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]

            nw = nws[pointNumber]
            det = dets[pointNumber]
            
            nwi = np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]

            row0 = html.Tr([html.Td("S/N"), html.Td('%.2f' % (ncobs.ncs[nwi].p['amps'][det]))])
            row1 = html.Tr([html.Td("X Centroid [px]"), html.Td('%.2f' % (ncobs.ncs[nwi].p['x'][det]))])
            row2 = html.Tr([html.Td("Y Centroid [px]"), html.Td('%.2f' % (ncobs.ncs[nwi].p['y'][det]))])
            row3 = html.Tr([html.Td("X FWHM [px]"), html.Td('%.2f' % (ncobs.ncs[nwi].p['fwhmx'][det]))])
            row4 = html.Tr([html.Td("Y FWHM [px]"), html.Td('%.2f' % (ncobs.ncs[nwi].p['fwhmy'][det]))])
            row5 = html.Tr([html.Td("Frequency [MHz]"), html.Td('%.2f' % (ncobs.ncs[nwi].f[det]))])
            row6 = html.Tr([html.Td("Network"), html.Td(int(nw))])
            row7 = html.Tr([html.Td("Detector Number on Network"), html.Td(int(det))])

            table_body = [html.Tbody([row0,row1, row2, row3, row4,row5,row6,row7])]
            return table_header + table_body
             
           
        #Makes the 1.1 mm S/N histogram for the currently plotted networks.
        #works in the same way as the array plot in that it assembles lists
        #for the networks picked by the checklist.
        @app.callback(
            Output(p1100_h0.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p1100_h0(obsnum,checklist):
            
            #Same as the array plot, but we only need one parameter.
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
                'marker': { 'color': '#C0392B',
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                #'title': "1.1 mm Array",
                'xaxis': {'title': 'S/N'},
                'yaxis': {'title': 'N'},
                'autosize': True,

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
        
        
        #Makes the 1.1 mm x centroid histogram for the currently plotted networks.
        #works in the same way as the array plot in that it assembles lists
        #for the networks picked by the checklist.
        @app.callback(
            Output(p1100_h1.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p1100_h1(obsnum,checklist):
            
            #Same as the array plot, but we only need one parameter.
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
                'marker': { 'color': '#884EA0', #set color equal to a variable
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                #'title': "1.1 mm Array",
                'xaxis': {'title': 'X Centroid [px]'},
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
        
        #Makes the 1.1 mm y centroid histogram for the currently plotted networks.
        #works in the same way as the array plot in that it assembles lists
        #for the networks picked by the checklist.
        @app.callback(
            Output(p1100_h2.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p1100_h2(obsnum,checklist):
            
            #Same as the array plot, but we only need one parameter.
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
                'marker': { 'color': '#2471A3', #set color equal to a variable
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                #'title': "1.1 mm Array",
                'xaxis': {'title': 'Y Centroid [px]'},
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
        
        
        #Makes the 1.1 mm fwhm_x histogram for the currently plotted networks.
        #works in the same way as the array plot in that it assembles lists
        #for the networks picked by the checklist.
        @app.callback(
            Output(p1100_h3.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p1100_h3(obsnum,checklist):
            
            #Same as the array plot, but we only need one parameter.
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
                'marker': { 'color': '#17A589', #set color equal to a variable
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                #'title': "1.1 mm Array",
                'xaxis': {'title': 'X FWHM [px]'},
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
        
        
        
        #Makes the 1.1 mm fwhm_y histogram for the currently plotted networks.
        #works in the same way as the array plot in that it assembles lists
        #for the networks picked by the checklist.
        @app.callback(
            Output(p1100_h4.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p1100_h4(obsnum,checklist):
            
            #Same as the array plot, but we only need one parameter.
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
                'marker': { 'color': '#229954', #set color equal to a variable
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                #'title': "1.1 mm Array",
                'xaxis': {'title': 'Y FWMH [px]'},
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
        
        
        #Makes the 1.1 mm freq histogram for the currently plotted networks.
        #works in the same way as the array plot in that it assembles lists
        #for the networks picked by the checklist.
        @app.callback(
            Output(p1100_h5.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p1100_h5(obsnum,checklist):
            
            #Same as the array plot, but we only need one parameter.
            h = []            
            for i in range(len(checklist)):
                if int(checklist[i]) in [0,1,4,5,6,7,8]:
                    h.extend(ncobs.ncs[int(checklist[i])].f)
                    
            figure={
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
                #'title': "1.1 mm Array",
                'xaxis': {'title': 'Frequency [MHz]'},
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
        
        
        
        #Make a tab for the 1.4 mm plots
        a1400_container = array_tab_container.child(dcc.Tab,label='1.4mm').child(dbc.Row)
        
        #A container for the 1.4 mm table
        t1400_container = a1400_container.child(dbc.Col,style={'margin':0})#.child(dbc.Jumbotron, [html.H1(children="Selected Detector")],style={"width": "25%"})
        
        #Make a tab for the 1.4 mm beammap and slice plots
        #a1400_b_tab = a1400_container.child(dbc.Col,style={'width':'40%', 'margin':0, 'textAlign': 'center'}).child(dcc.Tabs,vertical=True)
        
        a1400_b_tab = t1400_container.child(dbc.Tabs,style={'margin':0, 'textAlign': 'center'})
        
        #Tabs and graphs for the 1.4 mm beammap and slice plots
        #p1400_b = a1400_b_tab.child(dcc.Tab,label='beammap').child(dbc.Col).child(dcc.Graph)
        #p1400_b2 = a1400_b_tab.child(dcc.Tab,label='y-slice').child(dbc.Col).child(dcc.Graph)
        #p1400_b3 = a1400_b_tab.child(dcc.Tab,label='x-slice').child(dbc.Col).child(dcc.Graph)
        
        p1400_b = a1400_b_tab.child(dbc.Tab,label='beammap').child(dcc.Graph,figure = {
                'data': [{
                    #'x': list(range(ncobs.nrows)),
                    #'y': list(range(ncobs.ncols)),
                    'z': np.zeros([ncobs.nrows,ncobs.ncols]),
                    'textposition': 'top',
                    'customdata': 0,
                    'type': 'scatter',
                    'mode': 'markers+text',
                    'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
                    'colorscale':'Gray', # one of plotly colorscales
                    'colorbar': {'title': 'N/A'},
                    'unselected': {
                        'marker': { 'opacity': 0.3 },
                        'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                    }
                }],
                'layout': {
                    'width': b_width, 
                    'height': b_height,
                    #'title': "1.4 mm Array",
                    'xaxis': {'title': 'x (pixels)'},
                    'yaxis': {'title': 'y (pixels)'},
                    'dragmode': 'select',
                    'hovermode': True,
                }})
        p1400_b2 = a1400_b_tab.child(dbc.Tab,label='y slice').child(dcc.Graph, figure={'layout': {
                    'width': b_width, 
                    'height': b_height,
                    'xaxis': {'title': 'y (pixels)'},
                    'yaxis': {'title': 'S/N'},
                    'dragmode': 'select',
                    'hovermode': True,
                }})
        p1400_b3 = a1400_b_tab.child(dbc.Tab, label='x slice').child(dcc.Graph, figure={'layout': {
                    'width': b_width, 
                    'height': b_height,
                    #'title': "2.0 mm Array",
                    'xaxis': {'title': 'x (pixels)'},
                    'yaxis': {'title': 'S/N'},
                    'dragmode': 'select',
                    'hovermode': True,
                }})
        
        
        t1400 = t1400_container.child(dbc.Table,bordered=True, striped=True, hover=True,responsive=True,style={'width': '100%'})

        t1400.children = table_header + table_body

        #This creates a graph for the 1.4 mm array plot
        #p1400 = a1400_container.child(dbc.Col,style={'width':'50%', 'margin':0, 'textAlign': 'center'}).child(dcc.Graph)
        p1400_container = a1400_container.child(dbc.Col,style={'width': '100%', 'margin':0, 'textAlign': 'center'}).child(dbc.Card).child(dbc.CardBody,style={"marginRight": 75,"marginTop": 0, "marginBottom": 0,'width': '100%'}).child(dbc.CardHeader,html.H5("1.4mm Array"))
        

        #Inside of the 1.4 mm container, make a dropdown to control the
        #plotted axes for the array plot.  Separate for each array.   Defaults
        #to x, y
        drp_container = p1400_container.child(dbc.Row)
        drp_1400=drp_container.child(dbc.Col,style={'width':'10%', 'margin':0, 'textAlign': 'center'}).child(dcc.Dropdown, options=[
            {'label': 'S/N', 'value': 'amps'},
            {'label': 'x', 'value': 'x'},
            {'label': 'y', 'value': 'y'},
            {'label': 'fwhmx', 'value': 'fwhmx'},
            {'label': 'fwhmy', 'value': 'fwhmy'},
            {'label': 'Freq', 'value': 'f'}

        ],
        value='x',
        multi=False)
        
        drp_1400_2=drp_container.child(dbc.Col,style={'width':'10%', 'margin':0, 'textAlign': 'center'}).child(dcc.Dropdown, options=[
            {'label': 'S/N', 'value': 'amps'},
            {'label': 'x', 'value': 'x'},
            {'label': 'y', 'value': 'y'},
            {'label': 'fwhmx', 'value': 'fwhmx'},
            {'label': 'fwhmy', 'value': 'fwhmy'},
            {'label': 'Freq', 'value': 'f'}

        ],
        value='y',
        multi=False)
        
        drp_1400_3=drp_container.child(dbc.Col,style={'width':'10%', 'margin':0, 'textAlign': 'center'}).child(dcc.Dropdown, options=[
            {'label': 'NW', 'value': 'NW'},
            {'label': 'S/N', 'value': 'amps'},
            {'label': 'x', 'value': 'x'},
            {'label': 'y', 'value': 'y'},
            {'label': 'fwhmx', 'value': 'fwhmx'},
            {'label': 'fwhmy', 'value': 'fwhmy'},
            {'label': 'Freq', 'value': 'f'}
        ],
        value='NW',
        multi=False)
        
        p1400 = p1400_container.child(dcc.Graph, figure={
                'layout': {
                    #'title': "1.4 mm Array",
                    'autosize': False,
                    'automargin': False,
                    'xaxis': {'range': [0,max(ncobs.nrows,ncobs.ncols)]},
                    'yaxis': {'range': [0,max(ncobs.nrows,ncobs.ncols)]},
                    'width': a_width, 
                    'height': a_height,
                    'dragmode': 'select',
                    'hovermode': 'closest',
                    'editable': True,
                    'animate': True,
                    'clickmode': 'event+select',
                    'plot_bgcolor': "#F9F9F9",
                    'paper_bgcolor': "#F9F9F9",
                }})
        
        #Create a tab for the 1.4 mm histograms        
        a1400_h_tab = a1400_container.child(dbc.Row)
        
        #Set up each histogram for the 1.4 mm array.  Each is a new column.
        p1400_h0 = a1400_container.child(dbc.Row).child(dcc.Graph,align="center")
        p1400_h1 = a1400_container.child(dbc.Row).child(dcc.Graph,align="center")
        p1400_h2 = a1400_container.child(dbc.Row).child(dcc.Graph,align="center")
        p1400_h3 = a1400_container.child(dbc.Row).child(dcc.Graph,align="center")
        p1400_h4 = a1400_container.child(dbc.Row).child(dcc.Graph,align="center")
        p1400_h5 = a1400_container.child(dbc.Row).child(dcc.Graph,align="center")

        
        #This creates the array plot for the 1.4 mm array.  It takes the
        #dropdown and checklist as input.  From the checklist, it creates
        #lists of all selected parameters from the dropdown using only the
        #networks selected.  This then updates 'data' in the figure.
        @app.callback(
            Output(p1400.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(drp_1400.id, 'value'),
             Input(drp_1400_2.id, 'value'),
             Input(drp_1400_3.id, 'value'),
             Input(nw_checklist.id, 'value'),
             Input(p1400_h0.id,'selectedData'),
             Input(p1400_h1.id,'selectedData'),
             Input(p1400_h2.id,'selectedData'),
             Input(p1400_h3.id,'selectedData'),
             Input(p1400_h4.id,'selectedData'),
             Input(p1400_h5.id,'selectedData'),]) 
        def update_a1400(obsnum,value,value_2,value_3,checklist,clickData_h0,clickData_h1,clickData_h2,
                         clickData_h3,clickData_h4,clickData_h5):
            #Lists for the x,y, and color
            x = []
            y = []
            z = []
            
            p = {}
            f = []
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] = []
    
            #Fill up x,y, and z for all detectors in the given networks.
            for i in range(len(checklist)):
                if int(checklist[i]) in [9,10,11,12]:
                    for j in range(len(ncobs.pnames)):
                        p[ncobs.pnames[j]].extend(ncobs.ncs[int(checklist[i])].p[ncobs.pnames[j]])
                    f.extend(ncobs.ncs[int(checklist[i])].f)

                    #x.extend(ncobs.ncs[int(checklist[i])].p[value[0]])
                    #y.extend(ncobs.ncs[int(checklist[i])].p[value[1]])
                    
                    #if len(value)==2:
                    if value_3 == 'NW':
                        if value!='f':
                            z.extend(np.ones(len(ncobs.ncs[int(checklist[i])].p[value]))*int(checklist[i]))  
                        else:
                            z.extend(np.ones(len(ncobs.ncs[int(checklist[i])].f))*int(checklist[i]))
                         
                        ''' 
                           if value!='f':
                             char_arr = np.chararray(len(ncobs.ncs[int(checklist[i])].p[value]),unicode=True,itemsize=7)
                             char_arr[:] = colors[int(checklist[i])]
                        else:
                            char_arr = np.chararray(len(ncobs.ncs[int(checklist[i])].f),unicode=True,itemsize=7)
                            char_arr[:] = colors[int(checklist[i])]
                        z.extend(char_arr)
                        print(z)
                        '''
                        
            #if len(value) == 3:
            if value_3 !='NW':
                if value_3 !='f':
                    z = np.array(p[value_3])
                else:
                    z = np.array(f)
            #Each block from the try to the z = statement take the limits
            #from a histogram selection and limit the parameters to only
            #those points that fall into that selection.
            try:
                print('clickData',clickData_h0['range']['x'])
                amp_temp = np.array(p['amps'])
                ai = np.where((amp_temp>=clickData_h0['range']['x'][0]) & (amp_temp<=clickData_h0['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['amps'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            
            z = np.array(z)[ai]
            
            try:
                print('clickData',clickData_h1['range']['x'])
                amp_temp = np.array(p['x'])
                ai = np.where((amp_temp>=clickData_h1['range']['x'][0]) & (amp_temp<=clickData_h1['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['x'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            
            z = np.array(z)[ai]
            
            
            try:
                print('clickData',clickData_h2['range']['x'])
                amp_temp = np.array(p['y'])
                ai = np.where((amp_temp>=clickData_h2['range']['x'][0]) & (amp_temp<=clickData_h2['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['y'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            
            z = np.array(z)[ai]
            
        
            try:
                print('clickData',clickData_h3['range']['x'])
                amp_temp = np.array(p['fwhmx'])
                ai = np.where((amp_temp>=clickData_h3['range']['x'][0]) & (amp_temp<=clickData_h3['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmx'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            
            z = np.array(z)[ai]
            
            try:
                print('clickData',clickData_h4['range']['x'])
                amp_temp = np.array(p['fwhmy'])
                ai = np.where((amp_temp>=clickData_h4['range']['x'][0]) & (amp_temp<=clickData_h4['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmy'])))
                
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            
            z = np.array(z)[ai]
            
            
            try:
                print('clickData',clickData_h5['range']['x'])
                amp_temp = np.array(f)
                ai = np.where((amp_temp>=clickData_h5['range']['x'][0]) & (amp_temp<=clickData_h5['range']['x'][1]))[0]
            except:
                ai = list(range(len(f)))
                
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            
            z = np.array(z)[ai]
            
            if value!='f':
                x = np.array(p[value])
            else:
                x = np.array(f)
            if value_2!='f':
                y = np.array(p[value_2])
            else:
                y = np.array(f)
            
            if (str(value) == 'x') or (str(value) == 'y'):
                xmax = max(ncobs.nrows,ncobs.ncols)
                xmin = 0
            else:
                xmax = np.max(x)
                xmin = np.min(x)
                
            if (str(value_2) == 'x') or (str(value_2) == 'y'):
                ymax = max(ncobs.nrows,ncobs.ncols)
                ymin = 0
            else:
                ymax = np.max(y)
                ymin = np.min(y)
            
            if value=='amps':
                label='S/N'  
            elif value=='x':
                label='X Centroid [px]'
            elif value=='y':
                label='Y Centroid [px]'
            elif value=='fwhmx':
                label='X FWHM [px]'
            elif value=='fwhmy':
                label='Y FWHM [px]'
            elif value=='f':
                label='Freq [MHz]'
                
            if value_2=='amps':
                label_2='S/N'  
            elif value_2=='x':
                label_2='X Centroid [px]'
            elif value_2=='y':
                label_2='Y Centroid [px]'
            elif value_2=='fwhmx':
                label_2='X FWHM [px]'
            elif value_2=='fwhmy':
                label_2='Y FWHM [px]'
            elif value_2=='f':
                label_2='Freq [MHz]'
            
            if value_3=='amps':
                label_3='S/N'  
            elif value_3=='x':
                label_3='X Centroid [px]'
            elif value_3=='y':
                label_3='Y Centroid [px]'
            elif value_3=='fwhmx':
                label_3='X FWHM [px]'
            elif value_3=='fwhmy':
                label_3='Y FWHM [px]'
            elif value_3=='f':
                label_3='Freq [MHz]'
            elif value_3=='NW':
                label_3 = 'NW'
            
            #return dict with all plotting options
            figure={
                'data': [{
                    'x': x,
                    'y': y,
                    'textposition': 'top',
                    'customdata': 0,
                    'type': 'scatter',
                    'mode': 'markers',
                    'marker': {
                        'color': z,
                        #'colorscale':'Viridis', # one of plotly colorscales
                        'showscale': True,
                        'colorbar': {'title': label_3},

                        #'size': 5 
                    },
                    'unselected': {
                        'marker': { 'opacity': 1.0 },
                        'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                    },
                    'selected': {
                        'marker': {'color': 'red'},
                        'textfont': { 'color': 'rgba(1,0,0,0)' }
                    }
                }],
                'layout': {
                    #'title': "1.4 mm Array",
                    'autosize': False,
                    'automargin': False,
                    'xaxis': {'title': label,'range': [xmin,xmax]},
                    'yaxis': {'title': label_2,'range': [ymin,ymax]},
                    'width': a_width, 
                    'height': a_height,
                    'dragmode': 'select',
                    'hovermode': 'closest',
                    'editable': True,
                    'animate': True,
                    'clickmode': 'event+select',
                    'plot_bgcolor': "#F9F9F9",
                    'paper_bgcolor': "#F9F9F9",
                }}
            
            return figure
        
        
        #Creates the beammap plot for the 1.4 mm array.  It takes the
        #clickData from the 1.4 mm array plot as input along with the checklist
        #It then finds the corresponding detector by assembling network and
        #detector lists and plots that beammap.  The beammap matrix is assembled
        #at the start when ncobs is created.
        @app.callback(
            Output(p1400_b.id, 'figure'),
            [Input(p1400.id,'selectedData'),
             Input(nw_checklist.id, 'value'),
             Input(p1400_h0.id,'selectedData'),
             Input(p1400_h1.id,'selectedData'),
             Input(p1400_h2.id,'selectedData'),
             Input(p1400_h3.id,'selectedData'),
             Input(p1400_h4.id,'selectedData'),
             Input(p1400_h5.id,'selectedData'),])
        def update_b1400(clickData,checklist,clickData_h0,clickData_h1,clickData_h2,
                         clickData_h3,clickData_h4,clickData_h5):
            #try:
                #pointNumber is the number in a list of all plotted points of
                #the nearest data point to where was clicked
            try:
                pointNumber = clickData['points'][0]['pointNumber']
                print(pointNumber)
            except:
                pass
                
            p = {}
            f = []
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] = []
            
            #Fill up x,y, and z for all detectors in the given networks.
            for i in range(len(checklist)):
                if int(checklist[i]) in [9,10,11,12]:
                    for j in range(len(ncobs.pnames)):
                        p[ncobs.pnames[j]].extend(ncobs.ncs[int(checklist[i])].p[ncobs.pnames[j]])
                    f.extend(ncobs.ncs[int(checklist[i])].f)
                
                #Make lists of the networks and detector numbers so we can
                #find out what network pointNumber is in and what detector
                #number in that network it corresponds to.
                dets = []
                nws = []
                
                for i in range(len(checklist)):
                    ci = int(checklist[i])
                    if ci in [9,10,11,12]:
                        nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                        dets.extend(range(ncobs.ncs[ci].ndets))
             
                        
            print('nws',len(nws))

            try:
                print('clickData',clickData_h0['range']['x'])
                amp_temp = np.array(p['amps'])
                ai = np.where((amp_temp>=clickData_h0['range']['x'][0]) & (amp_temp<=clickData_h0['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['amps'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            try:
                print('clickData',clickData_h1['range']['x'])
                amp_temp = np.array(p['x'])
                ai = np.where((amp_temp>=clickData_h1['range']['x'][0]) & (amp_temp<=clickData_h1['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['x'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h2['range']['x'])
                amp_temp = np.array(p['y'])
                ai = np.where((amp_temp>=clickData_h2['range']['x'][0]) & (amp_temp<=clickData_h2['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['y'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

        
            try:
                print('clickData',clickData_h3['range']['x'])
                amp_temp = np.array(p['fwhmx'])
                ai = np.where((amp_temp>=clickData_h3['range']['x'][0]) & (amp_temp<=clickData_h3['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmx'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h4['range']['x'])
                amp_temp = np.array(p['fwhmy'])
                ai = np.where((amp_temp>=clickData_h4['range']['x'][0]) & (amp_temp<=clickData_h4['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmy'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            
            try:
                print('clickData',clickData_h5['range']['x'])
                amp_temp = np.array(f)
                ai = np.where((amp_temp>=clickData_h5['range']['x'][0]) & (amp_temp<=clickData_h5['range']['x'][1]))[0]
            except:
                ai = list(range(len(f)))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            nw = nws[pointNumber]
            det = dets[pointNumber]
                
            #Get the map for the corresponding detector that was clicked                
            z = ncobs.ncs[np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]].x_onoff[:,:,det]

            #return the figure dict
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
                    'colorscale':'Viridis', # one of plotly colorscales
                    'colorbar': {'title': 'S/N'},
                    'unselected': {
                        'marker': { 'opacity': 0.3 },
                        'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                    }
                }],
                'layout': {
                    'width': b_width, 
                    'height': b_height,
                    #'title': "1.4 mm Array",
                    'xaxis': {'title': 'x (pixels)'},
                    'yaxis': {'title': 'y (pixels)'},
                    'dragmode': 'select',
                    'hovermode': True,
                }}
            
            return figure
        
        
        #Creates the y slice plot for the 1.4 mm array.  It takes the
        #clickData from the 1.4 mm array plot as input along with the checklist
        #It then finds the corresponding detector by assembling network and
        #detector lists and plots that y slice.  The beammap matrix is assembled
        #at the start when ncobs is created.
        @app.callback(
            Output(p1400_b2.id, 'figure'),
            [Input(p1400.id,'selectedData'),
             Input(nw_checklist.id, 'value'),
             Input(p1400_h0.id,'selectedData'),
             Input(p1400_h1.id,'selectedData'),
             Input(p1400_h2.id,'selectedData'),
             Input(p1400_h3.id,'selectedData'),
             Input(p1400_h4.id,'selectedData'),
             Input(p1400_h5.id,'selectedData')])
        def update_b1400_2(clickData,checklist,clickData_h0,clickData_h1,clickData_h2,
                         clickData_h3,clickData_h4,clickData_h5):
                 #try:
                #pointNumber is the number in a list of all plotted points of
                #the nearest data point to where was clicked
            try:
                pointNumber = clickData['points'][0]['pointNumber']
                print(pointNumber)
            except:
                pass
                
            p = {}
            f = []
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] = []
            
            #Fill up x,y, and z for all detectors in the given networks.
            for i in range(len(checklist)):
                if int(checklist[i]) in [9,10,11,12]:
                    for j in range(len(ncobs.pnames)):
                        p[ncobs.pnames[j]].extend(ncobs.ncs[int(checklist[i])].p[ncobs.pnames[j]])
                    f.extend(ncobs.ncs[int(checklist[i])].f)
                
                #Make lists of the networks and detector numbers so we can
                #find out what network pointNumber is in and what detector
                #number in that network it corresponds to.
                dets = []
                nws = []
                
                for i in range(len(checklist)):
                    ci = int(checklist[i])
                    if ci in [9,10,11,12]:
                        nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                        dets.extend(range(ncobs.ncs[ci].ndets))
             
                        
            print('nws',len(nws))

            try:
                print('clickData',clickData_h0['range']['x'])
                amp_temp = np.array(p['amps'])
                ai = np.where((amp_temp>=clickData_h0['range']['x'][0]) & (amp_temp<=clickData_h0['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['amps'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            try:
                print('clickData',clickData_h1['range']['x'])
                amp_temp = np.array(p['x'])
                ai = np.where((amp_temp>=clickData_h1['range']['x'][0]) & (amp_temp<=clickData_h1['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['x'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h2['range']['x'])
                amp_temp = np.array(p['y'])
                ai = np.where((amp_temp>=clickData_h2['range']['x'][0]) & (amp_temp<=clickData_h2['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['y'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

        
            try:
                print('clickData',clickData_h3['range']['x'])
                amp_temp = np.array(p['fwhmx'])
                ai = np.where((amp_temp>=clickData_h3['range']['x'][0]) & (amp_temp<=clickData_h3['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmx'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h4['range']['x'])
                amp_temp = np.array(p['fwhmy'])
                ai = np.where((amp_temp>=clickData_h4['range']['x'][0]) & (amp_temp<=clickData_h4['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmy'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            
            try:
                print('clickData',clickData_h5['range']['x'])
                amp_temp = np.array(f)
                ai = np.where((amp_temp>=clickData_h5['range']['x'][0]) & (amp_temp<=clickData_h5['range']['x'][1]))[0]
            except:
                ai = list(range(len(f)))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            nw = nws[pointNumber]
            det = dets[pointNumber]
                
            #Get the map for the corresponding detector that was clicked                
            z = ncobs.ncs[np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]].x_onoff[:,:,det]
            
            nwi = np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]
            
            #To find which row and column to slice on, we find the x,y
            #centroid and round to the nearest element in the matrix.
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
                    'colorscale':'Viridis', # one of plotly colorscales
                    'unselected': {
                        'marker': { 'opacity': 0.3 },
                        'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                    }
                }],
                'layout': {
                    'width': b_width, 
                    'height': b_height,
                    #'title': "1.4 mm Array",
                    'xaxis': {'title': 'y (pixels)'},
                    'yaxis': {'title': 'S/N'},
                    'dragmode': 'select',
                    'hovermode': True,
                }}
            
            return figure
            
         
        #Creates the x slice plot for the 1.4 mm array.  It takes the
        #clickData from the 1.4 mm array plot as input along with the checklist
        #It then finds the corresponding detector by assembling network and
        #detector lists and plots that x slice.  The beammap matrix is assembled
        #at the start when ncobs is created.
        @app.callback(
            Output(p1400_b3.id, 'figure'),
            [Input(p1400.id,'selectedData'),
             Input(nw_checklist.id, 'value'),
             Input(p1400_h0.id,'selectedData'),
             Input(p1400_h1.id,'selectedData'),
             Input(p1400_h2.id,'selectedData'),
             Input(p1400_h3.id,'selectedData'),
             Input(p1400_h4.id,'selectedData'),
             Input(p1400_h5.id,'selectedData')])
        def update_b1400_3(clickData,checklist,clickData_h0,clickData_h1,clickData_h2,
                         clickData_h3,clickData_h4,clickData_h5):
                      #try:
                #pointNumber is the number in a list of all plotted points of
                #the nearest data point to where was clicked
            try:
                pointNumber = clickData['points'][0]['pointNumber']
                print(pointNumber)
            except:
                pass
                
            p = {}
            f = []
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] = []
            
            #Fill up x,y, and z for all detectors in the given networks.
            for i in range(len(checklist)):
                if int(checklist[i]) in [9,10,11,12]:
                    for j in range(len(ncobs.pnames)):
                        p[ncobs.pnames[j]].extend(ncobs.ncs[int(checklist[i])].p[ncobs.pnames[j]])
                    f.extend(ncobs.ncs[int(checklist[i])].f)
                
                #Make lists of the networks and detector numbers so we can
                #find out what network pointNumber is in and what detector
                #number in that network it corresponds to.
                dets = []
                nws = []
                
                for i in range(len(checklist)):
                    ci = int(checklist[i])
                    if ci in [9,10,11,12]:
                        nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                        dets.extend(range(ncobs.ncs[ci].ndets))
             
                        
            print('nws',len(nws))

            try:
                print('clickData',clickData_h0['range']['x'])
                amp_temp = np.array(p['amps'])
                ai = np.where((amp_temp>=clickData_h0['range']['x'][0]) & (amp_temp<=clickData_h0['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['amps'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            try:
                print('clickData',clickData_h1['range']['x'])
                amp_temp = np.array(p['x'])
                ai = np.where((amp_temp>=clickData_h1['range']['x'][0]) & (amp_temp<=clickData_h1['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['x'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h2['range']['x'])
                amp_temp = np.array(p['y'])
                ai = np.where((amp_temp>=clickData_h2['range']['x'][0]) & (amp_temp<=clickData_h2['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['y'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

        
            try:
                print('clickData',clickData_h3['range']['x'])
                amp_temp = np.array(p['fwhmx'])
                ai = np.where((amp_temp>=clickData_h3['range']['x'][0]) & (amp_temp<=clickData_h3['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmx'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h4['range']['x'])
                amp_temp = np.array(p['fwhmy'])
                ai = np.where((amp_temp>=clickData_h4['range']['x'][0]) & (amp_temp<=clickData_h4['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmy'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            
            try:
                print('clickData',clickData_h5['range']['x'])
                amp_temp = np.array(f)
                ai = np.where((amp_temp>=clickData_h5['range']['x'][0]) & (amp_temp<=clickData_h5['range']['x'][1]))[0]
            except:
                ai = list(range(len(f)))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            nw = nws[pointNumber]
            det = dets[pointNumber]
                
            #Get the map for the corresponding detector that was clicked                
            z = ncobs.ncs[np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]].x_onoff[:,:,det]
            
            nwi = np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]
                            
            #To find which row and column to slice on, we find the x,y
            #centroid and round to the nearest element in the matrix.
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
                    'colorscale':'Viridis', # one of plotly colorscales
                    'unselected': {
                        'marker': { 'opacity': 1.0 },
                        'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                    },
                    'selected': {'marker': {'color': 'red'}}
                }],
                'layout': {
                    'width': b_width, 
                    'height': b_height,
                    #'title': "1.4 mm Array",
                     'xaxis': {'title': 'x (pixels)'},
                     'yaxis': {'title': 'S/N'},
                    'dragmode': 'select',
                    'hovermode': True,
                }}
            
            return figure
            
        #This callback updates the table for a selected detector and is
        #triggered on the selection of new data in the 1.4 mm array plot.  It
        #pulls the parameters from the nocbs object.  The detector and
        #network are calculated in the same way as the beammap plot.
        @app.callback(
            Output(t1400.id, 'children'),
            [Input(p1400.id,'selectedData'),
             Input(nw_checklist.id, 'value'),
             Input(p1400_h0.id,'selectedData'),
             Input(p1400_h1.id,'selectedData'),
             Input(p1400_h2.id,'selectedData'),
             Input(p1400_h3.id,'selectedData'),
             Input(p1400_h4.id,'selectedData'),
             Input(p1400_h5.id,'selectedData')])
        def update_t1400(clickData,checklist,clickData_h0,clickData_h1,clickData_h2,
                         clickData_h3,clickData_h4,clickData_h5):
                      #try:
                #pointNumber is the number in a list of all plotted points of
                #the nearest data point to where was clicked
            try:
                pointNumber = clickData['points'][0]['pointNumber']
                print(pointNumber)
            except:
                pass
                
            p = {}
            f = []
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] = []
            
            #Fill up x,y, and z for all detectors in the given networks.
            for i in range(len(checklist)):
                if int(checklist[i]) in [9,10,11,12]:
                    for j in range(len(ncobs.pnames)):
                        p[ncobs.pnames[j]].extend(ncobs.ncs[int(checklist[i])].p[ncobs.pnames[j]])
                    f.extend(ncobs.ncs[int(checklist[i])].f)
                
                #Make lists of the networks and detector numbers so we can
                #find out what network pointNumber is in and what detector
                #number in that network it corresponds to.
                dets = []
                nws = []
                
                for i in range(len(checklist)):
                    ci = int(checklist[i])
                    if ci in [9,10,11,12]:
                        nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                        dets.extend(range(ncobs.ncs[ci].ndets))
             
                        
            print('nws',len(nws))

            try:
                print('clickData',clickData_h0['range']['x'])
                amp_temp = np.array(p['amps'])
                ai = np.where((amp_temp>=clickData_h0['range']['x'][0]) & (amp_temp<=clickData_h0['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['amps'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            try:
                print('clickData',clickData_h1['range']['x'])
                amp_temp = np.array(p['x'])
                ai = np.where((amp_temp>=clickData_h1['range']['x'][0]) & (amp_temp<=clickData_h1['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['x'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h2['range']['x'])
                amp_temp = np.array(p['y'])
                ai = np.where((amp_temp>=clickData_h2['range']['x'][0]) & (amp_temp<=clickData_h2['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['y'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

        
            try:
                print('clickData',clickData_h3['range']['x'])
                amp_temp = np.array(p['fwhmx'])
                ai = np.where((amp_temp>=clickData_h3['range']['x'][0]) & (amp_temp<=clickData_h3['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmx'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h4['range']['x'])
                amp_temp = np.array(p['fwhmy'])
                ai = np.where((amp_temp>=clickData_h4['range']['x'][0]) & (amp_temp<=clickData_h4['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmy'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            
            try:
                print('clickData',clickData_h5['range']['x'])
                amp_temp = np.array(f)
                ai = np.where((amp_temp>=clickData_h5['range']['x'][0]) & (amp_temp<=clickData_h5['range']['x'][1]))[0]
            except:
                ai = list(range(len(f)))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]

            nw = nws[pointNumber]
            det = dets[pointNumber]
            
            nwi = np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]

            row0 = html.Tr([html.Td("S/N"), html.Td('%.2f' % (ncobs.ncs[nwi].p['amps'][det]))])
            row1 = html.Tr([html.Td("X Centroid [px]"), html.Td('%.2f' % (ncobs.ncs[nwi].p['x'][det]))])
            row2 = html.Tr([html.Td("Y Centroid [px]"), html.Td('%.2f' % (ncobs.ncs[nwi].p['y'][det]))])
            row3 = html.Tr([html.Td("X FWHM [px]"), html.Td('%.2f' % (ncobs.ncs[nwi].p['fwhmx'][det]))])
            row4 = html.Tr([html.Td("Y FWHM [px]"), html.Td('%.2f' % (ncobs.ncs[nwi].p['fwhmy'][det]))])
            row5 = html.Tr([html.Td("Frequency [MHz]"), html.Td('%.2f' % (ncobs.ncs[nwi].f[det]))])
            row6 = html.Tr([html.Td("Network"), html.Td(int(nw))])
            row7 = html.Tr([html.Td("Detector Number on Network"), html.Td(int(det))])

            table_body = [html.Tbody([row0,row1, row2, row3, row4,row5,row6,row7])]
            return table_header + table_body
             
           
        #Makes the 1.4 mm S/N histogram for the currently plotted networks.
        #works in the same way as the array plot in that it assembles lists
        #for the networks picked by the checklist.
        @app.callback(
            Output(p1400_h0.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p1400_h0(obsnum,checklist):
            
            #Same as the array plot, but we only need one parameter.
            h = []            
            for i in range(len(checklist)):
                if int(checklist[i]) in [9,10,11,12]:
                    h.extend(ncobs.ncs[int(checklist[i])].p['amps'])
                    
            figure={
                'data': [{
                'x': h,
                #'text': 0,
                'textposition': 'top',
                'customdata': 0,
                'type': 'histogram',
                'mode': 'markers+text',
                'marker': { 'color': '#C0392B',
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                #'title': "1.4 mm Array",
                'xaxis': {'title': 'S/N'},
                'yaxis': {'title': 'N'},
                'autosize': True,

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
        
        
        #Makes the 1.4 mm x centroid histogram for the currently plotted networks.
        #works in the same way as the array plot in that it assembles lists
        #for the networks picked by the checklist.
        @app.callback(
            Output(p1400_h1.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p1400_h1(obsnum,checklist):
            
            #Same as the array plot, but we only need one parameter.
            h = []            
            for i in range(len(checklist)):
                if int(checklist[i]) in [9,10,11,12]:
                    h.extend(ncobs.ncs[int(checklist[i])].p['x'])
                    
            figure={
                'data': [{
                'x': h,
                #'text': 0,
                'textposition': 'top',
                'customdata': 0,
                'type': 'histogram',
                'mode': 'markers+text',
                'marker': { 'color': '#884EA0', #set color equal to a variable
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                #'title': "1.4 mm Array",
                'xaxis': {'title': 'X Centroid [px]'},
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
        
        #Makes the 1.4 mm y centroid histogram for the currently plotted networks.
        #works in the same way as the array plot in that it assembles lists
        #for the networks picked by the checklist.
        @app.callback(
            Output(p1400_h2.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p1400_h2(obsnum,checklist):
            
            #Same as the array plot, but we only need one parameter.
            h = []            
            for i in range(len(checklist)):
                if int(checklist[i]) in [9,10,11,12]:
                    h.extend(ncobs.ncs[int(checklist[i])].p['y'])
                    
            figure={
                'data': [{
                'x': h,
                #'text': 0,
                'textposition': 'top',
                'customdata': 0,
                'type': 'histogram',
                'mode': 'markers+text',
                'marker': { 'color': '#2471A3', #set color equal to a variable
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                #'title': "1.4 mm Array",
                'xaxis': {'title': 'Y Centroid [px]'},
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
        
        
        #Makes the 1.4 mm fwhm_x histogram for the currently plotted networks.
        #works in the same way as the array plot in that it assembles lists
        #for the networks picked by the checklist.
        @app.callback(
            Output(p1400_h3.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p1400_h3(obsnum,checklist):
            
            #Same as the array plot, but we only need one parameter.
            h = []            
            for i in range(len(checklist)):
                if int(checklist[i]) in [9,10,11,12]:
                    h.extend(ncobs.ncs[int(checklist[i])].p['fwhmx'])
                    
            figure={
                'data': [{
                'x': h,
                #'text': 0,
                'textposition': 'top',
                'customdata': 0,
                'type': 'histogram',
                'mode': 'markers+text',
                'marker': { 'color': '#17A589', #set color equal to a variable
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                #'title': "1.4 mm Array",
                'xaxis': {'title': 'X FWHM [px]'},
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
        
        
        
        #Makes the 1.4 mm fwhm_y histogram for the currently plotted networks.
        #works in the same way as the array plot in that it assembles lists
        #for the networks picked by the checklist.
        @app.callback(
            Output(p1400_h4.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p1400_h4(obsnum,checklist):
            
            #Same as the array plot, but we only need one parameter.
            h = []            
            for i in range(len(checklist)):
                if int(checklist[i]) in [9,10,11,12]:
                    h.extend(ncobs.ncs[int(checklist[i])].p['fwhmy'])
                    
            figure={
                'data': [{
                'x': h,
                #'text': 0,
                'textposition': 'top',
                'customdata': 0,
                'type': 'histogram',
                'mode': 'markers+text',
                'marker': { 'color': '#229954', #set color equal to a variable
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                #'title': "1.4 mm Array",
                'xaxis': {'title': 'Y FWMH [px]'},
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
        
        
        #Makes the 1.4 mm freq histogram for the currently plotted networks.
        #works in the same way as the array plot in that it assembles lists
        #for the networks picked by the checklist.
        @app.callback(
            Output(p1400_h5.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p1400_h5(obsnum,checklist):
            
            #Same as the array plot, but we only need one parameter.
            h = []            
            for i in range(len(checklist)):
                if int(checklist[i]) in [9,10,11,12]:
                    h.extend(ncobs.ncs[int(checklist[i])].f)
                    
            figure={
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
                #'title': "1.4 mm Array",
                'xaxis': {'title': 'Frequency [MHz]'},
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
        
        
        #Make a tab for the 2.0 mm plots
        a2000_container = array_tab_container.child(dcc.Tab,label='2.0mm').child(dbc.Row)
        
        #A container for the 2.0 mm table
        t2000_container = a2000_container.child(dbc.Col,style={'margin':0})#.child(dbc.Jumbotron, [html.H1(children="Selected Detector")],style={"width": "25%"})
        
        #Make a tab for the 2.0 mm beammap and slice plots
        #a2000_b_tab = a2000_container.child(dbc.Col,style={'width':'40%', 'margin':0, 'textAlign': 'center'}).child(dcc.Tabs,vertical=True)
        
        a2000_b_tab = t2000_container.child(dbc.Tabs,style={'margin':0, 'textAlign': 'center'})
        
        #Tabs and graphs for the 2.0 mm beammap and slice plots
        #p2000_b = a2000_b_tab.child(dcc.Tab,label='beammap').child(dbc.Col).child(dcc.Graph)
        #p2000_b2 = a2000_b_tab.child(dcc.Tab,label='y-slice').child(dbc.Col).child(dcc.Graph)
        #p2000_b3 = a2000_b_tab.child(dcc.Tab,label='x-slice').child(dbc.Col).child(dcc.Graph)
        
        p2000_b = a2000_b_tab.child(dbc.Tab,label='beammap').child(dcc.Graph,figure = {
                'data': [{
                    #'x': list(range(ncobs.nrows)),
                    #'y': list(range(ncobs.ncols)),
                    'z': np.zeros([ncobs.nrows,ncobs.ncols]),
                    'textposition': 'top',
                    'customdata': 0,
                    'type': 'scatter',
                    'mode': 'markers+text',
                    'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
                    'colorscale':'Gray', # one of plotly colorscales
                    'colorbar': {'title': 'N/A'},
                    'unselected': {
                        'marker': { 'opacity': 0.3 },
                        'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                    }
                }],
                'layout': {
                    'width': b_width, 
                    'height': b_height,
                    #'title': "2.0 mm Array",
                    'xaxis': {'title': 'x (pixels)'},
                    'yaxis': {'title': 'y (pixels)'},
                    'dragmode': 'select',
                    'hovermode': True,
                }})
        p2000_b2 = a2000_b_tab.child(dbc.Tab,label='y slice').child(dcc.Graph, figure={'layout': {
                    'width': b_width, 
                    'height': b_height,
                    'xaxis': {'title': 'y (pixels)'},
                    'yaxis': {'title': 'S/N'},
                    'dragmode': 'select',
                    'hovermode': True,
                }})
        p2000_b3 = a2000_b_tab.child(dbc.Tab, label='x slice').child(dcc.Graph, figure={'layout': {
                    'width': b_width, 
                    'height': b_height,
                    #'title': "2.0 mm Array",
                    'xaxis': {'title': 'x (pixels)'},
                    'yaxis': {'title': 'S/N'},
                    'dragmode': 'select',
                    'hovermode': True,
                }})
        
        
        t2000 = t2000_container.child(dbc.Table,bordered=True, striped=True, hover=True,responsive=True,style={'width': '100%'})

        t2000.children = table_header + table_body

        #This creates a graph for the 2.0 mm array plot
        #p2000 = a2000_container.child(dbc.Col,style={'width':'50%', 'margin':0, 'textAlign': 'center'}).child(dcc.Graph)
        p2000_container = a2000_container.child(dbc.Col,style={'width': '100%', 'margin':0, 'textAlign': 'center'}).child(dbc.Card).child(dbc.CardBody,style={"marginRight": 75,"marginTop": 0, "marginBottom": 0,'width': '100%'}).child(dbc.CardHeader,html.H5("2.0mm Array"))
        

        #Inside of the 2.0 mm container, make a dropdown to control the
        #plotted axes for the array plot.  Separate for each array.   Defaults
        #to x, y
        drp_container = p2000_container.child(dbc.Row)
        drp_2000=drp_container.child(dbc.Col,style={'width':'10%', 'margin':0, 'textAlign': 'center'}).child(dcc.Dropdown, options=[
            {'label': 'S/N', 'value': 'amps'},
            {'label': 'x', 'value': 'x'},
            {'label': 'y', 'value': 'y'},
            {'label': 'fwhmx', 'value': 'fwhmx'},
            {'label': 'fwhmy', 'value': 'fwhmy'},
            {'label': 'Freq', 'value': 'f'}

        ],
        value='x',
        multi=False)
        
        drp_2000_2=drp_container.child(dbc.Col,style={'width':'10%', 'margin':0, 'textAlign': 'center'}).child(dcc.Dropdown, options=[
            {'label': 'S/N', 'value': 'amps'},
            {'label': 'x', 'value': 'x'},
            {'label': 'y', 'value': 'y'},
            {'label': 'fwhmx', 'value': 'fwhmx'},
            {'label': 'fwhmy', 'value': 'fwhmy'},
            {'label': 'Freq', 'value': 'f'}

        ],
        value='y',
        multi=False)
        
        drp_2000_3=drp_container.child(dbc.Col,style={'width':'10%', 'margin':0, 'textAlign': 'center'}).child(dcc.Dropdown, options=[
            {'label': 'NW', 'value': 'NW'},
            {'label': 'S/N', 'value': 'amps'},
            {'label': 'x', 'value': 'x'},
            {'label': 'y', 'value': 'y'},
            {'label': 'fwhmx', 'value': 'fwhmx'},
            {'label': 'fwhmy', 'value': 'fwhmy'},
            {'label': 'Freq', 'value': 'f'}
        ],
        value='NW',
        multi=False)
        
        p2000 = p2000_container.child(dcc.Graph, figure={
                'layout': {
                    #'title': "2.0 mm Array",
                    'autosize': False,
                    'automargin': False,
                    'xaxis': {'range': [0,max(ncobs.nrows,ncobs.ncols)]},
                    'yaxis': {'range': [0,max(ncobs.nrows,ncobs.ncols)]},
                    'width': a_width, 
                    'height': a_height,
                    'dragmode': 'select',
                    'hovermode': 'closest',
                    'editable': True,
                    'animate': True,
                    'clickmode': 'event+select',
                    'plot_bgcolor': "#F9F9F9",
                    'paper_bgcolor': "#F9F9F9",
                }})
        
        #Create a tab for the 2.0 mm histograms        
        a2000_h_tab = a2000_container.child(dbc.Row)
        
        #Set up each histogram for the 2.0 mm array.  Each is a new column.
        p2000_h0 = a2000_container.child(dbc.Row).child(dcc.Graph,align="center")
        p2000_h1 = a2000_container.child(dbc.Row).child(dcc.Graph,align="center")
        p2000_h2 = a2000_container.child(dbc.Row).child(dcc.Graph,align="center")
        p2000_h3 = a2000_container.child(dbc.Row).child(dcc.Graph,align="center")
        p2000_h4 = a2000_container.child(dbc.Row).child(dcc.Graph,align="center")
        p2000_h5 = a2000_container.child(dbc.Row).child(dcc.Graph,align="center")

        
        #This creates the array plot for the 2.0 mm array.  It takes the
        #dropdown and checklist as input.  From the checklist, it creates
        #lists of all selected parameters from the dropdown using only the
        #networks selected.  This then updates 'data' in the figure.
        @app.callback(
            Output(p2000.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(drp_2000.id, 'value'),
             Input(drp_2000_2.id, 'value'),
             Input(drp_2000_3.id, 'value'),
             Input(nw_checklist.id, 'value'),
             Input(p2000_h0.id,'selectedData'),
             Input(p2000_h1.id,'selectedData'),
             Input(p2000_h2.id,'selectedData'),
             Input(p2000_h3.id,'selectedData'),
             Input(p2000_h4.id,'selectedData'),
             Input(p2000_h5.id,'selectedData'),]) 
        def update_a2000(obsnum,value,value_2,value_3,checklist,clickData_h0,clickData_h1,clickData_h2,
                         clickData_h3,clickData_h4,clickData_h5):
            #Lists for the x,y, and color
            x = []
            y = []
            z = []
            
            p = {}
            f = []
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] = []
    
            #Fill up x,y, and z for all detectors in the given networks.
            for i in range(len(checklist)):
                if int(checklist[i]) in [2,3]:
                    for j in range(len(ncobs.pnames)):
                        p[ncobs.pnames[j]].extend(ncobs.ncs[int(checklist[i])].p[ncobs.pnames[j]])
                    f.extend(ncobs.ncs[int(checklist[i])].f)

                    #x.extend(ncobs.ncs[int(checklist[i])].p[value[0]])
                    #y.extend(ncobs.ncs[int(checklist[i])].p[value[1]])
                    
                    #if len(value)==2:
                    if value_3 == 'NW':
                        if value!='f':
                            z.extend(np.ones(len(ncobs.ncs[int(checklist[i])].p[value]))*int(checklist[i]))  
                        else:
                            z.extend(np.ones(len(ncobs.ncs[int(checklist[i])].f))*int(checklist[i]))
                         
                        ''' 
                           if value!='f':
                             char_arr = np.chararray(len(ncobs.ncs[int(checklist[i])].p[value]),unicode=True,itemsize=7)
                             char_arr[:] = colors[int(checklist[i])]
                        else:
                            char_arr = np.chararray(len(ncobs.ncs[int(checklist[i])].f),unicode=True,itemsize=7)
                            char_arr[:] = colors[int(checklist[i])]
                        z.extend(char_arr)
                        print(z)
                        '''
                        
            #if len(value) == 3:
            if value_3 !='NW':
                if value_3 !='f':
                    z = np.array(p[value_3])
                else:
                    z = np.array(f)
            #Each block from the try to the z = statement take the limits
            #from a histogram selection and limit the parameters to only
            #those points that fall into that selection.
            try:
                print('clickData',clickData_h0['range']['x'])
                amp_temp = np.array(p['amps'])
                ai = np.where((amp_temp>=clickData_h0['range']['x'][0]) & (amp_temp<=clickData_h0['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['amps'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            
            z = np.array(z)[ai]
            
            try:
                print('clickData',clickData_h1['range']['x'])
                amp_temp = np.array(p['x'])
                ai = np.where((amp_temp>=clickData_h1['range']['x'][0]) & (amp_temp<=clickData_h1['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['x'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            
            z = np.array(z)[ai]
            
            
            try:
                print('clickData',clickData_h2['range']['x'])
                amp_temp = np.array(p['y'])
                ai = np.where((amp_temp>=clickData_h2['range']['x'][0]) & (amp_temp<=clickData_h2['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['y'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            
            z = np.array(z)[ai]
            
        
            try:
                print('clickData',clickData_h3['range']['x'])
                amp_temp = np.array(p['fwhmx'])
                ai = np.where((amp_temp>=clickData_h3['range']['x'][0]) & (amp_temp<=clickData_h3['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmx'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            
            z = np.array(z)[ai]
            
            try:
                print('clickData',clickData_h4['range']['x'])
                amp_temp = np.array(p['fwhmy'])
                ai = np.where((amp_temp>=clickData_h4['range']['x'][0]) & (amp_temp<=clickData_h4['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmy'])))
                
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            
            z = np.array(z)[ai]
            
            
            try:
                print('clickData',clickData_h5['range']['x'])
                amp_temp = np.array(f)
                ai = np.where((amp_temp>=clickData_h5['range']['x'][0]) & (amp_temp<=clickData_h5['range']['x'][1]))[0]
            except:
                ai = list(range(len(f)))
                
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            f = np.array(f)[ai]
            
            z = np.array(z)[ai]
            
            if value!='f':
                x = np.array(p[value])
            else:
                x = np.array(f)
            if value_2!='f':
                y = np.array(p[value_2])
            else:
                y = np.array(f)
            
            if (str(value) == 'x') or (str(value) == 'y'):
                xmax = max(ncobs.nrows,ncobs.ncols)
                xmin = 0
            else:
                xmax = np.max(x)
                xmin = np.min(x)
                
            if (str(value_2) == 'x') or (str(value_2) == 'y'):
                ymax = max(ncobs.nrows,ncobs.ncols)
                ymin = 0
            else:
                ymax = np.max(y)
                ymin = np.min(y)
            
            if value=='amps':
                label='S/N'  
            elif value=='x':
                label='X Centroid [px]'
            elif value=='y':
                label='Y Centroid [px]'
            elif value=='fwhmx':
                label='X FWHM [px]'
            elif value=='fwhmy':
                label='Y FWHM [px]'
            elif value=='f':
                label='Freq [MHz]'
                
            if value_2=='amps':
                label_2='S/N'  
            elif value_2=='x':
                label_2='X Centroid [px]'
            elif value_2=='y':
                label_2='Y Centroid [px]'
            elif value_2=='fwhmx':
                label_2='X FWHM [px]'
            elif value_2=='fwhmy':
                label_2='Y FWHM [px]'
            elif value_2=='f':
                label_2='Freq [MHz]'
            
            if value_3=='amps':
                label_3='S/N'  
            elif value_3=='x':
                label_3='X Centroid [px]'
            elif value_3=='y':
                label_3='Y Centroid [px]'
            elif value_3=='fwhmx':
                label_3='X FWHM [px]'
            elif value_3=='fwhmy':
                label_3='Y FWHM [px]'
            elif value_3=='f':
                label_3='Freq [MHz]'
            elif value_3=='NW':
                label_3 = 'NW'
            
            #return dict with all plotting options
            figure={
                'data': [{
                    'x': x,
                    'y': y,
                    'textposition': 'top',
                    'customdata': 0,
                    'type': 'scatter',
                    'mode': 'markers',
                    'marker': {
                        'color': z,
                        #'colorscale':'Viridis', # one of plotly colorscales
                        'showscale': True,
                        'colorbar': {'title': label_3},

                        #'size': 5 
                    },
                    'unselected': {
                        'marker': { 'opacity': 1.0 },
                        'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                    },
                    'selected': {
                        'marker': {'color': 'red'},
                        'textfont': { 'color': 'rgba(1,0,0,0)' }
                    }
                }],
                'layout': {
                    #'title': "2.0 mm Array",
                    'autosize': False,
                    'automargin': False,
                    'xaxis': {'title': label,'range': [xmin,xmax]},
                    'yaxis': {'title': label_2,'range': [ymin,ymax]},
                    'width': a_width, 
                    'height': a_height,
                    'dragmode': 'select',
                    'hovermode': 'closest',
                    'editable': True,
                    'animate': True,
                    'clickmode': 'event+select',
                    'plot_bgcolor': "#F9F9F9",
                    'paper_bgcolor': "#F9F9F9",
                }}
            
            return figure
        
        
        #Creates the beammap plot for the 2.0 mm array.  It takes the
        #clickData from the 2.0 mm array plot as input along with the checklist
        #It then finds the corresponding detector by assembling network and
        #detector lists and plots that beammap.  The beammap matrix is assembled
        #at the start when ncobs is created.
        @app.callback(
            Output(p2000_b.id, 'figure'),
            [Input(p2000.id,'selectedData'),
             Input(nw_checklist.id, 'value'),
             Input(p2000_h0.id,'selectedData'),
             Input(p2000_h1.id,'selectedData'),
             Input(p2000_h2.id,'selectedData'),
             Input(p2000_h3.id,'selectedData'),
             Input(p2000_h4.id,'selectedData'),
             Input(p2000_h5.id,'selectedData'),])
        def update_b2000(clickData,checklist,clickData_h0,clickData_h1,clickData_h2,
                         clickData_h3,clickData_h4,clickData_h5):
            #try:
                #pointNumber is the number in a list of all plotted points of
                #the nearest data point to where was clicked
            try:
                pointNumber = clickData['points'][0]['pointNumber']
                print(pointNumber)
            except:
                pass
                
            p = {}
            f = []
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] = []
            
            #Fill up x,y, and z for all detectors in the given networks.
            for i in range(len(checklist)):
                if int(checklist[i]) in [2,3]:
                    for j in range(len(ncobs.pnames)):
                        p[ncobs.pnames[j]].extend(ncobs.ncs[int(checklist[i])].p[ncobs.pnames[j]])
                    f.extend(ncobs.ncs[int(checklist[i])].f)
                
                #Make lists of the networks and detector numbers so we can
                #find out what network pointNumber is in and what detector
                #number in that network it corresponds to.
                dets = []
                nws = []
                
                for i in range(len(checklist)):
                    ci = int(checklist[i])
                    if ci in [2,3]:
                        nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                        dets.extend(range(ncobs.ncs[ci].ndets))
             
                        
            print('nws',len(nws))

            try:
                print('clickData',clickData_h0['range']['x'])
                amp_temp = np.array(p['amps'])
                ai = np.where((amp_temp>=clickData_h0['range']['x'][0]) & (amp_temp<=clickData_h0['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['amps'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            try:
                print('clickData',clickData_h1['range']['x'])
                amp_temp = np.array(p['x'])
                ai = np.where((amp_temp>=clickData_h1['range']['x'][0]) & (amp_temp<=clickData_h1['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['x'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h2['range']['x'])
                amp_temp = np.array(p['y'])
                ai = np.where((amp_temp>=clickData_h2['range']['x'][0]) & (amp_temp<=clickData_h2['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['y'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

        
            try:
                print('clickData',clickData_h3['range']['x'])
                amp_temp = np.array(p['fwhmx'])
                ai = np.where((amp_temp>=clickData_h3['range']['x'][0]) & (amp_temp<=clickData_h3['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmx'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h4['range']['x'])
                amp_temp = np.array(p['fwhmy'])
                ai = np.where((amp_temp>=clickData_h4['range']['x'][0]) & (amp_temp<=clickData_h4['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmy'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            
            try:
                print('clickData',clickData_h5['range']['x'])
                amp_temp = np.array(f)
                ai = np.where((amp_temp>=clickData_h5['range']['x'][0]) & (amp_temp<=clickData_h5['range']['x'][1]))[0]
            except:
                ai = list(range(len(f)))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            nw = nws[pointNumber]
            det = dets[pointNumber]
                
            #Get the map for the corresponding detector that was clicked                
            z = ncobs.ncs[np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]].x_onoff[:,:,det]

            #return the figure dict
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
                    'colorscale':'Viridis', # one of plotly colorscales
                    'colorbar': {'title': 'S/N'},
                    'unselected': {
                        'marker': { 'opacity': 0.3 },
                        'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                    }
                }],
                'layout': {
                    'width': b_width, 
                    'height': b_height,
                    #'title': "2.0 mm Array",
                    'xaxis': {'title': 'x (pixels)'},
                    'yaxis': {'title': 'y (pixels)'},
                    'dragmode': 'select',
                    'hovermode': True,
                }}
            
            return figure
        
        
        #Creates the y slice plot for the 2.0 mm array.  It takes the
        #clickData from the 2.0 mm array plot as input along with the checklist
        #It then finds the corresponding detector by assembling network and
        #detector lists and plots that y slice.  The beammap matrix is assembled
        #at the start when ncobs is created.
        @app.callback(
            Output(p2000_b2.id, 'figure'),
            [Input(p2000.id,'selectedData'),
             Input(nw_checklist.id, 'value'),
             Input(p2000_h0.id,'selectedData'),
             Input(p2000_h1.id,'selectedData'),
             Input(p2000_h2.id,'selectedData'),
             Input(p2000_h3.id,'selectedData'),
             Input(p2000_h4.id,'selectedData'),
             Input(p2000_h5.id,'selectedData')])
        def update_b2000_2(clickData,checklist,clickData_h0,clickData_h1,clickData_h2,
                         clickData_h3,clickData_h4,clickData_h5):
                 #try:
                #pointNumber is the number in a list of all plotted points of
                #the nearest data point to where was clicked
            try:
                pointNumber = clickData['points'][0]['pointNumber']
                print(pointNumber)
            except:
                pass
                
            p = {}
            f = []
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] = []
            
            #Fill up x,y, and z for all detectors in the given networks.
            for i in range(len(checklist)):
                if int(checklist[i]) in [2,3]:
                    for j in range(len(ncobs.pnames)):
                        p[ncobs.pnames[j]].extend(ncobs.ncs[int(checklist[i])].p[ncobs.pnames[j]])
                    f.extend(ncobs.ncs[int(checklist[i])].f)
                
                #Make lists of the networks and detector numbers so we can
                #find out what network pointNumber is in and what detector
                #number in that network it corresponds to.
                dets = []
                nws = []
                
                for i in range(len(checklist)):
                    ci = int(checklist[i])
                    if ci in [2,3]:
                        nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                        dets.extend(range(ncobs.ncs[ci].ndets))
             
                        
            print('nws',len(nws))

            try:
                print('clickData',clickData_h0['range']['x'])
                amp_temp = np.array(p['amps'])
                ai = np.where((amp_temp>=clickData_h0['range']['x'][0]) & (amp_temp<=clickData_h0['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['amps'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            try:
                print('clickData',clickData_h1['range']['x'])
                amp_temp = np.array(p['x'])
                ai = np.where((amp_temp>=clickData_h1['range']['x'][0]) & (amp_temp<=clickData_h1['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['x'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h2['range']['x'])
                amp_temp = np.array(p['y'])
                ai = np.where((amp_temp>=clickData_h2['range']['x'][0]) & (amp_temp<=clickData_h2['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['y'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

        
            try:
                print('clickData',clickData_h3['range']['x'])
                amp_temp = np.array(p['fwhmx'])
                ai = np.where((amp_temp>=clickData_h3['range']['x'][0]) & (amp_temp<=clickData_h3['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmx'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h4['range']['x'])
                amp_temp = np.array(p['fwhmy'])
                ai = np.where((amp_temp>=clickData_h4['range']['x'][0]) & (amp_temp<=clickData_h4['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmy'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            
            try:
                print('clickData',clickData_h5['range']['x'])
                amp_temp = np.array(f)
                ai = np.where((amp_temp>=clickData_h5['range']['x'][0]) & (amp_temp<=clickData_h5['range']['x'][1]))[0]
            except:
                ai = list(range(len(f)))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            nw = nws[pointNumber]
            det = dets[pointNumber]
                
            #Get the map for the corresponding detector that was clicked                
            z = ncobs.ncs[np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]].x_onoff[:,:,det]
            
            nwi = np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]
            
            #To find which row and column to slice on, we find the x,y
            #centroid and round to the nearest element in the matrix.
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
                    'colorscale':'Viridis', # one of plotly colorscales
                    'unselected': {
                        'marker': { 'opacity': 0.3 },
                        'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                    }
                }],
                'layout': {
                    'width': b_width, 
                    'height': b_height,
                    #'title': "2.0 mm Array",
                    'xaxis': {'title': 'y (pixels)'},
                    'yaxis': {'title': 'S/N'},
                    'dragmode': 'select',
                    'hovermode': True,
                }}
            
            return figure
            
         
        #Creates the x slice plot for the 2.0 mm array.  It takes the
        #clickData from the 2.0 mm array plot as input along with the checklist
        #It then finds the corresponding detector by assembling network and
        #detector lists and plots that x slice.  The beammap matrix is assembled
        #at the start when ncobs is created.
        @app.callback(
            Output(p2000_b3.id, 'figure'),
            [Input(p2000.id,'selectedData'),
             Input(nw_checklist.id, 'value'),
             Input(p2000_h0.id,'selectedData'),
             Input(p2000_h1.id,'selectedData'),
             Input(p2000_h2.id,'selectedData'),
             Input(p2000_h3.id,'selectedData'),
             Input(p2000_h4.id,'selectedData'),
             Input(p2000_h5.id,'selectedData')])
        def update_b2000_3(clickData,checklist,clickData_h0,clickData_h1,clickData_h2,
                         clickData_h3,clickData_h4,clickData_h5):
                      #try:
                #pointNumber is the number in a list of all plotted points of
                #the nearest data point to where was clicked
            try:
                pointNumber = clickData['points'][0]['pointNumber']
                print(pointNumber)
            except:
                pass
                
            p = {}
            f = []
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] = []
            
            #Fill up x,y, and z for all detectors in the given networks.
            for i in range(len(checklist)):
                if int(checklist[i]) in [2,3]:
                    for j in range(len(ncobs.pnames)):
                        p[ncobs.pnames[j]].extend(ncobs.ncs[int(checklist[i])].p[ncobs.pnames[j]])
                    f.extend(ncobs.ncs[int(checklist[i])].f)
                
                #Make lists of the networks and detector numbers so we can
                #find out what network pointNumber is in and what detector
                #number in that network it corresponds to.
                dets = []
                nws = []
                
                for i in range(len(checklist)):
                    ci = int(checklist[i])
                    if ci in [2,3]:
                        nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                        dets.extend(range(ncobs.ncs[ci].ndets))
             
                        
            print('nws',len(nws))

            try:
                print('clickData',clickData_h0['range']['x'])
                amp_temp = np.array(p['amps'])
                ai = np.where((amp_temp>=clickData_h0['range']['x'][0]) & (amp_temp<=clickData_h0['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['amps'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            try:
                print('clickData',clickData_h1['range']['x'])
                amp_temp = np.array(p['x'])
                ai = np.where((amp_temp>=clickData_h1['range']['x'][0]) & (amp_temp<=clickData_h1['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['x'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h2['range']['x'])
                amp_temp = np.array(p['y'])
                ai = np.where((amp_temp>=clickData_h2['range']['x'][0]) & (amp_temp<=clickData_h2['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['y'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

        
            try:
                print('clickData',clickData_h3['range']['x'])
                amp_temp = np.array(p['fwhmx'])
                ai = np.where((amp_temp>=clickData_h3['range']['x'][0]) & (amp_temp<=clickData_h3['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmx'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h4['range']['x'])
                amp_temp = np.array(p['fwhmy'])
                ai = np.where((amp_temp>=clickData_h4['range']['x'][0]) & (amp_temp<=clickData_h4['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmy'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            
            try:
                print('clickData',clickData_h5['range']['x'])
                amp_temp = np.array(f)
                ai = np.where((amp_temp>=clickData_h5['range']['x'][0]) & (amp_temp<=clickData_h5['range']['x'][1]))[0]
            except:
                ai = list(range(len(f)))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            nw = nws[pointNumber]
            det = dets[pointNumber]
                
            #Get the map for the corresponding detector that was clicked                
            z = ncobs.ncs[np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]].x_onoff[:,:,det]
            
            nwi = np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]
                            
            #To find which row and column to slice on, we find the x,y
            #centroid and round to the nearest element in the matrix.
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
                    'colorscale':'Viridis', # one of plotly colorscales
                    'unselected': {
                        'marker': { 'opacity': 1.0 },
                        'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                    },
                    'selected': {'marker': {'color': 'red'}}
                }],
                'layout': {
                    'width': b_width, 
                    'height': b_height,
                    #'title': "2.0 mm Array",
                     'xaxis': {'title': 'x (pixels)'},
                     'yaxis': {'title': 'S/N'},
                    'dragmode': 'select',
                    'hovermode': True,
                }}
            
            return figure
            
        #This callback updates the table for a selected detector and is
        #triggered on the selection of new data in the 2.0 mm array plot.  It
        #pulls the parameters from the nocbs object.  The detector and
        #network are calculated in the same way as the beammap plot.
        @app.callback(
            Output(t2000.id, 'children'),
            [Input(p2000.id,'selectedData'),
             Input(nw_checklist.id, 'value'),
             Input(p2000_h0.id,'selectedData'),
             Input(p2000_h1.id,'selectedData'),
             Input(p2000_h2.id,'selectedData'),
             Input(p2000_h3.id,'selectedData'),
             Input(p2000_h4.id,'selectedData'),
             Input(p2000_h5.id,'selectedData')])
        def update_t2000(clickData,checklist,clickData_h0,clickData_h1,clickData_h2,
                         clickData_h3,clickData_h4,clickData_h5):
                      #try:
                #pointNumber is the number in a list of all plotted points of
                #the nearest data point to where was clicked
            try:
                pointNumber = clickData['points'][0]['pointNumber']
                print(pointNumber)
            except:
                pass
                
            p = {}
            f = []
            for i in range(len(ncobs.pnames)):
                p[ncobs.pnames[i]] = []
            
            #Fill up x,y, and z for all detectors in the given networks.
            for i in range(len(checklist)):
                if int(checklist[i]) in [2,3]:
                    for j in range(len(ncobs.pnames)):
                        p[ncobs.pnames[j]].extend(ncobs.ncs[int(checklist[i])].p[ncobs.pnames[j]])
                    f.extend(ncobs.ncs[int(checklist[i])].f)
                
                #Make lists of the networks and detector numbers so we can
                #find out what network pointNumber is in and what detector
                #number in that network it corresponds to.
                dets = []
                nws = []
                
                for i in range(len(checklist)):
                    ci = int(checklist[i])
                    if ci in [2,3]:
                        nws.extend(np.ones(ncobs.ncs[ci].ndets)*int(ncobs.nws[ci]))
                        dets.extend(range(ncobs.ncs[ci].ndets))
             
                        
            print('nws',len(nws))

            try:
                print('clickData',clickData_h0['range']['x'])
                amp_temp = np.array(p['amps'])
                ai = np.where((amp_temp>=clickData_h0['range']['x'][0]) & (amp_temp<=clickData_h0['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['amps'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            try:
                print('clickData',clickData_h1['range']['x'])
                amp_temp = np.array(p['x'])
                ai = np.where((amp_temp>=clickData_h1['range']['x'][0]) & (amp_temp<=clickData_h1['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['x'])))
                
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
            
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h2['range']['x'])
                amp_temp = np.array(p['y'])
                ai = np.where((amp_temp>=clickData_h2['range']['x'][0]) & (amp_temp<=clickData_h2['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['y'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]

            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

        
            try:
                print('clickData',clickData_h3['range']['x'])
                amp_temp = np.array(p['fwhmx'])
                ai = np.where((amp_temp>=clickData_h3['range']['x'][0]) & (amp_temp<=clickData_h3['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmx'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))

            
            try:
                print('clickData',clickData_h4['range']['x'])
                amp_temp = np.array(p['fwhmy'])
                ai = np.where((amp_temp>=clickData_h4['range']['x'][0]) & (amp_temp<=clickData_h4['range']['x'][1]))[0]
            except:
                ai = list(range(len(p['fwhmy'])))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]
            
            print('nws',len(nws))
            
            
            try:
                print('clickData',clickData_h5['range']['x'])
                amp_temp = np.array(f)
                ai = np.where((amp_temp>=clickData_h5['range']['x'][0]) & (amp_temp<=clickData_h5['range']['x'][1]))[0]
            except:
                ai = list(range(len(f)))
            
            for i in range(len(ncobs.pnames)):
                print(len(p[ncobs.pnames[i]]))
                p[ncobs.pnames[i]] =  np.array(p[ncobs.pnames[i]])[ai]
                
            f = np.array(f)[ai]
                
            dets = np.array(dets)[ai]
            nws = np.array(nws)[ai]

            nw = nws[pointNumber]
            det = dets[pointNumber]
            
            nwi = np.where(np.array(ncobs.nws) == str(int(nw)))[0][0]

            row0 = html.Tr([html.Td("S/N"), html.Td('%.2f' % (ncobs.ncs[nwi].p['amps'][det]))])
            row1 = html.Tr([html.Td("X Centroid [px]"), html.Td('%.2f' % (ncobs.ncs[nwi].p['x'][det]))])
            row2 = html.Tr([html.Td("Y Centroid [px]"), html.Td('%.2f' % (ncobs.ncs[nwi].p['y'][det]))])
            row3 = html.Tr([html.Td("X FWHM [px]"), html.Td('%.2f' % (ncobs.ncs[nwi].p['fwhmx'][det]))])
            row4 = html.Tr([html.Td("Y FWHM [px]"), html.Td('%.2f' % (ncobs.ncs[nwi].p['fwhmy'][det]))])
            row5 = html.Tr([html.Td("Frequency [MHz]"), html.Td('%.2f' % (ncobs.ncs[nwi].f[det]))])
            row6 = html.Tr([html.Td("Network"), html.Td(int(nw))])
            row7 = html.Tr([html.Td("Detector Number on Network"), html.Td(int(det))])

            table_body = [html.Tbody([row0,row1, row2, row3, row4,row5,row6,row7])]
            return table_header + table_body
             
           
        #Makes the 2.0 mm S/N histogram for the currently plotted networks.
        #works in the same way as the array plot in that it assembles lists
        #for the networks picked by the checklist.
        @app.callback(
            Output(p2000_h0.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p2000_h0(obsnum,checklist):
            
            #Same as the array plot, but we only need one parameter.
            h = []            
            for i in range(len(checklist)):
                if int(checklist[i]) in [2,3]:
                    h.extend(ncobs.ncs[int(checklist[i])].p['amps'])
                    
            figure={
                'data': [{
                'x': h,
                #'text': 0,
                'textposition': 'top',
                'customdata': 0,
                'type': 'histogram',
                'mode': 'markers+text',
                'marker': { 'color': '#C0392B',
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                #'title': "2.0 mm Array",
                'xaxis': {'title': 'S/N'},
                'yaxis': {'title': 'N'},
                'autosize': True,

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
        
        
        #Makes the 2.0 mm x centroid histogram for the currently plotted networks.
        #works in the same way as the array plot in that it assembles lists
        #for the networks picked by the checklist.
        @app.callback(
            Output(p2000_h1.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p2000_h1(obsnum,checklist):
            
            #Same as the array plot, but we only need one parameter.
            h = []            
            for i in range(len(checklist)):
                if int(checklist[i]) in [2,3]:
                    h.extend(ncobs.ncs[int(checklist[i])].p['x'])
                    
            figure={
                'data': [{
                'x': h,
                #'text': 0,
                'textposition': 'top',
                'customdata': 0,
                'type': 'histogram',
                'mode': 'markers+text',
                'marker': { 'color': '#884EA0', #set color equal to a variable
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                #'title': "2.0 mm Array",
                'xaxis': {'title': 'X Centroid [px]'},
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
        
        #Makes the 2.0 mm y centroid histogram for the currently plotted networks.
        #works in the same way as the array plot in that it assembles lists
        #for the networks picked by the checklist.
        @app.callback(
            Output(p2000_h2.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p2000_h2(obsnum,checklist):
            
            #Same as the array plot, but we only need one parameter.
            h = []            
            for i in range(len(checklist)):
                if int(checklist[i]) in [2,3]:
                    h.extend(ncobs.ncs[int(checklist[i])].p['y'])
                    
            figure={
                'data': [{
                'x': h,
                #'text': 0,
                'textposition': 'top',
                'customdata': 0,
                'type': 'histogram',
                'mode': 'markers+text',
                'marker': { 'color': '#2471A3', #set color equal to a variable
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                #'title': "2.0 mm Array",
                'xaxis': {'title': 'Y Centroid [px]'},
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
        
        
        #Makes the 2.0 mm fwhm_x histogram for the currently plotted networks.
        #works in the same way as the array plot in that it assembles lists
        #for the networks picked by the checklist.
        @app.callback(
            Output(p2000_h3.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p2000_h3(obsnum,checklist):
            
            #Same as the array plot, but we only need one parameter.
            h = []            
            for i in range(len(checklist)):
                if int(checklist[i]) in [2,3]:
                    h.extend(ncobs.ncs[int(checklist[i])].p['fwhmx'])
                    
            figure={
                'data': [{
                'x': h,
                #'text': 0,
                'textposition': 'top',
                'customdata': 0,
                'type': 'histogram',
                'mode': 'markers+text',
                'marker': { 'color': '#17A589', #set color equal to a variable
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                #'title': "2.0 mm Array",
                'xaxis': {'title': 'X FWHM [px]'},
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
        
        
        
        #Makes the 2.0 mm fwhm_y histogram for the currently plotted networks.
        #works in the same way as the array plot in that it assembles lists
        #for the networks picked by the checklist.
        @app.callback(
            Output(p2000_h4.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p2000_h4(obsnum,checklist):
            
            #Same as the array plot, but we only need one parameter.
            h = []            
            for i in range(len(checklist)):
                if int(checklist[i]) in [2,3]:
                    h.extend(ncobs.ncs[int(checklist[i])].p['fwhmy'])
                    
            figure={
                'data': [{
                'x': h,
                #'text': 0,
                'textposition': 'top',
                'customdata': 0,
                'type': 'histogram',
                'mode': 'markers+text',
                'marker': { 'color': '#229954', #set color equal to a variable
                'showscale': False,
                'size': 2 },
                'unselected': {
                    'marker': { 'opacity': 0.3 },
                    'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
                }
                }],
                'layout': {
                #'title': "2.0 mm Array",
                'xaxis': {'title': 'Y FWMH [px]'},
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
        
        
        #Makes the 2.0 mm freq histogram for the currently plotted networks.
        #works in the same way as the array plot in that it assembles lists
        #for the networks picked by the checklist.
        @app.callback(
            Output(p2000_h5.id,component_property='figure'),
            [Input(files.id, 'children'),
             Input(nw_checklist.id, 'value')]) 
        def update_p2000_h5(obsnum,checklist):
            
            #Same as the array plot, but we only need one parameter.
            h = []            
            for i in range(len(checklist)):
                if int(checklist[i]) in [2,3]:
                    h.extend(ncobs.ncs[int(checklist[i])].f)
                    
            figure={
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
                #'title': "2.0 mm Array",
                'xaxis': {'title': 'Frequency [MHz]'},
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
