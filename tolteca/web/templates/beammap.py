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

class obs:
    
    def _init_(self):
        pass
    
    def setup(self,obsnum,nrows,ncols,path,sampfreq,order,transpose,files=None):
        self.obsnum = str(obsnum)
        self.nrows = nrows
        self.ncols = ncols
        self.path = path
        self.sampfreq = sampfreq
        self.obsnum = obsnum
        self.files = files
        
        self.a1100_nws = np.array([0,1,2,3,4,5,6])
        self.a1400_nws = np.array([7,8,9,10])
        
        self.pnames = ['x','y','fwhmx','fwhmy','amps','snr']
        self.cs = [0,1,2,3,4,5,6,7,8,9,10,11]

        self.get_obs_data(order,transpose)
        
    def get_obs_data(self,order,transpose):
        if self.files == None:
            self.beammap_files = np.sort(glob.glob(self.path+str(self.obsnum)+'/*.nc'))
        else:
            self.beammap_files = self.files
        self.raw_files = np.sort(glob.glob(self.path[:-6]+'/data/'+str(self.obsnum)+'/*processed.nc'))
        
        print('Getting data from %i files for obsnum %s' %(len(self.beammap_files),self.obsnum))

        self.ncs = []
        self.nws = []
        
        self.ndets_1100 = []
        self.ndets_1400 = []

        self.tdets_1100 = 0
        self.tdets_1400 = 0

        for f in self.beammap_files:
            nw = re.findall(r'\d+', f)   
            self.nws.append(nw[-1])
            print('on nw ' + self.nws[-1])
            #dx,dy,df = self.get_design(int(self.nws[-1]))
            nc = ncdata(f,self.obsnum,self.nrows,self.ncols,self.nws[-1],self.path,self.sampfreq,order,transpose)
            self.ncs.append(nc)
            if self.nws[-1] in str(self.a1100_nws):
                self.tdets_1100 = self.tdets_1100 + self.ncs[-1].ndets
                self.ndets_1100.append(self.ncs[-1].ndets)

            elif self.nws[-1] in str(self.a1400_nws):
                self.tdets_1400 = self.tdets_1400 + self.ncs[-1].ndets
                self.ndets_1400.append(self.ncs[-1].ndets)

            
        self.p1100 = {}
        self.p1400 = {}
        
        print(self.nws)
        
        for i in range(len(self.pnames)):
            self.p1100[self.pnames[i]] = np.zeros(self.tdets_1100)
            self.p1400[self.pnames[i]] = np.zeros(self.tdets_1400)
            
        print(self.tdets_1100,self.tdets_1400)
        
        m = 0
        n = 0
        for i in range(len(self.nws)):
            if self.nws[i] in str(self.a1100_nws):
                print(self.nws[i])
                for k in range(self.ncs[i].ndets):
                    for j in range(len(self.pnames)):
                        self.p1100[self.pnames[j]][m] = self.ncs[i].p[self.pnames[j]][k]
                    m = m + 1
                        
            elif self.nws[i] in str(self.a1400_nws):
                    for k in range(self.ncs[i].ndets):
                        for j in range(len(self.pnames)):
                            self.p1400[self.pnames[j]][n] = self.ncs[i].p[self.pnames[j]][k]
                        n = n + 1
                        
        self.a1100_maps = np.zeros([self.nrows*self.ncols,self.tdets_1100])
        self.a1400_maps = np.zeros([self.nrows*self.ncols,self.tdets_1400])

        self.cs_1100 = np.zeros(self.tdets_1100)
        self.cs_1400 = np.zeros(self.tdets_1400)
        
        self.nws_1100 = np.zeros(self.tdets_1100)
        self.nws_1400 = np.zeros(self.tdets_1400)
        
        self.detn_1100 = np.zeros(self.tdets_1100)
        self.detn_1400 = np.zeros(self.tdets_1400)

        m = 0
        n = 0
        for i in range(len(self.nws)):
            if self.nws[i] in str(self.a1100_nws):
                for j in range(self.ncs[i].ndets):
                  self.a1100_maps[:,m] = self.ncs[i].ncfile['x_onoff'][:,j]
                  self.cs_1100[m] = self.cs[int(self.nws[i])]
                  self.nws_1100[m] = int(self.nws[i])
                  self.detn_1100[m] = j
                  m = m + 1
            elif self.nws[i] in str(self.a1400_nws):
                for j in range(self.ncs[i].ndets):
                  self.a1400_maps[:,n] = self.ncs[i].ncfile['x_onoff'][:,j]
                  self.cs_1400[n] = self.cs[int(self.nws[i])]
                  self.nws_1400[n] = int(self.nws[i])
                  self.detn_1400[n] = j
                  n = n + 1
        
             
class ncdata:
    def __init__(self, ncfile_name,obsnum,nrows,ncols,nw,path,sampfreq,order,transpose):
        self.ncfile_name = ncfile_name
        self.obsnum = str(obsnum)
        self.nrows = nrows
        self.ncols = ncols
        self.nw = nw
        self.path = path
        self.sampfreq = sampfreq
        self.dx = 0#x
        self.dy = 0#dy
        self.df = 0#df
        
        self.beammap_files = np.sort(glob.glob(self.path+str(obsnum)+'/*toltec'+nw+'.nc'))
        self.raw_files = np.sort(glob.glob(self.path[:-6]+'/data/'+str(obsnum)+'/toltec'+nw+'*.nc'))

        self.pnames = ['x','y','fwhmx','fwhmy','amps','snr']
        self.nc_pnames = ["amplitude", "FWHM_x", "FWHM_y", "offset_x", "offset_y"] #, "bolo_name"]
        
        self.map_names = ['x_onoff', 'r_onoff', 'x_off', 'r_off', 'x_on', 'r_on', 'xmap', 'rmap']
        
        self.get_nc_data(order,transpose)
        
    def get_nc_data(self,order,transpose):
        self.ncfile = netCDF4.Dataset(self.ncfile_name)
        
        self.ndets = len(self.ncfile.dimensions['ndet'])
        
        self.indices = list(range(self.ndets))
        self.bad_indices = []

        self.get_params()
        self.get_f()
    
    def get_params(self):
        self.p = {}
        
        for i in range(len(self.pnames)):
            self.p[self.pnames[i]] = np.ones(self.ndets)*-99
        
        for i in range(self.ndets):
            map_fit = 'map_fits'+str(i)
            self.p['x'][i] = self.ncfile[map_fit].getncattr('offset_x')
            self.p['y'][i] = self.ncfile[map_fit].getncattr('offset_y')
            self.p['fwhmx'][i] = self.ncfile[map_fit].getncattr('FWHM_x')
            self.p['fwhmy'][i] = self.ncfile[map_fit].getncattr('FWHM_y')
            self.p['amps'][i] = self.ncfile[map_fit].getncattr('amplitude')
                        
    def get_f(self):
        try:
            nc_f = netCDF4.Dataset(self.raw_files[0])
            self.f = np.array(nc_f['Header.Toltec.ToneFreq'][:]) + np.array(nc_f['Header.Toltec.LoFreq'])
            self.f = self.f[0]/10**6
        except:
            self.f = np.zeros(self.ndets)
        
        try:            
            self.f = (self.ncfile['tone_freq'][:] +  self.ncfile['LoFreq'])/10**6.
        except:
            self.f = np.zeros(self.ndets)


class beammap(ComponentTemplate):
    _component_cls = dbc.Container
    fluid = True

    def setup_layout(self, app):
        title = self.title_text
        #header = self.child(dbc.Row).child(dbc.Col).child(dbc.Jumbotron,style={'color':'g'},fluid=True,background=None)
        body = self.child(dbc.Row).child(dbc.Col)

        '''header.children = [
                html.H1(f'{title}'),
                html.P(
                    'This is a subtitle'),
                
                ]
        '''
        

        
        button_container = body.child(dbc.Row)
        '''
        obsnum_input = button_container.child(dbc.Col).child(
                dcc.Input,
                placeholder="Enter Obsnum: ",
                type='number',
                value=0)
        '''
        
        path_input = button_container.child(dbc.Col).child(
                dcc.Input,
                placeholder="Enter File Path: ",
                type='string',
                value='/Users/mmccrackan/toltec/data/tests/wyatt/coadd_telecon2/')
        
        
        nw_checklist = button_container.child(dbc.Col).child(dcc.Checklist,
    options=[
        {'label': 'network 0', 'value': '0'},
        {'label': 'network 1', 'value': '1'},
        {'label': 'network 2', 'value': '2'},
        {'label': 'network 3', 'value': '3'},
        {'label': 'network 4', 'value': '4'},
        {'label': 'network 5', 'value': '5'},
        {'label': 'network 6', 'value': '6'},
        {'label': 'network 7', 'value': '7'},
        {'label': 'network 8', 'value': '8'},
        {'label': 'network 9', 'value': '9'},
        {'label': 'network 10', 'value': '10'},
        {'label': 'network 11', 'value': '11'},

    ],
    value=[],
    labelStyle={'display': 'inline-block'}
)  
                
        '''upload = button_container.child(dbc.Col).child(dcc.Upload,children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),style={
            'width': '70%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '1px'
        },
        # Allow multiple files to be uploaded
        multiple=True)
        '''
        
            
        ticker_container = button_container.child(dbc.Col).child(html.Div, className='d-flex')
        ticker_container.child(
                dbc.Label("Files Found:", className='mr-2'))
        ticker = ticker_container.child(html.Div, 'N/A')
        
        @app.callback(Output(ticker.id, 'children'),
              #[Input(upload.id, 'contents'),
              [Input(path_input.id,'value'),
              Input(nw_checklist.id,'value')])#,
              #[State(upload.id, 'filename'),
               #State(upload.id, 'last_modified')])
        #def update_output(list_of_contents, path, value, list_of_names, list_of_dates):
            
            
        def update_output(path, value):
            #if value == []:
            file_list = glob.glob(path + '/*.nc')
            file_list_short = []

            nrows = 21
            ncols = 25
            sf = 488.281/4
            obsnum = 'none'
            
            beammap_files = []
            
            for i in range(len(file_list)):
                file_list_short.append(file_list[i][len(path):] + ', ')
                nw = re.findall(r'\d+', file_list[i])
                print('nw',nw)
                if nw[-1] in value:
                    beammap_files.append(file_list[i])
                
            ncobs.setup(obsnum,nrows,ncols,path,sf,order='C',transpose=False,files=beammap_files)
            
            return np.sort(file_list_short)
        
        
        
        @app.callback(Output(nw_checklist.id, 'value'),
                      [Input(path_input.id,'value')])
        def clear_checklist(path):
            return []


            '''
            try:
                beammap_files = []
                for i in range(len(list_of_names)):
                    beammap_files.append(path+list_of_names[i])
                    
                print(beammap_files)
                
                
                file_list = glob.glob(path + '/*')
                #beammap_files = []
                

                nrows = 21
                ncols = 25
                sf = 488.281/4
                obsnum = 'none'
                ncobs.setup(obsnum,nrows,ncols,path,sf,order='C',transpose=False,files=beammap_files)
                
                return list_of_names
            except:
                print('No files found')
            
        '''
        table_header = [html.Thead(html.Tr([html.Th("Parameter"), html.Th("Value")]))]
        
        a_width=600
        a_height=600
        
        b_width=400
        b_height=400
        
        hist_width = 500 
        hist_height = 400

        plot_tab_container = body.child(dbc.Row).child(dcc.Tabs,vertical=True)
        p1100_container = plot_tab_container.child(dcc.Tab,label='1.1mm').child(dbc.Row)
        
        p1100 = p1100_container.child(dbc.Col).child(dcc.Graph,align="center",figure={
        'layout': {
            'width': a_width, 
            'height': a_height,
            #'margin': {'l': 0, 'r': 0, 'b': 15, 't': 5},
            'dragmode': 'select',
            'hovermode': 'closest',
            'editable': True,
            'animate': True,
            'clickmode': 'event+select'
            # Display a rectangle to highlight the previously selected region
            }})
        
    
        
        drp_1100=p1100_container.child(dbc.Col).child(dcc.Dropdown,
        options=[
            {'label': 'S/N', 'value': 'amps'},
            {'label': 'x', 'value': 'x'},
            {'label': 'y', 'value': 'y'},
            {'label': 'fwhmx', 'value': 'fwhmx'},
            {'label': 'fwhmy', 'value': 'fwhmy'}
        ],
        value=['x','y'],
        multi=True
    )
    
        
        p1100_bmap_tab = p1100_container.child(dcc.Tabs)
        
        p1100_bmap = p1100_bmap_tab.child(dcc.Tab,label='beammap').child(dbc.Col).child(dcc.Graph)
        p1100_bmap2 = p1100_bmap_tab.child(dcc.Tab,label='y-slice').child(dbc.Col).child(dcc.Graph)
        p1100_bmap3 = p1100_bmap_tab.child(dcc.Tab,label='x-slice').child(dbc.Col).child(dcc.Graph)
        
        t1100 = p1100_container.child(dbc.Table, bordered=True,
dark=False, hover=True, responsive=True,striped=True,width=100)

        
        p1100_hists_tab = p1100_container.child(dbc.Row)
        p1100_hist0 = p1100_hists_tab.child(dbc.Row).child(dbc.Col).child(dcc.Graph,align="center")
        p1100_hist1 = p1100_hists_tab.child(dbc.Col).child(dcc.Graph,align="center")
        p1100_hist2 = p1100_hists_tab.child(dbc.Col).child(dcc.Graph,align="center")
        p1100_hist3 = p1100_hists_tab.child(dbc.Col).child(dcc.Graph,align="center")
        p1100_hist4 = p1100_hists_tab.child(dbc.Col).child(dcc.Graph,align="center")

                      
        @app.callback(
            Output(p1100.id,component_property='figure'),
            [Input(ticker.id, 'children'),
             Input(drp_1100.id, 'value')]
            ) 
        def update_a1100(obsnum,value):
            if len(value) == 3:
                color_arr = ncobs.p1100[value[2]]
            else:
                color_arr = ncobs.cs_1100            
            figure={
            'data': [{
            'x': ncobs.p1100[value[0]],
            'y': ncobs.p1100[value[1]],
            #'text': 0,
            'textposition': 'top',
            'customdata': 0,
            'type': 'scatter',
            'mode': 'markers+text',
            #'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
            'marker': { 'color': color_arr, #set color equal to a variable
            'colorscale':'Viridis', # one of plotly colorscales
            'showscale': True,
            'size': 5 },
            'unselected': {
                'marker': { 'opacity': 0.3 },
                # make text transparent when not selected
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
            # Display a rectangle to highlight the previously selected region
            }}
            
            return figure
        
            
        @app.callback(
            Output(p1100_bmap.id, 'figure'),
            [Input(p1100.id,'selectedData')])
        def update_b1100(clickData):
            pointNumber = clickData['points'][0]['pointNumber']
    
            bmap = ncobs.a1100_maps[:,pointNumber]
            bmap = np.reshape(bmap,(ncobs.nrows,ncobs.ncols),order='C')
            bmap[::2,:]= np.flip(bmap[::2,:],axis=1)
        
            return {
        'data': [{
            'x': list(range(ncobs.nrows)),
            'y': list(range(ncobs.ncols)),
            'z': bmap,
            #'text': 0,
            'textposition': 'top',
            'customdata': 0,
            'type': 'heatmap',
            'mode': 'markers+text',
            'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
            'colorscale':'tropic', # one of plotly colorscales
            'unselected': {
                'marker': { 'opacity': 0.3 },
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
            }
        }],
        'layout': {
            'width': b_width, 
            'height': b_height,
            'title': "1.1 mm Array",
            'xaxis': {'title': 'x (pixels)'},
            'yaxis': {'title': 'y (pixels)'},
            #'autosize': True,
            #'margin': {'l': 700, 'r': 50, 'b': 15, 't': 5},
            'dragmode': 'select',
            'hovermode': True,
            # Display a rectangle to highlight the previously selected region
            }}
        
        
        
        @app.callback(
            Output(p1100_bmap2.id, 'figure'),
            [Input(p1100.id,'selectedData')])
        def update_b1100_2(clickData):
            pointNumber = clickData['points'][0]['pointNumber']
    
            bmap = ncobs.a1100_maps[:,pointNumber]
            bmap = np.reshape(bmap,(ncobs.nrows,ncobs.ncols),order='C')
            bmap[::2,:]= np.flip(bmap[::2,:],axis=1)
        
            row = int(np.round(ncobs.p1100['x'][pointNumber]))
            col = int(np.round(ncobs.p1100['y'][pointNumber]))
            
            print(row,col,0,0,0,0)

        
            return {
        'data': [{
            'x': list(range(ncobs.nrows)),
            'y': bmap[:,row],
            'z': bmap,
            #'text': 0,
            'textposition': 'top',
            'customdata': 0,
            'type': 'line',
            'mode': 'lines+text',
            'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
            'unselected': {
                'marker': { 'opacity': 0.3 },
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
            }
        }],
        'layout': {
            'width': b_width, 
            'height': b_height,
            'title': "1.1 mm Array",
            'xaxis': {'title': 'y (pixels)'},
            'yaxis': {'title': 'S/N'},
            #'autosize': True,
            #'margin': {'l': 700, 'r': 50, 'b': 15, 't': 5},
            'dragmode': 'select',
            'hovermode': True,
            # Display a rectangle to highlight the previously selected region
            }}
        
        
        @app.callback(
            Output(p1100_bmap3.id, 'figure'),
            [Input(p1100.id,'selectedData')])
        def update_b1100_3(clickData):
            pointNumber = clickData['points'][0]['pointNumber']
    
            bmap = ncobs.a1100_maps[:,pointNumber]
            bmap = np.reshape(bmap,(ncobs.nrows,ncobs.ncols),order='C')
            bmap[::2,:]= np.flip(bmap[::2,:],axis=1)
        
            row = int(np.round(ncobs.p1100['x'][pointNumber]))
            col = int(np.round(ncobs.p1100['y'][pointNumber]))
            
            print(row,col,0,0,0,0)

        
            return {
        'data': [{
            'x': list(range(ncobs.ncols)),
            'y': bmap[col,:],
            'z': bmap,
            #'text': 0,
            'textposition': 'top',
            'customdata': 0,
            'type': 'line',
            'mode': 'lines+text',
            'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
            'unselected': {
                'marker': { 'opacity': 0.3 },
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
            }
        }],
        'layout': {
            'width': b_width, 
            'height': b_height,
            'title': "1.1 mm Array",
            'xaxis': {'title': 'x (pixels)'},
            'yaxis': {'title': 'S/N'},
            #'autosize': True,
            #'margin': {'l': 700, 'r': 50, 'b': 15, 't': 5},
            'dragmode': 'select',
            'hovermode': True,
            # Display a rectangle to highlight the previously selected region
            }}
        
        
        @app.callback(
            Output(t1100.id, 'children'),
            [Input(p1100.id,'selectedData')])
        def update_t1100(clickData):
            det = clickData['points'][0]['pointNumber']

            row0 = html.Tr([html.Td("S/N"), html.Td(ncobs.p1100['amps'][det])])
            row1 = html.Tr([html.Td("x"), html.Td(ncobs.p1100['x'][det])])
            row2 = html.Tr([html.Td("y"), html.Td(ncobs.p1100['y'][det])])
            row3 = html.Tr([html.Td("fwhm_x"), html.Td(ncobs.p1100['fwhmx'][det])])
            row4 = html.Tr([html.Td("fwhm_y"), html.Td(ncobs.p1100['fwhmy'][det])])
            row5 = html.Tr([html.Td("f"), html.Td('N/A')])
            row6 = html.Tr([html.Td("nw"), html.Td(ncobs.nws_1100[det])])
            row7 = html.Tr([html.Td("det"), html.Td(ncobs.detn_1100[det])])

            table_body = [html.Tbody([row0,row1, row2, row3, row4,row5,row6,row7])]
            return table_header + table_body
        
        
        @app.callback(
            Output(p1100_hist0.id,component_property='figure'),
            [Input(ticker.id, 'children')]
            ) 
        def update_p1100_hist0(obsnum):
            figure={
            'data': [{
            'x': ncobs.p1100['amps'],
            'y': ncobs.p1100['y'],
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
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
            }
            }],
            'layout': {
            'title': "1.1 mm Array",
            'xaxis': {'title': 'S/N'},
            'yaxis': {'title': 'N'},
            'width': hist_width, 
            'height': hist_height,
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
            Output(p1100_hist1.id,component_property='figure'),
            [Input(ticker.id, 'children')]
            ) 
        def update_p1100_hist1(obsnum):
            figure={
            'data': [{
            'x': ncobs.p1100['x'],
            'y': ncobs.p1100['y'],
            #'text': 0,
            'textposition': 'top',
            'customdata': 0,
            'type': 'histogram',
            'mode': 'markers+text',
            #'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
            'marker': { 'color': 'g', #set color equal to a variable
            'colorscale':'Viridis', # one of plotly colorscales
            'showscale': False,
            'size': 2 },
            'unselected': {
                'marker': { 'opacity': 0.3 },
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
            }
            }],
            'layout': {
            'title': "1.1 mm Array",
            'xaxis': {'title': 'x'},
            'yaxis': {'title': 'N'},
            'width': hist_width, 
            'height': hist_height,
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
            Output(p1100_hist2.id,component_property='figure'),
            [Input(ticker.id, 'children')]
            ) 
        def update_p1100_hist2(obsnum):
            figure={
            'data': [{
            'x': ncobs.p1100['y'],
            'y': ncobs.p1100['y'],
            #'text': 0,
            'textposition': 'top',
            'customdata': 0,
            'type': 'histogram',
            'mode': 'markers+text',
            #'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
            'marker': { 'color': 'g', #set color equal to a variable
            'colorscale':'Viridis', # one of plotly colorscales
            'showscale': False,
            'size': 2 },
            'unselected': {
                'marker': { 'opacity': 0.3 },
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
            }
            }],
            'layout': {
            'title': "1.1 mm Array",
            'xaxis': {'title': 'y'},
            'yaxis': {'title': 'N'},
            'width': hist_width, 
            'height': hist_height,
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
            Output(p1100_hist3.id,component_property='figure'),
            [Input(ticker.id, 'children')]
            ) 
        def update_p1100_hist3(obsnum):
            figure={
            'data': [{
            'x': ncobs.p1100['fwhmx'],
            'y': ncobs.p1100['y'],
            #'text': 0,
            'textposition': 'top',
            'customdata': 0,
            'type': 'histogram',
            'mode': 'markers+text',
            #'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
            'marker': { 'color': 'g', #set color equal to a variable
            'colorscale':'Viridis', # one of plotly colorscales
            'showscale': False,
            'size': 2 },
            'unselected': {
                'marker': { 'opacity': 0.3 },
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
            }
            }],
            'layout': {
            'title': "1.1 mm Array",
            'xaxis': {'title': 'fwhmx'},
            'yaxis': {'title': 'N'},
            'width': hist_width, 
            'height': hist_height,
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
            Output(p1100_hist4.id,component_property='figure'),
            [Input(ticker.id, 'children')]
            ) 
        def update_p1100_hist4(obsnum):
            figure={
            'data': [{
            'x': ncobs.p1100['fwhmy'],
            'y': ncobs.p1100['y'],
            #'text': 0,
            'textposition': 'top',
            'customdata': 0,
            'type': 'histogram',
            'mode': 'markers+text',
            #'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
            'marker': { 'color': 'g', #set color equal to a variable
            'colorscale':'Viridis', # one of plotly colorscales
            'showscale': False,
            'size': 2 },
            'unselected': {
                'marker': { 'opacity': 0.3 },
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
            }
            }],
            'layout': {
            'title': "1.1 mm Array",
            'xaxis': {'title': 'fwhmy'},
            'yaxis': {'title': 'N'},
            'width': hist_width, 
            'height': hist_height,
            #'margin': {'l': 0, 'r': 0, 'b': 15, 't': 5},
            'dragmode': 'select',
            'hovermode': 'closest',
            'editable': True,
            'animate': True,
            'clickmode': 'event+select'
            # Display a rectangle to highlight the previously selected region
            }}
            
            return figure
        
        

        '''-----------------------------------------------------------------'''
        p1400_container = plot_tab_container.child(dcc.Tab,label='1.4mm').child(dbc.Row)
        p1400 = p1400_container.child(dbc.Col).child(dcc.Graph,figure={
        'layout': {
            'width': a_width, 
            'height': a_height,
            #'margin': {'l': 0, 'r': 0, 'b': 15, 't': 5},
            'dragmode': 'select',
            'hovermode': 'closest',
            'editable': True,
            'animate': True,
            'clickmode': 'event+select'
            # Display a rectangle to highlight the previously selected region
            }})
        
        
        drp_1400=p1400_container.child(dbc.Col).child(dcc.Dropdown,
        options=[
            {'label': 'S/N', 'value': 'amps'},
            {'label': 'x', 'value': 'x'},
            {'label': 'y', 'value': 'y'},
            {'label': 'fwhmx', 'value': 'fwhmx'},
            {'label': 'fwhmy', 'value': 'fwhmy'}
        ],
        value=['x','y'],
        multi=True
    )
        
        p1400_bmap_tab = p1400_container.child(dcc.Tabs)
        
        p1400_bmap = p1400_bmap_tab.child(dcc.Tab,label='beammap').child(dbc.Col).child(dcc.Graph)
        p1400_bmap2 = p1400_bmap_tab.child(dcc.Tab,label='y-slice').child(dbc.Col).child(dcc.Graph)
        p1400_bmap3 = p1400_bmap_tab.child(dcc.Tab,label='x-slice').child(dbc.Col).child(dcc.Graph)
        
        t1400 = p1400_container.child(dbc.Table, bordered=True,
dark=False, hover=True, responsive=True,striped=True,width=100)

        
        p1400_hists_tab = p1400_container.child(dbc.Row)
        p1400_hist0 = p1400_hists_tab.child(dbc.Row).child(dbc.Col).child(dcc.Graph)
        p1400_hist1 = p1400_hists_tab.child(dbc.Row).child(dbc.Col).child(dcc.Graph)
        p1400_hist2 = p1400_hists_tab.child(dbc.Row).child(dbc.Col).child(dcc.Graph)
        p1400_hist3 = p1400_hists_tab.child(dbc.Row).child(dbc.Col).child(dcc.Graph)
        p1400_hist4 = p1400_hists_tab.child(dbc.Row).child(dbc.Col).child(dcc.Graph)

                      
        @app.callback(
            Output(p1400.id,component_property='figure'),
            [Input(ticker.id, 'children'),
             Input(drp_1400.id,'value')]
            ) 
        def update_a1400(obsnum,value):
            if len(value) == 3:
                color_arr = ncobs.p1400[value[2]]
            else:
                color_arr = ncobs.cs_1400
            figure={
            'data': [{
            'x': ncobs.p1400[value[0]],
            'y': ncobs.p1400[value[1]],
            #'text': 0,
            'textposition': 'top',
            'customdata': 0,
            'type': 'scatter',
            'mode': 'markers+text',
            #'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
            'marker': { 'color': color_arr, #set color equal to a variable
            'colorscale':'Viridis', # one of plotly colorscales
            'showscale': True,
            'size': 5 },
            'unselected': {
                'marker': { 'opacity': 0.3 },
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
            }
            }],
            'layout': {
            'title': "1.4 mm Array",
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
            # Display a rectangle to highlight the previously selected region
            }}
            
            return figure
        
            
        @app.callback(
            Output(p1400_bmap.id, 'figure'),
            [Input(p1400.id,'selectedData')])
        def update_b1400(clickData):
            pointNumber = clickData['points'][0]['pointNumber']
    
            bmap = ncobs.a1400_maps[:,pointNumber]
            bmap = np.reshape(bmap,(ncobs.nrows,ncobs.ncols),order='C')
            bmap[::2,:]= np.flip(bmap[::2,:],axis=1)
        
            return {
        'data': [{
            'x': list(range(ncobs.nrows)),
            'y': list(range(ncobs.ncols)),
            'z': bmap,
            #'text': 0,
            'textposition': 'top',
            'customdata': 0,
            'type': 'heatmap',
            'mode': 'markers+text',
            'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
            'unselected': {
                'marker': { 'opacity': 0.3 },
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
            }
        }],
        'layout': {
            'width': b_width, 
            'height': b_height,
            'title': "1.4 mm Array",
            'xaxis': {'title': 'x (pixels)'},
            'yaxis': {'title': 'y (pixels)'},
            #'autosize': True,
            #'margin': {'l': 700, 'r': 50, 'b': 15, 't': 5},
            'dragmode': 'select',
            'hovermode': True,
            # Display a rectangle to highlight the previously selected region
            }}
        
        
        
        @app.callback(
            Output(p1400_bmap2.id, 'figure'),
            [Input(p1400.id,'selectedData')])
        def update_b1400_2(clickData):
            pointNumber = clickData['points'][0]['pointNumber']
    
            bmap = ncobs.a1400_maps[:,pointNumber]
            bmap = np.reshape(bmap,(ncobs.nrows,ncobs.ncols),order='C')
            bmap[::2,:]= np.flip(bmap[::2,:],axis=1)
        
            row = int(np.round(ncobs.p1400['x'][pointNumber]))
            col = int(np.round(ncobs.p1400['y'][pointNumber]))
            
            print(row,col,0,0,0,0)

        
            return {
        'data': [{
            'x': list(range(ncobs.nrows)),
            'y': bmap[:,row],
            'z': bmap,
            #'text': 0,
            'textposition': 'top',
            'customdata': 0,
            'type': 'line',
            'mode': 'lines+text',
            'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
            'unselected': {
                'marker': { 'opacity': 0.3 },
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
            }
        }],
        'layout': {
            'width': b_width, 
            'height': b_height,
            'title': "1.4 mm Array",
            'xaxis': {'title': 'y (pixels)'},
            'yaxis': {'title': 'S/N'},
            #'autosize': True,
            #'margin': {'l': 700, 'r': 50, 'b': 15, 't': 5},
            'dragmode': 'select',
            'hovermode': True,
            # Display a rectangle to highlight the previously selected region
            }}
        
        
        @app.callback(
            Output(p1400_bmap3.id, 'figure'),
            [Input(p1400.id,'selectedData')])
        def update_b1400_3(clickData):
            pointNumber = clickData['points'][0]['pointNumber']
    
            bmap = ncobs.a1400_maps[:,pointNumber]
            bmap = np.reshape(bmap,(ncobs.nrows,ncobs.ncols),order='C')
            bmap[::2,:]= np.flip(bmap[::2,:],axis=1)
        
            row = int(np.round(ncobs.p1400['x'][pointNumber]))
            col = int(np.round(ncobs.p1400['y'][pointNumber]))
            
            print(row,col,0,0,0,0)

        
            return {
        'data': [{
            'x': list(range(ncobs.ncols)),
            'y': bmap[col,:],
            'z': bmap,
            #'text': 0,
            'textposition': 'top',
            'customdata': 0,
            'type': 'line',
            'mode': 'lines+text',
            'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
            'unselected': {
                'marker': { 'opacity': 0.3 },
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
            }
        }],
        'layout': {
            'width': b_width, 
            'height': b_height,
            'title': "1.4 mm Array",
            'xaxis': {'title': 'x (pixels)'},
            'yaxis': {'title': 'S/N'},
            #'autosize': True,
            #'margin': {'l': 700, 'r': 50, 'b': 15, 't': 5},
            'dragmode': 'select',
            'hovermode': True,
            # Display a rectangle to highlight the previously selected region
            }}
        
        
        @app.callback(
            Output(t1400.id, 'children'),
            [Input(p1400.id,'selectedData')])
        def update_t1400(clickData):
            det = clickData['points'][0]['pointNumber']

            row0 = html.Tr([html.Td("S/N"), html.Td(ncobs.p1400['amps'][det])])
            row1 = html.Tr([html.Td("x"), html.Td(ncobs.p1400['x'][det])])
            row2 = html.Tr([html.Td("y"), html.Td(ncobs.p1400['y'][det])])
            row3 = html.Tr([html.Td("fwhm_x"), html.Td(ncobs.p1400['fwhmx'][det])])
            row4 = html.Tr([html.Td("fwhm_y"), html.Td(ncobs.p1400['fwhmy'][det])])
            row5 = html.Tr([html.Td("f"), html.Td('N/A')])
            row6 = html.Tr([html.Td("nw"), html.Td(ncobs.nws_1400[det])])
            row7 = html.Tr([html.Td("det"), html.Td(ncobs.detn_1400[det])])

            table_body = [html.Tbody([row0,row1, row2, row3, row4,row5,row6,row7])]
            return table_header + table_body
        
        
        @app.callback(
            Output(p1400_hist0.id,component_property='figure'),
            [Input(ticker.id, 'children')]
            ) 
        def update_p1400_hist0(obsnum):
            figure={
            'data': [{
            'x': ncobs.p1400['amps'],
            'y': ncobs.p1400['y'],
            #'text': 0,
            'textposition': 'top',
            'customdata': 0,
            'type': 'histogram',
            'mode': 'markers+text',
            #'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
            'marker': { 'color': 'g', #set color equal to a variable
            'colorscale':'Viridis', # one of plotly colorscales
            'showscale': False,
            'size': 2 },
            'unselected': {
                'marker': { 'opacity': 0.3 },
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
            }
            }],
            'layout': {
            'title': "1.4 mm Array",
            'xaxis': {'title': 'S/N'},
            'yaxis': {'title': 'N'},
            'width': hist_width, 
            'height': hist_height,
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
            Output(p1400_hist1.id,component_property='figure'),
            [Input(ticker.id, 'children')]
            ) 
        def update_p1400_hist1(obsnum):
            figure={
            'data': [{
            'x': ncobs.p1400['x'],
            'y': ncobs.p1400['y'],
            #'text': 0,
            'textposition': 'top',
            'customdata': 0,
            'type': 'histogram',
            'mode': 'markers+text',
            #'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
            'marker': { 'color': 'g', #set color equal to a variable
            'colorscale':'Viridis', # one of plotly colorscales
            'showscale': False,
            'size': 2 },
            'unselected': {
                'marker': { 'opacity': 0.3 },
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
            }
            }],
            'layout': {
            'title': "1.4 mm Array",
            'xaxis': {'title': 'x'},
            'yaxis': {'title': 'N'},
            'width': hist_width, 
            'height': hist_height,
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
            Output(p1400_hist2.id,component_property='figure'),
            [Input(ticker.id, 'children')]
            ) 
        def update_p1400_hist2(obsnum):
            figure={
            'data': [{
            'x': ncobs.p1400['y'],
            'y': ncobs.p1400['y'],
            #'text': 0,
            'textposition': 'top',
            'customdata': 0,
            'type': 'histogram',
            'mode': 'markers+text',
            #'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
            'marker': { 'color': 'g', #set color equal to a variable
            'colorscale':'Viridis', # one of plotly colorscales
            'showscale': False,
            'size': 2 },
            'unselected': {
                'marker': { 'opacity': 0.3 },
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
            }
            }],
            'layout': {
            'title': "1.4 mm Array",
            'xaxis': {'title': 'y'},
            'yaxis': {'title': 'N'},
            'width': hist_width, 
            'height': hist_height,
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
            Output(p1400_hist3.id,component_property='figure'),
            [Input(ticker.id, 'children')]
            ) 
        def update_p1400_hist3(obsnum):
            figure={
            'data': [{
            'x': ncobs.p1400['fwhmx'],
            'y': ncobs.p1400['y'],
            #'text': 0,
            'textposition': 'top',
            'customdata': 0,
            'type': 'histogram',
            'mode': 'markers+text',
            #'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
            'marker': { 'color': 'g', #set color equal to a variable
            'colorscale':'Viridis', # one of plotly colorscales
            'showscale': False,
            'size': 2 },
            'unselected': {
                'marker': { 'opacity': 0.3 },
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
            }
            }],
            'layout': {
            'title': "1.4 mm Array",
            'xaxis': {'title': 'fwhmx'},
            'yaxis': {'title': 'N'},
            'width': hist_width, 
            'height': hist_height,
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
            Output(p1400_hist4.id,component_property='figure'),
            [Input(ticker.id, 'children')]
            ) 
        def update_p1400_hist4(obsnum):
            figure={
            'data': [{
            'x': ncobs.p1400['fwhmy'],
            'y': ncobs.p1400['y'],
            #'text': 0,
            'textposition': 'top',
            'customdata': 0,
            'type': 'histogram',
            'mode': 'markers+text',
            #'marker': { 'color': 'rgba(0, 116, 217, 0.7)', 'size': 5 },
            'marker': { 'color': 'g', #set color equal to a variable
            'colorscale':'Viridis', # one of plotly colorscales
            'showscale': False,
            'size': 2 },
            'unselected': {
                'marker': { 'opacity': 0.3 },
                # make text transparent when not selected
                'textfont': { 'color': 'rgba(0, 0, 0, 0)' }
            }
            }],
            'layout': {
            'title': "1.4 mm Array",
            'xaxis': {'title': 'fwhmy'},
            'yaxis': {'title': 'N'},
            'width': hist_width, 
            'height': hist_height,
            #'margin': {'l': 0, 'r': 0, 'b': 15, 't': 5},
            'dragmode': 'select',
            'hovermode': 'closest',
            'editable': True,
            'animate': True,
            'clickmode': 'event+select'
            # Display a rectangle to highlight the previously selected region
            }}
            
            return figure        
        
        
        
        
        p2000_container = plot_tab_container.child(dcc.Tab,label='2.0mm').child(dbc.Row,label='2.0mm')


       
        
       
        ncobs = obs()

        

extensions = [
    {
        'module': 'dasha.web.extensions.dasha',
        'config': {
            'template': beammap,
            'title_text': ' Beammap Dasha Page',
            'color': 'blue'
            }
        },
    ]
