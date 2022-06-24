#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:29:07 2022

@author: mmccrackan
"""

from dash_component_template import ComponentTemplate
from dash import html 
import dash_bootstrap_components as dbc
from dash import dcc
from dash.dependencies import Output, Input, State, MATCH, ALL
from dash import no_update, exceptions

from dash import dash_table

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from astropy.io.misc.yaml import load as yaml_load
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.modeling import models

import numpy as np
import functools
import copy
import sys
import copy

import pandas as pd
import glob

import netCDF4

class CitlaliData:
    def __init__(self):
        self.nws = []
        self.arrays = []
        self.path = ''
        self.param_names = ['Amplitude','X Centroid', 'Y Centroid', 'FWHM X',
                            'FWMH Y','Angle']
        
        self.data = {}
        
    def load_fits(self,path):
        files = np.sort(glob.glob(path + '/*beammap*.fits'))
        
        self.hdus = {}
        arrays = {'a1100': 0,
                  'a1400': 1,
                  'a2000': 2}
        
        for f in files:
            for arr in arrays.keys():
                if arr in f:
                    self.hdus[arrays[arr]] = fits.open(f)
                    
    def load_apt(self,path):
        print(path)
        files = np.sort(glob.glob(path + '/apt*.ecsv'))
        print(files)
        self.apt = Table.read(files[0])
        self.apt['uid'] = range(len(self.apt['array']))
        
        self.apt_colnames = ['uid','array','nw','amp','x_t','y_t','a_fwhm',
                             'b_fwhm','angle']#self.apt.colnames
        self.param_names = self.apt_colnames


class Plotter:
    def __init__(self):
        
        self.Data = None

        self.nws = np.array(range(13))
        self.param_names = ['X Centroid', 'Y Centroid', 'FWHM X',
                            'FWMH Y', 'S/N']
        self.a1100_nws = [0, 1, 2, 3, 4, 5, 6]
        self.a1400_nws = [7, 8, 9, 10]
        self.a2000_nws = [11, 12]
        
        self.apt_colnames = ['uid','array','nw','amp','x_t','y_t','a_fwhm',
                             'b_fwhm','angle']
        self.param_names = self.apt_colnames#['UID','Array','NW','Amplitude','x','y','a_fwhm',
                             #'b_fwhm','Theta']
        
        self.fwhm_to_std = 1/(2*np.sqrt(2*np.log(2)))
   
    def get_data(self, Data):
        self.Data = Data
        self.apt_colnames = self.Data.apt_colnames
        self.param_names = self.apt_colnames


    def update_scatter(self, obsnum, networks, 
                       x_axis, y_axis, z_axis, 
                       s_axis, #a1100_scale, a1400_scale, a2000_scale, options,
                       h0_SD, h1_SD, h2_SD, h3_SD, h4_SD, h5_SD,clickedPoint):
        
        layout = go.Layout(
            margin=go.layout.Margin(
                l=0, #left margin
                r=0, #right margin
                b=0, #bottom margin
                t=0, #top margin
                ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)')

        fig = go.Figure(layout=layout)

        hn_SD = [h0_SD, h1_SD, h2_SD, h3_SD, h4_SD, h5_SD]

        hn_x_lower = []
        hn_x_upper = []

        for i in range(len(hn_SD)):
            if hn_SD[i] is not None:
                hn_x_lower.append(hn_SD[i]['range']['x'][0])
                hn_x_upper.append(hn_SD[i]['range']['x'][1])
            else:
                hn_x_lower.append(None)
                hn_x_upper.append(None)

        axes = [x_axis, y_axis, z_axis]
        #scales = [a1100_scale, a1400_scale, a2000_scale]
        
        print(self.Data)

        if (obsnum is None) or (None in axes) or (self.Data is None):
            return no_update

        '''for i in range(len(scales)):
            if scales[i] == None:
                scales[i] = 1.0

        a1100_scale = float(scales[0])
        a1400_scale = float(scales[1])
        a2000_scale = float(scales[2])
        '''

        if not networks:
            return [fig,]
        #else:
            #obsnum = int(obsnum)
        data_to_plot = {}
        for p in self.apt_colnames:
            data_to_plot[p] = np.array([])
            for nw in networks:
                d = np.array(self.Data.apt[self.Data.apt['nw']==nw][p])
                data_to_plot[p] = np.concatenate((data_to_plot[p],d))
        
        #scale = np.ones(len(data_to_plot['array']))
        
        '''if (x_axis == 'x_t' or y_axis == 'x_t' or x_axis == 'y_t' or y_axis == 'y_t'):
            arr = np.where(self.Data.apt['array'] == 0)[0]
            scale[arr] = a1100_scale
            
            arr = np.where(self.Data.apt['array'] == 1)[0]
            scale[arr] = a1400_scale

            arr = np.where(self.Data.apt['array'] == 2)[0]
            scale[arr] = a2000_scale
            '''
            
        j = 0
        for i in range(len(self.apt_colnames)):
            if self.apt_colnames[i] != 'uid' and self.apt_colnames[i] != 'array' and self.apt_colnames[i] != 'nw':
                if hn_x_lower[j] != None and hn_x_upper[j] != None:
                    d = np.array(data_to_plot[self.apt_colnames[i]])
                    for k in range(len(self.apt_colnames)):
                        data_to_plot[self.apt_colnames[k]] = data_to_plot[self.apt_colnames[k]][(d > hn_x_lower[j]) & (d < hn_x_upper[j])]
                j = j + 1

        '''if 'Subtract Mean' in options:
            data_to_plot['x_t'] = np.array(scale)*(data_to_plot['x_t'] - np.mean(data_to_plot['x_t']))
            data_to_plot['y_t'] = np.array(scale)*(data_to_plot['y_t'] - np.mean(data_to_plot['y_t']))
        else:
            data_to_plot['x_t'] = np.array(scale)*(data_to_plot['x_t'])
            data_to_plot['y_t'] = np.array(scale)*(data_to_plot['y_t'])
            '''
            
        try:
            point_number = int(clickedPoint['points'][0]['pointNumber'])
            point_x = float(clickedPoint['points'][0]['x'])
            point_y = float(clickedPoint['points'][0]['y'])
            det = 0
            for i in range(len(self.Data.apt[x_axis])):
                if self.Data.apt[x_axis][i] == point_x and self.Data.apt[y_axis][i] == point_y:
                    det = i
        except:
            det = None
        
        if z_axis == 'nw':# or z_axis == 'array':
            colorscale = 'Jet'
        else:
            colorscale = 'Jet'
                            
        if s_axis == 'N/A':
            fig.add_trace(go.Scatter(x=data_to_plot[x_axis],
                                     y=data_to_plot[y_axis],
                                     mode='markers',
                                     #name='array',
                                     marker=dict(
                                         color=data_to_plot[z_axis],
                                         colorbar=dict(title=z_axis,exponentformat='e'),
                                         colorscale=colorscale),
                                     selected={
                                         'marker': {
                                             'color': 'black',
                                             #'symbol': 'X'
                                             }}))
        else:
            fig.add_trace(go.Scatter(x=data_to_plot[x_axis],
                                     y=data_to_plot[y_axis],
                                     mode='markers',
                                     #name='array',
                                     marker=dict(
                                         size=10*data_to_plot[s_axis]/np.max(data_to_plot[s_axis]),
                                         color=data_to_plot[z_axis],
                                         colorbar=dict(title=z_axis,exponentformat='e'),
                                         colorscale=colorscale),
                                     selected={
                                         'marker': {
                                             'color': 'black', 
                                             #'symbol': 'X'
                                             }}))
        if (x_axis == 'x_t' or x_axis=='y_t') and (y_axis == 'x_t' or y_axis=='y_t'):
            fig['layout']['yaxis']['scaleanchor']='x'
            
        if det != None:
            fig.add_hline(y=self.Data.apt[y_axis][det], line_width=2, line_dash="dash", line_color="black")
            fig.add_vline(x=self.Data.apt[x_axis][det], line_width=2, line_dash="dash", line_color="black")

        fig.update_coloraxes(colorbar=dict(lenmode='fraction',len=0.1))

        fig.update_layout(showlegend=False,
                          xaxis=dict(title=x_axis),
                          yaxis=dict(title=y_axis),
                          clickmode='event+select',
                          margin=go.layout.Margin(
                                    l=0, #left margin
                                    r=0, #right margin
                                    b=0, #bottom margin
                                    t=0), #top margin
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)')


        return [fig,]

    def update_beammap(self, obsnum, networks, x_axis, y_axis,
                       clickedPoint, maptype, h0_SD,
                       h1_SD, h2_SD, h3_SD, h4_SD, h5_SD):
        
        layout = go.Layout(
            margin=go.layout.Margin(
                l=0, #left margin
                r=0, #right margin
                b=0, #bottom margin
                t=0, #top margin
                ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)')
        
        try:
            point_number = int(clickedPoint['points'][0]['pointNumber'])
            point_x = float(clickedPoint['points'][0]['x'])
            point_y = float(clickedPoint['points'][0]['y'])

        except:
            return no_update

        hn_SD = [h0_SD, h1_SD, h2_SD, h3_SD, h4_SD, h5_SD]

        hn_x_lower = []
        hn_x_upper = []

        for i in range(len(hn_SD)):
            if hn_SD[i] is not None:
                hn_x_lower.append(hn_SD[i]['range']['x'][0])
                hn_x_upper.append(hn_SD[i]['range']['x'][1])
            else:
                hn_x_lower.append(None)
                hn_x_upper.append(None)

        if (obsnum is None) or (self.Data is None):
            return no_update

        if not networks:
            return [go.Figure(layout=layout),]

        else:
            #obsnum = int(obsnum)
            data_to_plot = {}                
            for p in self.apt_colnames:
                data_to_plot[p] = np.array([])
                for nw in networks:
                    d = np.array(self.Data.apt[self.Data.apt['nw']==nw][p])
                    data_to_plot[p] = np.concatenate((data_to_plot[p],d))
    
            j = 0
            for i in range(len(self.apt_colnames)):
                if self.apt_colnames[i] != 'uid' and self.apt_colnames[i] != 'array' and self.apt_colnames[i] != 'nw':
                    if hn_x_lower[j] != None and hn_x_upper[j] != None:
                        d = np.array(data_to_plot[self.apt_colnames[i]])
                        for k in range(len(self.apt_colnames)):
                            data_to_plot[self.apt_colnames[k]] = data_to_plot[self.apt_colnames[k]][(d > hn_x_lower[j]) & (d < hn_x_upper[j])]
                    j = j + 1

            det = 0
            for i in range(len(self.Data.apt['x_t'])):
                if self.Data.apt[x_axis][i] == point_x and self.Data.apt[y_axis][i] == point_y:
                    det = i

            arr = self.Data.apt['array'][det]
            uid = self.Data.apt['uid'][det]
            
            gauss2d = models.Gaussian2D(amplitude=self.Data.apt['amp'][det],
                                 x_mean=self.Data.apt['x_t'][det],
                                 y_mean=self.Data.apt['y_t'][det],
                                 x_stddev=self.Data.apt['a_fwhm'][det]*self.fwhm_to_std,
                                 y_stddev=self.Data.apt['b_fwhm'][det]*self.fwhm_to_std,
                                 theta=self.Data.apt['angle'][det])
            
            if (maptype == 'signal') or (maptype == 'weight'):
                map_to_plot = self.Data.hdus[arr][maptype+'_'+str(int(uid))+'_I'].data[0,0,:,:]
                wcs = WCS(self.Data.hdus[arr][maptype+'_'+str(int(uid))+'_I'].header).sub(2)
            elif maptype == 'sig2noise':
                map_to_plot = self.Data.hdus[arr]['signal'+'_'+str(int(uid))+'_I'].data[0,0,:,:]
                map_to_plot = map_to_plot*np.sqrt(self.Data.hdus[arr]['weight'+'_'+str(int(uid))+'_I'].data[0,0,:,:])
                wcs = WCS(self.Data.hdus[arr]['signal'+'_'+str(int(uid))+'_I'].header).sub(2)

             
            map_name = maptype+'_'+str(int(uid))+'_I'
   
            crpix1, crpix2 = wcs.wcs.crpix
            ny,nx = map_to_plot.shape
                
            x = np.linspace(-nx/2, nx/2, nx)
            y = np.linspace(-ny/2, ny/2, ny)
            
            xx,yy = np.meshgrid(x,y)
            g = gauss2d(xx,yy)

            fig = px.imshow(img=map_to_plot,origin='lower', 
                            x=x,
                            y=y,
                            color_continuous_scale="turbid",title=map_name)
            fig.update_layout(title=dict(y=0.81,x=0.47))
            #fig.add_hline(y=self.Data.apt['y_t'][det], line_width=1, line_dash="dash", line_color="black")
            #fig.add_vline(x=self.Data.apt['x_t'][det], line_width=1, line_dash="dash", line_color="black")
            
            fig.update_coloraxes(colorbar=dict(lenmode='fraction',len=0.5,exponentformat='e'))

            #fig.add_trace(go.Contour(z=g,contours_coloring='lines', x=x,
             #                        y=y,colorscale='gray',showscale=False))
            fig.update_layout(xaxis=dict(title='x'),
                              yaxis=dict(title='y'),
                              margin=go.layout.Margin(
                                    l=0, #left margin
                                    r=0, #right margin
                                    b=0, #bottom margin
                                    t=0, #top margin
                                    ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)')

            return [fig,]

    def update_hist(self, obsnum, networks):

        figs = []
        j = 0
        for i in range(len(self.apt_colnames)):
            if self.apt_colnames[i] != 'uid' and self.apt_colnames[i] != 'array' and self.apt_colnames[i] != 'nw':
                figs.append(go.Figure())
                figs[-1].update_layout(margin=go.layout.Margin(
                                        l=0, #left margin
                                        r=0, #right margin
                                        b=50, #bottom margin
                                        t=50, #top margin
                                        ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)')
                j = j + 1

        if (obsnum is None) or (self.Data is None):
            return no_update

        if not networks:
            return figs
        else:
            #obsnum = int(obsnum)
            data_to_plot = {}
            for p in self.apt_colnames:
                data_to_plot[p] = np.array([])
                for nw in networks:
                    d = np.array(self.Data.apt[self.Data.apt['nw']==nw][p])
                    data_to_plot[p] = np.concatenate((data_to_plot[p],d))
        j = 0
        for i in range(len(self.apt_colnames)):
            if self.apt_colnames[i] != 'uid' and self.apt_colnames[i] != 'array' and self.apt_colnames[i] != 'nw':
                figs[j].add_trace(go.Histogram(x=data_to_plot[self.apt_colnames[i]][data_to_plot['array']==0],histnorm='percent'))
                figs[j].add_trace(go.Histogram(x=data_to_plot[self.apt_colnames[i]][data_to_plot['array']==1],histnorm='percent'))
                figs[j].add_trace(go.Histogram(x=data_to_plot[self.apt_colnames[i]][data_to_plot['array']==2],histnorm='percent'))
                figs[j].update_layout(autosize=True, showlegend=False)
                figs[j].update_xaxes(title_text=self.param_names[i])
                figs[0].update_yaxes(title_text='Percent')
                figs[j].update_layout(margin=go.layout.Margin(
                                        l=0, #left margin
                                        r=0, #right margin
                                        b=50, #bottom margin
                                        t=50, #top margin
                                        ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)')
                j = j + 1

        return figs
    
    
    
    def update_scatter_3D(self, obsnum, networks, 
                       x_axis, y_axis, z_axis, 
                       s_axis, #a1100_scale, a1400_scale, a2000_scale, options,
                       h0_SD, h1_SD, h2_SD, h3_SD, h4_SD, h5_SD,clickedPoint):
        
        layout = go.Layout(
            margin=go.layout.Margin(
                l=0, #left margin
                r=0, #right margin
                b=0, #bottom margin
                t=0, #top margin
                ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)')
    
        fig = go.Figure(layout=layout)
    
        hn_SD = [h0_SD, h1_SD, h2_SD, h3_SD, h4_SD, h5_SD]
    
        hn_x_lower = []
        hn_x_upper = []
    
        for i in range(len(hn_SD)):
            if hn_SD[i] is not None:
                hn_x_lower.append(hn_SD[i]['range']['x'][0])
                hn_x_upper.append(hn_SD[i]['range']['x'][1])
            else:
                hn_x_lower.append(None)
                hn_x_upper.append(None)
    
        axes = [x_axis, y_axis, z_axis]
        #scales = [a1100_scale, a1400_scale, a2000_scale]
        
        print(self.Data)
    
        if (obsnum is None) or (None in axes) or (self.Data is None):
            return no_update
    
        '''for i in range(len(scales)):
            if scales[i] == None:
                scales[i] = 1.0
    
        a1100_scale = float(scales[0])
        a1400_scale = float(scales[1])
        a2000_scale = float(scales[2])
        '''
    
        if not networks:
            return [fig,]
        #else:
            #obsnum = int(obsnum)
        data_to_plot = {}
        for p in self.apt_colnames:
            data_to_plot[p] = np.array([])
            for nw in networks:
                d = np.array(self.Data.apt[self.Data.apt['nw']==nw][p])
                data_to_plot[p] = np.concatenate((data_to_plot[p],d))
        
        #scale = np.ones(len(data_to_plot['array']))
        
        '''if (x_axis == 'x_t' or y_axis == 'x_t' or x_axis == 'y_t' or y_axis == 'y_t'):
            arr = np.where(self.Data.apt['array'] == 0)[0]
            scale[arr] = a1100_scale
            
            arr = np.where(self.Data.apt['array'] == 1)[0]
            scale[arr] = a1400_scale
    
            arr = np.where(self.Data.apt['array'] == 2)[0]
            scale[arr] = a2000_scale
            '''
            
        j = 0
        for i in range(len(self.apt_colnames)):
            if self.apt_colnames[i] != 'uid' and self.apt_colnames[i] != 'array' and self.apt_colnames[i] != 'nw':
                if hn_x_lower[j] != None and hn_x_upper[j] != None:
                    d = np.array(data_to_plot[self.apt_colnames[i]])
                    for k in range(len(self.apt_colnames)):
                        data_to_plot[self.apt_colnames[k]] = data_to_plot[self.apt_colnames[k]][(d > hn_x_lower[j]) & (d < hn_x_upper[j])]
                j = j + 1
    
        '''if 'Subtract Mean' in options:
            data_to_plot['x_t'] = np.array(scale)*(data_to_plot['x_t'] - np.mean(data_to_plot['x_t']))
            data_to_plot['y_t'] = np.array(scale)*(data_to_plot['y_t'] - np.mean(data_to_plot['y_t']))
        else:
            data_to_plot['x_t'] = np.array(scale)*(data_to_plot['x_t'])
            data_to_plot['y_t'] = np.array(scale)*(data_to_plot['y_t'])
            '''
            
        try:
            point_number = int(clickedPoint['points'][0]['pointNumber'])
            point_x = float(clickedPoint['points'][0]['x'])
            point_y = float(clickedPoint['points'][0]['y'])
            det = 0
            for i in range(len(self.Data.apt[x_axis])):
                if self.Data.apt[x_axis][i] == point_x and self.Data.apt[y_axis][i] == point_y:
                    det = i
        except:
            det = None
        
        if z_axis == 'nw':# or z_axis == 'array':
            colorscale = 'Jet'
        else:
            colorscale = 'Jet'
                            
        if s_axis == 'N/A':
            fig.add_trace(go.Scatter3d(x=data_to_plot[x_axis],
                                     y=data_to_plot[y_axis],
                                     z = data_to_plot[z_axis],
                                     mode='markers',
                                     #name='array',
                                     #marker=dict(
                                         #color=data_to_plot[z_axis],
                                         #colorbar=dict(title=z_axis,exponentformat='e'),
                                         #colorscale=colorscale
                                      #   ),
                                     #selected={
                                      #   'marker': {
                                       #      'color': 'black',
                                             #'symbol': 'X'
                                             #}}
                                       ))
        else:
            fig.add_trace(go.Scatter3d(x=data_to_plot[x_axis],
                                     y=data_to_plot[y_axis],
                                     z = data_to_plot[z_axis],
                                     mode='markers',
                                     #name='array',
                                     marker=dict(
                                         size=10*data_to_plot[s_axis]/np.max(data_to_plot[s_axis]),
                                         #color=data_to_plot[z_axis],
                                         #colorbar=dict(title=z_axis,exponentformat='e'),
                                         #colorscale=colorscale
                                         ),
                                     selected={
                                         'marker': {
                                             'color': 'black', 
                                             #'symbol': 'X'
                                             }}))
        if (x_axis == 'x_t' or x_axis=='y_t') and (y_axis == 'x_t' or y_axis=='y_t'):
            fig['layout']['yaxis']['scaleanchor']='x'
            
        if det != None:
            fig.add_hline(y=self.Data.apt[y_axis][det], line_width=2, line_dash="dash", line_color="black")
            fig.add_vline(x=self.Data.apt[x_axis][det], line_width=2, line_dash="dash", line_color="black")
    
        fig.update_coloraxes(colorbar=dict(lenmode='fraction',len=0.1))
    
        fig.update_layout(showlegend=False,
                          xaxis=dict(title=x_axis),
                          yaxis=dict(title=y_axis),
                          clickmode='event+select',
                          margin=go.layout.Margin(
                                    l=0, #left margin
                                    r=0, #right margin
                                    b=0, #bottom margin
                                    t=0), #top margin
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)')
    
    
        return [fig,]