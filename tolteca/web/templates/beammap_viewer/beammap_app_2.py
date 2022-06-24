#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:38:12 2022

@author: mmccrackan
"""

from dash_component_template import ComponentTemplate
#from dasha.web.templates.common import (
#    CollapseContent, LabeledDropdown, ButtonListPager)
from dash import html 
import dash_bootstrap_components as dbc
from dash import dcc
from dash.dependencies import Output, Input, State, MATCH, ALL
from dash import no_update, exceptions

from dash import dash_table

import plotly.express as px
import plotly.graph_objects as go

from astropy.io.misc.yaml import load as yaml_load

import numpy as np
import functools
import copy
import sys
import os
import glob

import json

import tollan.utils.fmt as t_fmt

sys.path.append('/Users/mmccrackan/toltec/python/wyatt/beammap_dasha/')
from beammap_plot_2 import Plotter, CitlaliData

class beammap(ComponentTemplate):
    fluid = True
    
    class Meta:
        component_cls = dbc.Container
        
    def __init__(self, title_text, *args, **kwargs):
        self.selectedObsList = []
        super().__init__(*args, **kwargs)
        self.title_text = title_text
        
    def setup_layout(self, app):
        self.fluid = True
        
        def make_tooltip(target,msg,placement):
            return dbc.Tooltip(
                msg,
                target=target,
                placement=placement)
        
        interval = self.child(dcc.Interval,
            interval=5*1000, # in milliseconds
            n_intervals=0)
                
        # class that holds plotting functions
        BP = Plotter()
        
        config_file = '/Users/mmccrackan/toltec/python/wyatt/beammap_dasha/config.yaml'
        
        input_files = []

        with open(config_file, 'r') as fo:
            indexfile = yaml_load(fo)
            data_dir = indexfile['inputs'][0]['path']
                    
        sub_dirs = [x[0] for x in os.walk(data_dir)]
            
        for i in range(1,len(sub_dirs)):
            fits_files = glob.glob(sub_dirs[i] +'/*.fits')
            if fits_files:
                input_files.append({'label': sub_dirs[i],#.rsplit('/', 1)[-1], 
                                    'value': sub_dirs[i]})
    
        # header for title
        header = self.child(dbc.Row)
        title = header.child(dbc.Col).child(html.H2,"Citlali Beammap Viewer")
        
        '''from astropy.io import fits
        f = fits.open('/Users/mmccrackan/toltec/temp/polarization/point_source/toast/redu_toast/redu00/000001/toltec_simu_a1100_pointing_000001.fits')
        
        hd = {}
        for i in range(len(f[0].header.cards)):
            hd[f[0].header.cards[i].keyword] =  f[0].header.cards[i].value
            
        test = header.child(html.P,t_fmt.pformat_yaml(hd),
                            style={'white-space': 'pre-line',
                                   'fontSize': 14})
        '''
        
        data_input_group = header.child(dbc.Col).child(dbc.InputGroup)
        #data_group = data_input_group.child(dbc.Row)
        data_label = data_input_group.child(dbc.InputGroupText,"Input Files")
        data_select = data_input_group.child(dbc.Select, options=input_files)

        # horizontal line
        hr = header.child(dbc.Row).child(html.Hr)
        
        checklist_card = header.child(dbc.Col,width="auto").child(html.Div).child(dbc.Row).child(dbc.Col).child(dbc.Card,color="light") 
        checklist_ch = checklist_card.child(dbc.CardHeader,"Select array(s)")
        checklist_cb = checklist_card.child(dbc.CardBody)
        
        checklist_input_group = checklist_cb.child(dbc.InputGroup)
        checlist_col = checklist_input_group.child(dbc.Col)
        #checklist_label = checlist_col.child(dbc.InputGroupText, 
        #                                      'Select array(s)')
        
        checklist_presets = checlist_col.child(dbc.Checklist, inline=True, 
                                               persistence=False)
        checklist_presets.options = [
                {'label': 'All', 'value': 'all'},
                {'label': '1.1mm', 'value': '1.1 mm Array'},
                {'label': '1.4mm', 'value': '1.4 mm Array'},
                {'label': '2.0mm', 'value': '2.0 mm Array'},
                ]

        checklist_presets.value = []
        
        #nw_checklist_card = header.child(dbc.Col,width="auto").child(html.Div).child(dbc.Row).child(dbc.Col).child(dbc.Card,color="light") 
        #nw_checklist_ch = nw_checklist_card.child(dbc.CardHeader,"Select NW(s)")
        #nw_checklist_cb = nw_checklist_card.child(dbc.CardBody)
        
        nw_checklist_col = checklist_cb.child(dbc.Col)
        #nw_checklist_label = nw_checklist_col.child(dbc.InputGroupText, 
        #                                      'Select NW(s)')
        # make three button groups
        nw_select = nw_checklist_col.child(dbc.Checklist, inline=True, 
                                           persistence=False,className='w-auto')
        nw_checklist_options = [{
            'label': f'N{i}', 'value': i}
                for i in range(13)]
        
        nw_select.options = nw_checklist_options
        
        array_names = ['1.1 mm Array', '1.4 mm Array', '2.0 mm Array']
        preset_networks_map = dict()
        preset_networks_map['1.1 mm Array'] = set(o['value'] for o in nw_checklist_options[0:7])
        preset_networks_map['1.4 mm Array'] = set(o['value'] for o in nw_checklist_options[7:11])
        preset_networks_map['2.0 mm Array'] = set(o['value'] for o in nw_checklist_options[11:13])
        preset_networks_map['all'] = functools.reduce(set.union, (preset_networks_map[k] for k in array_names))
        
        # init layout for figures
        layout = go.Layout(
            margin=go.layout.Margin(
                l=0, #left margin
                r=0, #right margin
                b=0, #bottom margin
                t=0, #top margin
                ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)')
        
        # axis options
        axis_options = []
        for pn in range(len(BP.param_names)):
            axis_options.append({'label': BP.param_names[pn], 
                                 'value': BP.apt_colnames[pn]})
            
        s_options = []
        for pn in range(len(BP.param_names)):
            s_options.append({'label': BP.param_names[pn], 
                                 'value': BP.apt_colnames[pn]})
        s_options.append({'label': 'N/A', 
                             'value': 'N/A'})
            
        
        # main body
        body = self.child(dbc.Row)
        
        # row for control, scatter, and beammap cards
        card_row = body.child(dbc.Row)

        # control card
        ctrl_card = card_row.child(dbc.Col,width="auto").child(html.Div).child(dbc.Row).child(dbc.Col).child(dbc.Card,color="light") 
        ctrl_ch = ctrl_card.child(dbc.CardHeader,"Controls")
        ctrl_cb = ctrl_card.child(dbc.CardBody)

        scatter_options_container = ctrl_cb.child(dbc.Row)
        scatter_input_group = scatter_options_container.child(dbc.InputGroup).child(dbc.Col)
        
        b_group = scatter_input_group.child(dbc.Row)
        bmp_radioitem = b_group.child(dcc.RadioItems, inline=True, 
                                      value='signal', style={"margin-bottom": "15px"})
        
        #tt = make_tooltip(bmp_radioitem.id, "Controls for plots", "bottom")        
        
        bmp_radioitem.options = [
            {'label': 'Signal', 'value': 'signal'},
            {'label': 'Weight', 'value': 'weight'},
            {'label': 'S/N', 'value': 'sig2noise'},
            ]
        
        x_group = scatter_input_group.child(dbc.Row)
        x_axis_label = x_group.child(dbc.InputGroupText,"x axis")
        x_axis_select = x_group.child(dbc.Select, options=axis_options, 
                                      value='x_t', style={"margin-bottom": "15px"})
        
        y_group = scatter_input_group.child(dbc.Row)
        y_axis_label = y_group.child(dbc.InputGroupText,"y axis")
        y_axis_select = y_group.child(dbc.Select, options=axis_options,
                                                value='y_t', style={"margin-bottom": "15px"})
        
        z_group = scatter_input_group.child(dbc.Row)
        z_axis_label = z_group.child(dbc.InputGroupText,"z axis")
        z_axis_select = z_group.child(dbc.Select, options=axis_options,
                                                value='amp', style={"margin-bottom": "15px"})
        
        s_group = scatter_input_group.child(dbc.Row)
        s_axis_label = s_group.child(dbc.InputGroupText,"marker size")
        s_axis_select = s_group.child(dbc.Select, options=s_options,
                                                value='N/A',style={"margin-bottom": "15px"})
        
        flx_group = scatter_input_group.child(dbc.Row)
        flx_axis_label = flx_group.child(dbc.InputGroupText,"Units")
        flx_axis_select = flx_group.child(dbc.Select,
                                                value='Xs',style={"margin-bottom": "15px"})
        
        flx_axis_select.options = [
            {'label': 'Xs', 'value': 'Xs'},
            {'label': 'mJy/beam', 'value': 'mJy/beam'},
            {'label': 'MJy/Sr', 'value': 'MJy/Sr'},
            {'label': 'uK/arcmin^2', 'value': 'uK/arcmin^2'},
            ]
        
        # scatter card
        scatter_card = card_row.child(dbc.Col).child(html.Div).child(dbc.Row).child(dbc.Col).child(dbc.Card,color="light") 
        scatter_ch = scatter_card.child(dbc.CardHeader,"Scatter Plot")
        scatter_cb = scatter_card.child(dbc.CardBody)
        scatter_fig = scatter_cb.child(dcc.Graph, figure=go.Figure(layout=layout))
        
        # beammap card
        bmp_card = card_row.child(dbc.Col).child(html.Div).child(dbc.Row).child(dbc.Col).child(dbc.Card,color="light") 
        bmp_ch = bmp_card.child(dbc.CardHeader,"Beammap Plot")
        bmp_cb = bmp_card.child(dbc.CardBody)
        bmp_fig = bmp_cb.child(dcc.Graph, figure=go.Figure(layout=layout))
        
        # histogram card
        hist_card = body.child(dbc.Row).child(dbc.Col).child(html.Div).child(dbc.Row).child(dbc.Col).child(dbc.Card,color="light") 
        hist_ch = hist_card.child(dbc.CardHeader,"Histogram Plots")
        hist_cb = hist_card.child(dbc.CardBody)
        hist_cb_row = hist_cb.child(dbc.Row)
        hist_fig_0 = hist_cb_row.child(dbc.Col).child(dcc.Graph, 
                                                      figure=go.Figure(layout=layout))
        hist_fig_1 = hist_cb_row.child(dbc.Col).child(dcc.Graph, 
                                                      figure=go.Figure(layout=layout))
        hist_fig_2 = hist_cb_row.child(dbc.Col).child(dcc.Graph, 
                                                      figure=go.Figure(layout=layout))
        hist_fig_3 = hist_cb_row.child(dbc.Col).child(dcc.Graph, 
                                                      figure=go.Figure(layout=layout))
        hist_fig_4 = hist_cb_row.child(dbc.Col).child(dcc.Graph, 
                                                      figure=go.Figure(layout=layout))
        hist_fig_5 = hist_cb_row.child(dbc.Col).child(dcc.Graph, 
                                                      figure=go.Figure(layout=layout))
        
                
        @app.callback(Output(data_select.id, 'options'),
              Input(interval.id, 'n_intervals'))
        def update_files(n):
            
            with open(config_file, 'r') as fo:
                indexfile = yaml_load(fo)
                data_dir = indexfile['inputs'][0]['path']
                        
            sub_dirs = [x[0] for x in os.walk(data_dir)]
                
            for i in range(1,len(sub_dirs)):
                exists = False
                for j in range(len(input_files)):
                    if input_files[j]['value'] == sub_dirs[i]:
                        exists = True
                if exists == False:
                    fits_files = glob.glob(sub_dirs[i] +'/*.fits')
                    if fits_files:
                        input_files.append({'label': sub_dirs[i],#.rsplit('/', 1)[-1], 
                                            'value': sub_dirs[i]})
                    
            j = 0
            for i in range(len(input_files)):
                if input_files[i]['value'] not in sub_dirs:
                    input_files.pop(j)
                    j = j - 1
                else:
                    j = j + 1
            #lon, lat, alt = satellite.get_lonlatalt(datetime.datetime.now())
            return input_files
        
        # a callback to update the check state
        @app.callback(
                [
                    Output(nw_select.id, "options"),
                    Output(nw_select.id, "value"),
                    ],
                [
                    Input(checklist_presets.id, "value"),
                    Input(data_select.id, "value"),
                ]
            )
        def load_data(preset_values, path):
            # load data
            if path is not None:
                Data = CitlaliData()
                Data.load_fits(path)
                Data.load_apt(path)
                BP.get_data(Data)
            
            # this is all the nws
            nw_values = set()
            for pv in preset_values:
                nw_values = nw_values.union(preset_networks_map[pv])
            options = [o for o in nw_checklist_options if o['value'] in nw_values]
            values = list(nw_values)

            return options, values
        
        app.callback(
            [Output(scatter_fig.id, "figure")],
            [Input(data_select.id, "value"),
             Input(nw_select.id, "value"),
             Input(x_axis_select.id, "value"),
             Input(y_axis_select.id, "value"),
             Input(z_axis_select.id, "value"),
             Input(s_axis_select.id, "value"),
             #Input(a1100_scale.id, "value"),
             #Input(a1400_scale.id, "value"),
             #Input(a2000_scale.id, "value"),
             Input(hist_fig_0.id, "selectedData"),
             Input(hist_fig_1.id, "selectedData"),
             Input(hist_fig_2.id, "selectedData"),
             Input(hist_fig_3.id, "selectedData"),
             Input(hist_fig_4.id, "selectedData"),
             Input(hist_fig_5.id, "selectedData"),
             Input(scatter_fig.id, "clickData")],
            )(functools.partial(BP.update_scatter))

        app.callback(
            [Output(bmp_fig.id, "figure")],
            [Input(data_select.id, "value"),
             Input(nw_select.id, "value"),
             Input(x_axis_select.id, "value"),
             Input(y_axis_select.id, "value"),
             Input(scatter_fig.id, "clickData"),
             Input(bmp_radioitem.id, "value"),
             Input(hist_fig_0.id, "selectedData"),
             Input(hist_fig_1.id, "selectedData"),
             Input(hist_fig_2.id, "selectedData"),
             Input(hist_fig_3.id, "selectedData"),
             Input(hist_fig_4.id, "selectedData"),
             Input(hist_fig_5.id, "selectedData")]
            )(functools.partial(BP.update_beammap))
        
        app.callback(
            [Output(hist_fig_0.id, "figure"),
             Output(hist_fig_1.id, "figure"),
             Output(hist_fig_2.id, "figure"),
             Output(hist_fig_3.id, "figure"),
             Output(hist_fig_4.id, "figure"),
             Output(hist_fig_5.id, "figure")],
            [Input(data_select.id, "value"),
             Input(nw_select.id, "value")]
            )(functools.partial(BP.update_hist))

DASHA_SITE = {
    'extensions': [
        {
            'module': 'dasha.web.extensions.dasha',
            'config': {
                'template': beammap,
                'title_text': 'Beammap',
                }
            },
        ]
    }