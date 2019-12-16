#! /usr/bin/env python

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from .. import get_current_dash_app


app = get_current_dash_app()


def get_layout(**kwargs):
    '''Returns the layout that contains a table view to the source.'''

    canvas_layout = html.Div([
        html.Div(className='row', children=[
            html.Div(className='two columns', style={'margin-top': '2%'}, children=[
            html.Div(className='row', style={'margin-top': 30}, children=[
                html.Div(className='six columns', children=[
                    html.H6('Rows'),
                    dcc.Dropdown(
                        id='rows',
                        options=[{
                            'label': i,
                            'value': i
                        } for i in [1,2,3,4]],
                        placeholder='Select number of rows...',
                        clearable=False,
                        value=2
                    ),
                ]),
                html.Div(className='six columns', children=[
                    html.H6('Columns'),
                    dcc.Dropdown(
                        id='columns',
                        options=[{
                            'label': i,
                            'value': i
                        } for i in [1,2,3]],
                        placeholder='Select number of columns...',
                        clearable=False,
                        value=3
                    ),
                ])
            ]),
        ]),
        html.Div(className='ten columns', id='layout-div', style={'border-style': 'solid', 'border-color': 'gray'}, children=[])

    ])
    ])
    return html.Div([
        dbc.Row(dbc.Col(html.H1("Kids View"))),
        dbc.Row(dbc.Col(canvas_layout)),
        ])


@app.callback(
    Output('layout-div', 'children'),
    [Input('rows', 'value'),
    Input('columns', 'value')])
def configure_layout(rows, cols):

    mapping = {1: 'twelve columns', 2: 'six columns', 3: 'four columns', 4: 'three columns'}
    sizing = {1: '40vw', 2: '35vw', 3: '23vw'}

    layout = [html.Div(className='row', children=[
        html.Div(className=mapping[cols], children=[
            dcc.Graph(
                id='test{}'.format(i+1+j*cols),
                config={'displayModeBar': False},
                style={'width': sizing[cols], 'height': sizing[cols]}
            ),

        ]) for i in range(cols)
    ]) for j in range(rows)]

    return layout

#Max layout is 3 X 4
for k in range(1,13):

    @app.callback(
        [Output('test{}'.format(k), 'figure'),
        Output('test{}'.format(k), 'style')],
        [Input('columns', 'value')])
    def create_graph(cols):

        sizing = {1: '40vw', 2: '35vw', 3: '23vw'}

        style = {
            'width': sizing[cols],
            'height': sizing[cols],
        }
        fig = {'data': [], 'layout': {}}
        return [fig, style]
