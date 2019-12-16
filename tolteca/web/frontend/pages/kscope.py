#! /usr/bin/env python

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ClientsideFunction
from tolteca.utils.log import timeit, get_logger
from ...backend import dataframe_from_db, cache
from .. import get_current_dash_app
from ..common import TableViewComponent
import dash
import plotly
from pathlib import Path


app = get_current_dash_app()
logger = get_logger()

UPDATE_INTERVAL = 1000  # ms
N_RECORDS_LATEST = 1


src = {
    'label': 'kscope',
    'title': 'KScope',
    'bind': 'lmt_toltec',
    'table': 'toltec.toltec',
    'query': f'select {{use_cols}} from {{table}} a'
             f' {{join}} {{group}}'
             f' order by {{order}} limit {N_RECORDS_LATEST}',
    'use_cols': ', '.join([
        # 'GROUP_CONCAT(a.id SEPARATOR ",") AS id',
        'CONCAT(a.Obsnum, "_", a.SubObsNum, "_", a.ScanNum) as ObsNum',
        # 'TIMESTAMP(a.Date, a.Time) as DateTime',
        'GROUP_CONCAT('
        'a.RoachIndex order by a.RoachIndex SEPARATOR ",")'
        ' AS RoachIndex',
        'CONCAT("clip", GROUP_CONCAT('
        'distinct right(a.HostName, 1) order by a.RoachIndex SEPARATOR "/"))'
        ' AS HostName',
        'b.label as ObsType',
        # 'c.label as Master',
        'd.Entry as Comment',
        ]),
    'join': f"inner join toltec.obstypes b on a.ObsType = b.id"
            # f" inner join toltec.masters c on a.Master = c.id"
            f" inner join lmtmc_notes.userlog d on a.Obsnum = d.Obsnum",
    # 'group': ''
    'group': 'group by a.ObsNum',
    'order': 'a.id desc',
    'query_params': {'parse_dates': ["DateTime"]},
    }


class NcScope(object):

    logger = get_logger()

    def __init__(self, source):
        self._source = source

    @cache.memoize(timeout=60)
    @classmethod
    def from_file(cls, filepath):
        cls.logger.debug(f"create {cls} for {filepath}")
        return cls(source=Path(filepath))

    def iter_data(size):
        head = 0
        tail = 100
        def iter():
            while head < tail:
                yield range(head, head + size)
                head += size


class KidsScope(NcScope):

    logger = get_logger()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_db(cls, **kwargs):
        return cls(source=None)


class Thermetry(NcScope):

    logger = get_logger()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_db(cls, **kwargs):
        return cls(source=None)


tbn = TableViewComponent(src['label'])
tbn.add_component(
        'timer',
        lambda id_: dcc.Interval(id_, interval=UPDATE_INTERVAL))
tbn.add_component(
        'entry_updated',
        lambda id_: dcc.Store(id_, data=True))


interfaces = [
        f'toltec{i}' for i in range(13)
        ]


def interfaces_from_entry(entry):
    return [f'toltec{i}' for i in entry['RoachIndex'].split(",")]


interface_options = [{'label': interface,
                      'value': interface}
                     for interface in interfaces]


def get_layout(**kwargs):
    '''Returns the layout that contains a table view to the source.'''
    controls = html.Div([
            # dbc.Col(html.Div([
            #             html.Label(
            #                 'Filter by tone range:',
            #                 # className="control_label"
            #             ),
            #             dcc.RangeSlider(
            #                 id='tone-range-slider',
            #                 min=0,
            #                 max=1000,
            #                 value=[0, 10],
            #                 allowCross=False,
            #                 # className="dcc_control"
            #                 tooltip={'always_visible': True},
            #             ),
            #             ])),
            # interface
            dbc.Row(dbc.Col(html.Div([
                        dcc.Dropdown(
                            id='interface-dropdown',
                            options=interface_options,
                            multi=True,
                            value=interfaces,
                            # className="dcc_control"
                        ),
                        ])),
                    ),
            ], style={
                    'padding': '2em 0',
                    })

    def _table_view():
        try:
            df = dataframe_from_db(
                    src['bind'], src['query'].format(**src),
                    **src['query_params'])
        except Exception as e:
            logger.error(e, exc_info=True)
            return html.Div(dbc.Alert(
                        "Query Failed", color="danger"),
                        style={
                            'padding': '15px 0px 0px 0px',
                            })
        return html.Div([
            tbn.components(
                src['title'],
                columns=[
                    {"name": i, "id": i} for i in df.columns],
                data=df.to_dict("records"),
                filter_action='none',
                # fixed_rows=None,
                style_table={
                    'height': 'auto',
                },
                )
            ])
    graph_view = html.Div([
        html.Label(id='entry-updated'),
        dcc.Graph(id='kscope-graph')
        ])
    return html.Div([
        dbc.Row([dbc.Col(_table_view()), ]),
        dbc.Row([dbc.Col(controls), ]),
        dbc.Row([dbc.Col(graph_view), ]),
        ])


@timeit
@app.callback([
        Output(tbn.table, 'data'),
        Output(tbn.is_loading, 'children'),
        Output(tbn.entry_updated, 'data'),
        ], [
        Input(tbn.timer, 'n_intervals')], [
        State(tbn.table, 'data'),
        ])
def update(n_intervals, data):
    try:
        df = dataframe_from_db(
            src['bind'],
            src['query'].format(**src),
            **src['query_params'])
        lastest = df.iloc[0]
        entry_updated = lastest['ObsNum'] != data[0]['ObsNum']
        return df.to_dict("records"), "", entry_updated
    except Exception as e:
        logger.error(e, exc_info=True)
        return list(), html.Div(dbc.Alert(
                    "Refresh Failed", color="danger"),
                    style={
                        'padding': '15px 0px 0px 0px',
                        }), False


@timeit
@app.callback([
        Output('entry-updated', 'children'),
        Output('kscope-graph', 'figure')
        ], [
        Input(tbn.entry_updated, 'data')], [
        ])
def entry_update(entry_updated):
    if entry_updated:
        kscope = KScope.from_db()
        fig = plotly.tools.make_subplots(rows=2, cols=1, vertical_spacing=0.2)
        fig['layout']['margin'] = {
            'l': 30, 'r': 10, 'b': 30, 't': 10
        }
        fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
        import numpy as np
        fig.append_trace({
            'x': np.arange(10),
            'y': np.sin(np.arange(10)),
            'name': 'Altitude',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, 1, 1)
        fig.append_trace({
            'x': np.arange(20),
            'y': np.tan(np.arange(20)),
            'text': np.arange(20),
            'name': 'Longitude vs Latitude',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, 2, 1)
        return html.Div("Updated"), fig
    raise dash.exceptions.PreventUpdate()


# update list
app.clientside_callback(
        ClientsideFunction(
            namespace='tolteca',
            function_name='interface_from_latest_data',
        ),
        Output('interface-dropdown', 'options'),
        [
            Input(tbn.entry_updated, 'data'),
        ],
        [
            State(tbn.table, 'data'),
            State('interface-dropdown', 'options'),
        ]
    )
