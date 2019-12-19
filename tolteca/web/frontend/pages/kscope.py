#! /usr/bin/env python

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ClientsideFunction
from tolteca.utils.fmt import pformat_dict
from tolteca.utils.log import timeit, get_logger
from tolteca.io.toltec import NcFileIO
from ...backend import dataframe_from_db
from .. import get_current_dash_app
from ..common import TableViewComponent
import dash
from plotly.subplots import make_subplots
from .ncscope import NcScope
from functools import lru_cache
from pathlib import Path
import numpy as np


app = get_current_dash_app()
logger = get_logger()

title_text = 'KScope'
title_icon = 'fas fa-stethoscope'

UPDATE_INTERVAL = 5 * 1000  # ms

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
        'c.label as Master',
        'd.Entry as Comment',
        ]),
    'join': f"inner join toltec.obstypes b on a.ObsType = b.id"
            f" inner join toltec.masters c on a.Master = c.id"
            f" left join lmtmc_notes.userlog d on a.Obsnum = d.Obsnum",
    # 'group': ''
    'group': 'group by a.ObsNum',
    'order': 'a.id desc',
    'query_params': {'parse_dates': ["DateTime"]},
    }

data_rootpaths = {
        'clipa': '/clipa/toltec',
        'clipo': '/clipo/toltec',
        }

# data_rootpaths = {
#         'clipa': '/Users/ma/Codes/toltec/kids/test_data2/clipa',
#         'clipo': '/Users/ma/Codes/toltec/kids/test_data2/clipo',
#         }

roach_ids = list(range(13))


def get_data_rootpath(roach_id):
    if roach_id in range(0, 7):
        return Path(data_rootpaths['clipa'])
    elif roach_id in range(7, 13):
        return Path(data_rootpaths['clipo'])
    raise RuntimeError(f"unknown roach_id {roach_id}")


def roach_ids_from_toltecdb_entry(entry):
    return [int(i) for i in entry['RoachIndex'].split(",")]


class KScope(NcScope):

    logger = get_logger()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.io = NcFileIO(self.nc)

    @classmethod
    @lru_cache(maxsize=128)
    def from_filepath(cls, filepath):
        cls.logger.info(f"create sclope for {filepath}")
        return cls(source=filepath)

    @classmethod
    @timeit
    def from_toltecdb_entry(cls, entry):
        roach_ids = roach_ids_from_toltecdb_entry(entry)
        master = entry['Master'].lower()
        result = []
        for roach_id in roach_ids:
            rootpath = get_data_rootpath(roach_id)
            path = rootpath.joinpath(
                    f'{master}/toltec{roach_id}/toltec{roach_id}.nc').resolve()
            cls.logger.info(f"get scope for {path}")
            result.append(cls.from_filepath(path))
        return result

    @timeit
    def get_iqts(self, time, tone_slice=None):

        # make sure we update the meta
        self.io.sync()

        var = self.io.nm.getvar
        meta = self.io.meta

        n_samples = int(time * meta['fsmp'])

        time_total = meta['ntimes_all'] / meta['fsmp']
        self.logger.debug(f"current total length {time_total}")

        if n_samples < 0:
            mask = slice(n_samples, None)
            add_time = time_total
        else:
            mask = slice(None, n_samples)
            add_time = 0.

        if tone_slice is None:
            tone_slice = slice()

        self.logger.debug(
                f"get iqs n_samples={n_samples}"
                f" {time}/{time_total} mask={mask}")
        # iqs have dim [tone, time] after .T
        iqs = var('is')[mask, tone_slice].T + \
            1.j * var('qs')[mask, tone_slice].T
        ts = np.arange(iqs.shape[-1]) / meta['fsmp'] + time + add_time
        self.logger.debug(f"got iqs.shape {iqs.shape} ts.shape {ts.shape}")
        return iqs, ts


tbn = TableViewComponent(src['label'])
tbn.add_components_factory(
        'timer',
        lambda id_: dcc.Interval(id_, interval=UPDATE_INTERVAL))
tbn.add_components_factory(
        'entry_updated',
        lambda id_: dcc.Store(id_, data=True))

interface_options = [{
    'label': 'toltec{i}',
    'value': i} for i in roach_ids]


@timeit("kscope.get_layout")
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
            # interface select
            dbc.Row(dbc.Col(html.Div([
                        dcc.Dropdown(
                            id='interface-dropdown',
                            options=interface_options,
                            multi=True,
                            value=roach_ids,
                            # className="dcc_control"
                        ),
                        ])),
                    ),
            ], style={
                    'padding': '1em 0',
                    })

    def _table_view():
        try:
            df = timeit(dataframe_from_db)(
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
        # dcc.Graph(id='kscope-graph')
        html.Div(id='graph-content'),
        ])

    file_info = html.Div([
                dbc.Button(
                        "File Info",
                        id="file-info-toggle",
                        className="mb-2",
                        size='sm',
                        color='info',
                    ),
                dbc.Collapse(
                    html.Div(id='file-info-content'),
                    id="file-info-collapse"
                    ),
            ])

    return html.Div([
        dbc.Row([dbc.Col(_table_view()), ]),
        dbc.Row([dbc.Col(controls), ]),
        dbc.Row([dbc.Col(file_info), ]),
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


@app.callback([
        Output('entry-updated', 'children'),
        Output('file-info-content', 'children'),
        Output('graph-content', 'children'),
        # Output('kscope-graph', 'figure')
        ], [
        Input(tbn.entry_updated, 'data'),
        Input('interface-dropdown', 'value'),
        ], [
        State(tbn.table, 'data'),
        ])
@timeit
def entry_update(entry_updated, use_roach_ids, data):
    # if entry_updated:
    if True:
        entry = data[0]
        use_roach_ids = set(roach_ids_from_toltecdb_entry(entry)).intersection(
                set(use_roach_ids))
        logger.info(f"update for new entry {entry} with {use_roach_ids}")
        entry['RoachIndex'] = ','.join(map(str, use_roach_ids))
        scopes = KScope.from_toltecdb_entry(entry)
        if len(scopes) == 0:
            raise dash.exceptions.PreventUpdate("no scopes found.")

        def make_info_card(scope):
            return dbc.Card([
                        # dbc.CardHeader(dbc.Button(scope.io.meta['interface'])),
                        dbc.CardBody([
                            html.H6(scope.io.meta['interface']),
                            html.Pre(pformat_dict(scope.io.meta))
                            ])
                        ])

        def make_plot(scope):

            graph_fs = dcc.Graph(figure={
                'data': [{
                        'x': np.arange(10),
                        'y': np.sin(np.arange(10)),
                        'name': 'Tones',
                        'mode': 'markers',
                        'type': 'scatter'
                    }],
                'layout': dict(
                    uirevision=True,
                    yaxis={
                        'autorange': True,
                        'title': 'Qr'
                        },
                    xaxis={
                        'autorange': True,
                        'title': 'Frequency (MHz)'
                        },
                    )})

            iqs, ts = scope.get_iqts(-1., tone_slice=range(10))
            print(iqs.shape)
            print(ts.shape)
            n_panels = 1
            fig = make_subplots(
                rows=n_panels, cols=1)
            fig.update_layout(
                    uirevision=True,
                    yaxis={
                        'autorange': True,
                        'title': 'I'
                        },
                    xaxis={
                        'autorange': True,
                        'title': 'time (s)'
                        },
                    )
            for i in range(iqs.shape[0]):
                fig.append_trace({
                        'x': ts,
                        'y': iqs.real[i, :],
                        'name': f'tone{i}',
                        'mode': 'lines+markers',
                        'type': 'scattergl',
                        'marker': dict(size=2),
                        'line': dict(width=0.5),
                    }, 1, 1)

            graph_iqt = dcc.Graph(figure=fig, animate=False)

            return html.Div([
                    # dbc.Row([dbc.Col(graph_fs), ]),
                    dbc.Row([dbc.Col(graph_iqt), ]),
                ])

        def make_plot_card(scope):
            return dbc.Card([
                        # dbc.CardHeader(dbc.Button(scope.io.meta['interface'])),
                        dbc.CardBody([
                            dbc.Badge(
                                '{interface}_{obsid}_{subobsid}'
                                '_{scanid}_{kindstr}'.format(
                                    **scope.io.meta)),
                            make_plot(scope)
                            ])
                        ])

        # cards_container = dbc.CarDeck
        cards_container = html.Div
        info_cards = cards_container([make_info_card(s) for s in scopes])
        plot_cards = cards_container([make_plot_card(s) for s in scopes])
        return html.Div("Updated"), info_cards, plot_cards
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


app.clientside_callback(
        ClientsideFunction(
            namespace='ui',
            function_name='toggleWithClick',
        ),
        Output("file-info-collapse", 'is_open'),
        [Input("file-info-toggle", "n_clicks")],
        [State("file-info-collapse", 'is_open')],
    )
