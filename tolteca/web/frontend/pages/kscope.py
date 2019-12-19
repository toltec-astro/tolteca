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
from ..common import TableViewComponent, SimpleComponent
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


TABLE_UPDATE_INTERVAL = 5 * 1000  # ms
GRAPH_UPDATE_INTERVAL = 1 * 1000  # ms
N_RECORDS_LATEST = 1
ROACH_IDS_AVAILABLE = list(range(13))


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

# data_rootpaths = {
#         'clipa': '/clipa/toltec',
#         'clipo': '/clipo/toltec',
#         }

data_rootpaths = {
        'clipa': '/Users/ma/Codes/toltec/kids/test_data2/clipa',
        'clipo': '/Users/ma/Codes/toltec/kids/test_data2/clipo',
        }


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
    def get_iq_t(self, time_slice, tone_slice=None):

        # make sure we update the meta
        self.logger.debug("sync file")
        self.io.sync()

        var = self.io.nm.getvar
        meta = self.io.meta

        n_samples_all = meta['ntimes_all']
        time_total = n_samples_all / meta['fsmp']

        self.logger.debug(
                f"time_total={time_total}s"
                f" n_samples_all={n_samples_all}")

        def time_to_sample(v):
            if v is not None:
                v = int(v * meta['fsmp'])
                if v > n_samples_all:
                    v = n_samples_all
                if v < -n_samples_all:
                    v = -n_samples_all
                return v
            return None

        sample_slice = slice(*(
            time_to_sample(getattr(time_slice, p))
            for p in ('start', 'stop', 'step')
            ))

        self.logger.debug(
                f"time_slice={time_slice} sample_slice={sample_slice}")

        if tone_slice is None:
            tone_slice = slice()

        # iqs have dim [tone, time] after .T
        iqs = var('is')[sample_slice, tone_slice].T + \
            1.j * var('qs')[sample_slice, tone_slice].T
        ts = np.arange(*sample_slice.indices(n_samples_all)) / meta['fsmp']
        self.logger.debug(f"got iqs.shape {iqs.shape} ts.shape {ts.shape}")
        return iqs, ts

    @property
    def title(self):
        return '{interface}_{obsid}_{subobsid}' \
               '_{scanid}_{kindstr}'.format(**self.io.meta)


tbn = TableViewComponent(src['label'])
tbn.add_components_factory(
        'timer',
        lambda id_: dcc.Interval(id_, interval=TABLE_UPDATE_INTERVAL))
tbn.add_components_factory(
        'entry_changed',
        lambda id_: dcc.Store(id_, data=True))


def get_table_data():
    return timeit(dataframe_from_db)(
            src['bind'], src['query'].format(**src),
            **src['query_params'])


def get_table_view():
    try:
        df = get_table_data()
    except Exception as e:
        logger.error(e, exc_info=True)
        return html.Div(dbc.Alert(
                    "Query Failed", color="danger"),
                    style={
                        'padding': '15px 0px 0px 0px',
                        }), None
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
        ]), df


class KScopeComponent(SimpleComponent):

    component_label = 'kscope-graph'
    logger = get_logger()

    def __init__(self, label):
        super().__init__(label, ('iq_t', ))

    def components(self, scope):

        try:
            iqs, ts = scope.get_iq_t(slice(-10., None), tone_slice=range(10))
        except Exception:
            self.logger.debug("Unable to read data from scope", exc_info=True)
            return html.Div(dbc.Alert(
                    "Unable to read data", color="danger"),
                    style={
                        'padding': '15px 0px 0px 0px',
                        })

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
        # add all traces
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

        graph_iqt = dcc.Graph(self.iq_t, figure=fig, animate=True)

        return html.Div([
                # dbc.Row([dbc.Col(graph_fs), ]),
                dbc.Row([dbc.Col(graph_iqt), ]),
            ])


kscope_components_factory = {
        i: KScopeComponent(f'toltec{i}') for i in ROACH_IDS_AVAILABLE
        }


def make_kscope_graphs(scope):

    pass
    # graph_fs = dcc.Graph(figure={
    #     'data': [{
    #             'x': np.arange(10),
    #             'y': np.sin(np.arange(10)),
    #             'name': 'Tones',
    #             'mode': 'markers',
    #             'type': 'scatter'
    #         }],
    #     'layout': dict(
    #         uirevision=True,
    #         yaxis={
    #             'autorange': True,
    #             'title': 'Qr'
    #             },
    #         xaxis={
    #             'autorange': True,
    #             'title': 'Frequency (MHz)'
    #             },
    #         )})


def get_controls(entry):

    roach_ids = roach_ids_from_toltecdb_entry(entry)
    interface_options = [{
        'label': f'toltec{i}',
        'value': i} for i in roach_ids]

    return html.Div([
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
            dbc.Row(dbc.Col(
                        dcc.Dropdown(
                            id='interface-dropdown',
                            options=interface_options,
                            multi=True,
                            value=roach_ids,
                        ))
                    ),
            ], style={
                    'padding': '1em 0',
                    })


def get_layout(**kwargs):

    table_view, df = get_table_view()

    controls = get_controls(df.iloc[0])

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

    graph_view = html.Div([
        html.Div(id='graph-view-content'),
        ])

    return html.Div([
        dbc.Row([dbc.Col(table_view), ]),
        dbc.Row([dbc.Col(controls), ]),
        dbc.Row([dbc.Col(file_info), ]),
        dbc.Row([dbc.Col(graph_view), ]),
        ])


# update control for interface options
app.clientside_callback(
        ClientsideFunction(
            namespace='tolteca',
            function_name='interface_from_latest_data',
        ),
        Output('interface-dropdown', 'options'),
        [
            Input(tbn.entry_changed, 'data'),
        ],
        [
            State(tbn.table, 'data'),
            State('interface-dropdown', 'options'),
        ]
    )


@timeit
@app.callback([
        Output(tbn.table, 'data'),
        Output(tbn.is_loading, 'children'),
        Output(tbn.entry_changed, 'data'),
        ], [
        Input(tbn.timer, 'n_intervals')], [
        State(tbn.table, 'data'),
        ])
def table_update(n_intervals, data):
    try:
        df = get_table_data()
        data_new = df.to_dict("records")
        entry_changed = data_new[0]['ObsNum'] != data[0]['ObsNum']
        return data_new, "", entry_changed
    except Exception as e:
        logger.error(e, exc_info=True)
        return data, html.Div(dbc.Alert(
                    "Refresh Failed", color="danger"),
                    style={
                        'padding': '15px 0px 0px 0px',
                        }), False


@app.callback([
        Output('file-info-content', 'children'),
        Output('graph-view-content', 'children'),
        ], [
        Input(tbn.entry_changed, 'data'),
        Input('interface-dropdown', 'value'),
        ], [
        State(tbn.table, 'data'),
        ])
@timeit
def on_entry_changed(entry_changed, use_roach_ids, data):
    entry = data[0]
    if not use_roach_ids:
        raise dash.exceptions.PreventUpdate("No interface specified")
    use_roach_ids = set(roach_ids_from_toltecdb_entry(entry)).intersection(
            set(use_roach_ids))
    logger.info(f"update graph for new entry {entry} with {use_roach_ids}")
    entry['RoachIndex'] = ','.join(map(str, use_roach_ids))
    scopes = KScope.from_toltecdb_entry(entry)
    if len(scopes) == 0:
        raise dash.exceptions.PreventUpdate("No scopes found")

    def make_info_card(scope):
        return dbc.Card([
                    # dbc.CardHeader(dbc.Button(scope.io.meta['interface'])),
                    dbc.CardBody([
                        html.H6(scope.title),
                        html.Pre(pformat_dict(scope.io.meta))
                        ])
                    ], style={
                        'min-width': '40rem'})

    def make_graph_card(scope):
        roachid = scope.io.meta['roachid']
        return dbc.Card([
                # dbc.CardHeader(dbc.Button(scope.io.meta['interface'])),
                dbc.CardBody([
                    dbc.Badge(scope.title),
                    kscope_components_factory[roachid].components(scope),
                    ])
                ])

    info_cards = dbc.CardDeck([make_info_card(s) for s in scopes])
    graph_cards = html.Div([make_graph_card(s) for s in scopes])
    return info_cards, graph_cards


app.clientside_callback(
        ClientsideFunction(
            namespace='ui',
            function_name='toggleWithClick',
        ),
        Output("file-info-collapse", 'is_open'),
        [Input("file-info-toggle", "n_clicks")],
        [State("file-info-collapse", 'is_open')],
    )
