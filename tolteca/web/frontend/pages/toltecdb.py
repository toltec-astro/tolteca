import dash_bootstrap_components as dbc
# import dash_core_components as dcc
import dash_html_components as html
# import dash_table

# from pandas import DataFrame
# import pandas as pd
# import dash_bootstrap_components as dbc
from dash.dependencies import Input, State, Output
from ...backend import dataframe_from_db
from .. import get_current_dash_app
from tolteca.utils.log import timeit, get_logger
from tolteca.utils.fmt import pformat_dict
from ..common import TableViewComponent, SyncedListComponent
from collections import OrderedDict


app = get_current_dash_app()
logger = get_logger()


UPDATE_INTERVAL = 4000  # ms
N_RECORDS_INIT = 500
N_RECORDS = 500


def odict_from_list(l, key):
    return OrderedDict([(v[key], v) for v in l])


source_common = {
            'query_init': f'select {{use_cols}} from {{table}} a'
                          f' {{join}} {{group}}'
                          f' order by {{order}} limit {N_RECORDS_INIT}',
            'query_update': f'select {{use_cols}} from {{table}} a'
                            f' {{join}} {{group}}'
                            f' where a.id >= {{id_since}}'
                            f' order by {{order}} limit {N_RECORDS}',
            'query_params': {'parse_dates': ["DateTime"]},
        }
sources = odict_from_list(map(lambda d: d.update(source_common) or d, [
        {
            'label': 'user_log',
            'title': 'User Log',
            'bind': 'lmt_toltec',
            'table': 'lmtmc_notes.userlog',
            'use_cols': ', '.join([
                'a.id',
                'TIMESTAMP(a.Date, a.Time) as DateTime',
                'a.Obsnum',
                'a.Entry', 'a.Keyword', ]),
            'join': "",
            'group': '',
            'order': 'a.id desc',
            },
        {
            'label': 'toltec_files',
            'title': 'TolTEC Files',
            'bind': 'lmt_toltec',
            'table': 'toltec.toltec',
            'use_cols': ', '.join([
                # 'GROUP_CONCAT(a.id SEPARATOR ",") AS id',
                'max(a.id) as id',
                'a.Obsnum', 'a.SubObsNum', 'a.ScanNum',
                'TIMESTAMP(a.Date, a.Time) as DateTime',
                'GROUP_CONCAT('
                'a.RoachIndex order by a.RoachIndex SEPARATOR ",")'
                ' AS RoachIndex',
                # 'a.RoachIndex',
                'GROUP_CONCAT('
                'distinct a.HostName order by a.RoachIndex SEPARATOR ",")'
                ' AS HostName',
                # 'a.HostName',
                'b.label as ObsType',
                'c.label as Master',
                ]),
            'join': f"inner join toltec.obstypes b on a.ObsType = b.id"
                    f" inner join toltec.masters c on a.Master = c.id",
            # 'group': ''
            'group': 'group by a.ObsNum',
            'order': 'a.id desc'
            },
    ]), key='label')


logger.debug(f"sources: {pformat_dict(sources)}")


# setup layout factory and callbacks
for src in sources.values():

    src['_synced_list'] = sln = SyncedListComponent(src['label'])
    src['_table_view'] = tbn = TableViewComponent(src['label'])

    sln.make_callbacks(
            app,
            data_component=(tbn.table, 'data'),
            cb_namespace='tolteca',
            cb_state='array_summary',
            cb_commit='array_concat',
            )

    @timeit
    @app.callback([
            Output(sln.items, 'data'),
            Output(tbn.is_loading, 'children')
            ], [
            Input(sln.timer, 'n_intervals')], [
            State(sln.state, 'data')
            ])
    def update(n_intervals, state):
        # it is critical to make sure the body does not refer to
        # mutable global states
        src = sources[state['label']]
        try:
            nrows = state['size']
            first_row_id = state['first']['id']
            # print(first_row_id)
            if nrows < N_RECORDS:
                first_row_id -= N_RECORDS - nrows
            # if state['update_extra'] is not None:
            #     first_row_id -= state['update_extra']
        except Exception:
            return list(), html.Div(dbc.Alert(
                    "Refresh Failed", color="danger"),
                    style={
                        'padding': '15px 0px 0px 0px',
                        })
        else:
            df = dataframe_from_db(
                src['bind'],
                src['query_update'].format(id_since=first_row_id + 1, **src),
                **src['query_params'])
            return df.to_dict("records"), ""


def _get_layout(src):
    try:
        df = dataframe_from_db(
                src['bind'], src['query_init'].format(**src),
                **src['query_params'])
    except Exception as e:
        logger.error(e, exc_info=True)
        return html.Div(dbc.Alert(
                    "Query Failed", color="danger"),
                    style={
                        'padding': '15px 0px 0px 0px',
                        })

    slc = src['_synced_list'].components(interval=UPDATE_INTERVAL)

    components = src['_table_view'].components(
            src['title'],
            additional_components=slc,
            columns=[
                {"name": i, "id": i} for i in df.columns],
            data=df.to_dict("records"),
            )
    return components


def get_layout(**kwargs):
    '''Returns the layout that contains a table view to the source.'''

    components = []

    width = 12 / len(sources)
    for src in sources.values():
        components.append(dbc.Col(
            _get_layout(src), width=12, lg=width))

    return html.Div([dbc.Row(components)])
