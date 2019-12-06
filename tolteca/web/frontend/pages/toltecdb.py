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
from tolteca.utils.log import timeit
from ..common import TableViewComponent, SyncedListComponent


app = get_current_dash_app()


UPDATE_INTERVAL = 4000  # ms
N_RECORDS_INIT = 50
N_RECORDS = 50
QUERY_PARAMS = {'parse_dates': ["Date"]}

sources = [
        {
            'label': 'user_log',
            'title': 'User Log',
            'bind': 'lmt_toltec',
            'query': '',
            'query_init': f'select * from lmtmc_notes.userlog'
                          f' order by id desc limit {N_RECORDS_INIT}',
            'query_update': f'select * from lmtmc_notes.userlog a'
                            f' where a.id >= {{id_since}}'
                            f' order by a.id desc limit {N_RECORDS}',
            'query_params': QUERY_PARAMS,
            },
        {
            'label': 'toltec_files',
            'title': 'TolTEC Files',
            'bind': 'lmt_toltec',
            'query': '',
            'query_init': f'select * from toltec.toltec'
                          f' order by id desc limit {N_RECORDS_INIT}',
            'query_update': f'select * from toltec.toltec b'
                            f' where b.id >= {{id_since}}'
                            f' order by b.id desc limit {N_RECORDS}',
            'query_params': QUERY_PARAMS,
            },
    ]


_sources_dict = {s['label']: s for s in sources}


# setup layout factory and callbacks
for src in sources:

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
        src = _sources_dict[state['label']]
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
                src['query_update'].format(id_since=first_row_id + 1),
                **src['query_params'])
            return df.to_dict("records"), ""


def _get_layout(src):
    try:
        df = dataframe_from_db(
                src['bind'], src['query_init'],
                **src['query_params'])
    except Exception:
        return html.Div("Unable to get data.")

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
    for src in sources:
        components.append(dbc.Col(
            _get_layout(src), width=12, lg=width))

    return html.Div([dbc.Row(components)])
