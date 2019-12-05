import dash_core_components as dcc
import dash_html_components as html
import dash_table

# from pandas import DataFrame
import pandas as pd
# import dash_bootstrap_components as dbc
from dash.dependencies import Input, State, Output, ClientsideFunction
from ...backend import db, cache
from .. import get_current_dash_app
from tolteca.utils.log import timeit


app = get_current_dash_app()


@cache.memoize(timeout=0)
def filter_table_cols(colnames):
    # result = ['id', ]
    result = ['id', 'ObsNum', ]
    ignored = ('Time', 'FileName', 'id')
    for colname in colnames:
        if colname not in result and colname not in ignored:
            result.append(colname)
    return result


ctx = 'lmt-toltec-database'
title = 'TolTEC Files'
update_interval = 4000  # ms
n_records_init = 50
n_records = 200


@timeit
def _get_toltec_files(n_records, filter_cols, id_since=None):

    session = db.create_scoped_session(
            options={'bind': db.get_engine(app.server, 'lmt_toltec')})

    if id_since is None:
        where = ''
    else:
        where = f'where a.id > {id_since}'
    query = f"select a.*,b.Entry,b.User from toltec.toltec a" \
            f" left join lmtmc_notes.userlog b" \
            f" on a.ObsNum = b.ObsNum" \
            f" {where} order by a.id desc" \
            f" limit {n_records};"

    df = pd.read_sql_query(
            query,
            con=session.bind,
            parse_dates=['Date'],
            )
    if len(df) > 0:
        df['Date'] = df['Date'] + df['Time']
    if filter_cols is not None:
        df = df[filter_cols(df.columns)]
    return df


@timeit
@cache.memoize(timeout=60)
def get_toltec_files():
    return _get_toltec_files(
            n_records=n_records_init,
            filter_cols=filter_table_cols,
            )


@timeit
@cache.memoize(timeout=1)
def update_toltec_files(id_since):
    return _get_toltec_files(
            n_records=n_records,
            filter_cols=filter_table_cols,
            id_since=id_since
            )


@timeit
def get_layout(**kwargs):
    df = get_toltec_files()
    data = df.to_dict("records")
    lo_table = dash_table.DataTable(
            filter_action="native",
            # sort_action="native",
            sort_mode="multi",
            # column_selectable="single",
            # row_selectable="multi",
            id=f'{ctx}-table',
            # virtualization=True,
            # persistence=True,
            # persistence_type='session',
            columns=[
                {"name": i, "id": i} for i in df.columns],
            data=data,
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#eeeeee'
                }
            ],
            style_table={
                'border': 'thin lightgrey solid'
            },
            style_data={
                'whiteSpace': 'normal',
                # 'height': 'auto'
            },
            style_header={
                'backgroundColor': '#aaaaaa',
                'fontWeight': 'bold'
            },
            fixed_rows={'headers': True, 'data': 0},
            style_cell={
                'textAlign': 'left',
                'padding': '5px',
                'max-width': '500px',
                'min-width': '60px',
            },
            )
    return html.Div([
            html.Div(
                [
                    html.H1(
                        title,
                        style={
                            'display': 'inline-block',
                            'line-height': '70px',
                            'vertical-align': 'middle',
                            }
                        ),
                    dcc.Loading(
                        id="table-is-loading",
                        children=[
                                html.Div(id="loading-output-1")
                            ],
                        style={
                            'display': 'inline-block',
                            'padding': '0 1em',
                            'vertical-align': 'middle',
                            },
                        ),
                ]),
            html.Div(
                id='table-wrapper',
                children=[
                    lo_table,
                    dcc.Interval(
                        id='table-update', interval=update_interval),
                    dcc.Store(id='table-update-first-row-id'),
                    dcc.Store(id='table-update-nrows'),
                    dcc.Store(id='table-update-new-rows'),
                    # dcc.Store(
                    #     id='table-data', storage_type='session',
                    #     data=data
                    #     ),
                    ],
            )
            # lo_table,
        ])


app.clientside_callback(
    ClientsideFunction(
        namespace='tolteca',
        function_name='array_first_id'
    ),
    Output('table-update-first-row-id', 'data'),
    [Input(f'{ctx}-table', 'data')]
)


app.clientside_callback(
    ClientsideFunction(
        namespace='tolteca',
        function_name='array_size'
    ),
    Output('table-update-nrows', 'data'),
    [Input(f'{ctx}-table', 'data')]
)


app.clientside_callback(
    ClientsideFunction(
        namespace='tolteca',
        function_name='array_concat'
    ),
    Output(f'{ctx}-table', 'data'),
    [
        Input('table-update-new-rows', 'data'),
    ],
    [State(f'{ctx}-table', 'data')]
)


@timeit
@app.callback([
        Output('table-update-new-rows', 'data'),
        Output('table-is-loading', 'children')
        ], [
        Input('table-update', 'n_intervals')], [
        State('table-update-first-row-id', 'data'),
        State('table-update-nrows', 'data'),
        ])
def refresh_table(n_intervals, first_row_id, nrows):
    if nrows is not None and nrows < n_records:
        first_row_id -= n_records - nrows
    df = update_toltec_files(first_row_id)
    return df.to_dict("records"), ""
