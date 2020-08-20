#! /usr/bin/env python


from dasha.web.templates import ComponentTemplate
from dasha.web.templates.collapsecontent import CollapseContent
import dash_html_components as html
from ...tasks.dbrt import dbrt
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
from dasha.web.extensions.db import dataframe_from_db
from tollan.utils.fmt import pformat_yaml
from tollan.utils.log import get_logger
import dash
import textwrap
from dash_table import DataTable
from tolteca.fs.toltec import ToltecDataset


class DBQueryView(ComponentTemplate):
    """This is a view that allow user query the databases for entries.

    """
    _component_cls = html.Div

    logger = get_logger()

    def __init__(self, *args, file_search_paths=None, **kwargs):
        super().__init__(*args, **kwargs)
        # these paths are used to locate the data file paths returned
        # by the query
        self._file_search_paths = file_search_paths

    def setup_layout(self, app):

        # We just use the abstract name `container` here to refer to this
        # component itself in order to make it easier to move the layout
        # defined here to a different parent later
        container = self

        # it is a good practice to always use a grid system to manage the child
        # components, unless we know there will be only one child or the
        # children are simple.
        # also note that "row" and "col" does not translate a grid of
        # items on the web page, because the columns in one row by default
        # may be automatically wrapped, due to the `width` setting.
        # a row of three cols with each of them width=12 (100% of the view
        # port width), it will appears as three "rows" on the web page.
        # input container to provide user input

        # here we use a two column set up, left for input, right for output
        container_row = container.child(dbc.Row)
        input_container = container_row.child(dbc.Col, width=4)
        # output container to provide the query result
        output_container = container_row.child(dbc.Col, width=8)

        # we can choose to define the children for these containers here
        # but for readability we better put the definition of sublayout
        # in separate functions

        db_select_drp, db_query_ta, db_query_submit_btn, db_error_container = \
            self._setup_input_container(
                app, input_container)
        query_output_details_container, dataset_table_view = \
            self._setup_output_container(
                app, output_container)

        self.dataset_index_store = dataset_index_store = \
            container.child(dcc.Store)

        # this is to trigger setup_layout of any child templates
        # without this, callbacks defined in those child templates are
        # not registered.
        # If we only use the standard components from Dash as the children,
        # in which case this line can be omitted but not recommented to do so.
        super().setup_layout(app)

        # it is the convention that we define callbacks *after* the
        # above class setup_layout call. This is to allow all
        # dynamic attributes of child templates to be created (which is
        # done when we call the super().setup_layout(). However, again,
        # if the callback does not depends on those attributes, it is
        # OK to put it anywhere as long as all the related components are
        # defined before it.

        @app.callback(
                [
                    Output(query_output_details_container.id, 'children'),
                    Output(dataset_table_view.id, 'columns'),
                    Output(dataset_table_view.id, 'data'),
                    Output(dataset_index_store.id, 'data'),
                    Output(db_error_container.id, 'children'),
                    ],
                [
                    Input(db_select_drp.id, 'value'),
                    Input(db_query_submit_btn.id, 'n_clicks'),
                    ],
                [
                    State(db_query_ta.id, 'value'),
                    ]
                )
        def on_db_query_change(db_name, n_clicks, query_str):
            db = dbrt[db_name]
            table_names = db.tables.keys()
            if query_str is None:
                return [dash.no_update, ] * 5

            dataset_columns = None
            dataset_data = None
            dataset_index_store_data = None

            try:
                query_result = dataframe_from_db(query_str, db_name)
                # build the dataset table from files
                # here we just do a glob of the file name in these
                # folders
                if (self._file_search_paths is None) or (
                        'ObsNum' not in query_result.columns):
                    pass
                else:
                    self.logger.debug("create file list")
                    filepaths = []
                    # here we just glob all related files with all
                    # networks and obsnums
                    for obsnum, roachid in zip(
                            query_result['ObsNum'],
                            query_result['RoachIndex']):
                        patterns = [
                            f'toltec{roachid}/toltec{roachid}_{obsnum:06d}_*',
                            f'toltec{roachid}_{obsnum:06d}_*',
                            ]
                        for r in self._file_search_paths:
                            for pattern in patterns:
                                filepaths.extend(list(r.glob(pattern)))
                    if len(filepaths) > 0:
                        # build the dataset
                        dataset = ToltecDataset.from_files(*filepaths)
                        self.logger.debug(f'dataset:\n{dataset}')
                        use_cols = ['nwid', 'obsid', 'subobsid', 'scanid', 'ut', 'kindstr']
                        t = dataset.index_table[use_cols]
                        t['source'] = list(map(str, dataset.index_table['source']))
                        dataset_df = t.to_pandas()
                        print(dataset_df)
                        print(dataset_df.columns)
                        dataset_columns = [{'name': c, 'id': c} for c in dataset_df.columns]
                        dataset_data = dataset_df.to_dict('records')
                        dataset_index_store_data = dataset_data
                error_notify = ""
            except Exception as e:
                self.logger.debug(
                        f'error execute query: {query_str}', exc_info=True)
                query_result = f"{e}"
                error_notify = dbc.Alert(
                    f'Query failed: {e.__class__.__name__}',
                    color='danger'
                    )
            return (
                    html.Pre(pformat_yaml({
                        'tables': list(table_names),
                        'query_result': query_result,
                        })),
                    dataset_columns,
                    dataset_data,
                    dataset_index_store_data,
                    error_notify)

    def _setup_input_container(self, app, container):
        # note we used the name container again here to denote
        # the parent of this sub-layout.
        # this allows it easiser to refactor the code later

        container_row = container.child(dbc.Row)

        # set up an input group to select the db
        # width=12 actually make this row a vertical stack of columns
        db_select_container = container_row.child(
                dbc.Col, width=12, className='mb-2')

        db_select_igrp = db_select_container.child(
                dbc.InputGroup, size='sm', className='w-auto')
        db_select_igrp.child(
                dbc.InputGroupAddon(
                    "Select DB:", addon_type="prepend"))
        # here we get the options from the dbrt object
        # which is initialized at the server starting time.
        # it should contain a map to all available databases
        db_select_drp = db_select_igrp.child(
                dbc.Select,
                options=[
                    {'label': k, 'value': k}
                    for k in dbrt.keys()
                    ], value=next(iter(dbrt.keys())))

        # set up an input group to enter the query string
        db_query_container = container_row.child(
                dbc.Col, width=12, className='mb-2')
        db_query_igrp = db_query_container.child(
                dbc.InputGroup, size='sm', className='w-auto')
        # db_query_igrp.child(
        #         dbc.InputGroupAddon(
        #             "Query:", addon_type="prepend"))
        db_query_ta = db_query_igrp.child(
                dbc.Textarea, rows=10,
                style={
                    "font-family": "monospace"})

        db_query_submit_btn = container_row.child(
                dbc.Col, width=3, className='mb-2').child(
                        dbc.Button, "Submit", color="primary", size='sm')

        # an error display area
        db_error_container = container_row.child(
                dbc.Col, width=12, className='mb-2')

        @app.callback(
                Output(db_query_ta.id, 'value'),
                [
                    Input(db_select_drp.id, 'value')
                    ]
                )
        def _rest_db_quqery_ta(value):
            table_names = list(dbrt[value].tables.keys())
            if value == 'toltec':
                default_table = 'toltec_r1'
            else:
                default_table = table_names[0]
            info = pformat_yaml({
                'Tables': table_names}).strip('\n')
            return f"""
/* Enter sql to query "{value}".
{textwrap.indent(info, ' ' * 3)}
*/
select * from {default_table} order by id desc limit 100
""".strip()
        # we return the key components defined here to the setup_layout
        # func body so that we can make callbacks using them
        return (
                db_select_drp, db_query_ta, db_query_submit_btn,
                db_error_container)

    def _setup_output_container(self, app, container):

        details_container = container.child(
                CollapseContent(button_text='Details ...')).content

        dataset_table_view = container.child(
                DataTable,
                # style_table={'overflowX': 'scroll'},
                page_current=0,
                page_size=10,
                page_action='custom',
                style_table={
                    'overflowX': 'auto',
                    'width': '100%',
                    'overflowY': 'auto',
                    'height': '25vh',
                },
                )
        return details_container, dataset_table_view
