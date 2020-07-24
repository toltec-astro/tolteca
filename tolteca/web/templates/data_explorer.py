#! /usr/bin/env python


from dasha.web.templates import ComponentTemplate
import dash_html_components as html
import dash_bootstrap_components as dbc
from .common.dbqueryview import DBQueryView
from pathlib import Path
import pandas
from astropy.table import Table
import dash
from plotly.subplots import make_subplots
import dash_core_components as dcc
from dash.dependencies import Input, Output


class DataExplorer(ComponentTemplate):

    _component_cls = html.Div

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_layout(self, app):

        container = self

        # we create two major sub-layouts: the data selection section, and
        # the data view section
        db_query_view = self._setup_selection_section(
                app, container.child(dbc.Row).child(dbc.Col))

        all_files_graph = self._setup_view_section(
                app, container.child(dbc.Row).child(dbc.Col))

        super().setup_layout(app)

        self._setup_all_files_graph_callback(
                app, db_query_view.dataset_index_store, all_files_graph)

    def _setup_selection_section(self, app, container):

        # this function contains the code to setup the
        # selection widgets
        db_query_view = container.child(DBQueryView(file_search_paths=[
            Path('/data')
            ]))
        return db_query_view

    def _setup_view_section(self, app, container):

        # this function contains the code to setup the
        # graphs and etc. to view the data

        container.child(dbc.Row).child(dbc.Col).child(html.Hr())

        all_files_graph = container.child(
                dcc.Graph,
                )

        return all_files_graph

    def _setup_all_files_graph_callback(
            self, app, dataset_index_store, all_files_graph):

        @app.callback(
            Output(all_files_graph.id, 'figure'),
            [
                Input(dataset_index_store.id, 'data')
                ]
                )
        def update_kids_info(dataset_index_store_data):
            if dataset_index_store_data is None:
                return dash.no_update
            dataset_index_table = Table.from_pandas(
                    pandas.DataFrame.from_records(dataset_index_store_data))
            fig = make_subplots(
                    rows=1, cols=1, start_cell="top-left",
                    x_title='ObsNum',
                    y_title='ROACH Index',
                    )
            fig.update_layout(
                    uirevision=True,
                    showlegend=False,
                    margin=dict(t=60),
                    )
            trace = {
                    'name': 'all files',
                    'type': 'scatter',
                    'x': dataset_index_table['obsid'],
                    'y': dataset_index_table['nwid']
                    }
            fig.add_trace(trace, 1, 1)
            return fig
