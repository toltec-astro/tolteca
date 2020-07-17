#! /usr/bin/env python


from dasha.web.templates import ComponentTemplate
from dasha.web.templates.collapsecontent import CollapseContent
from dasha.web.templates.dataframeview import (
        DataFrameView, default_dash_table_kwargs)
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State, ClientsideFunction
import itertools
from tollan.utils import to_typed, rupdate
from tollan.utils.fmt import pformat_yaml
import dash_table
from plotly.subplots import make_subplots


from ..tasks.toltecdb import get_toltec_file_info


class AutoDriveView(ComponentTemplate):

    _component_cls = dbc.Container

    def setup_layout(self, app):

        container = self
        header = container.child(dbc.Row)
        title_container = header.child(dbc.Col)
        title_container.child(html.H1, 'Autodrive Plot')
        body = self.child(dbc.Row).child(dbc.Col)

        self.setup_section_select(app, body.child(dbc.Row).child(dbc.Col))

        self.setup_section_viewer(app, body.child(dbc.Row).child(dbc.Col))

    def setup_section_viewer(
            self, app, container,
            runid_dropdown, nwid_dropdown, toneid_dropdown):
        graph = container.child(
                dcc.Graph, animate=False,
                )

        @app.callback(
                Output(graph.id, 'figure'),
                [
                    Input(runid_dropdown.id, 'value'),
                    Input(nwid_dropdown.id, 'value'),
                    Input(toneid_dropdown.id, 'value'),
                    ]
                )
        def update_plot(runid, nwid, toneid)
            fig = make_subplots(
                    rows=2, cols=1,
                    # shared_xaxes=True,
                    # vertical_spacing=0.02,
                    # row_heights=[0.2, 0.6, 0.2],
                    )
            fig.update_layout(
                    title=f'Autodrive {runid}',
                    uirevision=True,
                    height=1000,
                    )

       @app.callback(
                [
                    Output(self.plot_data_view.id, 'data'),
                    Output(viewer_debug.id, 'children'),
                    Output(viewer_debug_button.id, 'children'),
                    Output(fig_sources_store.id, 'data'),
                    ],
                [
                    Input(viewer_timer.id, 'n_intervals'),
                ], [
                    State(fig_sources_store.id, 'data'),
                    ]
                )
        def update_viewer_data(n_intervals, fig_params):
            result = get_viewer_data()
            result = sorted(
                        filter(
                            lambda r: len(r['meta']['filepaths']) > 0,
                            result.values()),
                        key=lambda d: d['meta']['Obsnum'], reverse=True)
            debug = f"{len(result)} data objects:\n{pformat_yaml(result)}"
            debug_button = f'{len(result)} data objects'

            rupdate(fig_params, {
                'model_params_hist': {
                    'data': [
                        {
                            'xaxis': 'x2',
                            'yaxis': 'y',
                            'type': 'histogram',
                            'x': r['model_params']['Qr'],
                            } for r in result if 'model_params' in r
                        ] + [
                        {
                            'xaxis': 'x2',
                            'yaxis': 'y2',
                            'type': 'scattergl',
                            'mode': 'markers',
                            'x': r['model_params']['Qr'],
                            'y': r['model_params']['fr'],
                            } for r in result if 'model_params' in r
                        ] + [
                        {
                            'xaxis': 'x3',
                            'yaxis': 'y3',
                            'type': 'histogram',
                            'x': r['model_params']['x0'],
                            'y': r['model_params']['fr'],
                            } for r in result if 'model_params' in r
                            ]

                    }
                })
            return [r['meta'] for r in result], debug, debug_button, fig_params



    def _setup_section_select(self, app, container)
        dropdown = container.child(
                dcc.Dropdown, multi=False, className='mb-2 px-0')

    @property
    def layout(self):
        return super().layout
