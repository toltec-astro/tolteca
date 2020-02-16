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


class KidsView(ComponentTemplate):
    _component_cls = html.Div

    _file_info_dict_keys = [
            'ObsType', 'Obsnum', 'SubObsNum', 'ScanNum',
            'Master', 'RoachIndex']
    _file_dropdown_label_keys = [
            'ObsType', 'Obsnum', 'SubObsNum', 'ScanNum']
    _file_info_join_char = '-'
    _plot_data_view_keys = [
            'ObsType', 'Obsnum', 'SubObsNum', 'ScanNum', 'Master', 'interface']

    @classmethod
    def _info_from_dropdown_value(cls, value):
        result = dict(
                zip(
                    cls._file_info_dict_keys,
                    map(
                        to_typed,
                        value.split(cls._file_info_join_char)
                        )
                    )
                )
        # reformat roach index
        result['interfaces'] = [
                f'toltec{i}' for i in str(
                    result.pop('RoachIndex')).split(',')]
        return result

    def _make_data_controls(self, container):

        fvc = container.child(dbc.Row).child(dbc.Col)
        self.file_view = fvc.child(DataFrameView(
                    data=get_toltec_file_info,
                    update_interval=1000.,
                    title_text='Files ...',
                    primary_key='id',
                    collapsible=True,
                    className='mb-2',
                    ))
        fvcc = self.file_view.content_container
        self.file_dropdown = fvcc.child(
                dcc.Dropdown, multi=True, className='mb-2 px-0')
        self.file_table_to_dropdown_params = fvcc.child(
                    dcc.Store, data={
                        'value_cols': self._file_info_dict_keys,
                        'label_cols': self._file_dropdown_label_keys,
                        'label_join': self._file_info_join_char,
                        'n_auto_select': 10,
                        }
                    )
        self.interface_dropdown = fvcc.child(
                dcc.Dropdown, multi=True, className='px-0 mb-2')
        self.file_interface_dropdown_params = fvcc.child(
                dcc.Store, data={
                    'value_sep': ',',
                    'sep': '-',
                    'pos': -1,
                    }
                )

    def _make_plot_controls(self, container):

        dvc = container.child(dbc.Row).child(dbc.Col)
        self.plot_data_view = dvc.child(html.Div, className='mb-2').child(
                dash_table.DataTable,
                columns=[
                    {'name': k, 'id': k} for k in self._plot_data_view_keys],
                **default_dash_table_kwargs)
        self.plot_data_dropdown = dvc.child(
                dcc.Dropdown, multi=True, className='mb-2 px-0')
        # self.plot_table_to_dropdown_params = dvc.child(
        #         dcc.Store, data={
        #             'value_cols': self._file_info_dict_keys,
        #             'label_cols': self._file_dropdown_label_keys,
        #             'label_join': self._file_info_join_char,
        #             'n_auto_select': 5,
        #             }
        #         )
        # self.plot_interface_dropdown = dvc.child(
        #         dcc.Dropdown, multi=True, className='px-0')
        # self.plot_interface_dropdown_params = dvc.child(
        #         dcc.Store, data={
        #             'value_sep': ',',
        #             'sep': '-',
        #             'pos': -1,
        #             }
        #         )

    def setup_layout(self, app):
        controls_container = self.child(dbc.Row)
        data_controls_container = controls_container.child(
                dbc.Col, lg=12).child(
                html.Div,
                className='border rounded py-0 px-2 mb-2',
                style={
                    'border-color': '#dcdcdc'
                    }
                )
        self._make_data_controls(data_controls_container)

        plot_controls_container = controls_container.child(
                dbc.Col, lg=12).child(
                html.Div
                )
        self._make_plot_controls(plot_controls_container)

        viewer_container = self.child(dbc.Row).child(dbc.Col).child(html.Div)
        viewer_dummy_output = viewer_container.child(
                html.Div, className='d-none')
        viewer_debug_container = viewer_container.child(
                CollapseContent(button_text='all data'))
        viewer_debug = viewer_debug_container.content.child(
                        html.Pre)
        viewer_debug_button = viewer_debug_container._button
        # viewer_datastore = viewer_container.child(dcc.Store)
        viewer_timer = viewer_container.child(dcc.Interval, interval=1000)
        fig_sources_store = viewer_container.child(dcc.Store, data=dict())
        fig_sources = [
                {
                    'label': 'model_params_hist',
                    'title': 'KIDs Model Params',
                    'fig_init': make_subplots(
                                rows=3, cols=1,
                                # shared_xaxes=True,
                                # vertical_spacing=0.02,
                                row_heights=[0.2, 0.6, 0.2],
                                )
                    },
                ]
        for src in fig_sources:
            src['fig_init'].update_layout(
                    title=src['title'],
                    uirevision=True,
                    height=1000,
                    )
            fig_sources_store.data[src['label']] = src['fig_init']
            graph = viewer_container.child(
                    dcc.Graph, figure=src['fig_init'], animate=False,
                    )
            key_store = viewer_container.child(dcc.Store, data=src['label'])
            app.clientside_callback(
                ClientsideFunction(
                    namespace='datastore',
                    function_name='getKey',
                    ),
                Output(graph.id, 'figure'),
                [
                    Input(fig_sources_store.id, "data")
                    ],
                [
                    State(key_store.id, 'data')
                    ]
                )

        super().setup_layout(app)

        app.clientside_callback(
            ClientsideFunction(
                namespace='ui',
                function_name='tableEntryToDropdown',
                ),
            [
                Output(self.file_dropdown.id, 'options'),
                Output(self.file_dropdown.id, 'value'),
                ],
            [
                Input(self.file_view._tbl.id, "derived_viewport_data")
                ],
            [
                State(self.file_table_to_dropdown_params.id, 'data')
                ]
            )

        app.clientside_callback(
            ClientsideFunction(
                namespace='tolteca',
                function_name='interfaceOptionsFromFileInfo',
                ),
            [
                Output(self.interface_dropdown.id, 'options'),
                Output(self.interface_dropdown.id, 'value'),
                ],
            [
                Input(self.file_dropdown.id, "value")
                ],
            [
                State(self.file_interface_dropdown_params.id, 'data')
                ]
            )

        from ..tasks.kidsview import (
                request_viewer_data,
                get_viewer_data)

        @app.callback(
                Output(viewer_dummy_output.id, 'children'),
                [
                    Input(self.file_dropdown.id, 'value'),
                    Input(self.interface_dropdown.id, 'value'),
                ],
                []
                )
        def make_requests(files, interfaces):
            if files is None or interfaces is None:
                return
            for (fileinfo, interface) in itertools.product(files, interfaces):
                info = self._info_from_dropdown_value(fileinfo)
                if interface in info['interfaces']:
                    info['interface'] = interface
                    request_viewer_data(info)
                else:
                    continue

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

    @property
    def layout(self):
        return super().layout
