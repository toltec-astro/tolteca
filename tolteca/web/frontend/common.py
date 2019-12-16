#! /usr/bin/env python

import dash_core_components as dcc
from dash.dependencies import Input, State, Output, ClientsideFunction
import dash_bootstrap_components as dbc
import dash_table
import dash_html_components as html


class SimpleComponent(object):

    def __init__(self, label, ids):
        self.label = label
        for id_ in ids:
            self.make_id(id_)
        self._components_factory = dict()

    def make_id(self, s):
        setattr(self, s, f'{self.label}-{self.component_label}-{s}')

    def add_component(self, s, f):
        self.make_id(s)
        self._components_factory[s] = f

    def components(self, **kwargs):
        return [
            f(getattr(self, k)) for k, f in self._components_factory.items()
            ]


class SyncedListComponent(SimpleComponent):

    component_label = 'synced-list-update'

    def __init__(self, label):
        super().__init__(label, ('timer', 'state', 'items'))

    def components(self, interval, update_extra=None):

        state_data = {
                'update_extra': update_extra,
                'label': self.label
                }

        result = super().components()
        result.extend([
                dcc.Interval(id=self.timer, interval=interval),
                dcc.Store(id=self.state, data=state_data),
                dcc.Store(id=self.items),
                ])
        return result

    def make_callbacks(
            self, app, data_component,
            cb_namespace, cb_state, cb_commit):
        """Setup callbacks that syncs client side data with server data.

        Parameters
        ----------
        data_component: 2-tuple
            The data component specified in form of `(id, prop)`.
        cb_namespace: str
            The namespace at which the client side callbacks are defined.
        cb_commit: str
            The function name to actually update the client side data.
        cb_state: str
            The function name to update the current state dict, which
            is passed to the server at given intervals for retrieving
            new items.
        """
        app.clientside_callback(
                ClientsideFunction(
                    namespace=cb_namespace,
                    function_name=cb_state
                ),
                Output(self.state, 'data'),
                [Input(*data_component)],
                [State(self.state, 'data')],
            )

        app.clientside_callback(
                ClientsideFunction(
                    namespace=cb_namespace,
                    function_name=cb_commit
                ),
                Output(*data_component),
                [
                    Input(self.items, 'data'),
                ],
                [State(*data_component)]
            )


class TableViewComponent(SimpleComponent):

    component_label = 'table-view'

    def __init__(self, label):
        super().__init__(label, ('table', 'is_loading', 'is_loading_trigger'))

    def components(self, title, additional_components=None, **kwargs):

        tbl_kwargs = dict(
                id=self.table,
                filter_action="native",
                sort_mode="multi",
                # sort_action="native",
                # column_selectable="single",
                # row_selectable="multi",
                # virtualization=True,
                # persistence=True,
                # persistence_type='session',
                page_action='none',
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
                },
                style_header={
                    'backgroundColor': '#aaaaaa',
                    'fontWeight': 'bold'
                },
                style_cell={
                    'textAlign': 'left',
                    'padding': '5px',
                    'max-width': '500px',
                    'min-width': '60px',
                },
                )
        tbl_kwargs.update(**kwargs)
        result = super().components()
        result.append(dash_table.DataTable(**tbl_kwargs))
        if additional_components is not None:
            result.extend(additional_components)
        return html.Div([
            # title row
            dbc.Row(
                [
                    dbc.Col(
                        html.H1(title, style={
                            'line-height': '80px',
                            'vertical-align': 'middle',
                            }),
                        width=4),
                    dbc.Col(
                        dcc.Loading(
                            id=self.is_loading,
                            children=[
                                    html.Div(id=self.is_loading_trigger)
                                ]),
                        width=2),
                    ]),
            # content row
            dbc.Row([dbc.Col(html.Div(result), width=12), ]),
            ])
