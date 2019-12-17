#! /usr/bin/env python

import dash_core_components as dcc
from dash.dependencies import Input, State, Output, ClientsideFunction
import dash_bootstrap_components as dbc
import dash_table
import dash_html_components as html
from dash.development.base_component import Component
from tolteca.utils.log import timeit, get_logger
from cached_property import cached_property
import importlib
# from .utils import get_url


def fa(className):
    return html.I(className=className)


class SimplePage(object):

    def __init__(self, label, module_prefix='.', route_prefix=''):
        self._label = label
        self._module_prefix = module_prefix
        self._route_prefix = route_prefix

    @property
    def pathname(self):
        return f'/{self._label}'

    @cached_property
    def page(self):
        return timeit(f"load page {self.pathname}")(
                importlib.import_module)(
                    f'{self._module_prefix}{self._label}',
                    package=__package__)

    def get_layout(self, **kwargs):
        layout = getattr(self.page, 'get_layout', None)
        if isinstance(layout, Component):
            return layout
        elif callable(layout):
            layout = layout(**kwargs)
            if isinstance(layout, Component):
                return layout
        raise ValueError(
                f"page {self.page} does not contain valid layout")

    @property
    def title(self):
        try:
            page = self.page
        except Exception:
            page = None
        title = getattr(page, 'title_text', self._label)
        icon = getattr(page, 'title_icon', 'fas fa-keyboard')
        return fa(icon), title

    @property
    def nav_link(self):
        return dbc.NavLink(
                self.title,
                href=f"{self._route_prefix}{self.pathname}",
                id=self.nav_link_id)

    @property
    def nav_link_id(self):
        return f"page-{self._label}-link"


class SimpleComponent(object):

    logger = get_logger()

    def __init__(self, label, ids):
        self.label = label
        for id_ in ids:
            self.make_id(id_)
        self._components_factory = dict()

    def make_id(self, s):
        setattr(self, s, f'{self.label}-{self.component_label}-{s}')

    def add_components_factory(self, s, f, use_ids=None):
        self.make_id(s)
        self._components_factory[s] = f
        if use_ids is not None:
            for s in use_ids:
                setattr(self, s, getattr(f, s))

    def make_components(self, s, **kwargs):
        self.logger.debug(f"make components {s} {kwargs}")
        f = self._components_factory[s]
        if isinstance(f, SimpleComponent):
            return f.components(**kwargs)
        return f(getattr(self, s), **kwargs)

    def components(self, **kwargs):
        return {
            s: self.make_components(s, **kwargs.get(s, dict()))
            for s in self._components_factory.keys()
            }


class SyncedListComponent(SimpleComponent):

    component_label = 'synced-list-update'

    def __init__(self, label):
        super().__init__(label, ('timer', 'state', 'items'))

    def components(self, interval, update_extra=None):

        state_data = {
                'update_extra': update_extra,
                'label': self.label
                }

        result = list(super().components().values())
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


class LiveTitleComponent(SimpleComponent):

    component_label = 'live-title'

    def __init__(self, label):
        super().__init__(label, ('text', 'is_loading', 'is_loading_trigger'))

    def components(self, title):

        return html.Div([
                html.H1(
                    title,
                    id=self.text,
                    className='d-inline-block align-middle',
                    style={
                        'line-height': '80px'
                        }
                    ),
                dcc.Loading(
                    id=self.is_loading,
                    children=[
                            html.Div(id=self.is_loading_trigger)
                        ],
                    className='d-inline-block btn align-middle',
                    ),
                ], className='px-0')


class TableViewComponent(SimpleComponent):

    component_label = 'table-view'

    def __init__(self, label):
        super().__init__(label, ('table', 'title'))
        self.add_components_factory(
                'title',
                LiveTitleComponent(self.title),
                use_ids=['is_loading', 'is_loading_trigger'])

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
        result = super().components(title={
            'title': title
            })
        title_components = result.pop("title")
        result = list(result.values())
        result.append(dash_table.DataTable(**tbl_kwargs))
        if additional_components is not None:
            result.extend(additional_components)
        return html.Div([
            # title row
            dbc.Row([
                dbc.Col(title_components),
                ]),
            # content row
            dbc.Row([dbc.Col(html.Div(result)), ]),
            ])
