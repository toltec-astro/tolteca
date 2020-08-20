#! /usr/bin/env python


from tollan.utils.log import get_logger
from dasha.web.templates import ComponentTemplate
import dash_bootstrap_components as dbc
import dash_html_components as html
from .common.basic_obs_select_view import BasicObsSelectView
from dash.dependencies import Output, Input
import numpy as np


class KidsExplorer(ComponentTemplate):
    _component_cls = html.Div

    logger = get_logger()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_layout(self, app):

        container = self
        control_container, view_container = container.grid(2, 1)

        ctx = self._setup_control(app, control_container, {})
        super().setup_layout(app)

        ctx = self._setup_view(app, control_container, ctx)

        # bods_select = ctx['bods_select']

    def _setup_control(self, app, container, ctx):
        bods_select_container = container.grid(1, 1)
        bods_select = bods_select_container.child(BasicObsSelectView())
        ctx.update({
            'bods_select': bods_select
            })
        return ctx

    def _setup_view(self, app, container, ctx):
        paging_container, graph_container = container.grid(2, 1)

        subplot_grid = (10, 2)

        n_subplots = np.prod(subplot_grid)

        @app.callback(
                Output(paging_container.id, 'children'),
                ctx['bods_select'].select_inputs
                )
        def update_paging(dataitem_value, network_value):
            bods = ctx['bods_select'].bods_from_select_inputs(
                    dataitem_value, network_value)
            n_subplots_total = bods.bod_list[0].meta["n_tones"]
            n_pages = n_subplots_total // n_subplots + (
                    n_subplots_total % n_subplots > 0)
            btns = [
                    dbc.Button(
                        f'{i * n_subplots}', color='link',
                        className='mr-1', size='sm')
                    for i in range(n_pages)
                    ]
            return btns
        return ctx
