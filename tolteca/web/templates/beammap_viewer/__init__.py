#!/usr/bin/env python

from dash_component_template import ComponentTemplate
from dash import html, dcc, Input, Output, State
import dash
import dash_bootstrap_components as dbc

from dasha.web.templates.common import (
        LabeledDropdown,
        LiveUpdateSection
        )
from dasha.web.templates.utils import PatternMatchingId, make_subplots

import yaml
import numpy as np
import astropy.units as u

from tollan.utils.log import get_logger
from tollan.utils.fmt import pformat_yaml

from ..common.misc import HeaderWithToltecLogo
from ..common.simple_basic_obs_select import KidsDataSelect


class BeammapViewer(ComponentTemplate):
    """A viewer for beammap data products."""

    class Meta:
        component_cls = dbc.Container

    logger = get_logger()

    def __init__(
            self,
            config_path=None,
            title_text='Live Viewer',
            **kwargs):
        kwargs.setdefault('fluid', True)
        super().__init__(**kwargs)
        cfg = self._config = self._load_config(config_path)
        self._title_text = title_text
        self.logger.debug(f"loaded viewer config:\n{pformat_yaml(cfg)}")
        
    @staticmethod
    def _load_config(config_path):
        config = {
            'reduced_file_search_paths': [
                '/data/data_lmt/toltec/reduced',
                ]
            }
        if config_path is not None:
            with open(config_path, 'r') as fo:
                config.update(yaml.load(fo))
        return config

    def setup_layout(self, app):

        container = self
        header_section, hr_container, body = container.grid(3, 1)
        hr_container.child(html.Hr())
        header_container = header_section.child(HeaderWithToltecLogo(
            logo_colwidth=4
            )).header_container

        title_container, controls_container = header_container.grid(2, 1)
        header = title_container.child(
                LiveUpdateSection(
                    title_component=html.H3(self._title_text),
                    interval_options=[2000, 5000],
                    interval_option_value=2000
                    ))
        obs_select = controls_container.child(
                KidsDataSelect(
                    multi=['nw', 'array'],
                    reduced_file_search_paths=self._config[
                        'reduced_file_search_paths']
                    )
                )
        obs_select.setup_live_update_section(
            app, header, query_kwargs={'obs_type': 'Nominal',
                                       'n_obs': 100})
        redu_select = controls_container.child(dbc.Row).child(
            LabeledDropdown(
                className='w-auto',
                label_text='Reduction ID',
                size='sm',
                placeholder='Select a reduction to view ...',
               )).dropdown

        obs_select_info_display = body.child(html.Pre)
        view_container = body.child(dbc.Col)

        super().setup_layout(app)

        @app.callback(
            [
                Output(obs_select_info_display.id, 'children'),
                Output(redu_select.id, 'options'),
            ],
            obs_select.inputs
        )
        def collect_reductions(obs_select_value):
            # glob the file system to collect reduced data
            print(obs_select_value)
            redu_dir = self._config.get('beammap_reduced_dir', '/some/path')
            redu_options = [
                {
                    'label': f"redu{i:02d}",
                    'value': f'path/to/redu{i:02d}'
                    }
                    for i in range(5)
                ]
            return str(obs_select_value), redu_options
        

        # An example callback that consumes the inputs above

        @app.callback(
            Output(view_container.id, 'children'),
            [Input(redu_select.id, 'value')]
        )
        def update_viewer(redu_path):
            # make plots using data in redu
            print(redu_path)
            return html.Pre(f'results of reduction in {redu_path}')