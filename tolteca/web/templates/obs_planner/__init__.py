#!/usr/bin/env python


from dataclasses import dataclass, field
from pathlib import Path
from typing import Union
from schema import Or

from dash_component_template import ComponentTemplate
# from dash import html, Input, Output
from dash import html
import dash_bootstrap_components as dbc

# from dasha.web.templates.common import (
#         LabeledDropdown, LabeledChecklist,
#         LabeledInput,
#         CollapseContent,
#         LiveUpdateSection
#     )

from tollan.utils.dataclass_schema import add_schema
# from tollan.utils.log import get_logger
from tollan.utils.fmt import pformat_yaml

from ....utils.common_schema import RelPathSchema


@add_schema
@dataclass
class ObsPlannerConfig(object):
    """The config class for `ObsPlanner`."""

    site_name: str = field(
        default='lmt',
        metadata={
            'description': 'The observing site name',
            'schema': Or("lmt", )
            }
        )
    instru_name: str = field(
        default='toltec',
        metadata={
            'description': 'The observing instrument name',
            'schema': Or("toltec", )
            }
        )
    pointing_catalog_path: Union[None, Path] = field(
        default=None,
        metadata={
            'description': 'The catalog path containing the pointing sources',
            'schema': RelPathSchema()
            }
        )
    # view related
    title_text: str = field(
        default='Obs Planner',
        metadata={
            'description': 'The title text of the page'}
        )

    class Meta:
        schema = {
            'ignore_extra_keys': False,
            'description': 'The config of obs planner.'
            }


class ObsPlanner(ComponentTemplate):
    """An observation Planner."""

    class Meta:
        component_cls = dbc.Container

    def __init__(self, config=None, **kwargs):
        if config is None:
            config = ObsPlannerConfig()
        kwargs.setdefault('fluid', True)
        super().__init__(**kwargs)
        self._config = config

    def setup_layout(self, app):
        cfg = self._config
        container = self
        container.child(html.H1(cfg.title_text))
        container.child(html.Pre(pformat_yaml(cfg.to_dict())))
        super().setup_layout(app)
