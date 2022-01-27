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
from ... import env_mapper


@env_mapper
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


class ObsSite(object):
    """A class provides info of observing site."""
    pass

    @classmethod
    def from_name(cls, name):
        """Return the site instance for `name`."""
        pass


class ObsInstru(object):
    """A class provides info of observing instrument."""
    pass

    @classmethod
    def from_name(cls, name):
        """Return the instru instance for `name`."""
        pass


class ObsPlanner(ComponentTemplate):
    """An observation Planner."""

    class Meta:
        component_cls = dbc.Container

    def __init__(
            self,
            site_name='lmt',
            instru_name='toltec',
            pointing_catalog_path=None,
            title_text='Obs Planner',
            **kwargs):
        kwargs.setdefault('fluid', True)
        super().__init__(**kwargs)
        self._site = ObsSite.from_name(site_name)
        self._instru = ObsInstru.from_name(instru_name)
        self._pointing_catalog_path = pointing_catalog_path
        self._title_text = title_text

    def setup_layout(self, app):
        container = self
        container.child(html.H1(self._title_text))
        init_info = {
            'site': self._site,
            'instru': self._instru,
            'pointing_catalog_path': self._pointing_catalog_path
            }
        container.child(html.Pre(pformat_yaml(init_info.to_dict())))
        super().setup_layout(app)
