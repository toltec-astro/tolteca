#!/usr/bin/env python

from dataclasses import dataclass, field
from pathlib import Path
from schema import Or
from typing import Union

from tollan.utils.dataclass_schema import add_schema

from .. import apps_registry, get_app_config
from ...utils.common_schema import RelPathSchema


@apps_registry.register('live_viewer')
@add_schema
@dataclass
class LiveViewerConfig():
    """The config class for the obs planner app."""

    site_name: str = field(
        default='lmt',
        metadata={
            'description': 'The observing site name.',
            'schema': Or("lmt", )
            }
        )
    instru_name: Union[None, str] = field(
        default=None,
        metadata={
            'description': 'The observing instrument name.',
            'schema': Or("toltec", 'sequoia', None)
            }
        )
    pointing_catalog_path: Union[None, Path] = field(
        default=None,
        metadata={
            'description': 'The catalog path containing the pointing sources.',
            'schema': Or(RelPathSchema(), None)
            }
        )
    js9_config_path: Union[None, Path] = field(
        default=None,
        metadata={
            'description': 'The json config file for JS9.',
            'schema': Or(RelPathSchema(), None)
            }
        )
    toltec_ocs3_url: str = field(
        default='socket://localhost:61559',
        metadata={
            'description': 'The ocs3 server url for TolTEC ICS.',
            }
        )
    lmt_ocs3_url: str = field(
        default='socket://localhost:61558',
        metadata={
            'description': 'The ocs3 server url for LMT.',
            }
        )
    title_text: str = field(
        default='Live Viewer',
        metadata={
            'description': 'The title text of the page.'
            }
        )


def DASHA_SITE():
    """The dasha site entry point.
    """

    from dash_js9 import JS9_SUPPORT

    dasha_config = get_app_config(LiveViewerConfig).to_dict()
    dasha_config.update({
        'template': 'tolteca.web.templates.live_viewer:LiveViewer',
        # 'THEME': dbc.themes.LUMEN,
        # 'ASSETS_IGNORE': 'bootstrap.*',
        # 'DEBUG': True,
        "EXTERNAL_SCRIPTS": [JS9_SUPPORT],
        })
    return {
        'extensions': [
            {
                'module': 'dasha.web.extensions.dasha',
                'config': dasha_config
                },
            ]
        }
