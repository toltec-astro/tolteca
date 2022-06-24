#!/usr/bin/env python

from dataclasses import dataclass, field
from pathlib import Path
from schema import Or
from typing import Optional

from tollan.utils.dataclass_schema import add_schema

from .. import apps_registry, get_app_config
from ...utils.common_schema import RelPathSchema


@apps_registry.register('beammap_viewer')
@add_schema
@dataclass
class BeammapViewerConfig():
    """The config class for the beammap viewer app."""

    config_path: Optional[Path] = field(
        default=None,
        metadata={
            'description': 'The standalone config file.',
            'schema': Or(RelPathSchema(), None)
            }
        )
    toltec_db_url: str = field(
        default='mysql+mysqldb://tolteca:tolteca@127.0.0.1:3307/toltec',
        metadata={
            'description': 'The toltec db url.'
            }
    )
    title_text: str = field(
        default='Beammap Viewer',
        metadata={
            'description': 'The title text of the page.'
            }
        )


def DASHA_SITE():
    """The dasha site entry point.
    """

    dasha_config = get_app_config(BeammapViewerConfig).to_dict()
    toltec_db_url = dasha_config.pop("toltec_db_url")
    dasha_config.update({
        'template': 'tolteca.web.templates.beammap_viewer:BeammapViewer',
        # 'THEME': dbc.themes.LUMEN,
        # 'ASSETS_IGNORE': 'bootstrap.*',
        # 'DEBUG': True,
        })
    return {
        'extensions': [
            {
                'module': 'dasha.web.extensions.db',
                'config': {
                    "SQLALCHEMY_TRACK_MODIFICATIONS": False,
                    "SQLALCHEMY_BINDS": {
                        'default': toltec_db_url,
                        'toltec': toltec_db_url,
                        },
                    }
                },
            {
                'module': 'dasha.web.extensions.dasha',
                'config': dasha_config
                },
            ]
        }
