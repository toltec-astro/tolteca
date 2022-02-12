#!/usr/bin/env python

import dash_bootstrap_components as dbc

import astropy.units as u
from pathlib import Path

from tollan.utils.dataclass_schema import add_schema
from dataclasses import dataclass, field
from schema import Or
from typing import Union

from .. import apps_registry, get_app_config
from ...utils.common_schema import PhysicalTypeSchema, RelPathSchema


@apps_registry.register('obs_planner_v0')
@add_schema
@dataclass
class ObsPlannerV0Config():
    """The config class for legacy obs planner app."""

    toltec_sensitivity_module_path: Path = field(
        metadata={
            'description': 'The path to locate toltec sensitivity module.',
            'schema': RelPathSchema()
            }
        )
    sma_pointing_catalog_path: Union[None, Path] = field(
        default=None,
        metadata={
            'description': 'The path to locate SMA pointing catalog.',
            'schema': Or(RelPathSchema(), None)
            }
        )
    raster_model_length_max: u.Quantity = field(
        default=20 << u.arcmin,
        metadata={
            'description': 'The maximum length of raster scan model.',
            'schema': PhysicalTypeSchema('angle')
            }

        )
    title_text: str = field(
        default='Obs Planner',
        metadata={
            'description': 'The title text of the page.'
            }
        )


def DASHA_SITE():
    """The dasha site entry point.
    """
    dasha_config = get_app_config(ObsPlannerV0Config).to_dict()
    dasha_config.update({
        'template': 'tolteca.web.templates.obsPlanner:ObsPlanner',
        'EXTERNAL_STYLESHEETS': [
            dbc.themes.MATERIA,
            ],
        'ASSETS_IGNORE': 'bootstrap.*',
        })
    return {
        'extensions': [
            {
                'module': 'dasha.web.extensions.dasha',
                'config': dasha_config
                },
            ]
        }
