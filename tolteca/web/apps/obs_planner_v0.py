#!/usr/bin/env python

import dash_bootstrap_components as dbc

import astropy.units as u

from tollan.utils.log import get_logger
from tollan.utils.dataclass_schema import DataclassNamespace
from schema import Optional, Schema

from .. import apps_registry
from ...utils.common_schema import PhysicalTypeSchema, RelPathSchema


@apps_registry.register('obs_planner_v0')
class ObsPlannerV0Config(DataclassNamespace):
    """The config class for legacy obs planner app."""

    _namespace_from_dict_schema = Schema({
        Optional(
            'raster_model_length_max',
            default=20 << u.arcmin,
            description='The maximum length of raster scan model.'):
        PhysicalTypeSchema('angle'),
        Optional(
            'pointing_catalog_path',
            default=None,
            description='The catalog path containing the pointing sources.'):
        RelPathSchema(),
        Optional(
            'title_text',
            default='Obs Planner',
            description='The title text of the page.'): str
        })

    logger = get_logger()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


DASHA_SITE = {
    'extensions': [
        {
            'module': 'dasha.web.extensions.dasha',
            'config': {
                'template': 'tolteca.web.templates.obsPlanner:ObsPlanner',
                'EXTERNAL_STYLESHEETS': [
                    dbc.themes.MATERIA,
                    ],
                'ASSETS_IGNORE': 'bootstrap.*'
                'site_name'
                },
            },
        ]
    }
