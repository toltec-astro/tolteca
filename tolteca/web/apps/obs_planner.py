#!/usr/bin/env python

import astropy.units as u

from tollan.utils.log import get_logger
from tollan.utils.dataclass_schema import DataclassNamespace
from schema import Optional, Schema, Or

from .. import apps_registry
from ...utils.common_schema import PhysicalTypeSchema, RelPathSchema


@apps_registry.register('obs_planner')
class ObsPlannerConfig(DataclassNamespace):
    """The config class for obs planner app."""

    _namespace_from_dict_schema = Schema({
        Optional(
            'raster_model_length_max',
            default=20 << u.arcmin,
            description='The maximum length of raster scan model.'):
        PhysicalTypeSchema('angle'),
        Optional(
            'site_name',
            default='lmt',
            description='The observing site name.'):
        Or("lmt", ),
        Optional(
            'instru_name',
            default='lmt',
            description='The observing instrument name.'):
        Or("toltec", ),
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
