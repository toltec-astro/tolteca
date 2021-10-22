#!/usr/bin/env python

from schema import Schema, Literal, Optional

from tollan.utils.dataclass_schema import DataclassNamespace

from ...utils.common_schema import RelPathSchema
from .. import sources_registry


@sources_registry.register('image')
class ImageSourceConfig(DataclassNamespace):
    """The config class for image source model created from FITS image file."""

    _namespace_from_dict_schema = Schema({
        Literal('filepath', description='The path to the FITS image file.'):
        RelPathSchema(),
        Literal(
            'grouping',
            description="The key to use in extname_map."): str,
        Literal(
            'extname_map',
            description="The assignment of extension to each group item."):
        dict,
        })


@sources_registry.register('point_source_catalog')
class PointSourceCatalogSourceConfig(DataclassNamespace):
    """The config class for point source catalog source model created from
    catalog file.
    """

    _namespace_from_dict_schema = Schema({
        Literal('filepath', description='The path to the catalog file.'):
        RelPathSchema(),
        Literal(
            'grouping',
            description="The key to use in colname_map."): str,
        Literal(
            'colname_map',
            description="The assignment of extension to each group item."):
        {
            Optional('name', default=None,
                     description='The column name of a unique identifier.'):
            str,
            Optional('coordinate', default=None,
                     description='The column names of the coordinate.'):
            list,
            Optional('ra', default=None,
                     description='The column name of the RA.'):
            str,
            Optional('dec', default=None,
                     description='The column name of the Dec.'):
            str,
            str: object
            },
        })
