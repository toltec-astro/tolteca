#!/usr/bin/env python

from schema import Schema, Literal, Optional, Or

from tollan.utils.dataclass_schema import DataclassNamespace

from ...utils.common_schema import RelPathSchema, DictTemplateListSchema
from .. import sources_registry
from .models import ImageSourceModel, CatalogSourceModel


def _make_data_map_value_schema_dict(base_dict=None):
    if base_dict is None:
        base_dict = dict()
    base_dict.update(
        {
            str: Or(
                str,
                {
                    Optional(
                        'I', default=None,
                        description='The Stokes I component.'): str,
                    Optional(
                        'Q', default=None,
                        description='The Stokes Q component.'): str,
                    Optional(
                        'U', default=None,
                        description='The Stokes U component.'): str,
                    }
                )
            })
    return base_dict


def _make_data_item_schema(data_key, description):
    data_key_schema = Literal(data_key, description=description)
    return DictTemplateListSchema(
            template_key=data_key,
            template_key_schema=data_key_schema,
            resolved_item_schema={
                data_key_schema: str,
                Optional(str, 'Key value to identify the data item.'): object
                },
            description='The list of data items to use.'
            )


@sources_registry.register('image')
class ImageSourceConfig(DataclassNamespace):
    """The config class for image source model created from FITS image file."""

    _namespace_from_dict_schema = Schema({
        Literal('filepath', description='The path to the FITS image file.'):
        RelPathSchema(),
        Literal(
            'data_exts',
            description=(
                "The assignments of FITS extensions to data item labels."
                )): _make_data_item_schema(
                    "extname", "The FITS extension name")
        })

    def __call__(self, cfg):
        return ImageSourceModel.from_file(
            self.filepath,
            data_exts=self.data_exts,
            )


@sources_registry.register('point_source_catalog')
class PointSourceCatalogSourceConfig(DataclassNamespace):
    """The config class for point source catalog source model created from
    catalog file.
    """

    _namespace_from_dict_schema = Schema({
        Literal('filepath', description='The path to the catalog file.'):
        RelPathSchema(),
        Optional(
            'name_col',
            default='source_name',
            description="The column for source names"): str,
        Optional(
            'pos_cols',
            default=['ra', 'dec'],
            description="The columns for coordinates"): [str],
        Literal(
            'data_cols',
            description=(
                "The assignments of columns to data item labels."
                )): _make_data_item_schema(
                    "colname", "The table column name"),
        })

    def __call__(self, cfg):
        return CatalogSourceModel.from_file(
            self.filepath,
            pos_cols=self.pos_cols,
            name_col=self.name_col,
            data_cols=self.data_cols,
            )
