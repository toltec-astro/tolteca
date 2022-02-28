#!/usr/bin/env python

from dataclasses import dataclass, field
from schema import Or, Literal, Schema

import astropy.units as u

from tollan.utils import odict_from_list
from tollan.utils.log import get_logger
from tollan.utils.dataclass_schema import add_schema, DataclassNamespace


@add_schema
@dataclass
class PresetDataItem(object):
    """The class to represent preset data item."""
    key: str = field(
        metadata={
            'description': 'The data key.',
            }
        )
    value: object = field(
        metadata={
            'description': 'The data value.',
            }
        )
    value_min: object = field(
        default=None,
        metadata={
            'description': 'minimum value.'
            }
        )
    value_max: object = field(
        default=None,
        metadata={
            'description': 'maximum value.'
            }
        )
    unit: str = field(
        default=None,
        metadata={
            'description': 'The data unit',
            }
        )
    label: str = field(
        default=None,
        metadata={
            'description': 'The label to display instead of key.'
            }
        )
    description: str = field(
        default=None,
        metadata={
            'description': 'The description of the data item.'
            }
        )
    component_type: str = field(
        default='number',
        metadata={
            'description': 'The input type.',
            'schema': Or('text', 'number'),
            }
        )
    component_kw: dict = field(
        default_factory=dict,
        metadata={
            'description': 'Additional kwargs passed to the component.'
            }
        )

    @property
    def quantity(self):
        return u.Quantity(self.value, unit=self.unit)

    @property
    def quantity_min(self):
        return u.Quantity(self.value_min, unit=self.unit)

    @property
    def quantity_max(self):
        return u.Quantity(self.value_max, unit=self.unit)


@add_schema
@dataclass
class Preset(object):
    """The class for preset."""

    type: str = field(
        metadata={
            'description': 'Type of the preset.',
            'schema': Or('mapping', 'reduce')
            }
        )
    key: str = field(
        metadata={
            'description': 'Unique identifier of the preset.',
            }
        )
    label: str = field(
        metadata={
            'description': 'Preset name to display.',
            }
        )
    description: str = field(
        default=None,
        metadata={
            'description': 'Description of the preset.',
            }
        )
    description_long: str = field(
        default=None,
        metadata={
            'description': 'Long description of the preset.',
            }
        )
    data: list = field(
        default_factory=list,
        metadata={
            'description': 'The preset data items.',
            'schema': [PresetDataItem.schema]
            }
        )

    def __post_init__(self, **kwargs):
        self._data_by_key = odict_from_list(self.data, key=lambda c: c.key)

    @property
    def unique_id(self):
        return (self.type, self.key)

    def get_data_item(self, key):
        return self._data_by_key[key]


class PresetsConfig(DataclassNamespace):
    """The config class for a list of presets."""

    _namespace_from_dict_schema = Schema({
        Literal(
            'presets',
            description='The presets list.'): [Preset.schema]
        })
    # _namespace_to_dict_schema = Schema({
    #     'presets': Use(lambda c: c.to_dict()),
    #     str: object
    #     })

    logger = get_logger()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._presets_by_uid = odict_from_list(
            self.presets, key=lambda x: x.unique_id)
        self._presets_by_type = odict_from_list(
            self.presets, key=lambda x: x.unique_id)

    def __iter__(self):
        yield from self.presets

    def get(self, type, key):
        uid = (type, key)
        if uid not in self._presets_by_uid:
            raise ValueError(f"invalid preset {uid}")
        return self._presets_by_uid[uid]
