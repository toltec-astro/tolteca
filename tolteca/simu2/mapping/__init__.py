#!/usr/bin/env python


from tollan.utils.log import get_logger
from tollan.utils.dataclass_schema import DataclassNamespace
from schema import Literal, Schema, Or, Use, Optional
from astropy.time import Time

from .. import mapping_registry
from ...utils.common_schema import RelPathSchema
from .lmt_tcs import LmtTcsTrajMappingModel
from .base import OffsetTrajMappingModel


@mapping_registry.register('lmt_tcs')
class LmtTcsMappingConfig(DataclassNamespace):
    """The config class for trajectory model created from LMT TCS file."""

    _namespace_from_dict_schema = Schema({
        Literal('filepath', description='The path to the tel.nc file.'):
        RelPathSchema(),
        })

    def __call__(self):
        # return the model object
        return LmtTcsTrajMappingModel(self.filepath)


def make_offset_mapping_model_config_cls(key, offset_mapping_model_cls):
    """A helper function to create config class for trajectory models created
    from offset mapping models.
    """
    mcls = offset_mapping_model_cls
    mname = f"{mcls.__module__}:{mcls.__qualname__}"

    from astropy.coordinates.baseframe import frame_transform_graph
    frame_names = frame_transform_graph.get_names()

    s = {
        Literal("target", description='The target name or sky coordinate.'):
        str,
        Optional(
            'target_frame',
            default='icrs',
            description='The frame the target coordinate is specified in.'):
            Or(*frame_names),
        Literal(
            "ref_frame",
            description='The reference frame to '
                        'interpret the mapping pattern.'):
        Or(frame_names),
        Literal(
            "t0",
            description='The starting time of the observation.'):
        Use(Time),
        }

    # extend the base schema with the model parameters
    # s.update(make_model_schema(model_cls))

    # create the config class
    def _new(cls, **kwargs):
        # return the model object
        m = OffsetTrajMappingModel(model=mcls(**kwargs), **kwargs)
        cls.logger.debug(f"resolved mapping model: {m}")
        return m

    config_cls = type(f"{key.capitalize()}Mapping", (DataclassNamespace, ), {
        '__new__': _new,
        '_namespace_from_dict_schema': Schema(s),
        'logger': get_logger(),
        })
    mapping_registry.register(key, aliases=[mname])(
        config_cls, dispatcher_description='The mapping trajectory type.')
    return config_cls
