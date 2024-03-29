#!/usr/bin/env python


from tollan.utils.log import get_logger
from tollan.utils.dataclass_schema import DataclassNamespace
from schema import Literal, Schema, Use, Optional
from astropy.time import Time

from .. import mapping_registry
from ...utils.common_schema import PhysicalTypeSchema, RelPathSchema
from .utils import _resolve_target, _get_frame_class
from .lmt_tcs import LmtTcsTrajMappingModel
from .raster import SkyRasterScanModel
from .lissajous import (
    SkyLissajousModel, SkyDoubleLissajousModel, SkyRastajousModel)
from .base import PatternKind


__all__ = ['PatternKind', 'LmtTcsTrajMappingModel']


@mapping_registry.register('lmt_tcs')
class LmtTcsMappingConfig(DataclassNamespace):
    """The config class for trajectory model created from LMT TCS file."""

    _namespace_from_dict_schema = Schema({
        Literal('filepath', description='The path to the tel.nc file.'):
        RelPathSchema(),
        })

    def __call__(self, cfg):
        # return the model object
        return LmtTcsTrajMappingModel(self.filepath)


def make_offset_mapping_model_config_cls(key, offset_mapping_model_cls):
    """A helper function to create config class for trajectory models created
    from offset mapping models.
    """
    mcls = offset_mapping_model_cls
    mname = f"{mcls.__module__}:{mcls.__qualname__}"

    # from astropy.coordinates.baseframe import frame_transform_graph
    # frame_names = frame_transform_graph.get_names()
    # the above is too much ... Just specify a manual list for now
    # frame_names = ['icrs', 'altaz', 'galactic']

    def _get_frame_inst(s):
        return _get_frame_class(s)()

    s = {
        Literal("target", description='The target name or sky coordinate.'):
        str,
        Optional(
            'target_frame',
            default=_get_frame_inst('icrs'),
            description='The frame the target coordinate is specified in.'):
        Use(_get_frame_inst),
        Literal(
            "ref_frame",
            description='The reference frame to '
                        'interpret the mapping pattern.'):
        Use(_get_frame_inst),
        Literal(
            "t0",
            description='The starting time of the observation.'):
        Use(Time),
        }

    # extend the schema with parameters

    for n in mcls.param_names:
        p = getattr(mcls, n)
        # we do not allow optional here so the config always
        # specify the mapping pattern params explicitly.
        v = PhysicalTypeSchema(str(p.unit.physical_type))
        s[Literal(n, description=p.__doc__)] = v

    # create the config class
    def _init(self, **kwargs):
        super(DataclassNamespace, self).__init__(**kwargs)
        # resolve target coord
        self.target_coord = _resolve_target(
            self.target, self.target_frame)

        # collect the model parameters and create the target model
        params = {k: getattr(self, k) for k in self.model_cls.param_names}
        self._m_cls = mcls
        self._m_params = params

    def _get_offset_model(self):
        return self._m_cls(**self._m_params)

    def _get_model(self, observer):
        return self.get_offset_model().get_traj_model(
            target=self.target_coord,
            ref_frame=self.ref_frame,
            t0=self.t0,
            observer=observer
            )

    def _call(self, cfg):
        return self.get_model(cfg.instrument.observer)

    config_cls = type(
        f"{key.capitalize()}MappingConfig",
        (DataclassNamespace, ), {
            '__init__': _init,
            '__call__': _call,
            '_namespace_from_dict_schema': Schema(s),
            '_namespace_to_dict_schema': Schema({
                'target_frame': Use(lambda f: f.name),
                'ref_frame': Use(lambda f: f.name),
                str: object
                }),
            'model_cls': mcls,
            'get_offset_model': _get_offset_model,
            'get_model': _get_model,
            'logger': get_logger(),
            })
    mapping_registry.register(
        key, aliases=[mname],
        dispatcher_description='The mapping trajectory type.')(config_cls)
    return config_cls


make_offset_mapping_model_config_cls(
    SkyRasterScanModel.pattern_name, SkyRasterScanModel)

make_offset_mapping_model_config_cls(
    SkyLissajousModel.pattern_name, SkyLissajousModel)

make_offset_mapping_model_config_cls(
    SkyDoubleLissajousModel.pattern_name, SkyDoubleLissajousModel)

make_offset_mapping_model_config_cls(
    SkyRastajousModel.pattern_name, SkyRastajousModel)
