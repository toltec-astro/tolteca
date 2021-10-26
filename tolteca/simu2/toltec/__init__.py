#!/usr/bin/env python


from schema import Schema, Optional
from tollan.utils.dataclass_schema import DataclassNamespace
from .. import instrument_registry
from .lmt import info as site_info


__all__ = ['site_info', 'ToltecObsSimulatorConfig']


@instrument_registry.register('toltec')
class ToltecObsSimulatorConfig(DataclassNamespace):
    """The config class for TolTEC observation simulator."""

    _namespace_from_dict_schema = Schema({
        Optional(
            'polarized',
            default=False,
            description='The toggle of whether to simulate polarized signal.'):
        bool,
        })
