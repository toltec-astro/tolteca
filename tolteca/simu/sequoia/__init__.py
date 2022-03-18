#!/usr/bin/env python


from schema import Schema, Optional, Or
from tollan.utils.dataclass_schema import DataclassNamespace
from tollan.utils.log import get_logger
from .. import instrument_registry
from ..lmt import lmt_info as site_info
from .sequoia_info import sequoia_info
from .simulator import SequoiaObsSimulator


__all__ = ['site_info', 'SequoiaObsSimulatorConfig']


@instrument_registry.register('sequoia')
class SequoiaObsSimulatorConfig(DataclassNamespace):
    """The config class for Sequoia observation simulator."""

    _namespace_from_dict_schema = Schema({
        Optional(
            'mode',
            default='s_wide',
            description='The spectrometer mode to use.'):
        Or(*sequoia_info['mode_names'])
        })

    logger = get_logger()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        simulator = self.simulator = SequoiaObsSimulator(mode=self.mode)
        self.observer = simulator.observer

    def __call__(self, cfg):
        return self.simulator
