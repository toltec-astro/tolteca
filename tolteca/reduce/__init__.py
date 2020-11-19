#! /usr/bin/env python


from schema import Optional
from ..utils import RuntimeContext, RuntimeContextError


__all__ = ['PipelineRuntimeError', 'PipelineRuntime']


class PipelineRuntimeError(RuntimeContextError):
    """Raise when errors occur in `PipelineRuntime`."""


class PipelineRuntime(RuntimeContext):
    """A class that manages the runtime of the reduction pipeline."""

    @classmethod
    def extend_config_schema(cls):
        # this defines the subschema relevant to the simulator.
        return {
            'reduce': {
                'instrument': {
                    'name': str,
                    'calobj': str,
                    Optional(object): object
                    },
                'sources': [{
                    'path': str,
                    'select': str,
                    Optional(object): object
                    }],
                object: object
                },
            }

    def run(self):

        cfg = self.config['reduce']
        cfg_rt = self.config['runtime']

        self.logger.debug(f"cfg: {cfg}")
        self.logger.debug(f"cfg_rt: {cfg_rt}")
        return locals()
