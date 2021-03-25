#! /usr/bin/env python

from tollan.utils.log import get_logger, timeit
from tollan.utils.schema import create_relpath_validator
from tollan.utils.registry import Registry, register_to
from tollan.utils.namespace import Namespace

from schema import Optional, Use, Schema

from ..utils import RuntimeContext, RuntimeContextError


__all__ = ['PipelineRuntimeError', 'PipelineRuntime']


_instru_pipeline_factory = Registry.create()
"""This holds the handler of the instrument pipeline config."""


@register_to(_instru_pipeline_factory, 'citlali')
def _ipf_citlali(cfg, cfg_rt):
    """Create and return `ToltecPipeline` from the config."""

    logger = get_logger()

    from ..cal import ToltecCalib
    from .toltec import Citlali

    path_validator = create_relpath_validator(cfg_rt['rootpath'])

    def get_calobj(p):
        return ToltecCalib.from_indexfile(path_validator(p))

    cfg = Schema({
        'name': 'citlali',
        Optional('config', default=None): dict,
        }).validate(cfg)

    cfg['pipeline'] = Citlali(
            binpath=cfg_rt['bindir'],
            version_specifiers=None,
            )
    logger.debug(f"pipeline config: {cfg}")
    return cfg


class PipelineRuntimeError(RuntimeContextError):
    """Raise when errors occur in `PipelineRuntime`."""
    pass


class PipelineRuntime(RuntimeContext):
    """A class that manages the runtime of the reduction pipeline."""

    @classmethod
    def extend_config_schema(cls):
        # this defines the subschema relevant to the simulator.
        return {
            'reduce': {
                'pipeline': {
                    'name': str,
                    'config': dict,
                    },
                'inputs': [{
                    'path': str,
                    'select': str,
                    Optional(object): object
                    }],
                'calobj': str,
                Optional('select', default=None): str
                },
            }

    def get_pipeline(self):
        """Return the data reduction pipeline object specified in the runtime
        config."""
        cfg = self.config['reduce']
        cfg_rt = self.config['runtime']
        ppl = _instru_pipeline_factory[cfg['pipeline']['name']](
                cfg['pipeline'], cfg_rt)
        return ppl

    def run(self):
        """Run the pipeline.

        Returns
        -------
        `PipelineResult` : The result context containing the reduced data.
        """

        ppl = self.get_pipeline()

        print(ppl)
        return locals()

    @timeit
    def cli_run(self, args=None):
        """Run the pipeline and save the result.
        """

        result = self.run()
        result.save(self.get_or_create_output_dir())


class PipelineResult(Namespace):
    """A class to hold the results of a pipeline run."""
    pass
