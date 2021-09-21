#! /usr/bin/env python

from tollan.utils.fmt import pformat_yaml
from tollan.utils.log import get_logger, timeit, logit
from tollan.utils.schema import create_relpath_validator
from tollan.utils.registry import Registry, register_to
from tollan.utils.namespace import Namespace
from cached_property import cached_property

import numpy as np
import yaml
from schema import Optional, Use, Schema

from ..utils import RuntimeContext, RuntimeContextError, get_pkg_data_path
from ..datamodels.toltec import BasicObsDataset


__all__ = ['PipelineRuntimeError', 'PipelineRuntime']


_instru_pipeline_factory = Registry.create()
"""This holds the handler of the instrument pipeline config."""


@register_to(_instru_pipeline_factory, 'citlali')
def _ipf_citlali(cfg, cfg_rt):
    """Create and return `ToltecPipeline` from the config."""

    logger = get_logger()

    from .toltec import Citlali

    cfg = Schema({
        'name': 'citlali',
        Optional('config', default=None): dict,
        }).validate(cfg)

    cfg['engine'] = Citlali(
            binpath=cfg_rt['bindir'],
            version_specifiers=None,
            )
    logger.debug(f"pipeline config: {pformat_yaml(cfg)}")
    return cfg


class PipelineRuntimeError(RuntimeContextError):
    """Raise when errors occur in `PipelineRuntime`."""
    pass


class PipelineRuntime(RuntimeContext):
    """A class that manages the runtime of the reduction pipeline."""

    @classmethod
    def config_schema(cls):
        # this defines the subschema relevant to the simulator.
        return Schema({
            'reduce': {
                'jobkey': str,
                'pipeline': {
                    'name': str,
                    'config': dict,
                    },
                'inputs': [{
                    'path': str,
                    # 'select': str,
                    Optional(object): object
                    }],
                # 'calobj': str,
                # Optional('select', default=None): str
                },
            str: object,
            })

    @cached_property
    def config(self):
        cfg = super().config
        return self.config_schema().validate(cfg)

    def get_pipeline_params(self):
        """Return the data reduction pipeline object specified in the runtime
        config."""
        cfg = self.config['reduce']
        cfg_rt = self.config['runtime_info']
        pl_params = _instru_pipeline_factory[cfg['pipeline']['name']](
                cfg['pipeline'], cfg_rt)
        return pl_params

    def get_input_dataset(self):
        """Return the input dataset."""
        cfg = self.config['reduce']
        cfg_rt = self.config['runtime_info']
        path_validator = create_relpath_validator(cfg_rt['rootpath'])

        def resolve_input_item(cfg):
            # index_table_filename='tolteca_index_table.ecsv'
            # index_table_path = datadir.joinpath(index_table_filename)
            cfg = Schema({
                'path': Use(path_validator),
                Optional('select', default=None): str
                }).validate(cfg)
            dataset = BasicObsDataset.from_files(
                    cfg['path'].glob('*'), open_=False)
            dataset = dataset.select(~np.equal(dataset['interface'], None))
            return dataset

        dataset = BasicObsDataset.vstack(
                map(resolve_input_item, cfg['inputs']))
        dataset.sort(['obsnum', 'interface'])
        # add special index columns for obsnums for backward selection
        for k in ['obsnum', 'subobsnum', 'scannum']:
            dataset[f'{k}_r'] = dataset[k].max() - dataset[k]
        self.logger.debug(f"collected {len(dataset)} data items")
        return dataset

    def run(self):
        """Run the pipeline.

        Returns
        -------
        `PipelineResult` : The result context containing the reduced data.
        """
        pl_params = self.get_pipeline_params()
        engine = pl_params['engine']
        input_dataset = self.get_input_dataset()
        self.logger.debug(f'{input_dataset}')
        with engine.proc_context(pl_params['config']) as proc:
            pl_info = proc(
                    input_dataset.select('obsnum_r == 0'),
                    self.get_or_create_output_dir()
                    )
        return pl_info

    @timeit
    def cli_run(self, args=None):
        """Run the pipeline and save the result.
        """
        self.run()
        # result = self.run()
        # result.save(self.get_or_create_output_dir())

    def get_or_create_output_dir(self):
        cfg = self.config['reduce']
        outdir = self.rootpath.joinpath(cfg['jobkey'])
        if not outdir.exists():
            with logit(self.logger.debug, 'create output dir'):
                outdir.mkdir(parents=True, exist_ok=True)
        return outdir


class PipelineResult(Namespace):
    """A class to hold the results of a pipeline run."""
    pass


def load_example_configs():

    example_dir = get_pkg_data_path().joinpath('examples')
    files = ['toltec_citlali_simple.yaml']

    def load_yaml(f):
        with open(f, 'r') as fo:
            return yaml.safe_load(fo)

    configs = {
            f'{f.stem}': load_yaml(f)
            for f in map(example_dir.joinpath, files)}
    return configs


example_configs = load_example_configs()
