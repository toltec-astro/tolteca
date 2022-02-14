#!/usr/bin/env python

from dataclasses import dataclass, field
from tollan.utils.dataclass_schema import add_schema
from tollan.utils.log import get_logger, timeit, logit
from tollan.utils.fmt import pformat_yaml
from ..utils.config_registry import ConfigRegistry
from ..utils.config_schema import add_config_schema
from ..utils import RuntimeBase, RuntimeBaseError
from ..utils.doc_helper import collect_config_item_types


__all__ = [
    'ReduConfig',
    'PipelineRuntime', 'PipelineRuntimeError',
    ]


inputs_registry = ConfigRegistry.create(
    name='InputsConfig',
    dispatcher_key='type',
    dispatcher_description='The input type.',
    dispatcher_key_is_optional=True
    )
"""The registry for ``reduce.inputs``."""


steps_registry = ConfigRegistry.create(
    name='StepsConfig',
    dispatcher_key='name',
    dispatcher_description='The reduction step name.'
    )
"""The registry for ``reduce.steps``."""


# Load submodules here to populate the registries
from . import engines as _  # noqa: F401, E402, F811
from . import toltec as _  # noqa: F401, E402, F811


@add_config_schema
@add_schema
@dataclass
class ReduConfig(object):
    """The config for `tolteca.reduce`."""

    jobkey: str = field(
        metadata={
            'description': 'The unique identifier the job.'
            }
        )
    inputs: list = field(
        default_factory=list,
        metadata={
            'description': 'The list contains input data for reduction.',
            'schema': list(inputs_registry.item_schemas),
            'pformat_schema_type': f"[<{inputs_registry.name}>, ...]"
            })
    steps: list = field(
        default_factory=list,
        metadata={
            'description': 'The list contains the defs of pipeline steps.',
            'schema': list(steps_registry.item_schemas),
            'pformat_schema_type': f"[<{steps_registry.name}>, ...]"
            })

    class Meta:
        schema = {
            'ignore_extra_keys': True,
            'description': 'The config dict for running reduction.'
            }
        config_key = 'reduce'

    def load_input_data(self):
        """Return loaded data objects from reduction config."""
        # This go through all registered data loaders and load the all the
        # input items. The loaded input items get aggregated on a per loader
        # basis
        data_collection = list()
        data_loaders = set()
        for input in self.inputs:
            data_collection.extend(input.load_all_data())
            data_loaders.update(input.get_data_loaders())
        aggregated_data_collection = list()
        for data_loader in data_loaders:
            aggregated_data_collection.extend(
                data_loader.aggregate(data_collection))
        return aggregated_data_collection

    def get_or_create_output_dir(self):
        logger = get_logger()
        rootpath = self.runtime_info.config_info.runtime_context_dir
        output_dir = rootpath.joinpath(self.jobkey)
        if not output_dir.exists():
            with logit(logger.debug, 'create output dir'):
                output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def get_log_file(self):
        return self.runtime_info.logdir.joinpath('reduce.log')


class PipelineRuntimeError(RuntimeBaseError):
    """Raise when errors occur in `PipelineRuntime`."""
    pass


class PipelineRuntime(RuntimeBase):
    """A class to run the data reduction pipeline.
    """

    config_cls = ReduConfig

    logger = get_logger()

    def run(self):
        """Run the reduction."""

        cfg = self.config

        self.logger.debug(
            f"run reduction with config dict: "
            f"{pformat_yaml(cfg.to_config_dict())}")
        # TODO
        # later on we'll enable the DAG like pipeline step
        # management facility but for now we'll just run each step
        # sequentially.
        n_steps = len(cfg.steps)
        tmp_data = cfg.load_input_data()
        self.logger.info(f"collected data from inputs: {tmp_data!r}")
        if len(cfg.steps) == 0:
            self.logger.warning("no pipeline steps found, nothing to do.")
        for i, step in enumerate(cfg.steps):
            with timeit(
                    f"run pipeline step [{i + 1}/{n_steps}] {step.name}",
                    level='INFO',
                    ):
                self.logger.debug(f"input data {tmp_data!r}")
                tmp_data = step.run(cfg, inputs=tmp_data)
                self.logger.debug(f"output data {tmp_data!r}")
        self.logger.info("work's done!")
        return tmp_data


# make a list of all redu config item types
redu_config_item_types = collect_config_item_types(list(locals().values()))
