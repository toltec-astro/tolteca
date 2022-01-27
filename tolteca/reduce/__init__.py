#!/usr/bin/env python

from dataclasses import dataclass, field, is_dataclass
from cached_property import cached_property
from tollan.utils.dataclass_schema import add_schema
from tollan.utils.log import get_logger, timeit, logit
from tollan.utils.fmt import pformat_yaml
from ..utils.config_registry import ConfigRegistry
from ..utils.config_schema import add_config_schema
from ..utils.runtime_context import RuntimeContext, RuntimeContextError
from ..utils import dict_from_cli_args


__all__ = ['ReduConfig', 'PipelineRuntime', 'PipelineRuntimeError']


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


class PipelineRuntimeError(RuntimeContextError):
    """Raise when errors occur in `PipelineRuntime`."""
    pass


class PipelineRuntime(RuntimeContext):
    """A class that manages the runtime context of the data reduction pipeline.

    This class drives the execution of the data reduction.
    """

    config_cls = ReduConfig

    logger = get_logger()

    @cached_property
    def redu_config(self):
        """Validate and return the reduction config object..

        The validated reduction config is cached.
        :meth:`PipelineRuntime.update`
        should be used to update the underlying config and re-validate.
        """
        return self.config_cls.from_config(
            self.config, rootpath=self.rootpath,
            runtime_info=self.runtime_info)

    def update(self, config):
        self.config_backend.update_override_config(config)
        if 'redu_config' in self.__dict__:
            del self.__dict__['redu_config']

    def cli_run(self, args=None):
        """Run the reduction with CLI and save the result.
        """
        if args is not None:
            _cli_cfg = dict_from_cli_args(args)
            # note the cli_cfg is under the namespace redu
            cli_cfg = {self.config_cls.config_key: _cli_cfg}
            if _cli_cfg:
                self.logger.info(
                    f"config specified with commandline arguments:\n"
                    f"{pformat_yaml(cli_cfg)}")
            self.update(cli_cfg)
            cfg = self.redu_config.to_config()
            # here we recursively check the cli_cfg and report
            # if any of the key is ignored by the schema and
            # throw an error

            def _check_ignored(key_prefix, d, c):
                if isinstance(d, dict) and isinstance(c, dict):
                    ignored = set(d.keys()) - set(c.keys())
                    ignored = [f'{key_prefix}.{k}' for k in ignored]
                    if len(ignored) > 0:
                        raise PipelineRuntimeError(
                            f"Invalid config items specified in "
                            f"the commandline: {ignored}")
                    for k in set(d.keys()).intersection(c.keys()):
                        _check_ignored(f'{key_prefix}{k}', d[k], c[k])
            _check_ignored('', cli_cfg, cfg)
        return self.run()

    def run(self):
        """Run the reduction."""

        cfg = self.redu_config

        self.logger.debug(
            f"run reduction with config dict: "
            f"{pformat_yaml(cfg.to_config())}")
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


# make a list of all reduce config item types
_locals = list(locals().values())
redu_config_item_types = list()
for v in _locals:
    if is_dataclass(v) and hasattr(v, 'schema'):
        redu_config_item_types.append(v)
    elif isinstance(v, ConfigRegistry):
        redu_config_item_types.extend(list(v.values()))
