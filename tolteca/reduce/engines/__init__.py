#!/usr/bin/env python


from .. import steps_registry
from ...utils.common_schema import RelPathSchema
from ...datamodels.toltec import BasicObsDataset
from tollan.utils.dataclass_schema import DataclassNamespace
from tollan.utils.log import get_logger, log_to_file
import os
import functools
from schema import Optional, Schema, Or
from .citlali import Citlali, CitlaliConfig


@steps_registry.register('citlali')
class CitlaliStepConfig(DataclassNamespace):
    """The config class for reduction with Citlali."""

    _namespace_from_dict_schema = Schema({
        Optional(
            'path',
            default=None,
            description='The path to find the executable.'):
        RelPathSchema(),
        Optional(
            'version',
            default=None,
            description='A version string or version predicate to specify the '
                        'version of citlali.'):
        str,
        Optional(
            'log_level',
            default='INFO',
            description='The log level for the Citlali call.'):
        Or("TRACE", "DEBUG", 'INFO'),
        Optional(
            'config',
            default=CitlaliConfig,
            description='The config dict passed to Citlali.'
            ):
        CitlaliConfig.schema
        })

    logger = get_logger()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _get_bindir_citlali_paths(bindir):
        paths = list()
        for p in [bindir.joinpath('citlali')] + list(bindir.glob("citlali_*")):
            if os.access(p, os.X_OK):
                paths.append(p)
        return paths

    def __call__(self, cfg):
        runtime_info = cfg.runtime_info

        # check the bindir for a list of possible citlali paths
        # create a list of paths to search for citlali executable
        # here we glob the bindir to find multiple versions of citlali
        # with format "citlali_*"
        path = self.path
        if path is not None:
            paths = [path]
        else:
            paths = []
        paths.extend(self._get_bindir_citlali_paths(runtime_info.bindir))
        self.logger.debug(f"create Citlali with paths={paths}")
        engine = Citlali(path=paths, version=self.version)
        if path is None or len(
                {'>', '<', '~'}.intersection(set(self.version or ''))) == 0:
            # we only check for update when the engine executable is requested
            # or a specific version is used
            engine.check_for_update()
        return engine

    def run(self, cfg, inputs=None):
        """Run this reduction step."""
        if inputs is None:
            inputs = cfg.load_input_data()
        # get bods
        bods = [
            input for input in inputs
            if isinstance(input, BasicObsDataset)
            ]
        if len(bods) == 0:
            self.logger.debug("no valid input for this step, skip")
            return None
        assert len(bods) == 1
        bods = bods[0]
        output_dir = cfg.get_or_create_output_dir()
        log_file = cfg.get_log_file()
        logger = get_logger("citlali")
        engine = self(cfg)
        self.logger.info(f'setup logging to file {log_file}')
        with log_to_file(
                filepath=log_file,
                level='DEBUG',
                disable_other_handlers=False
                ) as handler, engine.proc_context(
                    self.config
                    ) as proc:
            logger.propagate = False
            logger.addHandler(handler)
            return proc(
                dataset=bods,
                output_dir=output_dir,
                log_level=self.log_level,
                logger_func=logger.info)
