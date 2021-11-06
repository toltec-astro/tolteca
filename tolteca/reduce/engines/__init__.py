#!/usr/bin/env python


from .. import steps_registry
from ...utils.common_schema import RelPathSchema
from tollan.utils.dataclass_schema import DataclassNamespace
from tollan.utils.log import get_logger
import os
import functools
from schema import Optional, Schema
from .citlali import Citlali


@steps_registry.register('citlali')
class CitlaliConfig(DataclassNamespace):
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
            'config',
            default=dict,
            description='The config dict passed to Citlali.'
            ):
        dict
        })

    logger = get_logger()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _get_bindir_citlali_paths(bindir):
        paths = list()
        for p in bindir.glob("citlali_*"):
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
        engine = Citlali(path=paths, version=self.version)
        if path is None and len(
                {'>', '<'}.intersection(set(self.version or ''))) == 0:
            # we only check for update when the engine executable is requested
            # or a specific version is used
            engine.check_for_update()

    def run(self, cfg, inputs=None):
        """Run this reduction step."""
        if inputs is None:
            inputs = cfg.load_input_data()
        engine = self(cfg)
        return engine.run(inputs)
