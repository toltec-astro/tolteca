#! /usr/bin/env python

from ..utils import rreload
from ..utils.log import get_logger, timeit
from contextlib import ExitStack
import importlib


class CliRuntime(object):
    """This class manages all states for the CLI."""

    logger = get_logger()

    config = dict(
        default_modules=('io', 'db')
        )

    def __init__(self, site_config=None):
        self._exitstack = ExitStack()
        self._modules = dict()

    def __getattr__(self, name, *args):
        if name in self._modules:
            return self._modules[name]
        return super().__getattribute__(name, *args)

    def __exit__(self, *exc_details):
        self._exitstack.__exit__(*exc_details)

    def close(self):
        return self.__exit__(None)

    @staticmethod
    def _normalize_module_name(name):
        # normalize the name to be all relative to
        # the package root
        name = name.lstrip(".")
        qualname = f'..{name}'
        return name, qualname

    @property
    def modules(self):
        return self._modules

    def load_module(self, name):
        name, qualname = self._normalize_module_name(name)
        if name in self._modules:
            pass
        else:
            # relative to the package root
            self._modules[name] = timeit(importlib.import_module)(
                    qualname, package=__package__)
            self.logger.debug(
                    f"imported module {name}: "
                    f"{getattr(self, name)}")

    @timeit
    def load_modules(self, *args, with_default=True):
        if with_default:
            names = args + self.__class__.config.get('default_modules', ())
        else:
            names = args
        for name in names:
            self.load_module(name)
        self.logger.debug(
                f"imported {len(names)} modules: {names}")

    def reload_module(self, name):
        name, _ = self._normalize_module_name(name)
        self.logger.debug(
                f"reload module {name}: "
                f"{getattr(self, name)}")
        # recursively reload all submodules
        timeit(rreload)(self._modules[name])

    @timeit
    def reload_modules(self):
        for name in self._modules.keys():
            self.reload_module(name)
