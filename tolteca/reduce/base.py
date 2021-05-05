#! /usr/bin/env python

from tollan.utils.log import get_logger
from tollan.utils.registry import Registry


class PipelineEngine(object):
    """Base class for reduction pipeline engine."""

    logger = get_logger()
    _subclasses = Registry.create()

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls._subclasses.register(cls, cls)

    @property
    def version(self):
        """The version of the pipeline."""
        return self._version

    def proc_context(self, *args, **kwargs):
        """Implement to return the processing function."""
        return NotImplemented
