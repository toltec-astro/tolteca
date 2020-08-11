#! /usr/bin/env python

from tollan.utils.log import get_logger
from tollan.utils.registry import Registry


__all__ = ['IORegistry', 'io_registry', 'open_file']


class IORegistry(object):
    """A helper class to manage a registry for different IO classes.

    This is a very generic implementation that merely ties together a textual
    label (format), an identification function, and a factory (e.g, a class
    constructor)::

        label ->  identifier -> factory

    All registered labels are then valid ``format`` to be used in :meth:`open`,
    which returns the constructed object of the identified factory.
    """

    def __init__(self):
        self._io_classes = Registry.create()
        self._identifiers = Registry.create()

    @property
    def io_classes(self):
        """The mapping between label and io class."""
        return self._io_classes

    @property
    def identifiers(self):
        """The mapping between label and identifier."""
        return self._identifiers

    def clear(self):
        """Clear the registry."""
        self.io_classes.clear()
        self.identifiers.clear()

    def register(self, user_cls, label, identifier):
        """Register `user_cls` with `label` and `identifier`."""
        self.io_classes.register(label, user_cls)
        self.identifiers.register(label, identifier)

    def register_as(self, *args, **kwargs):
        """"Return a decorator that register the decorated class."""

        def decorator(user_cls):
            self.register(user_cls, *args, **kwargs)
            return user_cls
        return decorator

    def _get_identifier(self, format_):
        if format_ in self.identifiers:
            return self.identifiers[format_]
        raise RuntimeError(f"unknown format {format_}")

    def open(self, source, format=None, *args, **kwargs):
        """
        Return constructed object for given `source`.

        Parameters
        ----------
        source : str, object
            The source to open. This can be file, URL, or anything that may be
            identified by the identifiers registered.
        format : str, optional
            The format of the `source`. This is used to query the registry
            to get the identifier and the factory.
        *args **kwargs
            Additional arguments passed to the factory.
        """
        logger = get_logger()
        if format is not None:
            identifiers = [(format, self._get_identifier(format))]
        else:
            identifiers = list(self.identifiers.items())
        # logger.debug(f"try open with identifiers: {identifiers}")
        for format_, identifier in identifiers:
            if identifier(source):
                logger.debug(f"identified as format={format_}")
                result = self.io_classes[format_](source, *args, **kwargs)
                break
        else:
            supported_formats = list(self.identifiers.keys())
            if format is None:
                msg = (
                    f'unable to identify {source} format in '
                    f'{supported_formats}')
            else:
                msg = (
                    f'unable to open {source} with with format={format}. '
                    f" Supported formats: {supported_formats}")
            raise RuntimeError(msg)
        return result


io_registry = IORegistry()
"""A global IO registry instance."""


open_file = io_registry.open
"""A alias of `~tolteca.datamodels.io.registry.io_registry.open`"""
