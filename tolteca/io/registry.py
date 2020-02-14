#! /usr/bin/env python

from tollan.utils.log import get_logger
from tollan.utils.registry import Registry


_io_classes = Registry.create()
_identifiers = Registry.create()


def register_io_class(label, identifier=None):

    def decorator(cls):
        _io_classes.register(label, cls)
        if identifier is not None:
            _identifiers.register(label, identifier)
        return cls
    return decorator


def _get_identifier(format_):
    if format_ in _identifiers:
        return _identifiers[format_]
    else:
        raise RuntimeError(f"unknown format {format_}")


def open(source, format=None, *args, **kwargs):
    logger = get_logger()
    if format is not None:
        identifiers = [(format, _get_identifier(format))]
    else:
        identifiers = list(_identifiers.items())
    logger.debug(f"try open with identifiers: {identifiers}")
    for format_, identifier in identifiers:
        if identifier(source):
            logger.debug(f"identified as format={format_}")
            io = _io_classes[format_](source, *args, **kwargs)
            break
    else:
        raise RuntimeError(
                f"unable to open {source} with format={format}."
                " Supported formats: {_io_classes.keys()}")
    return io
