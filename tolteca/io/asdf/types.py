# -*- coding: utf-8 -*-

"""
Defines a ``ToltecaType`` used for ASDF file io.

All types are added automatically to ``_tolteca_types`` and the
ToltecaExtension.

"""

from asdf.types import ExtensionTypeMeta, CustomType


__all__ = ['ToltecaType', ]


_tolteca_types = set()


class ToltecaTypeMeta(ExtensionTypeMeta):
    """
    Keeps track of `ToltecaType` subclasses that are created so that they can
    be stored automatically by Astropy extensions for ASDF.
    """
    def __new__(mcls, name, bases, attrs):
        cls = super().__new__(mcls, name, bases, attrs)
        # Classes using this metaclass are automatically added to the list of
        # jwst types and JWSTExtensions.types.
        if issubclass(cls, ToltecaType):
            _tolteca_types.add(cls)
        return cls


class ToltecaType(CustomType, metaclass=ExtensionTypeMeta):
    """
    This class represents types that have schemas and tags
    implemented within `tolteca.io`.

    """
    organization = 'astro.umass.edu'
    standard = 'tolteca'

    def __init_subclass__(cls):
        super().__init_subclass__()
        _tolteca_types.add(cls)
