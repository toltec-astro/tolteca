#!/usr/bin/env python

from tollan.utils.dataclass_schema import DataclassNamespace
from schema import Schema
from .. import plots_registry


@plots_registry.register('visibility')
class VisibilityPlotConfig(DataclassNamespace):
    """The config class for visibility plot."""

    _namespace_from_dict_schema = Schema({})


@plots_registry.register('mapping')
class MappingPlotConfig(DataclassNamespace):
    """The config class for mapping trajectory plot."""

    _namespace_from_dict_schema = Schema({})
