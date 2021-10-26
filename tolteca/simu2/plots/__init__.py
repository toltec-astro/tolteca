#!/usr/bin/env python

from tollan.utils.dataclass_schema import DataclassNamespace
from schema import Schema, Optional
from .. import plots_registry
from .visibility import plot_visibility


_common_plotter_schema = {
    Optional('save', default=False, description='Save the figure.'): bool
    }


@plots_registry.register('visibility')
class VisibilityPlotConfig(DataclassNamespace):
    """The config class for visibility plot."""

    _namespace_from_dict_schema = Schema(dict(_common_plotter_schema))

    def __call__(self, cfg):
        """Make visibility plot for simulation config `cfg`."""
        target_coord = cfg.mapping.target_coord
        target_name = str(cfg.mapping.target)
        return plot_visibility(
            t0=cfg.mapping.t0,
            targets=[target_coord],
            target_names=[target_name],
            show=not self.save  # we handle the save later in the run() method
            )


@plots_registry.register('mapping')
class MappingPlotConfig(DataclassNamespace):
    """The config class for mapping trajectory plot."""

    _namespace_from_dict_schema = Schema(dict(_common_plotter_schema))
