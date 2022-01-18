#!/usr/bin/env python

from tollan.utils.dataclass_schema import DataclassNamespace
from schema import Schema, Optional
from astropy.time import Time
from tollan.utils.log import get_logger
from ...version import version
from .. import exports_registry
from .lmtot import LmtOTComposer


@exports_registry.register('lmtot')
class LmtOTExporterConfig(DataclassNamespace):
    """The config class for LMT OT script exporter."""

    _namespace_from_dict_schema = Schema({
        Optional('save', default=True, description='Save the exported data.'):
        bool
        })

    logger = get_logger()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lmtot_composer = LmtOTComposer()

    def __call__(self, cfg):
        """Make LMT OT script from simulator config `cfg`."""
        lmtot_composer = self._lmtot_composer
        runtime_info = cfg.runtime_info
        rootpath = runtime_info.config_info.runtime_context_dir
        preamble = (
            f'# LMT OT script generated with tolteca.simu v{version}\n'
            f'# Created at: {Time.now()}\n'
            f'# Created by: {runtime_info.username}@{runtime_info.hostname}\n'
            )
        steps = [preamble, ]
        steps.extend(lmtot_composer.make_setup_steps(
            instru_name=cfg.instrument.name))
        steps.extend(lmtot_composer.make_pointing_steps() or list())
        steps.extend(lmtot_composer.make_mapping_steps(
            mapping_model=cfg.mapping_model,
            target_name=cfg.mapping.target,
            t_exp=cfg.t_simu
            ))
        content = '\n'.join(steps)
        if self.save:
            output_filepath = rootpath.joinpath(f'{cfg.jobkey}.lmtot')
            with open(output_filepath, 'w') as fo:
                fo.write(content)
            self.logger.info(f"LMT OT script exported to {output_filepath}")
        return content
