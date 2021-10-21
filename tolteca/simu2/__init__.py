#!/usr/bin/env python

import astropy.units as u
from dataclasses import dataclass, field

from tollan.utils.dataclass_schema import add_schema
from tollan.utils.log import get_logger
from tollan.utils.fmt import pformat_yaml

from ..utils.common_schema import PhysicalTypeSchema
from ..utils.config_registry import ConfigRegistry
from ..utils.config_schema import add_config_schema
from ..utils.runtime_context import RuntimeContext, RuntimeContextError


__all__ = ['SimulatorRuntime', 'SimulatorRuntimeError']


@add_schema
@dataclass
class ObsParamsConfig(object):
    """The config class for ``simu.obs_params``."""

    t_exp: u.Quantity = field(
        default=10 << u.s,
        metadata={
            'description': 'The duration of the observation to simulate.',
            'schema': PhysicalTypeSchema("time"),
            }
        )

    f_smp_mapping: u.Quantity = field(
        default=12. << u.Hz,
        metadata={
            'description': 'The sampling frequency to '
                           'evaluate mapping models.',
            'schema': PhysicalTypeSchema("frequency"),
            }
        )
    f_smp_probing: u.Quantity = field(
        default=120. << u.Hz,
        metadata={
            'description': 'The sampling frequency '
                           'to evaluate detector signals.',
            'schema': PhysicalTypeSchema("frequency"),
            }
        )

    class Meta:
        schema = {
            'ignore_extra_keys': False,
            'description': 'The parameters related to observation.'
            }


@add_schema
@dataclass
class PerfParamsConfig(object):
    """The config class for ``simu.pef_params``."""

    chunk_len: u.Quantity = field(
        default=10 << u.s,
        metadata={
            'description': 'Chunk length to split the simulation to '
                           'reduce memory footprint.',
            'schema': PhysicalTypeSchema("time"),
            }
        )
    mapping_interp_len: u.Quantity = field(
        default=1 << u.s,
        metadata={
            'description': 'Interp length to speed-up mapping evaluation.',
            'schema': PhysicalTypeSchema("time"),
            }
        )
    erfa_interp_len: u.Quantity = field(
        default=300 << u.s,
        metadata={
            'description': 'Interp length to speed-up AltAZ to '
                           'ICRS coordinate transformation.',
            'schema': PhysicalTypeSchema("time"),
            }
        )
    anim_frame_rate: u.Quantity = field(
        default=300 << u.s,
        metadata={
            'description': 'Frame rate for plotting animation.',
            'schema': PhysicalTypeSchema("frequency"),
            }
        )

    class Meta:
        schema = {
            'ignore_extra_keys': False,
            'description': 'The parameters related to performance tuning.'
            }


mapping_registry = ConfigRegistry.create(
    name='MappingConfig',
    dispatcher_key='type',
    dispatcher_description='The mapping type.'
    )
"""The registry for ``simu.mapping``."""


instrument_registry = ConfigRegistry.create(
    name='InstrumentConfig',
    dispatcher_key='name',
    dispatcher_description='The instrument name.'
    )
"""The registry for ``simu.instrument``."""


sources_registry = ConfigRegistry.create(
    name='SourcesConfig',
    dispatcher_key='type',
    dispatcher_description='The simulator source type.'
    )
"""The registry for ``simu.sources``."""


plots_registry = ConfigRegistry.create(
    name='PlotsConfig',
    dispatcher_key='type',
    dispatcher_description='The plot type.'
    )
"""The registry for ``simu.plots``."""


exports_registry = ConfigRegistry.create(
    name='ExportsConfig',
    dispatcher_key='type',
    dispatcher_description='The export type.'
    )
"""The registry for ``simu.exports``."""

# Load submodules here to populate the registries
from . import mapping as _  # noqa: F401, E402, F811
from . import sources as _  # noqa: F401, E402, F811
from . import plots as _  # noqa: F401, E402, F811
from . import exports as _  # noqa: F401, E402, F811
from . import toltec as _  # noqa: F401, E402, F811
# from . import lmt as _  # noqa: F401, E402, F811


@add_config_schema
@add_schema
@dataclass
class SimuConfig(object):
    """The config for `tolteca.simu`."""

    jobkey: str = field(
        metadata={
            'description': 'The unique identifier the job.'
            }
        )
    instrument: dict = field(
        metadata={
            'description': 'The dict contains the instrument setting.',
            'schema': instrument_registry.schema,
            'pformat_schema_type': f'<{instrument_registry.name}>',
            })
    mapping: dict = field(
        metadata={
            'description': "The simulator mapping trajectory config.",
            'schema': mapping_registry.schema,
            'pformat_schema_type': f'<{mapping_registry.name}>'
            }
        )
    obs_params: ObsParamsConfig = field(
        metadata={
            'description': 'The dict contains the observation parameters.',
            })
    sources: list = field(
        default_factory=list,
        metadata={
            'description': 'The list contains input sources for simulation.',
            'schema': list(sources_registry.item_schemas),
            'pformat_schema_type': f"[<{sources_registry.name}>, ...]"
            })
    perf_params: PerfParamsConfig = field(
        default_factory=PerfParamsConfig,
        metadata={
            'description': 'The dict contains the performance related'
                           ' parameters.',
            })
    plots: list = field(
        default_factory=list,
        metadata={
            'description': 'The list contains config for plotting.',
            'schema': list(plots_registry.item_schemas),
            'pformat_schema_type': f"[<{plots_registry.name}>, ...]"
            })
    exports: list = field(
        default_factory=list,
        metadata={
            'description': 'The list contains config for exporting.',
            'schema': list(exports_registry.item_schemas),
            'pformat_schema_type': f"[<{exports_registry.name}>, ...]"
            })

    class Meta:
        schema = {
            'ignore_extra_keys': True,
            'description': 'The config dict for the simulator.'
            }
        config_key = 'simu'


class SimulatorRuntimeError(RuntimeContextError):
    """Raise when errors occur in `DatabaseRuntime`."""
    pass


class SimulatorRuntime(RuntimeContext):
    """A class that manages the runtime context of the simulator.

    This class drives the execution of the simulator.
    """

    logger = get_logger()

    @property
    def simu_config(self):
        return SimuConfig.from_config(self.config)

    def cli_run(self, args=None):
        """Run the simulator with CLI as save the result.
        """
        self.logger.debug(
            f"run simu with config: "
            f"{pformat_yaml(self.simu_config.to_config())}")
