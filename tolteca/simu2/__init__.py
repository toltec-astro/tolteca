#!/usr/bin/env python

import astropy.units as u
from typing import Union
from dataclasses import dataclass, field, is_dataclass
from cached_property import cached_property
import copy
from typing import ClassVar

from tollan.utils.dataclass_schema import add_schema
from tollan.utils.log import get_logger, logit, log_to_file
from tollan.utils.fmt import pformat_yaml
from tollan.utils import rupdate

from ..utils.common_schema import PhysicalTypeSchema
from ..utils.config_registry import ConfigRegistry
from ..utils.config_schema import add_config_schema
from ..utils.runtime_context import RuntimeContext, RuntimeContextError
from ..utils import config_from_cli_args


__all__ = ['SimulatorRuntime', 'SimulatorRuntimeError']


@add_schema
@dataclass
class ObsParamsConfig(object):
    """The config class for ``simu.obs_params``."""

    t_exp: Union[u.Quantity, None] = field(
        default=None,
        metadata={
            'description': 'The duration of the observation to simulate.',
            'schema': PhysicalTypeSchema('time'),
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
    catalog_model_render_pixel_size: u.Quantity = field(
        default=0.5 << u.arcsec,
        metadata={
            'description': 'Pixel size to render catalog source model.',
            'schema': PhysicalTypeSchema("angle"),
            }
        )
    mapping_eval_interp_len: Union[u.Quantity, None] = field(
        default=None,
        metadata={
            'description': 'Interp length to speed-up mapping evaluation.',
            'schema': PhysicalTypeSchema("time"),
            }
        )
    mapping_erfa_interp_len: u.Quantity = field(
        default=300 << u.s,
        metadata={
            'description': 'Interp length to speed-up AltAZ to '
                           'ICRS coordinate transformation.',
            'schema': PhysicalTypeSchema("time"),
            }
        )
    atm_eval_interp_alt_step: u.Quantity = field(
        default=6 << u.arcmin,
        metadata={
            'description': 'Interp altitude step to speed-up atm eval.',
            'schema': PhysicalTypeSchema("angle"),
            }
        )
    pre_eval_t_grid_size: int = field(
        default=100,
        metadata={
            'description': 'Size of time grid used for pre-eval calculations.',
            'schema': PhysicalTypeSchema("angle"),
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
    plot_only: bool = field(
        default=False,
        metadata={
            'description': 'Make plots of those defined in `plots`.'
            })

    class Meta:
        schema = {
            'ignore_extra_keys': True,
            'description': 'The config dict for the simulator.'
            }
        config_key = 'simu'

    logger: ClassVar = get_logger()

    def get_or_create_output_dir(self):
        logger = get_logger()
        rootpath = self.runtime_info.config_info.runtime_context_dir
        output_dir = rootpath.joinpath(self.jobkey)
        if not output_dir.exists():
            with logit(logger.debug, 'create output dir'):
                output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def get_log_file(self):
        return self.runtime_info.logdir.joinpath('simu.log')

    @cached_property
    def mapping_model(self):
        return self.mapping(self)

    @cached_property
    def source_models(self):
        return [s(self) for s in self.sources]

    @cached_property
    def simulator(self):
        return self.instrument(self)

    @cached_property
    def t_simu(self):
        """The length of the simulation.

        It equals `obs_params.t_exp` when set, otherwise ``t_pattern``
        of the mapping pattern is used.
        """
        t_simu = self.obs_params.t_exp
        if t_simu is None:
            t_pattern = self.mapping_model.t_pattern
            self.logger.debug(f"mapping pattern time: {t_pattern}")
            t_simu = t_pattern
            self.logger.info(f"use t_simu={t_simu} from mapping pattern")
        else:
            self.logger.info(f"use t_simu={t_simu} from obs_params")
        return t_simu


class SimulatorRuntimeError(RuntimeContextError):
    """Raise when errors occur in `SimulatorRuntime`."""
    pass


class SimulatorRuntime(RuntimeContext):
    """A class that manages the runtime context of the simulator.

    This class drives the execution of the simulator.
    """

    config_cls = SimuConfig

    logger = get_logger()

    @cached_property
    def simu_config(self):
        """Validate and return the simulator config object..

        The validated config is cached. :meth:`SimulatorRuntime.update`
        should be used to update the underlying config and re-validate.
        """
        return self.config_cls.from_config(
            self.config, rootpath=self.rootpath,
            runtime_info=self.runtime_info)

    def update(self, config):
        self.config_backend.update_override_config(config)
        if 'simu_config' in self.__dict__:
            del self.__dict__['simu_config']

    def cli_run(self, args=None):
        """Run the simulator with CLI as save the result.
        """
        if args is not None:
            _cli_cfg = config_from_cli_args(args)
            # note the cli_cfg is under the namespace simu
            cli_cfg = {self.config_cls.config_key: _cli_cfg}
            if _cli_cfg:
                self.logger.info(
                    f"config specified with commandline arguments:\n"
                    f"{pformat_yaml(cli_cfg)}")
            self.update(cli_cfg)
            cfg = self.simu_config.to_config()
            # here we recursively check the cli_cfg and report
            # if any of the key is ignored by the schema and
            # throw an error

            def _check_ignored(key_prefix, d, c):
                if isinstance(d, dict) and isinstance(c, dict):
                    ignored = set(d.keys()) - set(c.keys())
                    ignored = [f'{key_prefix}.{k}' for k in ignored]
                    if len(ignored) > 0:
                        raise SimulatorRuntimeError(
                            f"Invalid config items specified in "
                            f"the commandline: {ignored}")
                    for k in set(d.keys()).intersection(c.keys()):
                        _check_ignored(f'{key_prefix}{k}', d[k], c[k])
            _check_ignored('', cli_cfg, cfg)
        return self.run()

    def run(self):
        """Run the simulator."""

        cfg = self.simu_config

        self.logger.debug(
            f"run simu with config dict: "
            f"{pformat_yaml(cfg.to_config())}")

        if cfg.plot_only:
            results = []
            for plotter in cfg.plots:
                result = plotter(cfg)
                results.append(result)
                if plotter.save:
                    # TODO handle save here
                    pass
            return results

        # run simulator
        simu = cfg.simulator
        t_simu = cfg.t_simu
        mapping_model = cfg.mapping_model
        source_models = cfg.source_models
        output_dir = cfg.get_or_create_output_dir()

        self.logger.debug(
            f'run {simu} with:{{}}\n'.format(
                pformat_yaml({
                    'obs_params': cfg.obs_params.to_dict(),
                    'perf_params': cfg.perf_params.to_dict(),
                    })))
        self.logger.debug(
            'mapping:\n{}\n\nsources:\n{}\n'.format(
                mapping_model,
                '\n'.join(str(s) for s in source_models)
                )
            )
        self.logger.debug(
            f'simu output dir: {output_dir}\nsimu length={t_simu}'
            )
        # run the simulator
        log_file = cfg.get_log_file()
        self.logger.info(f'setup logging to file {log_file}')
        with log_to_file(
                filepath=log_file,
                level='DEBUG',
                disable_other_handlers=False
                ):
            output_ctx = simu.output_context(dirpath=output_dir)
            with output_ctx.open():
                self.logger.info(
                    f"write output to {output_ctx.rootpath}")
                # save the config file as YAML
                config_filepath = output_ctx.make_output_filename(
                    'tolteca', '.yaml')
                with open(config_filepath, 'w') as fo:
                    config = copy.deepcopy(self.config)
                    rupdate(config, self.simu_config.to_config())
                    self.yaml_dump(config, fo)
                # save mapping model meta
                output_ctx.write_mapping_meta(
                    mapping=mapping_model, simu_config=cfg)
                # save simulator meta
                output_ctx.write_sim_meta(simu_config=cfg)

                # run simulator for each chunk and save the data
                with simu.iter_eval_context(cfg) as (iter_eval, t_chunks):
                    n_chunks = len(t_chunks)
                    for ci, t in enumerate(t_chunks):
                        self.logger.info(
                            f"working on chunk {ci} of {n_chunks}")
                        output_ctx.write_sim_data(iter_eval(t))
        return output_dir

    def plot(self, type, **kwargs):
        """Make plot of type `type`."""
        if type not in plots_registry:
            raise ValueError(
                f"Invalid plot type {type}. "
                f"Available types: {plots_registry.keys()}")
        plotter = plots_registry[type].from_dict(kwargs)
        return plotter(self.simu_config)


# make a list of all simu config item types
_locals = list(locals().values())
simu_config_item_types = list()
for v in _locals:
    if is_dataclass(v) and hasattr(v, 'schema'):
        simu_config_item_types.append(v)
    elif isinstance(v, ConfigRegistry):
        simu_config_item_types.append(v)
