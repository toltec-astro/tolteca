#!/usr/bin/env python

# from wrapt import ObjectProxy
from dataclasses import dataclass, field
from cached_property import cached_property

from tollan.utils.dataclass_schema import add_schema

from ..utils.runtime_context import RuntimeContext, RuntimeContextError
from ..utils.config_registry import ConfigRegistry
from ..utils.config_schema import add_config_schema
from ..utils.env_schema import EnvMapper


__all__ = ['env_mapper', ]


env_mapper = EnvMapper(env_prefix='TOLTECA_WEB')
"""An env mapper instance shared by all apps."""


apps_registry = ConfigRegistry.create(
    name='AppsConfig',
    dispatcher_key='name',
    dispatcher_description='The app name.',
    dispatcher_key_is_optional=False
    )
"""The registry for ``web.apps``."""

# Load apps here to populate the registries
from . import apps as _  # noqa: F401, E402, F811


@add_config_schema
@add_schema
@dataclass
class WebConfig(object):
    """The config for `tolteca.web`."""

    apps: list = field(
        default_factory=list,
        metadata={
            'description': 'The list contains config for apps.',
            'schema': list(apps_registry.item_schemas),
            'pformat_schema_type': f"[<{apps_registry.name}>, ...]"
            })

    class Meta:
        schema = {
            'ignore_extra_keys': True,
            'description': 'The config dict for web apps.'
            }
        config_key = 'web'


class WebRuntime(RuntimeContext):
    """A class that manages the runtime context of web apps.

    """
    config_cls = WebConfig

    logger = get_logger()

    @cached_property
    def web_config(self):
        """Validate and return the web config object."""
        return self.config_cls.from_config(
            self.config, rootpath=self.rootpath,
            runtime_info=self.runtime_info)

    def update(self, config):
        self.config_backend.update_override_config(config)
        if 'redu_config' in self.__dict__:
            del self.__dict__['redu_config']

    def cli_run(self, args=None):
        """Run the reduction with CLI and save the result.
        """
        if args is not None:
            _cli_cfg = config_from_cli_args(args)
            # note the cli_cfg is under the namespace redu
            cli_cfg = {self.config_cls.config_key: _cli_cfg}
            if _cli_cfg:
                self.logger.info(
                    f"config specified with commandline arguments:\n"
                    f"{pformat_yaml(cli_cfg)}")
            self.update(cli_cfg)
            cfg = self.redu_config.to_config()
            # here we recursively check the cli_cfg and report
            # if any of the key is ignored by the schema and
            # throw an error

            def _check_ignored(key_prefix, d, c):
                if isinstance(d, dict) and isinstance(c, dict):
                    ignored = set(d.keys()) - set(c.keys())
                    ignored = [f'{key_prefix}.{k}' for k in ignored]
                    if len(ignored) > 0:
                        raise PipelineRuntimeError(
                            f"Invalid config items specified in "
                            f"the commandline: {ignored}")
                    for k in set(d.keys()).intersection(c.keys()):
                        _check_ignored(f'{key_prefix}{k}', d[k], c[k])
            _check_ignored('', cli_cfg, cfg)
        return self.run()



# @env_mapper
# @add_schema
# @dataclass
# class SiteConfig(object):
#     """The config class for `ObsPlanner`."""

#     site_name: str = field(
#         default='lmt',
#         metadata={
#             'description': 'The observing site name',
#             'schema': Or("lmt", )
#             }
#         )
#     instru_name: str = field(
#         default='toltec',
#         metadata={
#             'description': 'The observing instrument name',
#             'schema': Or("toltec", )
#             }
#         )
#     pointing_catalog_path: Union[None, Path] = field(
#         default=None,
#         metadata={
#             'description': 'The catalog path containing the pointing sources',
#             'schema': RelPathSchema()
#             }
#         )
#     # view related
#     title_text: str = field(
#         default='Obs Planner',
#         metadata={
#             'description': 'The title text of the page'}
#         )

#     class Meta:
#         schema = {
#             'ignore_extra_keys': False,
#             'description': 'The config of obs planner.'
#             }



