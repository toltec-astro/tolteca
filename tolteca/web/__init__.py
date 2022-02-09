#!/usr/bin/env python

# from wrapt import ObjectProxy
from dataclasses import dataclass, field

from tollan.utils.dataclass_schema import add_schema
from tollan.utils.log import get_logger
from tollan.utils.fmt import pformat_yaml

from ..utils import RuntimeBase, RuntimeBaseError
from ..utils.config_registry import ConfigRegistry
from ..utils.config_schema import add_config_schema
from ..utils.env_schema import EnvMapper


__all__ = ['env_mapper', 'WebRuntime', 'WebRuntimeError']


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


class WebRuntimeError(RuntimeBaseError):
    pass


class WebRuntime(RuntimeBase):
    """A class that manages the runtime context of web apps.

    """
    config_cls = WebConfig

    logger = get_logger()

    def run(self, app_name, ext_proc_name='flask'):
        cfg = self.config

        self.logger.debug(
            f"run web app={app_name} ext_proc={ext_proc_name}"
            f" with config dict: "
            f"{pformat_yaml(cfg.to_config_dict())}")
        # invoke dasha cli
        from dasha.cli import run_site
        dasha_site_name = f'tolteca.web.apps.{app_name}'
        return run_site(args=['-s', dasha_site_name, ext_proc_name])


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
