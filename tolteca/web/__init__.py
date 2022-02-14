#!/usr/bin/env python

from wrapt import ObjectProxy
import inspect
from dataclasses import dataclass, field

from tollan.utils.dataclass_schema import add_schema
from tollan.utils.log import get_logger
from tollan.utils.fmt import pformat_yaml
from tollan.utils import odict_from_list

from ..utils import RuntimeBase, RuntimeBaseError
from ..utils.config_registry import ConfigRegistry
from ..utils.config_schema import add_config_schema
from ..utils.env_loader import EnvLoader
from ..utils.doc_helper import collect_config_item_types


__all__ = [
    'env_loader', 'WebConfig', 'WebRuntime', 'WebRuntimeError',
    ]


env_loader = EnvLoader(root_namespace='TOLTECA_WEB')
"""An env loader instance shared by all apps."""


web_config = ObjectProxy(None)
"""
A proxy to the web config, which is made available
when `WebRuntime` instance is created.

This is consumed in the `tolteca.web.apps` submodule.
"""


class ConfigRegistryWithEnvLoader(ConfigRegistry):
    """A helper class to register config object with env loader."""

    def register(self, key, **kwargs):
        # we subclass the config registry to allow register the app
        # config classes with env loader
        # this will hook the app config from_dict method so it
        # pulls the env values before constructing the instance.
        cf = super().register(key, **kwargs)
        ef = env_loader.register_dataclass_schema(namespace=key.upper())

        def wrapped(item):
            return cf(ef(item))
        return wrapped


apps_registry = ConfigRegistryWithEnvLoader.create(
    name='AppsConfig',
    dispatcher_key='name',
    dispatcher_description='The app name.',
    dispatcher_key_is_optional=False
    )
"""The registry for ``web.apps``."""


def get_app_config(app_config_cls):
    """Return app config instance for given class."""
    logger = get_logger()

    if web_config:
        cfg = web_config.get_app_config(app_config_cls)
        if cfg is not None:
            logger.info(
                "load app config from web config:\n"
                f"{pformat_yaml(cfg.to_dict())}")
            return cfg
    # fall back to create a default one, if possible
    try:
        # note we use from_dict here to trigger env loader
        cfg = app_config_cls.from_dict({})
        logger.info(
            "load default app config:\n"
            f"{pformat_yaml(cfg.to_dict())}")
    except Exception as e:
        raise RuntimeError(
            f"cannot construct app config class {app_config_cls}: {e}")
    return cfg


# Load apps here to populate the registries
from . import apps as _  # noqa: F401, E402, F811

# we keep track of this dict in order to look up the namespace for
# collecting env vars
_app_name_by_config_cls = dict(
            (v, k) for k, v in apps_registry.items())


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

    def __post_init__(self, **kwargs):
        self._apps_by_name = odict_from_list(
            self.apps, key=lambda c: c.name)

    def get_app_config(self, arg):
        if isinstance(arg, str):
            return self._apps_by_name.get(arg, None)
        elif inspect.isclass(arg):
            arg = _app_name_by_config_cls.get(arg)
            return self.get_app_config(arg)
        raise ValueError(f"invalid arg: {arg}")


class WebRuntimeError(RuntimeBaseError):
    pass


class WebRuntime(RuntimeBase):
    """A class that manages the runtime context of web apps.

    """
    config_cls = WebConfig

    logger = get_logger()

    def run(self, app_name, ext_proc_name='flask'):
        web_config.__wrapped__ = self.config
        self.logger.debug(
            f"run web app={app_name} ext_proc={ext_proc_name}")
        # invoke dasha cli
        from dasha.cli import run_site
        dasha_site_name = f'tolteca.web.apps.{app_name}'
        return run_site(args=['-s', dasha_site_name, ext_proc_name])


web_config_item_types = collect_config_item_types(list(locals().values()))
