#!/usr/bin/env python

import copy
from schema import Optional

from tollan.utils.log import get_logger
from tollan.utils.env import EnvRegistry
from tollan.utils.fmt import pformat_yaml


__all__ = ['EnvLoader']


class EnvLoader(object):
    """
    A base class to manage environment variables.

    Parameters
    ----------
    root_namespace : str
        The namespace of all managed environment variables.
    """

    logger = get_logger()

    def __init__(self, root_namespace):
        self._root_namespace = root_namespace
        self._registry = EnvRegistry.create()
        self._register_info = dict()

    @property
    def root_namespace(self):
        return self._root_namespace

    def _attr_dict_from_env_var(self, dataclass_cls):
        # collect value from env var for dataclass_cls attr dict.
        if dataclass_cls not in self._register_info:
            raise ValueError(f"class {dataclass_cls} is not registered.")
        reg_info = self._register_info[dataclass_cls]
        # namespace = reg_info['namespace']
        env_name_map = reg_info['env_name_map']

        result = dict()
        for attr, env_name in env_name_map.items():
            v = self._registry.get(env_name, None)
            if v is not None:
                result[attr] = v
        self.logger.debug(
            f"collected attr dict from env:\n"
            f"{pformat_yaml(result)}")
        return result

    def register_dataclass_schema(self, namespace):
        """Return a decorator that collect env vars from dataclass with schema.
        """
        def decorator(dataclass_cls):
            if not hasattr(dataclass_cls, 'schema'):
                raise TypeError("{dataclass_cls} does not have schema.")
            schema = dataclass_cls.schema
            # for each schema item, we register an env var to the registry
            env_name_map = dict()
            for key_schema, value_schema in schema.schema.items():
                key = key_schema.schema
                env_name = f'{self.root_namespace}_{namespace}_{key}'.upper()
                desc = key_schema.description
                if isinstance(key_schema, Optional):
                    reg_args = (str(key_schema.default), )
                else:
                    reg_args = ()
                self._registry.register(env_name, desc, *reg_args)
                env_name_map[key] = env_name
            # keep track of registered class for later consume
            self._register_info[dataclass_cls] = dict(
                namespace=namespace, env_name_map=env_name_map)

            # hook the from_dict function so it handles
            # loading the fields with env
            _old_from_dict = dataclass_cls.from_dict

            def _new_from_dict(d):
                d = copy.copy(d)
                d.update(self._attr_dict_from_env_var(dataclass_cls))
                return _old_from_dict(d)
            dataclass_cls.from_dict = _new_from_dict
            return dataclass_cls
        return decorator

    def get(self, namespace, name, *args):
        e = f'{self.root_namespace}_{namespace}_{name}'
        return self._registry.get(e, *args)
