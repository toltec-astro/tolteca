#!/usr/bin/env python

from tollan.utils.env import EnvRegistry


class EnvMapper(object):
    """
    A base class to manage environment variables mapped to
    dataclasses with schema.
    """

    env_registry = EnvRegistry.create()

    def __init__(self, env_prefix):
        self._env_prefix = env_prefix

    def __call__(self, arg):
        if isinstance(arg, str):
            # arg is prefix, return decorator
            prefix = arg

            def decorator(dataclass_cls):
                if not hasattr(dataclass_cls, 'schema'):
                    raise TypeError("{dataclass_cls} does not have schema.")

                schema = dataclass_cls.schema
                # for each schema item, we register an env var to the registry
                for key_schema, value_schema in schema.schema.items():
                    print(key_schema, value_schema)
            return decorator
        # decorate arg with default prefix=''
        return self('')(arg)

    def register(self, *args):
        self.env_registry.register(*args)
