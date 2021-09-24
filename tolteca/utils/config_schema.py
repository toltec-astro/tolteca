#!/usr/bin/env python


# from tollan.utils.registry import Registry, register_to
import schema
from dataclasses import is_dataclass
from functools import lru_cache


MAX_CLASS_SCHEMA_CACHE_SIZE = 1024
"""Cache size for mapped config types."""


def config_schema(cls):
    """Create schema suitable to validate config dict.
    """
    # get the config_key attribute
    config_key = None
    if is_dataclass(cls):
        # retrieve the config key from the Meta class
        Meta = getattr(cls, 'Meta', None)
        if Meta is not None:
            config_key = getattr(Meta, 'config_key', None)
    else:
        config_key = getattr(cls, '_config_key', None)
    if config_key is None:
        raise TypeError("config key not found")
    return _config_schema(cls, config_key)


@lru_cache(maxsize=MAX_CLASS_SCHEMA_CACHE_SIZE)
def _config_schema(cls, config_key):
    return ConfigSchema.from_dataclass(
        dataclass_cls=cls,
        config_key=config_key
        )


class ConfigSchema(schema.Schema):
    """A subclass of schema that knows about config_key"""

    # note that the subclass's __init__ signature has to be
    # compatible with the base Schema class for recursive
    # so we use a factory method to create the instance
    def __init__(self, *args, **kwargs):
        dataclass_cls = kwargs.pop('dataclass_cls', None)
        config_key = kwargs.pop('config_key', None)
        kwargs.setdefault('ignore_extra_keys', True)
        super().__init__(*args, **kwargs)
        self._dataclass_cls = dataclass_cls
        self._config_key = config_key

    @classmethod
    def from_dataclass(cls, dataclass_cls, config_key):
        return cls(
            {config_key: dataclass_cls.schema},
            dataclass_cls=dataclass_cls,
            config_key=config_key,
            )

    @property
    def dataclass_cls(self):
        """Return the dataclass this schema is associated with."""
        return self._dataclass_cls

    @property
    def config_key(self):
        """Return the config key this schema is associated with."""
        return self._config_key

    def validate(self, config, create_instance=False):
        """Validate `config`, optionally create the dataclass instance.

        Parameters
        ----------

        config : dict
            The dict to validate

        create_instance: bool
            If True, return the instance created from validated dict.
        """
        # validate the data. Note that the create_instance is propagated
        # down to any nested DataclassSchema instance's validate method.
        return super().validate(config, create_instance=create_instance)

    def load(self, config):
        """Return the instance of :attr:`dataclass_cls` created from `data`
        after validation."""
        return self.validate(config, create_instance=True)[self.config_key]

    def dump(self, instance):
        """Return the config dict for `instance`."""
        if not isinstance(instance, self.dataclass_cls):
            raise TypeError("invalid type for dump")
        return {self.config_key, instance.to_dict()}


def add_config_schema(cls):
    """A decorator to add config schema and related methods to dataclass `cls`.
    """
    if any(hasattr(cls, a) for a in (
            'from_config', 'config_key', 'to_config')):
        raise TypeError(f"conflicted attribute exists on {cls}")
    cls.config_schema = config_schema(cls)
    cls.config_key = cls.config_schema.config_key
    cls.from_config = cls.config_schema.load
    cls.to_config = cls.config_schema.dump
    return cls


# class ConfigMapperMixin(object):
#     """A helper base class to map config dict to its subclasses.

#     """

#     def __init_subclass__(cls, *args, **kwargs):

#         # ensure the cls as _config_key attribute
#         if is_dataclass(cls):
#             # retrieve the config key from the Meta class
#             Meta = getattr(cls, 'Meta', None)
#             found_config_key = False
#             if Meta is not None:
#                 config_key = getattr(Meta, 'config_key', None)
#                 if config_key:
#                     found_config_key = True
#             if found_config_key:
#                 cls._config_key = config_key
#             else:
#                 raise TypeError("config key not found")
#         elif not hasattr(cls, '_config_key'):
#             raise TypeError("config key not found")
#         if not hasattr(cls, 'schema'):
#             raise TypeError("schema not found")

#     @classmethod
#     def config_schema(cls):
#         """Return a schema that can validate a config dict

#         """
#         key = cls._config_key
#         return Schema(
#             {
#                 Literal(key, description=cls.__doc__):
#                 cls.schema
#                 Use(cls.from_dict,
#                     error=f'{cls.schema} does not match {{}}')
#                 },
#             description='The schema to validate config dict for {cls}',
#             ignore_extra_keys=True)

#     @classmethod
#     def from_config(cls, config):
#         """Create instance from config dict.
#         """
#         s = cls.make_config_schema()
#         cfg = s.validate(config)
#         return cfg[cls._config_key]


# _config_mapper_dict = Registry.create()
# """This holds the map from top-level config dict keys to the respective
# handling classes.

# """


# def register_config_branch(key, dataclass_cls=None):
#     """Register a `dataclasses.dataclass` class to the config mapper
#     registry.

#     It can be used as either decorator or directly.
#     """
#     if dataclass_cls is None:
#         return register_to(_config_mapper_dict, key)
#     elif is_dataclass(dataclass_cls):
#         return _config_mapper_dict.register(key, dataclass_cls)
#     raise TypeError("{dataclass_cls} is not a dataclass.")
