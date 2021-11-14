#!/usr/bin/env python


# from tollan.utils.registry import Registry, register_to
import schema
from tollan.utils.dataclass_schema import get_meta_attr
from functools import lru_cache


MAX_CLASS_SCHEMA_CACHE_SIZE = 1024
"""Cache size for mapped config types."""


def config_schema(cls):
    """Create schema suitable to validate config dict.
    """
    config_key = get_meta_attr(cls, 'config_key')
    return _config_schema(cls, config_key)


@lru_cache(maxsize=MAX_CLASS_SCHEMA_CACHE_SIZE)
def _config_schema(cls, config_key):
    return ConfigSchema.from_dataclass(
        dataclass_cls=cls,
        config_key=config_key)


class ConfigSchema(schema.Schema):
    """A subclass of schema that maps data class to config dict."""

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
            config_key=config_key,)

    @property
    def dataclass_cls(self):
        """Return the dataclass this schema is associated with."""
        return self._dataclass_cls

    @property
    def config_key(self):
        """Return the config key this schema is associated with."""
        return self._config_key

    def validate(self, config, create_instance=False, **kwargs):
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
        return super().validate(
            config, create_instance=create_instance, **kwargs)

    def load(self, config, runtime_info=None, **kwargs):
        """Return the instance of :attr:`dataclass_cls` created from `data`
        after validation."""
        inst = self.validate(config, create_instance=True, **kwargs)[
            self.config_key]
        inst.runtime_info = runtime_info
        return inst

    def dump(self, instance):
        """Return the config dict for `instance`."""
        if not isinstance(instance, self.dataclass_cls):
            raise TypeError("invalid type for dump")
        return {self.config_key: instance.to_dict()}


def add_config_schema(cls):
    """A decorator to add config schema and related methods to dataclass `cls`.
    """
    if any(hasattr(cls, a) for a in (
            'from_config', 'config_key', 'to_config')):
        raise TypeError(f"conflicted attribute exists on {cls}")
    cls.config_schema = config_schema(cls)
    cls.config_key = cls.config_schema.config_key
    cls.from_config = cls.config_schema.load
    # this has to be done this way as the cls.config_schema itself is a
    # mound method
    cls.to_config = lambda a: cls.config_schema.dump(a)
    return cls
