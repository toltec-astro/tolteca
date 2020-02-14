#!/usr/bin/env python
# encoding: utf-8

from cached_property import cached_property
import inspect
import importlib
import functools
from .config import DB_CONFIG
from tollan.utils.fmt import pformat_dict
from tollan.utils.log import get_logger, logit, timeit
from .connection import DatabaseConnection


class ConfigMixin(object):

    logger = get_logger()

    def __init__(self, config, validate=True):
        if validate:
            self.validate_config(config)
        self._config = config
        self._config = config

    def validate_config(self, config):
        if self._has_config_validator(self):
            errors = self.config_validator(config)
            if errors:
                raise RuntimeError(f"invalid config: {errors}")

    @staticmethod
    def _has_config_validator(obj):
        return hasattr(obj, "config_validator")

    def __init_subclass__(cls, config_property_prefix="", **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._has_config_validator(cls):
            cls.config_validator.decorate_with_properties(
                    cls, prefix=config_property_prefix)


class ConfigValidator(object):

    logger = get_logger()

    def __init__(self):
        self._validators = list()
        self._keys = set()

    def __call__(self, config):
        errors = list()
        for v in self._validators:
            errors.extend(v(config))
        self.logger.debug(f"validated config: {pformat_dict(config)}")
        return errors

    def required(self, *args):
        def validate(config):
            errors = list()
            for a in args:
                if a not in config:
                    errors.append(f"missing required key {a}")
            return errors
        self._validators.append(validate)
        self._keys.update(set(args))
        return self

    def optional(self, **kwargs):
        def validate(config):
            for a, d in kwargs.items():
                if a not in config:
                    self.logger.debug(f"use default {a}={d}")
                    config[a] = d
            return list()
        self._validators.append(validate)
        self._keys.update(set(kwargs.keys()))
        return self

    def decorate_with_properties(self, cls, prefix=""):

        def getter(instance, key):
            # self.logger.debug(f"get {key} from {instance}")
            return instance._config[key]

        for key in self._keys:
            setattr(cls, f'{prefix}{key}', property(
                functools.partial(getter, key=key)))
            self.logger.debug(f"add property {prefix}{key} to {cls}")
        return cls


class DatabaseRuntime(ConfigMixin, config_property_prefix="_"):
    '''Class to hold database related states.'''

    config_validator = ConfigValidator() \
        .required("name", 'uri') \
        .optional(schema=None, tables_from_reflection=False)

    def __init__(self, config):
        super().__init__(config)
        self.logger = get_logger(f"db.{self._name}")

    @cached_property
    def connection(self):
        try:
            connection = DatabaseConnection(self._uri)
        except Exception as e:
            self.logger.error(
                    f"unable to connect to {self._name} {self._uri}: {e}")
        return connection

    @cached_property
    def session(self):
        return self.connection.Session()

    @cached_property
    def models(self):
        if self._schema is None:
            return None
        with logit(self.logger.debug, f"load database models {self._schema}"):
            try:
                m = timeit(
                    f"import models {self._schema}")(importlib.import_module)(
                        f".models.{self._schema}", __name__)
            except Exception as e:
                self.logger.warning(
                        f"unable to load models.{self._schema}: {e}")
            else:
                return m

    @cached_property
    def tables(self):
        if self._schema is None or self._tables_from_reflection:
            self.logger.debug("load tables with reflection")
            return None
        with logit(self.logger.debug, "load database tables {self._schema}"):
            try:
                m = timeit(
                    f"import tables {self._schema}")(importlib.import_module)(
                        f".tables.{self._schema}", __name__)
                m.create_tables(self.connection)
            except Exception as e:
                self.logger.warning(
                        f"unable to load tables.{self._schema}: {e}")
            else:
                return m

    @property
    def is_alive(self):
        if self.connection is not None and self.session is not None:
            try:
                self.session.connection()
            except Exception as e:
                self.logger.error(f'error querying database: {e}')
                return False
            else:
                return True
        else:
            self.logger.error(f'cannot connect to database')
            return False

    def modelclasses(self, name="default", lower=None):
        if name not in self.models:
            return dict()
        module = self.models[name]

        classes = dict()
        for model in inspect.getmembers(module, inspect.isclass):
            keyname = model[0].lower() if lower else model[0]
            if hasattr(model[1], '__tablename__'):
                classes[keyname] = model[1]
        return classes


def get_databases(config=DB_CONFIG):
    result = dict()
    for k, v in config.items():
        result[k] = DatabaseRuntime(dict(name=k, **v))
    return result
