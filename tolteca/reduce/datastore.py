#!/usr/bin/env python


from tollan.utils.dataclass_schema import DataclassNamespace
from tollan.utils.registry import Registry
from schema import Schema, Optional
from ..utils.common_schema import RelPathSchema
from . import inputs_registry


__all__ = ['LocalFileDataStore', 'DatabaseDataStore']


class DataLoaderRegistryMixin(object):

    @classmethod
    def register_data_loader(cls, key=None):

        def decorator(loader_cls):
            cls._data_loader_registry.register(key, loader_cls)
            return loader_cls

        return decorator

    @classmethod
    def get_data_loaders(cls):
        return list(cls._data_loader_registry.values())

    @classmethod
    def get_data_loader_names(cls):
        return list(cls._data_loader_registry.keys())

    def load_all_data(self):
        result = list()
        for k in self._data_loader_registry.keys():
            result.extend(self.load_data(k))
        return result

    def load_data(self, loader_name):
        if loader_name not in self._data_loader_registry:
            raise ValueError(f"invalid data loader name {loader_name}")
        loader = self._data_loader_registry[loader_name]
        if not loader.identify(self):
            return list()
        return loader.load(self)


@inputs_registry.register("localfile")
class LocalFileDataStore(DataclassNamespace, DataLoaderRegistryMixin):
    """A class describing data managed in local file system."""

    _namespace_from_dict_schema = Schema({
        Optional(
            'path',
            description='The root path of the data.'):
        RelPathSchema(),
        Optional(
            'select',
            default=None,
            description='The expression to select data to load.'
            ):
        str,
        Optional(
            'select_by_metadata',
            default=None,
            description='The expression to further select data by loaded metadata.'
            ):
        str,
        })

    _data_loader_registry = Registry.create()


@inputs_registry.register("database")
class DatabaseDataStore(DataclassNamespace, DataLoaderRegistryMixin):
    """A class describing data manged through a database."""

    _namespace_from_dict_schema = Schema({
        Optional(
            'uri',
            description='The toggle of whether to simulate polarized signal.'):
        RelPathSchema(),
        })

    _data_loader_registry = Registry.create()
