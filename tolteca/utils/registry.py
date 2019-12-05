#! /usr/bin/env python


from collections import OrderedDict
from ..utils.log import get_logger
import wrapt


class RegistryMixin(object):
    logger = get_logger()

    def register(self, label, item):
        if label in self:
            raise KeyError(
                f"label={label} exists in registry: {self[label]}")
        self[label] = item
        self.logger.debug(
                f"registered {item} as \"{label}\"")


def register_to(registry, key):
    def decorator(cls):
        registry.register(key(cls), cls)
        return cls

    return decorator


class Registry(wrapt.ObjectProxy, RegistryMixin):

    @classmethod
    def create(cls, container_cls=OrderedDict):
        return cls(container_cls())
