#!/usr/bin/env python

from tollan.utils.registry import Registry
from schema import Or, Literal, Optional
from tollan.utils.fmt import pformat_list
import textwrap
from tollan.utils.dataclass_schema import DataclassSchema


__all__ = ['ConfigRegistry']


class ConfigRegistry(Registry):
    """A helper class to register handler class for config items.
    """

    @classmethod
    def create(
            cls, name, dispatcher_key,
            dispatcher_description=None,
            dispatcher_key_is_optional=False
            ):
        inst = super().create()
        inst.name = name
        inst.dispatcher_key = dispatcher_key
        inst.dispatcher_description = dispatcher_description
        inst.dispatcher_key_is_optional = dispatcher_key_is_optional
        inst._register_info = dict()
        return inst

    def register(
            self, key, aliases=None,
            dispatcher_key=None,
            dispatcher_description=None,
            ):
        """Register(or return decorator to register) item with `key` with
        optional aliases."""
        if aliases is None:
            dispatcher_value_schema = key
        else:
            dispatcher_value_schema = Or(*([key] + list(aliases)))
        dispatcher_key = self.dispatcher_key
        if dispatcher_key is None:
            dispatcher_key = self.dispatcher_key
        if dispatcher_description is None:
            dispatcher_description = self.dispatcher_description
        self._register_info[key] = dict(aliases=aliases)

        def decorator(item):
            super(Registry, self).register(key, item)
            # add dispatch entry to schema
            # this update the schema in-place
            # self._add_dispather_entry_to_schema(
            #     item, dispatcher_key,
            #     dispatcher_description, dispatcher_value_schema,
            #     dispatcher_key_is_optional=self.dispatcher_key_is_optional,
            #     dispatcher_key_optional_default=key)
            # create a schema with dispatcher entry in it
            dispatcher_schema = self._create_schema_with_dispatcher_entry(
                item, dispatcher_key,
                dispatcher_description, dispatcher_value_schema,
                dispatcher_key_is_optional=self.dispatcher_key_is_optional,
                dispatcher_key_optional_default=key
                )
            self._register_info[key]['dispatcher_schema'] = dispatcher_schema
            if aliases is not None:
                # register the config item under the aliases
                for a in aliases:
                    super(Registry, self).register(a, item)
            return item
        return decorator

    @property
    def item_schemas(self):
        """The generator for all item schemas."""
        return (
            v['dispatcher_schema']
            for k, v in self._register_info.items())

    @property
    def schema(self):
        """The ``Or`` schema of all item schemas."""
        # build an or-schema for all factories
        return Or(*(self.item_schemas))

    def pformat_schema(self):
        """The pretty-formatted schema of registered factories."""
        # here we only show the original schema

        name = self.name
        desc = 'The sub-schemas for registered config items.'
        header = (
            f'{name}:\n  description:\n    {desc}'
            f'\n  {self.dispatcher_key}s:')
        toc_hdr = (f'{self.dispatcher_key}', 'choices', 'config class')
        toc_hdr = [toc_hdr, tuple('-' * len(h) for h in toc_hdr)]

        toc = pformat_list(
            toc_hdr + list(
                (k, aliases, self[k].__name__)
                for k, aliases in self._register_info.items()
                ),
            indent=4)
        body = textwrap.indent('\n'.join(
            [
                self[k].schema.pformat()
                for k in self._register_info.keys()]), prefix='  ')
        return f"{header}{toc}\n{body}"

    # @staticmethod
    # def _add_dispather_entry_to_schema(
    #         item, dispatcher_key, description, value_schema,
    #         dispatcher_key_is_optional=False,
    #         dispatcher_key_optional_default=None):
    #     # this will update the dict in-place
    #     # which may not be safe...
    #     # this juggling is to make the dispatcher key the first
    #     d = item.schema._schema
    #     d1 = item.schema._schema_orig = d.copy()
    #     d.clear()
    #     key_schema_kwargs = {'description': description}
    #     if dispatcher_key_is_optional:
    #         key_schema_cls = Optional
    #         key_schema_kwargs['default'] = dispatcher_key_optional_default
    #     else:
    #         key_schema_cls = Literal
    #     d[key_schema_cls(dispatcher_key, **key_schema_kwargs)] = value_schema
    #     d.update(d1)

    @staticmethod
    def _create_schema_with_dispatcher_entry(
            item, dispatcher_key, description, value_schema,
            dispatcher_key_is_optional=False,
            dispatcher_key_optional_default=None):
        # this juggling with copying d1 is to make the dispatcher key the
        # first
        d = item.schema._schema.copy()
        d1 = d.copy()
        d.clear()
        key_schema_kwargs = {'description': description}
        if dispatcher_key_is_optional:
            key_schema_cls = Optional
            key_schema_kwargs['default'] = dispatcher_key_optional_default
        else:
            key_schema_cls = Literal
        d[key_schema_cls(dispatcher_key, **key_schema_kwargs)] = value_schema
        d.update(d1)
        return DataclassSchema(d, dataclass_cls=item)
