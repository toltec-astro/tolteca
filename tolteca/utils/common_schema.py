#!/usr/bin/env python


import astropy.units as u
import schema
from tollan.utils import ensure_abspath, dict_product
from tollan.utils.log import get_logger
from tollan.utils.fmt import pformat_yaml

from schema import Or
import jinja2
import jinja2.meta


class PhysicalTypeSchema(schema.Schema):

    def __init__(self, physical_type):
        self._physical_type = physical_type
        super().__init__(
            Or(int, float, str),
            description=f"Quantity with physical_type="
                        f"\"{self._physical_type}\".")

    def validate(self, data, **kwargs):
        data = u.Quantity(super().validate(data, **kwargs))
        if self._physical_type is not None and \
                data.unit.physical_type != self._physical_type:
            raise schema.SchemaError(
                f"Quantity {data} does not have "
                f"expected physical type of {self._physical_type}")
        return data


class RelPathSchema(schema.Schema):
    def __init__(self, check_exists=True):
        super().__init__(str)
        self._check_exists = check_exists

    def validate(self, data, rootpath=None, _relpath_schema=True, **kwargs):
        data = super().validate(data, _relpath_schema=False, **kwargs)
        if _relpath_schema:
            if rootpath is not None:
                data = ensure_abspath(rootpath.joinpath(data))
            else:
                data = ensure_abspath(data)
            if self._check_exists:
                if data.exists():
                    return data
                raise schema.SchemaError(f"Path does not exist: {data}")
        return data


class DictTemplateListSchema(schema.Schema):
    """A schema that can resolve list of dict specified with templates."""

    logger = get_logger()

    def __init__(
            self,
            template_key, template_key_schema=None,
            resolved_item_schema=None,
            **kwargs):
        if template_key_schema is None:
            template_key_schema = schema.Literal(
                template_key, description='The generator key.')
        # init self for basic validation of generator
        item_schema = list()
        item_schema.append({
            template_key_schema: str,
            schema.Optional(
                str, description='List of values consumed by the generator.'):
            list,
            })
        if resolved_item_schema is not None:
            item_schema.append(resolved_item_schema)
        # we use a dummy schema to init self, to avoid recursive validate call
        super().__init__(None, **kwargs)
        self._template_key = template_key
        self._resolved_item_schema = resolved_item_schema
        self._item_schema = item_schema

    def validate(
            self, data, **kwargs):
        items = schema.Schema(self._item_schema).validate(data, **kwargs)
        self.logger.debug(
            f"list of dict items to be parsed: {pformat_yaml(items)}")
        # resolve the data_items
        result = list()
        for item in items:
            result.extend(
                self._resolve_item(item, self._template_key))
        self.logger.debug(
            f"parsed list of dict items: {pformat_yaml(result)}")
        return result

    @staticmethod
    def _resolve_item(item, template_key):
        env = jinja2.Environment()
        templ_str = item[template_key]
        ast = env.parse(templ_str)
        var_keys = jinja2.meta.find_undeclared_variables(ast)
        if not var_keys:
            # not a jinja2 template, return as is
            return [item]
        iter_item = dict()
        for key in var_keys:
            if key not in item:
                raise ValueError(f"template variable {key} not defined.")
            value = item[key]
            # value could be a str, which is used verbatim
            if isinstance(value, str):
                value = [value]
            iter_item[key] = value
        # render the template for all combinations of iter_vars
        result = list()
        templ = env.from_string(templ_str)
        for item in dict_product(**iter_item):
            item[template_key] = templ.render(**item)
            result.append(item)
        return result
