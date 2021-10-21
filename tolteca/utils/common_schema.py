#!/usr/bin/env python


import astropy.units as u
import schema
from tollan.utils import ensure_abspath


class PhysicalTypeSchema(schema.Schema):

    def __init__(self, physical_type):
        self._physical_type = physical_type
        super().__init__(
            str, description=f"Quantity with physical_type="
                             f"\"{self._physical_type}\".")

    def validate(self, data, **kwargs):
        data = u.Quantity(super().validate(data, **kwargs))
        if data.unit.physical_type != self._physical_type:
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
