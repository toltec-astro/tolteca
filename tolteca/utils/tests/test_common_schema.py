#!/usr/bin/env python

from tollan.utils.log import get_logger
from ..common_schema import PhysicalTypeSchema
import astropy.units as u
from schema import Schema, SchemaError
import pytest


def test_physical_type_schema():

    logger = get_logger()

    s = Schema({
        'a': PhysicalTypeSchema('frequency'),
        'b': PhysicalTypeSchema('speed'),
        'c': PhysicalTypeSchema('work'),
        })

    logger.debug(f"schema: {s}")

    d = s.validate({"a": '1 Hz', 'b': '2 m/s', 'c': '3 J'})
    assert d['a'] == 1 << u.Hz
    assert d['b'] == 2 << u.m / u.s
    assert d['c'] == 3 << u.J

    with pytest.raises(
            SchemaError, match='does not have expected physical type'):
        d = s.validate({"a": '1 s'})

    logger.debug(f"schema json: {s.json_schema(schema_id='test')}")
