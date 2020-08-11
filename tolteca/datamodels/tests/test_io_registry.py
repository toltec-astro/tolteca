#! /usr/bin/env python


from ..io.registry import IORegistry
import pytest


def test_io_registry_register():

    io_registry = IORegistry()

    class Factory(object):
        def __init__(self, source):
            self._source = source

        @property
        def source(self):
            return self._source

        @staticmethod
        def identify_source(source):
            return 'factory' in source

    io_registry.register(
            Factory,
            'test_format',
            Factory.identify_source)

    def identify_source_factory2(source):
        return source.endswith('factory2')

    @io_registry.register_as(
            'test_format2', identifier=identify_source_factory2)
    class Factory2(Factory):
        pass

    obj = io_registry.open('test_source.factory', format='test_format')
    assert type(obj) is Factory
    assert obj.source == 'test_source.factory'

    obj = io_registry.open('test_source.factory2', format='test_format2')
    assert type(obj) is Factory2
    assert obj.source == 'test_source.factory2'

    # auto detect
    obj = io_registry.open('test_source.factory2')
    # this will pick up the first that satisfies the identify function
    assert type(obj) is Factory
    assert obj.source == 'test_source.factory2'

    # invalid source
    with pytest.raises(RuntimeError, match='unable to identify.+'):
        obj = io_registry.open('test_source.invalid')

    # invalid format
    with pytest.raises(RuntimeError, match='unable to open.+'):
        obj = io_registry.open('test_source.factory', format='test_format2')

    io_registry.clear()
    assert not bool(io_registry.io_classes)
