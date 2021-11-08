#! /usr/bin/env python

from astropy.table import Table, vstack
from schema import Schema, Or, SchemaError
from astropy.io.misc import yaml
from astropy.utils.decorators import classproperty

from tollan.utils.registry import Registry

from ..base import CalibStack, CalibBase
from ...common.toltec import toltec_info
from ...common.lmt import lmt_info


__all__ = ['ToltecCalibBase']


_registry = Registry.create()
"""A registry to hold TolTEC calibration object types."""


class ToltecCalibBase(CalibBase):
    """Base class for calibration object for the TolTEC instrument.

    """

    info = {
            'toltec': toltec_info,
            'lmt': lmt_info
            }
    """Assorted instrument info for TolTEC and LMT."""

    array_names = info['toltec']['array_names']
    """The TolTEC array names."""

    interfaces = info['toltec']['interfaces']
    """The TolTEC instrument interface names."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _registry_key = getattr(cls, '_registry_key', None)
        if _registry_key is None:
            return
        _registry.register(_registry_key, cls)

    @property
    def data_index(self):
        """The index under the ``cls._registry_key``."""
        return self.index[self._registry_key]


class ToltecArrayProp(ToltecCalibBase):
    """Calibration object for TolTEC array property table."""

    _registry_key = 'array_prop_table'

    @classproperty
    def _index_schema(cls):
        # apt can be specified per-array or merged in the index file.
        return Schema({
            cls._registry_key: Or(
                {
                    # one table per array
                    Or(cls.array_names): {
                        'path': str,
                        }
                    },
                {
                    # single table
                    'path': str
                    },
                ),
            }, ignore_extra_keys=True)

    def _is_merged_apt(self):
        # apt can be specified per-array or merged in the index file.
        return 'path' in self.data_index

    def _make_merged_apt(cls, tbls):
        # strip the meta and attach array name and array index
        # to the table
        def _proc(t):
            m = t.meta
            t.meta = None
            t['array_name'] = m['name']
            t['array_name'].description = 'The array name.'
            t['array'] = m['index']
            t['array'].description = \
                'The array index (0=a1100, 1=a1400, 2=a2000).'
            return m

        meta_list = [_proc(t) for t in tbls]

        tbl = vstack(tbls, join_type='exact')
        # merge the meta data to a dict keyed off with the array_name
        for meta in meta_list:
            tbl.meta[meta['name']] = meta
        tbl.meta['array_names'] = [m['name'] for m in meta_list]
        return tbl

    def get(self, array_name=None):
        """Return the array property table for `array_name`.

        Parameters
        ----------
        array_name : {'a1100', 'a1400', 'a2000'}, optional
            The array to select. When None (the default),
            merge the array property tables for all the arrays.
        """
        if array_name is None:
            # stack the tables
            if self._is_merged_apt():
                filepath = self.resolve_path(
                        self.data_index['path'])
                tbl = Table.read(filepath, format='ascii')
            else:
                tbl = self._make_merged_apt([
                        self.get(array_name=n)
                        # this uses the array_names from index to
                        # allow apt with not all three arrays
                        for n in self.index['array_names']
                        ])
            return tbl
        if self._is_merged_apt():
            tbl = self.get(array_name=None)
            return tbl[tbl['array_name'] == array_name]
        filepath = self.resolve_path(
                self.data_index[array_name]['path'])
        return Table.read(filepath, format='ascii')


class ToltecPassband(ToltecCalibBase):
    """Calibration object for TolTEC passband table."""

    _registry_key = 'passband_table'

    @classproperty
    def _index_schema(cls):
        # apt can be specified per-array or merged in the index file.
        return Schema({
            cls._registry_key: {
                # one table per array
                array_name: {
                    'path': str,
                    }
                for array_name in cls.array_names
                },
            }, ignore_extra_keys=True)

    def get(self, array_name=None):
        """Return the passband for `array_name`.

        Parameters
        ----------
        array_name : {'a1100', 'a1400', 'a2000'}, optional
            The array to select. When None (the default),
            merge the passbands for all the arrays.
        """
        if array_name is None:
            # TODO add merged passband table
            raise NotImplementedError
        filepath = self.resolve_path(
                self.data_index[array_name]['path'])
        return Table.read(filepath, format='ascii')


class ToltecCalib(ToltecCalibBase, CalibStack):
    """A class to manage TolTEC calibration data."""

    def get_array_prop_table(self, **kwargs):
        return self.index['array_prop_table'].get(**kwargs)

    def get_passband(self, **kwargs):
        return self.index['passband'].get(**kwargs)

    @classmethod
    def from_indexfile(cls, indexfile):
        """Create calibration stack for TolTEC from index file.

        The appropriate calibration object(s) will be created from the
        specified index file by check the contents against the schema
        of known subclasses of `ToltecCalibBase`.

        """
        with open(indexfile, 'r') as fo:
            index = yaml.load(fo)

        _index = dict()
        for key, cls in _registry.items():
            try:
                calobj = cls(index=index)
            except SchemaError:
                continue
            _index[key] = calobj
        return cls(index=_index)
