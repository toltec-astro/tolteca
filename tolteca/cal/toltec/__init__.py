#! /usr/bin/env python

from astropy.table import Table, vstack
from schema import Schema, Optional, Or
from ..base import CalibStack, CalibBase


class ToltecArrayProp(CalibBase):
    """Calibration object for TolTEC array property table."""

    @property
    def array_names(self):
        return self.index['array_names']

    @classmethod
    def validate_index(cls, index):
        # apt can be specified per-array or merged in the index file.
        s = Schema({
            'array_prop_table': Or(
                {
                    # one table per array
                    str: {
                        'path': str,
                        }
                    },
                {
                    # single table
                    'path': str
                    },
                ),
            Optional(object): object
            })
        return s.validate(index)

    def _is_merged_apt(self):
        # apt can be specified per-array or merged in the index file.
        return 'path' in self.index['array_prop_table']

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
                        self.index['array_prop_table']['path'])
                tbl = Table.read(filepath, format='ascii')
            else:
                tbls = [
                        self.get(array_name=n)
                        for n in self.index['array_names']
                        ]

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

                metas = [_proc(t) for t in tbls]

                tbl = vstack(tbls, join_type='exact')
                # merge the meta data to a dict keyed off with the array_name
                for meta in metas:
                    tbl.meta[meta['name']] = meta
                    tbl.meta['array_names'] = self.index['array_names']
            return tbl
        if self._is_merged_apt():
            tbl = self.get(array_name=None)
            return tbl[tbl['array_name'] == array_name]
        filepath = self.resolve_path(
                self.index['array_prop_table'][array_name]['path'])
        return Table.read(filepath, format='ascii')


class ToltecPassband(CalibBase):
    """Calibration object for TolTEC passband table."""

    @property
    def array_names(self):
        return self.index['array_names']

    def get(self, array_name=None):
        """Return the passband for `array_name`.

        Parameters
        ----------
        array_name : {'a1100', 'a1400', 'a2000'}, optional
            The array to select. When None (the default),
            merge the passbands for all the arrays.
        """
        if array_name is None:
            return
        filepath = self.resolve_path(
                self.index['passbands'][array_name]['path'])
        return Table.read(filepath, format='ascii')


class ToltecCalib(CalibStack):
    """A class to manage TolTEC calibration data."""

    @property
    def array_names(self):
        return self.index['array_names']

    def get_array_prop_table(self, array_name=None):
        return self.index['array_prop_table'].get(array_name)

    def get_passband(self, array_name=None):
        return self.index['passband_table'].get(array_name)
