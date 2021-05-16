#! /usr/bin/env python

from astropy.table import Table, vstack
from ..base import CalibBase


class ToltecCalib(CalibBase):

    def get_array_prop_table(self, array_name=None):
        if array_name is None:
            # stack the tables
            tbls = [
                    self.get_array_prop_table(n)
                    for n in self.index['array_names']
                    ]

            # strip the meta and attach array name and array index
            def _proc(t):
                m = t.meta
                t.meta = None
                t['array_name'] = m['name']
                t['array'] = m['index']
                return m

            metas = [_proc(t) for t in tbls]

            tbl = vstack(tbls, join_type='exact')
            for meta in metas:
                tbl.meta[meta['name']] = meta
                tbl.meta['array_names'] = self.index['array_names']
            return tbl
        filepath = self._rootpath.joinpath(
                self.index['array_prop_table'][array_name]['path'])
        return Table.read(filepath.as_posix(), format='ascii')
