#! /usr/bin/env python

from urllib.parse import urlparse
from tollan.utils import file_uri_to_path
from pathlib import Path
from astropy.io.misc.yaml import load as yaml_load
from astropy.table import Table, vstack


class ToltecCalib(object):

    def __init__(self, indexfile):
        with open(indexfile, 'r') as fo:
            index = yaml_load(fo)
        self._index = index
        self._rootpath = indexfile.parent

    @property
    def index(self):
        return self._index

    def get_array_prop_table(self, array_name=None):
        if array_name is None:
            # stack the tables
            tbls = [
                    self.get_array_prop_table(n)
                    for n in self.index['array_names']
                    ]

            # strip the meta and attach array name
            def _proc(t):
                m = t.meta
                t.meta = None
                t['array_name'] = m['name']
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

    @classmethod
    def from_uri(cls, uri, **kwargs):

        u = urlparse(uri)

        m = getattr(cls, f"from_{u.scheme}", None)
        if m is None:
            raise ValueError("scheme {u.scheme} is not supported.")
        dispatch_uri = {
                'file': file_uri_to_path
                }.get(u.scheme, lambda v: v)
        return m(dispatch_uri(uri), **kwargs)

    @classmethod
    def from_indexfile(cls, indexfile):
        return cls(Path(indexfile))
