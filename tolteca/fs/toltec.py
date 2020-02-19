#! /usr/bin/env python

import re
from datetime import datetime
import os
from tollan.utils.log import get_logger
import numpy as np
from astropy.table import Table, Column
import numexpr as ne
from . import DataFileStore


class ToltecDataFileSpec(object):

    name = "toltec.1"

    @classmethod
    def _info_from_filename(cls, filename):
        re_toltec_file = (
            r'^(?P<interface>(?P<instru>toltec)(?P<nwid>\d+))_(?P<obsid>\d+)_'
            r'(?P<subobsid>\d+)_(?P<scanid>\d+)_(?P<ut>\d{4}_'
            r'\d{2}_\d{2}(?:_\d{2}_\d{2}_\d{2}))(?:_(?P<kindstr>[^\/.]+))?'
            r'\.(?P<fileext>.+)$')
        dispatch_toltec = {
            'nwid': int,
            'obsid': int,
            'subobsid': int,
            'scanid': int,
            'ut': lambda v: datetime.strptime(v, '%Y_%m_%d_%H_%M_%S'),
                }

        def post_toltec(info):
            if 'kindstr' not in info:
                info['kindstr'] = 'timestream'
            if info is not None and info['fileext'].lower() != "nc":
                info['kindstr'] = 'ancillary'

        re_wyatt_file = (
            r'^(?P<interface>(?P<instru>wyatt))'
            r'_(?P<ut>\d{4}-\d{2}-\d{2})'
            r'_(?P<obsid>\d+)_(?P<subobsid>\d+)_(?P<scanid>\d+)'
            r'(?:_(?P<kindstr>[^\/.]+))?'
            r'\.(?P<fileext>.+)$')
        dispatch_wyatt = {
            'obsid': int,
            'subobsid': int,
            'scanid': int,
            'ut': lambda v: datetime.strptime(v, '%Y-%m-%d'),
                }

        def post_wyatt(info):
            pass

        info = None
        for re_, dispatch, post in [
                (re_toltec_file, dispatch_toltec, post_toltec),
                (re_wyatt_file, dispatch_wyatt, post_wyatt)
                ]:
            m = re.match(re_, filename)
            if m is not None:
                info = {k: dispatch.get(k, lambda v: v)(v) for k, v
                        in m.groupdict().items()}
                post(info)
                break
        return info

    @classmethod
    def info_from_filename(cls, path, resolve=True):
        if resolve and path.is_symlink():
            path = path.resolve()
        info = cls._info_from_filename(path.name)
        if info is not None:
            info['source'] = path
            master = path.parent.name
            if master == f'{info["interface"]}':
                master = path.parent.parent.name
            info['master'] = master
        return info

    @staticmethod
    def runtime_datafile_links(path, master=None):
        runtime_link_patterns = ['toltec[0-9].nc', 'toltec[0-9][0-9].nc']
        files = []
        for pattern in runtime_link_patterns:
            if master is not None:
                pattern = os.path.join(
                        master, pattern.split('.')[0], pattern)
            files.extend(path.glob(pattern))
        return files


class ToltecDataFileStore(DataFileStore):

    spec = ToltecDataFileSpec

    def runtime_datafile_links(self, master=None):
        path = self.rootpath
        result = self.spec.runtime_datafile_links(path, master=None)
        for p, m in ((path, ''), (path.parent, path.name)):
            result = self.spec.runtime_datafile_links(p, master=m)
            if result:
                return result
        else:
            return list()


class ToltecDataset(object):
    """This class provides convenient access to a set of TolTEC data files.
    """
    logger = get_logger()
    spec = ToltecDataFileSpec

    _col_file_object = 'file_object'

    def __init__(self, index_table):
        self._index_table = index_table
        self._update_meta_from_file_objs()

    @staticmethod
    def _dispatch_dtype(v):
        if isinstance(v, int):
            return 'i8'
        elif isinstance(v, float):
            return 'd'
        elif isinstance(v, str):
            return 'S'

    @staticmethod
    def _col_score(c):
        hs_keys = ['nwid', 'obsid', 'subobsid', 'scanid']
        ls_keys = ['source', 'file_object']
        if c in hs_keys:
            return hs_keys.index(c) - len(hs_keys)
        if c in ls_keys:
            return ls_keys.index(c)
        return 0

    @classmethod
    def from_files(cls, *filepaths):
        if not filepaths:
            raise ValueError("no file specified")

        infolist = list(
                filter(
                    lambda i: i is not None,
                    map(cls.spec.info_from_filename,
                        map(DataFileStore._normalize_path, filepaths))))
        colnames = list(infolist[0].keys())
        # sort the keys to make it more human readable
        colnames.sort(key=lambda k: cls._col_score(k))

        dtypes = [
                cls._dispatch_dtype(infolist[0][c]) for c in colnames]
        tbl = Table(
                rows=[[fi[k] for k in colnames] for fi in infolist],
                names=colnames,
                dtype=dtypes
                )
        tbl.sort(['obsid', 'subobsid', 'scanid', 'nwid'])
        instance = cls(tbl)
        cls.logger.debug(
                f"loaded {instance}\n"
                f"({len(instance)} out of {len(filepaths)} input paths)")
        return instance

    def __repr__(self):
        tbl = self.index_table
        blacklist_cols = ['source', self._col_file_object, 'source_orig']
        use_cols = [c for c in tbl.colnames if c not in blacklist_cols]
        pformat_tbl = tbl[use_cols].pformat(max_width=-1)
        if pformat_tbl[-1].startswith("Length"):
            pformat_tbl = pformat_tbl[:-1]
        pformat_tbl = '\n'.join(pformat_tbl)
        return f"{self.__class__.__name__}" \
               f":\n{pformat_tbl}"

    @property
    def index_table(self):
        return self._index_table

    def __getitem__(self, col):
        return self.index_table[col]

    def __len__(self):
        return self.index_table.__len__()

    def select(self, cond):
        """Return a subset of the dataset using numpy-like `cond`."""
        tbl = self.index_table
        if isinstance(cond, str):
            _cond = ne.evaluate(
                    cond, local_dict={c: tbl[c] for c in tbl.colnames})
        else:
            _cond == cond
        tbl = tbl[_cond]
        if len(tbl) == 0:
            raise ValueError(f"no entries are selected by {cond}")
        instance = self.__class__(tbl)
        self.logger.debug(
                f"selected {instance}\n"
                f"({len(instance)} out of {len(self)} entries)")
        return instance

    def open_files(self):
        from ..io import open as open_file

        tbl = self.index_table
        tbl[self._col_file_object] = Column(
                [open_file(e['source']) for e in tbl],
                name=self._col_file_object,
                dtype=object
                )
        return self.__class__(tbl)

    def _update_meta_from_file_objs(self):
        logger = get_logger()
        tbl = self.index_table
        if self._col_file_object not in tbl.colnames:
            return
        fos = tbl[self._col_file_object]
        # update from object meta
        use_keys = None
        for fo in fos:
            if use_keys is None:
                use_keys = set(fo.meta.keys())
            else:
                use_keys = use_keys.union(set(fo.meta.keys()))
        if len(use_keys) > 0:
            logger.debug(f"update meta keys {use_keys}")
            for k in use_keys:
                tbl[k] = Column(
                        [fo.meta.get(k, None) for fo in fos],
                        dtype=self._dispatch_dtype(fos[0].meta[k]))
        colnames = sorted(tbl.colnames, key=lambda k: self._col_score(k))
        # we only pull the common keys in all of the file objects
        self._index_table = tbl[colnames]

    def write_index_table(self, *args, colfilter=None, **kwargs):
        tbl = self.index_table
        # exclude the file object
        use_cols = np.array([
                c for c in tbl.colnames
                if c != self._col_file_object])
        if colfilter is not None:
            use_cols = use_cols[colfilter]
        tbl[use_cols.tolist()].write(*args, **kwargs)
