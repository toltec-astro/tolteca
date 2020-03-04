#! /usr/bin/env python

import re
from datetime import datetime
import os
from tollan.utils.log import get_logger, logit
import numpy as np
from astropy.table import Table, Column, join
from . import DataFileStore, RemoteDataFileStore
from astropy.time import Time
import pickle
from ..utils import get_user_data_dir


class ToltecDataFileSpec(object):

    name = "toltec.1"

    @classmethod
    def _info_from_filename(cls, filename):
        re_toltec_file = (
            r'^(?P<interface>(?P<instru>toltec)(?P<nwid>\d+))_(?P<obsid>\d+)_'
            r'(?P<subobsid>\d+)_(?P<scanid>\d+)_(?P<ut>\d{4}_'
            r'\d{2}_\d{2}(?:_\d{2}_\d{2}_\d{2}))(?:_(?P<kindstr>[^\/.]+))?'
            r'\.(?P<fileext>.+)$')

        def parse_ut(v):
            result = Time(
                datetime.strptime(v, '%Y_%m_%d_%H_%M_%S'),
                scale='utc')
            result.format = 'isot'
            return result

        dispatch_toltec = {
            'nwid': int,
            'obsid': int,
            'subobsid': int,
            'scanid': int,
            'ut': parse_ut,
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
    """A class that provide access to TolTEC data files."""

    spec = ToltecDataFileSpec

    def __init__(
            self, rootpath, hostname=None, local_rootpath=None,
            **kwargs):
        if hostname is None:
            super().__init__(rootpath, **kwargs)
            self._remote = None
        else:
            config = kwargs.get('config', dict())
            if local_rootpath is None:
                local_rootpath = self._get_default_data_rootpath()
            super().__init__(local_rootpath)

            self._remote = RemoteDataFileStore(
                    f'ssh://{hostname}:{rootpath}', **config)

    @DataFileStore.accessor.getter
    def accessor(self):
        if self._remote is None:
            return DataFileStore.accessor.fget(self)
        return self._remote.accessor

    @DataFileStore.rootpath.getter
    def rootpath(self):
        if self._remote is None:
            return self.local_rootpath
        return self._remote.rootpath

    @property
    def local_rootpath(self):
        return DataFileStore.rootpath.fget(self)

    @DataFileStore.rootpath.setter
    def set_rootpath(self, value):
        self._remote.rootpath = value

    def __repr__(self):
        if self._remote is not None:
            return f'{self.__class__.__name__}("{self.rootpath}", ' \
                f'remote="{self._remote}")'
        else:
            return f'{self.__class__.__name__}("{self.rootpath}")'

    def glob(self, *patterns, **kwargs):
        if self._remote is not None:
            self._remote.check_sync(
                    dest=self.local_rootpath,
                    recursive=True, **kwargs)
        return super().glob(*patterns)

    @staticmethod
    def _get_default_data_rootpath():
        logger = get_logger()
        p = get_user_data_dir().joinpath('datastore')
        if not p.exists():
            with logit(logger.debug, f"create {p}"):
                p.mkdir(exist_ok=True, parents=True)
        return p

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

    def __init__(self, index_table):
        self._index_table = index_table
        self._update_meta_from_file_objs()

    @staticmethod
    def _dispatch_dtype(v):
        if isinstance(v, int):
            return 'i8', -99
        elif isinstance(v, float):
            return 'd', np.nan
        elif isinstance(v, str):
            return 'U', ""
        return None, None

    @staticmethod
    def _col_score(c):
        hs_keys = ['nwid', 'obsid', 'subobsid', 'scanid']
        ls_keys = ['source', 'file_obj', 'data_obj']
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
                cls._dispatch_dtype(infolist[0][c])[0] for c in colnames]
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
        blacklist_cols = ['source', 'source_orig', 'file_obj', 'data_obj']
        use_cols = [c for c in tbl.colnames if c not in blacklist_cols]
        pformat_tbl = tbl[use_cols].pformat(max_width=-1)
        if pformat_tbl[-1].startswith("Length"):
            pformat_tbl = pformat_tbl[:-1]
        pformat_tbl = '\n'.join(pformat_tbl)
        return f"{self.__class__.__name__}" \
               f":\n{pformat_tbl}"

    @property
    def index_table(self):
        """The index table of the dataset."""
        return self._index_table

    @property
    def file_objs(self):
        """The opened file objects of this dataset.

        `None` if the dataset is not created by `open_files`.
        """
        if "file_obj" not in self.index_table.colnames:
            return None
        return self.index_table['file_obj']

    @property
    def data_objs(self):
        """The created data objects of this dataset.

        `None` if the dataset is not created by `load_data`.
        """
        if 'data_obj' not in self.index_table.colnames:
            return None
        return self.index_table['data_obj']

    def __getitem__(self, arg):
        return self.index_table[arg]

    def __setitem__(self, arg, value):
        self.index_table[arg] = value

    def left_join(self, other, keys, cols):
        tbl = self.index_table
        other_tbl = other.index_table
        # make new cols
        use_cols = list(keys)
        for col in cols:
            if isinstance(col, tuple):
                on, nn = col
                other_tbl[nn] = other_tbl[on]
                use_cols.append(nn)
            else:
                use_cols.append(col)
        joined = join(
                tbl, other_tbl[use_cols],
                keys=keys, join_type='left')
        instance = self.__class__(joined)
        self.logger.debug(
                f"left joined {instance}")
        return instance

    def __len__(self):
        return self.index_table.__len__()

    def select(self, cond):
        """Return a subset of the dataset using numpy-like `cond`."""

        tbl = self.index_table

        if isinstance(cond, str):
            df = tbl.to_pandas()
            df.query(cond, inplace=True)
            tbl = Table.from_pandas(df)
            # _cond = ne.evaluate(
            #         cond, local_dict={c: tbl[c] for c in tbl.colnames})
        else:
            tbl = tbl[cond]
        if len(tbl) == 0:
            raise ValueError(f"no entries are selected by {cond}")
        instance = self.__class__(tbl)
        self.logger.debug(
                f"selected {instance}\n"
                f"({len(instance)} out of {len(self)} entries)")
        return instance

    def open_files(self):
        """Return an instance of `ToltecDataset` with the files opened."""
        from ..io import open as open_file

        tbl = self.index_table
        tbl['file_obj'] = [open_file(e['source']) for e in tbl]
        return self.__class__(tbl)

    def load_data(self, func=None):
        """Return an instance of `ToltecDataset` with the data loaded.

        Parameters
        ----------
        func: callable, optional
            If not None, `func` is used instead of the default
            `file_obj.read` method to get the actual data objects.
        """
        if self.file_objs is None:
            return self.open_files().load_data(func=func)
        tbl = self.index_table

        def _get_data(fo):
            if hasattr(fo, 'open'):
                fo.open()
            if func is not None:
                result = func(fo)
            else:
                result = fo.read()
            if hasattr(fo, 'close'):
                fo.close()
            return result

        tbl['data_obj'] = [
                _get_data(fo)
                for fo in self.file_objs
                ]
        return self.__class__(tbl)

    def _update_meta_from_file_objs(self):
        logger = get_logger()
        fos = self.file_objs
        if fos is None:
            return
        tbl = self.index_table
        # update from object meta
        use_keys = None
        for fo in fos:
            keys = filter(
                lambda k: not k.startswith('__'), fo.meta.keys())
            if use_keys is None:
                use_keys = set(keys)
            else:
                use_keys = use_keys.union(keys)

        def filter_cell(v):
            if not isinstance(v, (str, int, float, complex)):
                # logger.debug(f"ignore value of type {type(v)} {v}")
                return None
            else:
                return v
        if len(use_keys) > 0:
            logger.debug(f"update meta keys {use_keys}")
            for k in use_keys:
                col = [filter_cell(fo.meta.get(k, None)) for fo in fos]
                if all(c is None for c in col):
                    continue
                dtype = None
                for c in col:
                    dtype, fill = self._dispatch_dtype(c)
                if dtype is None:
                    continue
                if k in tbl.colnames:
                    # update
                    for i, c in enumerate(col):
                        if c is not None:
                            tbl[i][k] = c
                else:
                    tbl[k] = Column(
                        [fill if c is None else c for c in col],
                        dtype=dtype)
        colnames = sorted(tbl.colnames, key=lambda k: self._col_score(k))
        # we only pull the common keys in all of the file objects
        self._index_table = tbl[colnames]

    def write_index_table(self, *args, colfilter=None, **kwargs):
        """Write the index table to file using the `astropy.table.Table.write`
        function.
        """
        tbl = self.index_table
        # exclude the file object
        blacklist_cols = ['file_obj', 'data_obj', 'source_orig']
        use_cols = np.array([
                c for c in tbl.colnames
                if c not in blacklist_cols])
        if colfilter is not None:
            use_cols = use_cols[colfilter]
        tbl[use_cols.tolist()].write(*args, **kwargs)

    def dump(self, filepath):
        with open(filepath, 'wb') as fo:
            pickle.dump(self, fo)

    @classmethod
    def load(self, filepath):
        with open(filepath, 'rb') as fo:
            return pickle.load(fo)
