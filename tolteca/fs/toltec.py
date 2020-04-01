#! /usr/bin/env python

import re
from datetime import datetime
import os
from tollan.utils.log import get_logger, logit
import numpy as np
from astropy.table import Table, Column, join, vstack
from . import DataFileStore, RemoteDataFileStore
from astropy.time import Time
import pickle
from astropy.utils.metadata import MergeStrategy
from astropy.utils.metadata import enable_merge_strategies
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
        logger = get_logger()
        runtime_link_patterns = ['toltec[0-9].nc', 'toltec[0-9][0-9].nc']
        files = []
        for pattern in runtime_link_patterns:
            if master is not None:
                pattern = os.path.join(
                        master, pattern.split('.')[0], pattern)
            logger.debug(f"check runtime files in {path} {pattern}")
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
        if master is None:
            result = self.spec.runtime_datafile_links(
                    self.rootpath, master=None)
            return result
        return self.spec.runtime_datafile_links(
                self.rootpath.joinpath(master), master=None)
        # for p, m in ((path, ''), (path.parent, path.name)):
        #     result = self.spec.runtime_datafile_links(p, master=m)
        #     if result:
        #         return result
        # else:
        #     return list()


class ToltecDataset(object):
    """This class provides convenient access to a set of TolTEC data files.
    """
    logger = get_logger()
    spec = ToltecDataFileSpec

    def __init__(self, index_table, meta=None):
        self._index_table = index_table
        self._update_meta_from_file_objs()
        if meta is not None:
            self._index_table.meta.update(meta)

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

    class _MergeAsDict(MergeStrategy):
        types = (object, object)  # left side types

        @classmethod
        def merge(cls, left, right):
            return dict(left=left, right=right)

    class _MergeAsList(MergeStrategy):
        types = (object, object)  # left side types
        _ctx = '_ctx_merge_as_list'

        @classmethod
        def _add_ctx(cls, v):
            if not isinstance(v, list):
                v = [v, ]
            return (v, cls._ctx)

        @classmethod
        def _has_ctx(cls, v):
            try:
                return v[1] == cls._ctx
            except IndexError:
                return False

        @classmethod
        def merge(cls, left, right):

            if not cls._has_ctx(left):
                left = cls._add_ctx(left)

            if not cls._has_ctx(right):
                right = cls._add_ctx(right)
            return cls._add_ctx([left[0] + right[0]])

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
        exclude_cols = '.+_obj'

        def check_col(c):
            if c in blacklist_cols:
                return False
            if isinstance(exclude_cols, str):
                return re.match(exclude_cols, c) is None
            return True

        use_cols = [c for c in tbl.colnames if check_col(c)]
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
    def meta(self):
        """Meta data of the dataset."""
        return self._index_table.meta

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

    def _join(self, type_, other, keys, cols):
        logger = get_logger()
        tbl = self.index_table
        other_tbl = other.index_table
        # make new cols
        use_cols = list(keys)
        if isinstance(cols, str):
            cols = [cols, ]
        for col in cols:
            if isinstance(col, tuple):
                on, nn = col
                other_tbl[nn] = other_tbl[on]
                use_cols.append(nn)
            elif isinstance(col, str):
                if col in other_tbl.colnames:
                    use_cols.append(col)
                else:
                    # regex match
                    for c in other_tbl.colnames:
                        if re.match(col, c):
                            use_cols.append(c)
            else:
                raise ValueError(f"invalid column {col}")
        logger.debug(f"join_right_cols={use_cols}")
        with enable_merge_strategies(self._MergeAsDict):
            joined = join(
                    tbl, other_tbl[use_cols],
                    keys=keys, join_type=type_)
        instance = self.__class__(joined, meta={
            'join_keys': keys,
            'join_type': type_,
            'join_right_cols': use_cols,
            })
        self.logger.debug(
                f"{type_} joined {instance}")
        return instance

    def left_join(self, other, keys, cols):
        return self._join('left', other, keys, cols)

    def right_join(self, other, keys, cols):
        return self._join('right', other, keys, cols)

    @classmethod
    def vstack(cls, datasets, **kwargs):
        with enable_merge_strategies(cls._MergeAsList):
            return cls(vstack([d.index_table for d in datasets], **kwargs))

    def split(self, *keys):
        nvals = [len(np.unique(self[k])) for k in keys]
        use_keys = []
        for nv, k in zip(nvals, keys):
            if nv > 1:
                use_keys.append(k)
        keys = use_keys
        if len(keys) == 1:
            key = keys[0]
            for uv in np.unique(self[key]):
                query = f'{key} == {uv}'
                result = self.select(query)
                result.meta['split_value'] = uv
                yield result
        elif len(keys) == 0:
            yield self.__class__(self.index_table)
        else:
            yield self.split(keys[0]).split(keys[1:])
        return

    def __len__(self):
        return self.index_table.__len__()

    def select(self, cond):
        """Return a subset of the dataset using numpy-like `cond`."""

        tbl = self.index_table

        if isinstance(cond, str):
            df = tbl.to_pandas()
            df.query(cond, inplace=True)
            tbl = Table.from_pandas(df)
            cond_str = cond
            # _cond = ne.evaluate(
            #         cond, local_dict={c: tbl[c] for c in tbl.colnames})
        else:
            tbl = tbl[cond]
            cond_str = '<fancy index | mask>'
        if len(tbl) == 0:
            raise ValueError(f"no entries are selected by {cond}")
        instance = self.__class__(tbl, meta={
            'select_query': cond_str
            })
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

    def write_index_table(self, *args, exclude_cols=None, **kwargs):
        """Write the index table to file using the `astropy.table.Table.write`
        function.
        """
        tbl = self.index_table
        # exclude the file object
        blacklist_cols = ['file_obj', 'data_obj', 'source_orig']

        def check_col(c):
            if c in blacklist_cols:
                return False
            if isinstance(exclude_cols, str):
                return re.match(exclude_cols, c) is None
            return True
        use_cols = [
                c for c in tbl.colnames
                if check_col(c)]
        tbl[use_cols].write(*args, **kwargs)

    def dump(self, filepath):
        with open(filepath, 'wb') as fo:
            pickle.dump(self, fo)

    @classmethod
    def load(self, filepath):
        with open(filepath, 'rb') as fo:
            return pickle.load(fo)
