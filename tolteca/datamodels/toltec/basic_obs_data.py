#! /usr/bin/env python

from tollan.utils.log import get_logger
from astropy.table import Table, MaskedColumn, vstack, unique, join
import numpy as np
# from astropy.io import registry
from tollan.utils.log import logged_dict_update
from astropy.utils.decorators import sharedmethod
from ..io.registry import open_file
from ..io.base import DataFileIO
from ..fs.toltec import meta_from_source
from collections import defaultdict
import warnings
from tollan.utils.fmt import pformat_fancy_index
import dill


__all__ = ['BasicObsData', 'BasicObsDataset']


class BasicObsData(DataFileIO):
    """A class that provides unified IO interface to TolTEC basic
    obs data.

    This class acts as a simple wrapper around the data file IO classes defined
    in `~tolteca.datamodels.io.toltec` module, and provides an unified
    interface to handle high level information of data items that are of
    different types, or located on remote file systems.

    If the source points to a local file, the
    `file_obj` is available and can be used to access the data::

        >>> from tolteca.datamodels.toltec import BasicObsData
        >>> bod = BasicObsData('toltec0.nc')
        >>> with bod.open() as fo:
        >>>     # `fo` is a `~tolteca.datamodels.io.toltec.NcFileIO` instance.
        >>>     kidsdata = fo.read()

        The above can also be done via the `open_file` shortcut as follows::

        >>> with BasicObsData.open('toltec0.nc') as fo:
        >>>     kidsdata = fo.read()

        This can further be shortened as::

        >>> kidsdata = BasicObsData.read('toltec0.nc')

    When :attr:`file_loc` is a remote file, `file_obj` is ``None``, and
    (of course) attempting to open the file will raise `DataFileIOError`.

    In both cases, various information of the data is pulled and stored in
    :attr:`meta`. In particular, when :attr:`file_obj` is available (local
    file),  the meta data includes those from the header of the opened data
    file.

    Parameters
    ----------
    source : str, `~pathlib.Path`, `~tollan.utils.FileLoc`.
        The data file location. Remote locations can be specified in
        either URL or SFTP format ``<host>:<path>`.

    open_: bool
        If True, attemp to open the file to load meta data.
    """

    logger = get_logger()

    def __init__(self, source, open_=True):
        super().__init__()
        file_loc = self._file_loc = self._normalize_file_loc(source)
        if open_ and file_loc.is_local:
            file_obj = open_file(file_loc.path)
        else:
            file_obj = None
        self._file_loc = file_loc
        self._file_obj = file_obj
        self._update_meta()
        if self.file_obj is not None:
            # ensure that we automatically close the file
            # after the meta is updated.
            self.file_obj.close()

    @classmethod
    def update_meta_from_file_obj(cls, meta, file_obj):
        """Update `meta` with header of `file_obj`.

        This modifies `meta` in-place.
        """
        logged_dict_update(cls.logger.warning, meta, file_obj.meta)
        return meta

    @classmethod
    def update_meta_from_file_loc(cls, meta, file_loc):
        """Update `meta` with info encoded in `file_loc`.

        This modifies `meta` in-place.
        """
        try:
            logged_dict_update(
                cls.logger.warning, meta, meta_from_source(file_loc))
        except Exception:
            cls.logger.debug(
                    'unable to parse meta from file loc.',
                    exc_info=True)
        return meta

    def _update_meta(self):
        """Update :attr:`meta`."""

        self.update_meta_from_file_loc(self.meta, self.file_loc)
        if self.file_obj is not None:
            self.update_meta_from_file_obj(self.meta, self.file_obj)

    @sharedmethod
    def open(obj, *args, **kwargs):
        """Shortcut to open the file."""
        if isinstance(obj, BasicObsData):
            # instance
            return super().open()
        return obj(*args, **kwargs).open()

    @sharedmethod
    def read(obj, *args, **kwargs):
        """Shortcut to read the file."""
        if isinstance(obj, BasicObsData):
            # instance
            with super().open() as fo:
                return fo.read()
        with obj.open(*args, **kwargs) as fo:
            return fo.read()


class BasicObsDataset(object):
    """A helper class to access a set of TolTEC basic obs data items.

    An instance of this class is typically created from one of its
    ``from_*`` factory functions.

    This class is implemented as a thin wrapper around an index table (a
    `~astropy.table.Table` instance) that contains information to identify
    BOD items.

    Parameters
    ----------
    index_table : `~astropy.table.Table`
        The table that identifies a set of BOD items.

    bod_list : list
        A list of `BasicObsData` instances.

    include_meta_cols : str, list
        The columns to include in the index table, extracted from the
        loaded meta data.

    **kwargs :
        Passed to `BasicObsData` constructor.
    """

    logger = get_logger()

    def __init__(
            self,
            index_table=None,
            bod_list=None,
            include_meta_cols='intersection',
            **kwargs):
        if index_table is None and bod_list is None:
            raise ValueError('need one of index_table or bod_list.')
        if index_table is None:
            index_table = self._make_index_table(bod_list)
        elif bod_list is None:
            # when sliced, the tbl will contain the sliced bod list
            if '_bod' in index_table.colnames:
                bod_list = index_table['_bod']
                self._index_table = index_table
                self._bod_list = bod_list
                return
            else:
                bod_list = self._make_bod_list(index_table, **kwargs)
        assert len(index_table) == len(bod_list)
        self._index_table = index_table
        self._bod_list = bod_list
        self['_bod'] = self._bod_list
        if include_meta_cols is not None:
            self.add_meta_cols(include_meta_cols)

    logger = get_logger()

    _cols_bod_info = (
                'interface', 'obsnum', 'subobsnum', 'scannum',
                'master', 'repeat', 'source',
                )

    @classmethod
    def _validate_index_table(cls, tbl):
        cols_required = cls._cols_bod_info
        if not set(tbl.colnames).issuperset(set(cols_required)):
            raise ValueError(
                    "required columns missing in the input index_table.")
        return tbl

    @classmethod
    def _make_bod_list(cls, tbl, **kwargs):
        # this will update tbl in place.
        bods = []
        s = tbl['source']
        if isinstance(s, MaskedColumn):
            s = s.filled(None)
        for source in s:
            if source is None:
                bods.append(None)
                continue
            try:
                bod = BasicObsData(source, **kwargs)
            except Exception as e:
                cls.logger.debug(f"unable to create bod: {e}")
                bod = None
            bods.append(bod)

        return np.array(bods, dtype=object)

    @classmethod
    def _make_index_table(cls, bod_list):
        tbl = defaultdict(list)
        for bod in bod_list:
            # with bod.open():
            for col in cls._cols_bod_info:
                if col == 'source':
                    value = bod.file_loc.rsync_path
                else:
                    # default values for master and repeat
                    if col == 'master':
                        defval = 'unknown'
                    elif col == 'repeat':
                        defval = 0
                    else:
                        defval = None
                    value = bod.meta.get(col, defval)
                tbl[col].append(value)
        tbl = Table(data=tbl)
        return tbl

    def _add_meta_cols(self, keys):
        """Add meta data info to the index table.

        Parameters
        ----------
        keys : str
            The keys to add.
        """
        data = defaultdict(list)
        for k in keys:
            for bod in self:
                if bod is None:
                    continue
                data[k].append(bod.meta.get(k, None))
        for k in data.keys():
            try:
                self[k] = data[k]
            except Exception:
                pass

    def add_meta_cols(self, keys):
        """Add meta data info to the index table.

        Parameters
        ----------
        keys : list, str
            The keys to add. When string, this is passed as the option to
            :meth:`get_meta_keys` to get the keys.

        """
        if isinstance(keys, str):
            keys = self.get_meta_keys(keys)
        if keys is not None:
            self._add_meta_cols(keys)
        return self

    def get_meta_keys(self, option):
        """Return a set of meta keys.

        `option` can be one of:

        1. ``union``: The union of all meta keys in all BODs.
        2. ``intersection``: The intersection of all meta keys in all BODs.
        """
        keys = None
        for bod in self:
            if bod is None:
                continue
            k = bod.meta.keys()
            if keys is None:
                keys = set(k)
            else:
                if option == 'union':
                    keys = keys.union(k)
                elif option == 'intersection':
                    keys = keys.intersection(k)
                else:
                    raise ValueError(f"invalid option {option}.")
        return keys

    def read(self, *args, **kwargs):
        """Return a generator of data items."""
        for bod in self:
            if bod is None:
                yield None
            yield bod.read(*args, **kwargs)

    def read_all(self, *args, **kwargs):
        """Return a list of data items."""
        return list(self.read(*args, **kwargs))

    # iter op on the bod list
    def __iter__(self):
        return self._bod_list.__iter__()

    def __len__(self):
        return self._bod_list.__len__()

    # the bracket operators are passed on to the index table
    def __getitem__(self, arg):
        # here we can be smart in the return
        # when arg is colname is will return the column
        # otherwise it return an indextable
        if arg in self.index_table.colnames:
            return self.index_table[arg]
        if isinstance(arg, list) and any(
                a in self.index_table.colnames for a in arg):
            return self.index_table[arg]
        return self.__class__(self.index_table[arg])

    def __setitem__(self, arg, value):
        self.index_table[arg] = value

    @property
    def bod_list(self):
        return self._bod_list

    @property
    def meta(self):
        return self.index_table.meta

    @property
    def history(self):
        if 'history' not in self.meta:
            self.meta['history'] = []
        return self.meta['history']

    @property
    def index_table(self):
        """The index table."""
        return self._index_table

    def write_index_table(self, *args, **kwargs):
        tbl = self.index_table
        use_cols = [
            c for c in tbl.colnames
            if not (
                tbl.dtype[c].hasobject
                or tbl.dtype[c].ndim > 0
                )]
        return tbl[use_cols].write(*args, **kwargs)

    def dump(self, filepath):
        with open(filepath, 'wb') as fo:
            dill.dump(self, fo)

    @classmethod
    def load(self, filepath):
        with open(filepath, 'rb') as fo:
            return dill.load(fo)

    def select(self, cond, desc=None):
        """Return a subset of the dataset specified by `cond`

        When `cond` is a string, the index table is converted to
        `~pandas.DataFrame` and the selection is done by
        :meta:`~pandas.DataFrame.eval`.

        Parameters
        ----------
        cond : str or `~numpy.ndarray`
            The condition to create the subset.

        desc : str
            A description of this selection to include in the table meta.

        Returns
        -------
        `BasicObsDataset`
            The subset as a new dataset instance.
        """

        tbl = self.index_table

        if isinstance(cond, str):
            with warnings.catch_warnings():
                # this is to supress the ufunc size warning
                # and the numexpr
                warnings.simplefilter("ignore")
                use_cols = [
                    c for c in tbl.colnames
                    if not (
                        tbl.dtype[c].ndim > 0
                        )]
                df = tbl[use_cols].to_pandas()
                m = df.eval(cond).to_numpy(dtype=bool)
            cond_str = cond
        else:
            m = cond
            cond_str = pformat_fancy_index(cond)
        if len(tbl) == 0:
            raise ValueError(f"no entry meets the select cond {cond}")

        self.history.append({
                'bods_select': {
                    'cond_str': cond_str,
                    'description': desc if desc is not None else ""
                    }
                })
        inst = self._sliced_new(m)
        self.logger.debug(
                f"selected {inst}\n"
                f"({len(inst)} out of {len(self)} entries)")
        return inst

    def _sliced_new(self, item):
        return self.__class__(
                index_table=self._index_table[item],
                bod_list=np.asanyarray(self._bod_list)[item])

    def __repr__(self):
        pformat_tbl = self.index_table.pformat(max_width=-1)
        if pformat_tbl[-1].startswith("Length"):
            pformat_tbl = pformat_tbl[:-1]
        pformat_tbl = '\n'.join(pformat_tbl)
        return f"{self.__class__.__name__}" \
               f":\n{pformat_tbl}"

    @classmethod
    def from_files(cls, files, open_=True, **kwargs):
        if not files:
            raise ValueError("no file specified")
        bod_list = []
        for f in files:
            try:
                bod_list.append(BasicObsData(f, open_=open_))
            except Exception as e:
                cls.logger.debug(
                        f'ignored unknown file {f}: {e}', exc_info=False)
        return cls(bod_list=bod_list, **kwargs)

    @classmethod
    def from_index_table(cls, index_table, copy=True, meta=None):
        """Return a dataset from an index table.

        Parameters
        ----------
        index_table : `astropy.table.Table`, str
            The table or that path to the table that contains
            the BOD identifiers.

        copy : bool
            If True, a copy of the `index_table` is made.

        meta : dict
            Additional meta get stored in the table.
        """
        if not isinstance(index_table, Table):
            index_table = Table.read(index_table, format='ascii')
            copy = False  # we don't need to make the copy in this case
        if copy:
            index_table = index_table.copy()
        if meta is not None:
            index_table.meta.update(**meta)
        return cls(index_table=index_table)

    def sort(self, *args):
        self.index_table.sort(*args)
        self._bod_list = self['_bod']

    @classmethod
    def vstack(cls, instances):
        tbls = [inst.index_table for inst in instances]
        tbl = vstack(tbls)
        return cls(index_table=tbl)

    def unique(self, *args, **kwargs):
        return self.__class__(
                index_table=unique(self.index_table, *args, **kwargs))

    def join(self, other, *args, **kwargs):
        if isinstance(other, self.__class__):
            other_index_table = other.index_table
        else:
            other_index_table = other
        return self.__class__(
                index_table=join(self.index_table, other_index_table),
                *args, **kwargs)
