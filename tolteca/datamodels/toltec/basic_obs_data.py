#! /usr/bin/env python

from tollan.utils.log import get_logger
from astropy.table import Table, MaskedColumn
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

    """

    logger = get_logger()

    def __init__(self, source):
        file_loc = self._file_loc = self._normalize_file_loc(source)
        if file_loc.is_local:
            file_obj = open_file(file_loc.path)
        else:
            file_obj = None
        self._file_loc = file_loc
        self._file_obj = file_obj
        self._update_meta()

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
        logged_dict_update(
                cls.logger.warning, meta, meta_from_source(file_loc))
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

    cols_from_meta: str, list
        The columns to include in the index table, extracted from the
        loaded meta data.
    """

    def __init__(
            self,
            index_table=None,
            bod_list=None,
            include_meta_cols='intersection'):
        if index_table is None and bod_list is None:
            raise ValueError('need one of index_table or bod_list.')
        if index_table is None:
            index_table = self._make_index_table(bod_list)
        elif bod_list is None:
            bod_list = self._make_bod_list(index_table)
        assert len(index_table) == len(bod_list)
        self._index_table = index_table
        self._bod_list = bod_list
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
    def _make_bod_list(cls, tbl):
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
                bod = BasicObsData(source)
            except Exception:
                bod = None
            bods.append(bod)

        return np.array(bods, dtype=object)

    @classmethod
    def _make_index_table(cls, bod_list):
        tbl = defaultdict(list)
        for bod in bod_list:
            with bod.open():
                for col in cls._cols_bod_info:
                    if col == 'source':
                        value = bod.file_loc.path.as_posix()
                    else:
                        value = bod.meta.get(col, None)
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
        return self.index_table[arg]

    def __setitem__(self, arg, value):
        self.index_table[arg] = value

    @property
    def history(self):
        if 'history' not in self._index_table.meta:
            self._index_table.meta['history'] = []
        return self._index_table.meta['history']

    @property
    def index_table(self):
        """The index table."""
        return self._index_table

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
                df = tbl.to_pandas()
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
                bod_list=self._bod_list[item])

    def __repr__(self):
        pformat_tbl = self.index_table.pformat(max_width=-1)
        if pformat_tbl[-1].startswith("Length"):
            pformat_tbl = pformat_tbl[:-1]
        pformat_tbl = '\n'.join(pformat_tbl)
        return f"{self.__class__.__name__}" \
               f":\n{pformat_tbl}"

    @classmethod
    def from_files(cls, files, **kwargs):
        if not files:
            raise ValueError("no file specified")
        bod_list = [BasicObsData(f) for f in files]
        return cls(bod_list=bod_list, **kwargs)

    @classmethod
    def from_index_table(cls, index_table, copy=True, meta=None):
        """Return a dataset from an index table.

        Parameters
        ----------
        index_table : `astropy.table.Table`
            The table that contains the BOD identifiers.

        copy : bool
            If True, a copy of the `index_table` is made.

        meta : dict
            Additional meta get stored in the table.
        """
        if copy:
            index_table = index_table.copy()
        index_table.meta.update(**meta)
        return cls(index_table=index_table)
