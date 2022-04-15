#!/usr/bin/env python


from enum import Flag, auto
import warnings

import collections.abc
import dill

from astropy.table import Table

from tollan.utils import fileloc
from tollan.utils.fmt import pformat_fancy_index
from tollan.utils.log import get_logger

from ...utils import yaml_load


class DataItemKind(Flag):
    """Type for data product"""

    Unspecified = 0

    # content type
    TimeOrderedData = auto()
    Image = auto()
    Cube = auto()
    Catalog = auto()

    # calibration state
    Raw = auto()
    Calibrated = auto()

    # some common composite flags
    RawTimeOrderedData = Raw | TimeOrderedData
    CalibratedTimeOrderedData = Calibrated | TimeOrderedData
    CalibratedImage = Calibrated | Image


class ImageDataKind(Flag):
    """The image data kinds."""

    Unspecified = 0

    # signal
    Signal = auto()
    NoiseRealization = auto()
    Measurement = Signal | NoiseRealization

    # uncertainty
    Weight = auto()
    Variance = auto()
    RMS = auto()
    Uncertainty = Weight | Variance | RMS

    # ancillary
    Coverage = auto()
    Flag = auto()
    Kernel = auto()


class DataProd(object):
    """The class to hold data products."""

    data_item_kinds = DataItemKind.Unspecified
    """The data item kind flags.
    """

    logger = get_logger()

    def __init__(self, source):
        index, index_table = self._resolve_source(source)
        self._index = index
        self._index_table = index_table

    @classmethod
    def _resolve_source(cls, source):
        """Load index and index table from source."""

        def _index_table_to_index(tbl):
            use_cols = [
                c for c in tbl.colnames
                if not (
                    tbl.dtype[c].hasobject
                    or tbl.dtype[c].ndim > 0
                    )]
            index = dict({
                'meta': dict(**index_table.meta),
                'data_items': index_table[use_cols].to_pandas().to_dict(
                    orient='records')
                })
            return index

        def _index_to_index_table(index):
            # when data_item is data prod, explode the meta dict
            data_items = list()
            for data_item in index['data_items']:
                if isinstance(data_item, DataProd):
                    d = dict(**data_item.meta)
                    d['_data_item'] = data_item
                    data_items.append(d)
                else:
                    data_items.append(data_item)
            tbl = Table(rows=data_items)
            tbl.meta.update(**index['meta'])
            return tbl

        try:
            source = fileloc(source)
            # file, load file to index or index table
            if source.is_remote:
                raise NotImplementedError(
                    'Loading data product from remote source'
                    ' is not implemented yet.')
            try:
                index_table = Table.read(source, format='ascii.ecsv')
                index = _index_table_to_index(index_table)
            except ValueError:
                with open(source.path, 'r') as fo:
                    index = yaml_load(fo)
                index_table = _index_to_index_table(index)
        except ValueError:
            # already index or index table
            if isinstance(source, Table):
                index_table = source
                index = _index_table_to_index(index_table)
            elif isinstance(source, collections.abc.Mapping):
                index = source
                index_table = _index_to_index_table(index)
            else:
                raise ValueError(f"Invalid data prod source {source}")
        # attach some info to the runtime dict for further use.
        # index_table.meta['_dp_runtime']['source'] = source
        return index, index_table

    @property
    def index(self):
        """The index dict of the data product."""
        return self._index

    @property
    def meta(self):
        """The meta data dict of the data product."""
        return self.index_table.meta

    @property
    def history(self):
        if 'history' not in self.meta:
            self.meta['history'] = []
        return self.meta['history']

    @property
    def index_table(self):
        """Subclass implement this to return the index table.

        The index table should hold the data items in column ``data_item``.
        """
        return self._index_table

    @property
    def data_items(self):
        """Return the data items."""
        return self.index_table['_data_item']

    def __iter__(self):
        return self.data_items.__iter__()

    def __len__(self):
        return len(self.index_table)

    def copy(self):
        index_table = self.index_table.copy()
        return self.__class__(source=index_table)

    @classmethod
    def _get_sliced_type(cls):
        """Subclass implement to specify sliced object type returned by
        ``__getitem__``."""
        return None

    def _sliced_new(self, item_mask):
        sliced_type = self._get_sliced_type()
        if sliced_type is None:
            return self.index_table[item_mask]
        return sliced_type(source=self.index_table[item_mask])

    def __getitem__(self, arg):
        # the bracket operators are passed on to the index table
        # for column selection, we return the table, and
        # for row selection we return sliced object if
        # sliced type is defined
        if isinstance(arg, str):
            _arg = [arg]
        else:
            _arg = arg
        if all((a in self.index_table.colnames) for a in _arg):
            # column
            return self.index_table[arg]
        # row query, just do a sliced new
        return self._sliced_new(arg)

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
        """Return data product with subset of data items specified by `cond`.

        When `cond` is a string, the index table is converted to
        `~pandas.DataFrame` and the selection is done by
        :meth:`~pandas.DataFrame.eval`.

        Parameters
        ----------
        cond : str or `~numpy.ndarray`
            The condition to create the subset.

        desc : str
            A description of this selection to include in the index table meta.

        Returns
        -------
        `DataProduct` or None
            The data product containing subset of data items, or None if
            no entry is selected.
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
            return None

        self.history.append({
                'dp_select': {
                    'cond_str': cond_str,
                    'description': desc if desc is not None else ""
                    }
                })
        inst = self._sliced_new(m)
        self.logger.debug(
                f"selected {inst.pformat()}\n"
                f"({len(inst)} out of {len(self)} entries)")
        return inst

    def __repr__(self):
        return f"{self.__class__.__name__}(n_data_items={len(self)})"

    def pformat(self):
        pformat_tbl = self.index_table.pformat(max_width=-1)
        if pformat_tbl[-1].startswith("Length"):
            pformat_tbl = pformat_tbl[:-1]
        pformat_tbl = '\n'.join(pformat_tbl)
        return f"{self.__class__.__name__}" \
               f":\n{pformat_tbl}"
