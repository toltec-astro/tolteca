#! /usr/bin/env python


"""This module defines modules that manages various information shared
across workers through the IPC extension."""


from dasha.web.extensions.ipc import ipc
import pandas as pd
from dasha.web.extensions.cache import cache
from tollan.utils.log import get_logger


class SharedDataStore(object):

    def __init__(self, label):
        self._label = label

    @property
    def label(self):
        return self._label

    def _make_datastore_label(self, label):
        return f'{self._label}/{label}'

    def _get_datastore(self, label):
        return NotImplemented

    def __getitem__(self, label):
        return self._get_datastore(label)


class SharedDataStoreRedis(SharedDataStore):

    def _get_datastore(self, label):
        return ipc.get_or_create(
                'rejson', label=self._make_datastore_label(label))


class SharedToltecDataset(object):

    _datastore_cls = SharedDataStoreRedis
    from .. import tolteca_toltec_datastore as datafiles

    def __init__(self, label):
        self._datastore = self._datastore_cls(label)
        self._index_table = None

    @property
    def _index_table_store(self):
        return self._datastore['index_table']

    def _update_index_table(self, recreate=False):
        if self._index_table is None or recreate:
            self._index_table = self._index_table_store.get()
        self._index_table = self._index_table_store.get()
        # else:
        #     self._index_table = self._index_table_store.get_if_updated(
        #             self.index_table)

    def set_index_table(self, df):
        self._index_table_store.set(df.to_json())

    @property
    def index_table(self):
        self._update_index_table(recreate=False)
        return pd.read_json(self._index_table)

    @classmethod
    @cache.memoize(timeout=1)
    def files_from_info(cls, entry):
        logger = get_logger()
        pattern = \
            f'**/toltec*_' \
            f'{entry["Obsnum"]:06d}' \
            f'_{entry["SubObsNum"]:02d}_{entry["ScanNum"]:04d}*'
        rpath = SharedToltecDataset.datafiles.rootpath
        logger.info(f"query {rpath} with pattern {pattern}")
        return list(map(str, rpath.glob(pattern)))
