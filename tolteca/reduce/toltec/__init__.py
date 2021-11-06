#!/usr/bin/env python


from ..datastore import LocalFileDataStore
from ...datamodels.toltec import BasicObsDataset
from tollan.utils.namespace import NamespaceMixin
from tollan.utils import fileloc
from tollan.utils.log import get_logger
import numpy as np


@LocalFileDataStore.register_data_loader('toltec_bod')
class BasicObsDatasetLoader(object):
    """A class to collect TolTEC basic obs dataset from local files."""

    logger = get_logger()

    @staticmethod
    def _normalize_source(source):
        if isinstance(source, NamespaceMixin):
            source = source.path
        source = fileloc(source)
        return source

    @classmethod
    def identify(cls, source):
        """Identify if `BasicObsDataset` can be constructed from `source`."""
        if not isinstance(source, LocalFileDataStore):
            return False
        source = cls._normalize_source(source)
        if source.path.is_dir():
            return True
        return False

    @classmethod
    def load(cls, source):
        """Load `BasicObsDataset` from `source`."""
        source = cls._normalize_source(source)
        bods = BasicObsDataset.from_files(
                source.path.glob('*'), open_=False)
        # remove any file that does not have valid interface
        bods = bods.select(~np.equal(bods['interface'], None))
        return [bods]

    @classmethod
    def aggregate(cls, items):
        """Aggregate `BasicObsDataset` in `items` into a single one."""
        bods_list = [
            item for item in items if isinstance(item, BasicObsDataset)]
        bods = BasicObsDataset.vstack(bods_list)
        bods.sort(['obsnum', 'interface'])
        # add special index columns for obsnums for backward selection
        for k in ['obsnum', ]:
            bods[f'{k}_r'] = bods[k].max() - bods[k]
        cls.logger.debug(f"collected {len(bods)} data items")
        return [bods]
