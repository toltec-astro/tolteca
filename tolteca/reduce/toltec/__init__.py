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
            # by default we select the last item from the BODs
            # if select is not given
            select = source.select or 'obsnum_r == 0'
            source = source.path
        else:
            select = None
        source = fileloc(source)
        return source, select

    @classmethod
    def identify(cls, source):
        """Identify if `BasicObsDataset` can be constructed from `source`."""
        if not isinstance(source, LocalFileDataStore):
            return False
        source, _ = cls._normalize_source(source)
        if source.path.is_dir():
            return True
        return False

    @classmethod
    def load(cls, source, select=None):
        """Load `BasicObsDataset` from `source`."""
        source, default_select = cls._normalize_source(source)
        if select is None:
            select = default_select
        bods = BasicObsDataset.from_files(
                source.path.glob('*'), open_=False)
        # remove any file that does not have valid interface
        bods = bods.select(~np.equal(bods['interface'], None))
        if select is not None:
            # apply custom select
            bods = cls._normalize_bods(bods).select(select)
        return [bods]

    @classmethod
    def aggregate(cls, items):
        """Aggregate `BasicObsDataset` in `items` into a single one."""
        bods_list = [
            item for item in items if isinstance(item, BasicObsDataset)]
        bods = cls._normalize_bods(BasicObsDataset.vstack(bods_list))
        cls.logger.debug(f"collected {len(bods)} data items")
        return [bods]

    @classmethod
    def _normalize_bods(cls, bods):
        bods.sort(['obsnum', 'interface'])
        # add special index columns for obsnums for backward selection
        for k in ['obsnum', ]:
            bods[f'{k}_r'] = bods[k].max() - bods[k]
        return bods
