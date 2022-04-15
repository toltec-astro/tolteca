#! /usr/bin/env python


import re
import numpy as np

from ...common.toltec import toltec_info
from ..dp.base import DataProd, DataItemKind


__all__ = ['ToltecDataProd', ]


class ToltecDataProd(DataProd):
    """The base class for TolTEC data products.

    """
    @classmethod
    def collect_from_dir(cls, dirpath):
        # dive into dir looking for data products
        # TODO need to refactor this to make it extendable
        # to all sub types
        results = list()
        for p in dirpath.iterdir():
            if p.is_dir() and re.match(r'redu\d+', p.name):
                # citlali result
                results.extend(
                    ScienceDataProd.collect_from_citlali_output_dir(p))
        return cls(source={
            'meta': {
                'name': dirpath.name
                },
            'data_items': results
            })

    @classmethod
    def _get_sliced_type(cls):
        return ToltecDataProd


class ScienceDataProd(ToltecDataProd):
    """The class for TolTEC science data products."""

    def __init__(self, source):
        super().__init__(source)
        # index_table = self.index_table
        # TODO
        # implement the DataProd.open interface.
        # index_table['_data_item'] = index_table['filepath']

    @property
    def data_item_kinds(self):
        return np.logical_or.reduce(self.index_table['kind'], 0)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}'
            f'({self.meta["name"]}, id={self.meta["id"]})'
            )

    @classmethod
    def collect_from_citlali_output_dir(cls, dirpath):
        # collect fits images from dirpath
        results = list()
        for p in dirpath.iterdir():
            if p.is_dir() and re.match(r'\d{6}', p.name):
                # per-obs reduction
                # build the index
                data_items = [
                        {
                            'array_name': array_name,
                            'kind': DataItemKind.CalibratedImage,
                            'filepath': list(
                                p.glob(f'toltec_*_{array_name}*.fits'))[-1],
                            }
                        for array_name in toltec_info['array_names']
                        ]
                ctod = list(p.glob('toltec_*_timestream_*.nc'))
                if ctod:
                    ctod = ctod[-1]
                    data_items.append({
                        'array_name': None,
                        'kind': DataItemKind.CalibratedTimeOrderedData,
                        'filepath': ctod
                        })
                index = {
                    'data_items': data_items,
                    'meta': {
                        'name': f'm{int(p.name)}',
                        'id': int(f'{p.parent}'.split('u')[-1]),
                        }
                    }
                results.append(cls(source=index))
        return results
