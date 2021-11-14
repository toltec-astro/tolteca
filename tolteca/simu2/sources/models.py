#!/usr/bin/env python


from .base import SourceModel


__all__ = ['ImageSourceModel', 'CatalogSourceModel']


class ImageSourceModel(SourceModel):

    n_inputs = 2
    n_outputs = 1

    def evaluate(self, *args, **kwargs):
        pass

    @classmethod
    def from_fits(cls, filepath, extname_map, grouping):
        pass


class CatalogSourceModel(SourceModel):

    n_inputs = 2
    n_outputs = 1

    @classmethod
    def from_file(cls, filepath, colname_map, grouping):
        pass

    def evaluate(self, *args, **kwargs):
        pass
