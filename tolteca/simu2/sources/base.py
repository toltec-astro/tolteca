#!/usr/bin/env python

from astropy.modeling import Model


__all__ = ['SurfaceBrightnessModel', 'PowerLoadingModel']


class SourceModelBase(Model):
    pass


class SurfaceBrightnessModel(SourceModelBase):
    """The base class for simulator source specified as surface brightness.
    """

    fittable = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs = ("S", )


class PowerLoadingModel(SourceModelBase):
    """The base class for simulator source specified as power loading.
    """

    fittable = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs = ("P", )
