#!/usr/bin/env python

from astropy.modeling import Model


__all__ = ['SourceModel', ]


class SourceModel(Model):
    """The base class for simulator source models.
    """

    fittable = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = self.input_frame.axes_names
        self.outputs = ("S", )
