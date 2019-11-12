import logging
from copy import deepcopy

import numpy as np
from astropy import units as u
from astropy import constants as cnst
from astropy.nddata import NDDataRef
from astropy.utils.decorators import lazyproperty

from ..wcs import WCSAdapter, WCSWrapper
from .time_mixin import TimeStreamMixin

__all__ = ['TimeStream']


class TimeStream(TimeStreamMixin, NDDataRef):

    """
    Container for time stream data.

    Parameters
    ----------
    time_axis : `astropy.units.Quantity`
        Time information with the same shape as the last (or only)
        dimension of data.
    wcs : `astropy.wcs.WCS` or `gwcs.wcs.WCS`
        WCS information object.
    data : `numpy.ndarray`-like
        The time stream data.
    uncertainty : `astropy.nddata.NDUncertainty`
        Contains uncertainty information along with propagation rules.
    meta : dict
        Any user-specific information to be carried around.
    """
    def __init__(self, time_axis=None, data=None, **kwargs):
        unknown_kwargs = set(kwargs).difference(
            {'unit', 'uncertainty', 'meta', 'mask', 'copy'})

        if len(unknown_kwargs) > 0:
            raise ValueError("unknown kwarg(s): {}."
                             "".format(', '.join(map(str, unknown_kwargs))))

        data = kwargs.setdefault('data', None)
        kwargs.setdefault(
                'unit', data.unit if isinstance(data, u.Quantity) else None)

        # Parse time axis data as wcs object
        if not isinstance(time_axis, u.Quantity):
            raise ValueError("time axis must be a `Quantity` object.")
        wcs = WCSWrapper.from_array(time_axis)

        # Check shape
        if data is not None and time_axis is not None:
            if not time_axis.shape[0] == data.shape[-1]:
                raise ValueError(
                        'time axis ({}) and the last data axis ({})'
                        ' lengths must be the same'.format(
                            time_axis.shape[0], data.shape[-1]))

        super().__init__(wcs=wcs, data=data, **kwargs)

        if hasattr(self, 'uncertainty') and self.uncertainty is not None:
            if not self.data.shape == self.uncertainty.shape:
                raise ValueError(
                        "data ({}) and uncertainty ({}) shapes"
                        " must be the same.".format(
                            self.data.shape, self.uncertainty.shape))

    @property
    def time(self):
        return self.time_axis

    @property
    def shape(self):
        return self.data.shape
