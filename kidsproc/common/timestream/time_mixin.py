import numpy as np
from astropy.utils.decorators import lazyproperty


__all__ = ['TimeStreamMixin']


class TimeStreamMixin:
    @property
    def _time_axis_numpy_index(self):
        return self.data.ndim - 1 - self.wcs.wcs.spec

    @property
    def _time_axis_len(self):
        """
        How many elements are in the time dimension?
        """
        return self.data.shape[self._time_axis_numpy_index]

    @property
    def _data_with_time_axis_last(self):
        """
        Returns a view of the data with the time axis last
        """
        if self._time_axis_numpy_index == self.data.ndim - 1:
            return self.data
        else:
            return self.data.swapaxes(self._time_axis_numpy_index,
                                      self.data.ndim - 1)

    @property
    def _data_with_time_axis_first(self):
        """
        Returns a view of the data with the time axis first
        """
        if self._time_axis_numpy_index == 0:
            return self.data
        else:
            return self.data.swapaxes(self._time_axis_numpy_index, 0)

    @property
    def time_wcs(self):
        return self.wcs.axes.time

    @lazyproperty
    def time_axis(self):
        """
        Returns a Quantity array with the values of the time axis.
        """

        if len(self.data) > 0:
            time_axis = self.wcs.pixel_to_world(np.arange(self.data.shape[-1]))
        else:
            # After some discussion it was suggested to create the empty
            # spectral axis this way to better use the WCS infrastructure. This
            # is to prepare for a future where pixel_to_world might yield
            # something more than just a raw Quantity, which is planned for the
            # mid-term in astropy and possible gwcs.  Such changes might
            # necessitate a revisit of this code.
            dummy_timestream = self.__class__(
                    time_axis=[1, 2] * self.wcs.time_axis_unit,
                    flux=[1, 2] * self.data.unit)
            time_axis = dummy_timestream.wcs.pixel_to_world([0])[1:]
        return time_axis

    @property
    def time_axis_unit(self):
        """
        Returns the units of the spectral axis.
        """
        return self.wcs.time_axis_unit


class InplaceModificationMixin:
    pass
