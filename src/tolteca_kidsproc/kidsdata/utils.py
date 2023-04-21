from typing import Any, Callable

import astropy.units as u
from astropy.nddata import NDDataRef

__all__ = [
    "ExtendedNDDataRef",
]


class ExtendedNDDataRef(NDDataRef):
    """A little bit more than `~astropy.nddata.NDDataRef`.

    This allows slicing of extra data items.
    """

    _slice_extra: Callable
    meta: dict

    def __repr__(self):
        data = getattr(self, "_data", None)
        shape = data.shape if data is not None else "(empty)"
        return f"{self.__class__.__name__}{shape}"

    def __getitem__(self, item):
        """Implement slice of additional attributes along with this object.

        Define the list of attributes to slice in `_attr_to_slice` class
        attributes.

        """
        inst = super().__getitem__(item)
        inst.__dict__.update(self._slice_extra(item))
        return inst


class FrequencyDivisionMultiplexingDataRef(ExtendedNDDataRef):
    """A generic container for frequency division multiplexed data.

    The FDM technique is used in reading the data from large format KIDs
    array. This class defines a generic structure to handle multi-dimensional
    data generated this way.

    The data shall be organized such that the first axis is for
    the different readout channels characterized by a set of ``f_chans``.

    Parameters
    ----------
    f_chans : `astropy.units.Qunatity`
        The channel's reference frequency, typically the center of the sweep.
    data : `astropy.nddata.NDData`
        The data.
    kwargs :
        keyword arguments to pass to `astropy.nddata.NDDataRef`.
    """

    _f_chans: Any

    def _slice_extra(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        return {"_f_chans": self._f_chans[item[0]]}  # type: ignore

    def __init__(self, f_chans=None, data=None, **kwargs):
        # check lengths of axis
        if data is not None and f_chans is not None:
            if data.shape[0] != len(f_chans):
                raise ValueError(
                    f"data of shape {data.shape} is incompatible with the"
                    f" length of the tone frequencies {len(f_chans)}."
                )

        self._f_chans = f_chans
        super().__init__(data=data, **kwargs)

    @property
    def f_chans(self):
        """The channel tone frequencies."""
        return self._f_chans

    @property
    def n_chans(self):
        """The number of channels."""
        return len(self.f_chans)
