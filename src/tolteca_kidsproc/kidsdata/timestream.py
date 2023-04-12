from astropy.nddata import NDDataRef
from .utils import FrequencyDivisionMultiplexingDataRef
import astropy.units as u


__all__ = [
    "TimeStream",
]


class TimeStream(FrequencyDivisionMultiplexingDataRef):
    """A container class for time stream data.

    Parameters
    ----------
    I : `astropy.nddata.NDData`, `astropy.units.Quantity`
        The in-phase data, in (or assumed to be in) ADU.
    Q : `astropy.nddata.NDData`, `astropy.units.Quantity`
        The quadrature data, in (or assumed to be in) ADU.
    r : `astropy.nddata.NDData`
        The dimensionless ``radial`` tuning.
    x : `astropy.nddata.NDData`
        The dimensionless detuning.
    f_chans : `astropy.units.Quantity`
        The channel reference frequencies.
    kwargs :
        keyword arguments to pass to `astropy.nddata.NDDataRef`.
    """

    def __init__(self, f_chans=None, times=None, **kwargs):
        attrs_in_kwargs = set(kwargs.keys()).intersection(self._data_variables)
        if "data" in kwargs:
            # In cases of slicing, new objects will be initialized with `data`
            # instead of ``S21``. Ensure we grab the `data` argument.
            if len(attrs_in_kwargs) == 0:
                super().__init__(**kwargs)
                # additional attributes frequency and d21 will be added
                # by the _slice_extra call at the end of the slice
                # automatically
                return
            else:
                raise ValueError("data should not be specified.")

        # this is for normal construction
        data_vars = dict()  # this get updated to the instance
        for attr in attrs_in_kwargs:
            data_var = self._validate_data_var(attr, kwargs.pop(attr))
            data_vars[self._make_data_attr_name(attr)] = NDDataRef(
                data=data_var.value, unit=data_var.unit
            )
        # we need pass some thing as the data, here we use the first
        # data var
        # TODO this may be removed if we subclass data ref slicer
        # to allow data ref of None.
        data_key = next(iter(data_vars.keys()))
        kwargs["data"] = data_vars[data_key].data
        kwargs["unit"] = data_vars[data_key].unit
        super().__init__(f_chans=f_chans, **kwargs)
        self._times = times
        self.__dict__.update(data_vars)

    _I: NDDataRef
    _Q: NDDataRef
    _r: NDDataRef
    _x: NDDataRef
    _data_variables = {"I", "Q", "r", "x"}

    def __str__(self):
        if self.data is None:
            shape = "(empty)"
        else:
            shape = self.data.shape
        return f"{self.__class__.__name__}{shape}"

    @staticmethod
    def _make_data_attr_name(attr):
        return f"_{attr}"

    def _slice_extra(self, item):
        # this returns extra sliced attributes when slicing
        chan_slice = None
        time_slice = None
        if not isinstance(item, tuple):
            # normalize to a tuple
            item = (item,)
        if len(item) == 1:
            (chan_slice,) = item
        elif len(item) == 2:
            chan_slice, time_slice = item
        else:
            raise ValueError("too many slices.")
        result = super()._slice_extra(chan_slice)
        if self._times is not None:
            if time_slice is None:
                result["_times"] = self._times
            else:
                result["_times"] = self._times[time_slice]
        for attr in self._data_variables:
            attr = self._make_data_attr_name(attr)
            v = getattr(self, attr, None)
            if v is not None:
                result[attr] = v[item]
        return result

    @staticmethod
    def _validate_data_var(k, v):
        if k in ["I", "Q"]:
            check_unit = u.adu
        elif k in ["r", "x"]:
            check_unit = u.dimensionless_unscaled
        else:
            raise ValueError(f"unrecognized data variable {k}")
        if isinstance(v, u.Quantity):
            if v.unit != check_unit:
                raise ValueError(f"{k} unit has to be {check_unit}.")
            return v.to(check_unit)
        return v << check_unit

    @property
    def I(self):  # noqa: E743
        """The in-phase data."""
        return self._I.data << self._I.unit

    @property
    def Q(self):
        """The quadrature data."""
        return self._Q.data << self._Q.unit

    @property
    def r(self):  # noqa: E743
        """The dimensionless ``radial`` tuning."""
        return self._r.data << self._r.unit

    @property
    def x(self):
        """The dimensionless detuning."""
        return self._x.data << self._x.unit

    @property
    def times(self):
        return self._times
