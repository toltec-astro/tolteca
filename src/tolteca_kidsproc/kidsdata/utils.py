import itertools
from collections.abc import Callable
from functools import cached_property
from typing import ClassVar, Generic, NamedTuple, Protocol, TypeVar

import astropy.units as u
import numpy.typing as npt
from astropy.nddata import NDDataRef, NDUncertainty
from tollan.utils.general import ignore_unexpected_kwargs
from tollan.utils.log import logger
from tollan.utils.typing import get_physical_type_from_quantity_type

__all__ = [
    "NDDataProtocol",
    "ExtendedNDDataRef",
    "FrequencyDivisionMultiplexingMixin",
    "FDMData",
]


T = TypeVar("T")
QT = TypeVar("QT", bound=u.Quantity)
FrequencyQuantityType = u.Quantity["frequency"]


def validate_quantity(data, data_label="data", dtype=None, physical_type=None) -> QT:
    if not isinstance(data, u.Quantity):
        raise ValueError(f"{data_label} should be quantity.")  # noqa: TRY004
    if dtype is not None and data.dtype != dtype:
        raise ValueError(f"{data_label} dtype should be {dtype}, got {data.dtype}.")
    if physical_type is None:
        return data
    if isinstance(physical_type, str):
        physical_type = u.get_physical_type(physical_type)
    elif isinstance(physical_type, type(u.Quantity["dimensionless"])):
        physical_type = get_physical_type_from_quantity_type(physical_type)
    if data.unit.physical_type != physical_type:
        raise ValueError(
            f"{data_label} should have {physical_type=!s}.",
        )
    return data


class NDDataProtocol(Protocol):
    """A protocol class for NDData."""

    @property
    def data(self) -> npt.ArrayLike:
        """The data."""

    @property
    def unit(self) -> u.Unit:
        """The unit."""

    @property
    def uncertainty(self) -> NDUncertainty:
        """The uncertainty."""

    @property
    def quantity(self) -> u.Quantity:
        """The data as quantity."""

    def reset_cache(self):
        """Reset any cached property."""


def _do_mapped_slice(slice_mapper, data, item):
    if data is None:
        return data
    if not slice_mapper:
        # just slice assuming dimension match
        return data[item]
    # call mapper to map the slice
    if not isinstance(item, tuple):
        item = (item,)
    s = slice_mapper(item)
    if s is not None:
        return data[s]
    return data


SliceSpec = str | tuple[str, Callable[[tuple], tuple]]


class ExtendedNDDataRef(NDDataRef):
    """A little bit more than `~astropy.nddata.NDDataRef`.

    This allows slicing of extra data items on the class or on the meta.
    """

    _slice_attrs_default: ClassVar[tuple[SliceSpec, ...]] = ()
    _slice_meta_keys_default: ClassVar[tuple[SliceSpec, ...]] = ()
    _slice_meta_keys: list[SliceSpec]
    # this is used to delegage a given attr as the data object.
    _data_attr_name: str = None

    class InitArgs(NamedTuple):
        """A stuct to hold return value of ``_prepare_init_args``."""

        data: npt.ArrayLike
        kwargs: dict[str, None | npt.ArrayLike]
        attrs: dict[str, npt.ArrayLike]
        meta: dict[str, npt.ArrayLike]

    def __init__(
        self,
        data,
        slice_meta_keys=None,
        _slice=False,
        **kwargs,
    ):
        if _slice:
            ignore_unexpected_kwargs(super().__init__)(data, **kwargs)
        else:
            init_args = self._prepare_init_args(
                data,
                **kwargs,
            )
            super().__init__(init_args.data, **init_args.kwargs)
            self.__dict__.update(init_args.attrs)
            self.meta.update(init_args.meta)
        self._slice_meta_keys = slice_meta_keys or []

    @classmethod
    def _prepare_init_args(cls, data, **kwargs):
        return cls.InitArgs(
            data=data,
            kwargs=kwargs,
            attrs={},
            meta={},
        )

    @property
    def quantity(self):
        """Return data as quantity."""
        return self.data << self.unit

    def __repr__(self):
        if not hasattr(self, "_data"):
            body = "(uninitalized)"
        else:
            body = "(empty)" if self.data is None else self.data.shape
        return f"{self.__class__.__name__}{body}"

    def __getitem__(self, item):
        """Implement slice of additional attributes along with this object."""
        inst = super().__getitem__(item)
        attrs, meta = self._slice_extra(item)
        inst.__dict__.update(attrs)
        inst.meta.update(meta)
        return inst

    def _slice(self, item):
        kwargs = super()._slice(item)
        # add a marker key so constructor can implement special protocol
        # to handle slicing.
        kwargs["slice_meta_keys"] = self._slice_meta_keys
        kwargs["_slice"] = True
        # make a shallow copy of meta so later on we slice on it.
        # TODO: a deepcopy may be desired
        kwargs["meta"] = kwargs.pop("meta").copy()
        return kwargs

    def _slice_extra(self, item):
        """Return a dict of slided data.

        Subclass can extend this to have control over what data get sliced.
        """
        attrs = {}
        _missing = object()
        for spec in self._slice_attrs_default:
            if isinstance(spec, str):
                a, mapper = spec, None
            else:
                a, mapper = spec
            v = self.__dict__.get(a, _missing)
            if v is not _missing:
                attrs[a] = _do_mapped_slice(mapper, v, item)
        attrs["_data_attr_name"] = getattr(self, "_data_attr_name", None)
        meta = {}
        for spec in itertools.chain(
            self._slice_meta_keys_default,
            self._slice_meta_keys,
        ):
            if isinstance(spec, str):
                k, mapper = spec, None
            else:
                k, mapper = spec
            v = self.meta.get(k, _missing)
            if v is not _missing:
                meta[k] = _do_mapped_slice(mapper, v, item)

        return attrs, meta

    def __getattr__(self, name):
        if self._data_attr_name == name:
            return self.data << self.unit
        return self.__getattribute__(name)

    def reset_cache(self):
        """Reset any cached property."""
        for spec in self._slice_attrs_default:
            name = spec[0] if isinstance(spec, tuple) else spec
            if isinstance(getattr(self.__class__, name, None), cached_property) and (
                name in self.__dict__
            ):
                logger.debug(f"reset cached attr {name}")
                del self.__dict__[name]

    def add_slicable_meta(self, key, value, mapper=None):
        """Add meta data that can be propagated correctly when sliced."""
        spec_new = key if mapper is None else (key, mapper)
        for i, spec in enumerate(self._slice_meta_keys):
            k = spec if isinstance(spec, str) else spec[0]
            if k == key:
                # update existing with new mapper
                self._slice_meta_keys[i] = spec_new
                break
        else:
            # not found, add new
            self._slice_meta_keys.append(spec_new)
        self.meta[key] = value


class FrequencyDivisionMultiplexingMixin:
    """A mixin class for frequency division multiplexed data.

    The FDM technique is used in reading the data from large format KIDs
    array. This class defines a generic structure to handle multi-dimensional
    data generated this way.

    The data shall be organized such that the first axis is for
    the different readout channels characterized by a set of ``f_chans``.
    """

    _slice_attrs_default = (("_f_chans", lambda s: s[0]),)
    _f_chans: FrequencyQuantityType

    @classmethod
    def _prepare_init_args(cls, data, f_chans, **kwargs):
        # check lengths of axis
        if data.shape[0] != f_chans.shape[0]:
            raise ValueError(
                f"data of shape {data.shape} is incompatible with the"
                f" length of the channel frequencies {len(f_chans)}.",
            )
        if f_chans.ndim != 1:
            raise ValueError(f"Only 1-d f_chans is supported, got {f_chans.shape}.")
        f_chans = validate_quantity(
            f_chans,
            "f_chans",
            physical_type=FrequencyQuantityType,
        )
        return ExtendedNDDataRef.InitArgs(
            data=data,
            kwargs=kwargs,
            attrs={"_f_chans": f_chans},
            meta={},
        )

    @property
    def f_chans(self) -> FrequencyQuantityType:
        """The channel tone frequencies."""
        return self._f_chans

    @property
    def n_chans(self):
        """The number of channels."""
        return self.f_chans.shape[0]


class FDMData(FrequencyDivisionMultiplexingMixin, ExtendedNDDataRef, Generic[T]):
    """A generic container for FDM arbitury data.

    Parameters
    ----------
    f_chans : `astropy.units.Quantity`
        The channel reference frequencies. Typically the center of the sweep.
    kwargs :
        keyword arguments to pass to `astropy.nddata.NDDataRef`.
    """
