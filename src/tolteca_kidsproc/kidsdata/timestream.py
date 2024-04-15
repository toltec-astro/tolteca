from functools import cached_property

import astropy.units as u
import numpy as np
import numpy.typing as npt
from tollan.utils.np import make_complex
from typing_extensions import assert_never

from .utils import ExtendedNDDataRef, FDMData, FrequencyQuantityType, validate_quantity

__all__ = [
    "MultiTimeStream",
]

DataQuantityType = u.Quantity["dimensionless"]
TimeQuantityType = u.Quantity["time"]


class MultiPSD(FDMData):
    """A container class for PSD data."""

    # TODO: implement this


def _validate_complex_data_input(real, imag, comp, data_group):
    """Return validated data for compelx data input."""
    if real is None and imag is None and comp is None:
        return None, None, None
    dr, di, dc = data_group
    dl = f"{dc}[{dr},{di}]"
    if comp is None and (real is None or imag is None):
        raise ValueError(
            f"{dl} data require both {dr} and {di} "
            f"if complex {dc} is not provided.",
        )
    if comp is not None and sum([real is not None, imag is not None]) == 1:
        raise ValueError(
            f"{dl} require both {dr} and {di} if any of them is provided.",
        )
    if real is not None:
        if not isinstance(real, u.Quantity):
            real = real << u.dimensionless_unscaled
        real = validate_quantity(
            real,
            data_label=dr,
            physical_type=DataQuantityType,
        )
    if imag is not None:
        if not isinstance(imag, u.Quantity):
            imag = imag << u.dimensionless_unscaled
        imag = validate_quantity(
            imag,
            data_label=di,
            physical_type=DataQuantityType,
        )
    if comp is not None:
        if not isinstance(comp, u.Quantity):
            comp = comp << u.dimensionless_unscaled
        comp = validate_quantity(
            comp,
            data_label=dc,
            physical_type=DataQuantityType,
            dtype=complex,
        )
    return real, imag, comp


class MultiTimeStream(FDMData[u.Quantity]):
    """A container class for multiplexed time stream data.

    Parameters
    ----------
    I : `astropy.units.Quantity`
        The in-phase data.
    Q : `astropy.units.Quantity`
        The quadrature data.
    S21 : `astropy.units.Quantity`
        The S21 data.
    r : `astropy.nddata.NDData`
        The dimensionless noise.
    x : `astropy.nddata.NDData`
        The dimensionless detuning.
    X : `astropy.units.Quantity`
        The complex detuning.
    f_chans : `astropy.units.Quantity`
        The channel reference frequencies.
    times : `astropy.units.Quantity`
        The time coorindate.
    index : array-like
        The sample index.
    f_smp : `astropy.units.Quantity`
        The sample frequency.
    **kwargs :
        keyword arguments to pass to `astropy.nddata.NDDataRef`.
    """

    _slice_attrs_default = (
        "_I",
        "_Q",
        "_S21",
        "_S21_computed",
        "_r",
        "_x",
        "_X",
        "_X_computed",
        ("_times", lambda s: s[1] if len(s) > 1 else None),
        ("_f_smp", lambda _: None),
        ("_index", lambda s: s[1] if len(s) > 1 else None),
    ) + FDMData._slice_attrs_default  # noqa: SLF001
    _I: DataQuantityType
    _Q: DataQuantityType
    _S21: DataQuantityType
    _r: DataQuantityType
    _x: DataQuantityType
    _X: DataQuantityType
    _times: TimeQuantityType
    _f_smp: FrequencyQuantityType
    _index: npt.ArrayLike

    def __init__(  # noqa: PLR0913
        self,
        I=None,  # noqa: E741
        Q=None,
        S21=None,
        r=None,
        x=None,
        X=None,
        f_chans=None,
        times=None,
        f_smp=None,
        index=None,
        **kwargs,
    ):
        kwargs.setdefault("data", None)
        super().__init__(
            I=I,
            Q=Q,
            S21=S21,
            r=r,
            x=x,
            X=X,
            f_chans=f_chans,
            times=times,
            f_smp=f_smp,
            index=index,
            **kwargs,
        )

    @classmethod
    def _prepare_init_args(  # noqa: PLR0913
        cls,
        data,
        I,  # noqa: E741
        Q,
        S21,
        r,
        x,
        X,
        f_chans,
        times,
        f_smp,
        index,
        **kwargs,
    ):
        data_vars = {
            n: v
            for n, v in zip(
                ["_I", "_Q", "_S21", "_r", "_x", "_X"],
                (I, Q, S21, r, x, X),
                strict=True,
            )
            if v is not None
        }
        # map the first datavar as data
        data_attr_name, data_attr_var = next(iter(data_vars.items()))
        data_shapes = {n: v.shape for n, v in data_vars.items()}
        # check shape the same
        if len(set(data_shapes.values())) != 1:
            raise f"inconsistent data shapes: {data_shapes}"
        # check by group
        I, Q, S21 = _validate_complex_data_input(  # noqa: E741
            I,
            Q,
            S21,
            ("I", "Q", "S21"),
        )
        r, x, X = _validate_complex_data_input(r, x, X, ("r", "x", "X"))
        # check f_chans
        f_chans = FDMData._prepare_init_args(  # noqa: SLF001
            data_attr_var,
            f_chans,
        ).attrs["_f_chans"]

        # check one of index f_smp or times is set.
        if times is None and index is None:
            raise ValueError("one of time or index required.")
        if times is not None and index is not None and times.shape != index.shape:
            raise ValueError("time and index shape does not match.")
        if (times is not None and times.ndim != 1) or (
            index is not None and index.ndim != 1
        ):
            raise ValueError("times or index has to be 1-d.")
        if f_smp is not None:
            f_smp = validate_quantity(
                f_smp,
                "f_smp",
                physical_type=FrequencyQuantityType,
            )
        if times is not None:
            times = validate_quantity(times, "times", physical_type=TimeQuantityType)
        if index is not None and not np.issubdtype(index.dtype, np.integer):
            raise ValueError("index has to be int.")
        if times is None and index is not None and f_smp is not None:
            # compute time from index and f_smp
            times = (index / f_smp).to(u.s)
        if times is not None and index is None and f_smp is not None:
            # compute index from time and f_smp
            index = (times * f_smp).to_value(u.dimensionless_unscaled).astype(int)

        # pass the data attr as data
        attrs = {
            "_I": I,
            "_Q": Q,
            "_S21": S21,
            "_r": r,
            "_x": x,
            "_X": X,
            "_f_chans": f_chans,
            "_times": times,
            "_f_smp": f_smp,
            "_index": index,
            "_data_attr_name": data_attr_name,
        }
        # return the attrs without the data attr.
        data = attrs.pop(data_attr_name)
        return ExtendedNDDataRef.InitArgs(
            data=data,
            kwargs=kwargs,
            attrs=attrs,
            meta={"f_smp": f_smp},
        )

    def _get_data_from_group(self, real, imag, comp, name):  # noqa: PLR0911
        vr = getattr(self, real)
        vi = getattr(self, imag)
        vc = getattr(self, comp)
        if name == real:
            if vr is not None:
                return vr
            return vc.real if vc is not None else None
        if name == imag:
            if vi is not None:
                return vi
            return vc.imag if vc is not None else None
        if name == comp:
            if vc is not None:
                return vc
            if vr is not None and vi is not None:
                return getattr(self, f"{comp}_computed")
            return None
        assert_never()

    @property
    def I(self):  # noqa: E743
        """The in-phase data."""
        return self._get_data_from_group("_I", "_Q", "_S21", "_I")

    @property
    def Q(self):
        """The quadrature data."""
        return self._get_data_from_group("_I", "_Q", "_S21", "_Q")

    @property
    def S21(self):
        """The S21 data."""
        return self._get_data_from_group("_I", "_Q", "_S21", "_S21")

    @cached_property
    def _S21_computed(self):
        return make_complex(self._I, self._Q)

    @property
    def r(self):
        """The noise data."""
        return self._get_data_from_group("_r", "_x", "_X", "_r")

    @property
    def x(self):
        """The detuning data."""
        return self._get_data_from_group("_r", "_x", "_X", "_x")

    @property
    def X(self):
        """The quadrature data."""
        return self._get_data_from_group("_r", "_x", "_X", "_X")

    @cached_property
    def _X_computed(self):
        return make_complex(self._r, self._x)

    @property
    def times(self):
        """The time coordinates."""
        return self._times

    @property
    def f_smp(self):
        """The sample frequency."""
        return self._f_smp

    @property
    def index(self):
        """The sample indices."""
        return self._index
