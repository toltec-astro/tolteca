from typing import Any, cast

import astropy.units as u
import numpy as np
from astropy.nddata import NDDataRef, NDUncertainty
from scipy.ndimage import uniform_filter1d
from tollan.utils.log import logger, timeit
from tollan.utils.np import make_complex

from .utils import ExtendedNDDataRef, FrequencyDivisionMultiplexingDataRef

__all__ = [
    "Sweep",
    "MultiSweep",
]


class SweepMixin:
    """A mixin class for frequency sweep data."""

    uncertainty: NDUncertainty
    _frequency: NDDataRef
    _D21: NDDataRef

    @property
    def S21(self):
        """The S21."""
        return self.data << self.unit  # type: ignore

    @property
    def I(self):  # noqa: E743
        """The In-phase data."""
        return self.S21.real

    @property
    def Q(self):
        """The quadrature data."""
        return self.S21.imag

    @property
    def S21_unc(self):
        """The uncertainty of S21."""
        return self.uncertainty.quantity

    @property
    def I_unc(self):
        """The In-phase uncertainty."""
        return self.S21_unc.real

    @property
    def Q_unc(self):
        """The quadrature uncertainty."""
        return self.S21_unc.imag

    @property
    def frequency(self):
        """The frequency."""
        return self._frequency.data << self._frequency.unit

    @property
    def D21(self):
        """The D21."""
        return self._D21.data << self._D21.unit

    @staticmethod
    def _validate_S21(S21):
        # note that the returned item here is a quantity
        # which is different from the other _validate_* methods.
        if S21.dtype != complex:
            raise ValueError("S21 has to be complex.")
        if isinstance(S21, u.Quantity):
            if S21.unit != u.adu:
                raise ValueError("S21 unit has to be adu.")
            return S21.to(u.adu)
        return S21 << u.adu

    @classmethod
    def _validate_S21_unc(cls, S21_unc):
        return cls._validate_S21(S21_unc)

    @staticmethod
    def _validate_frequency(frequency):
        if (
            not isinstance(frequency, u.Quantity)
            or frequency.unit.physical_type != "frequency"  # type: ignore
        ):
            raise ValueError("frequency has to be a quantity with frequency unit.")
        return NDDataRef(data=frequency.value, unit=frequency.unit)

    @staticmethod
    def _validate_D21(D21):
        if not isinstance(D21, u.Quantity):
            raise TypeError("D21 has to be a quantity with unit.")
        if not D21.unit.is_equivalent(u.adu / u.Hz):  # type: ignore
            raise ValueError("invalid unit for D21.")
        return NDDataRef(data=D21.value, unit=D21.unit)


class Sweep(ExtendedNDDataRef, SweepMixin):
    """A container class for frequency sweep data.

    The response of the sweep is expressed as the ``S21`` parameter of the
    2-ports readout circuit, in its native analog-to-digit unit (ADU),
    as a function of the probing frequency.

    `S21` could also be a n-dim array for `Sweep`, in which case the
    data is for a collection of sweeps that share the same frequency grid.
    The data shall be organized such that the frequency varies along the last
    dimension. Note that this is different from `MultiSweep` in that
    for `MultiSweep`, each sweep has its own frequency grid.

    The sweep may also be provided with a `D21` spectrum, which is the
    magnitude of the derivative of the complex `S21` with respect to the
    frequency. The `D21` spectrum is a reduced form of `S21` thus does not
    carry all information that `S21` has, but it is easier to visualize
    when we only cares the resonance frequency and the quality factor.

    Parameters
    ----------
    S21 : `astropy.nddata.NDData`, `astropy.units.Quantity`
        The S21, in (or assumed to be in) ADU.
    frequency : `astropy.units.Quantity`
        The frequency.
    D21 : `astropy.units.Quantity`
        The D21 spectrum.
    extra_attrs : dict, optional
        A dict of extra data to attach to this sweep. They has to be
        of the same shape as `frequency`.
    kwargs :
        keyword arguments to pass to `astropy.nddata.NDDataRef`.
    """

    _extra_attrs_to_slice = None

    def _slice_extra(self, item):
        # this returns extra sliced attributes when slicing
        result = {
            "_frequency": self._frequency[item],
            "_D21": self._D21[item],
        }
        if self._extra_attrs_to_slice is not None:
            for a in self._extra_attrs_to_slice:
                result[a] = getattr(self, a)[item]
        return result

    def __init__(  # noqa: C901, PLR0912
        self,
        S21=None,
        frequency=None,
        D21=None,
        extra_attrs=None,
        **kwargs,
    ):
        if "data" in kwargs:
            # In cases of slicing, new objects will be initialized with `data`
            # instead of ``S21``. Ensure we grab the `data` argument.
            if S21 is None:
                super().__init__(**kwargs)
                # additional attributes frequency and d21 will be added
                # by the _slice_extra call at the end of the slice
                # automatically
                return
            raise ValueError("data should not be specified.")

        # this is for normal construction
        # we expect S21 and frequency to be always set
        # and d21 is optional
        # check dimensions
        if S21 is not None:
            if frequency is None:
                raise ValueError("S21 requires frequency.")
            if frequency.ndim > 0 and frequency.shape[-1] != S21.shape[-1]:
                raise ValueError("shape of frequency does not match S21.")
            if (
                D21 is not None
                and frequency.ndim > 0
                and frequency.shape[-1] != D21.shape[-1]
            ):
                raise ValueError("shape of frequency does not match D21.")

        if frequency is not None:
            frequency = self._validate_frequency(frequency)
            self._frequency = frequency

        if D21 is not None:
            D21 = self._validate_D21(D21)
            self._D21 = D21

        if S21 is not None:
            S21 = self._validate_S21(S21)
            kwargs["data"] = S21.data
            kwargs["unit"] = S21.unit
        elif D21 is not None:
            kwargs["data"] = D21.data
            kwargs["unit"] = D21.unit

        super().__init__(**kwargs)

        # handle extra attrs
        if extra_attrs is not None:
            # add extra data objects
            if any(hasattr(self, a) for a in extra_attrs):
                raise ValueError("name of extra_attrs conflicts with existing attts.")
            extra_attrs_to_slice = []
            for k, v in extra_attrs.items():
                if v.shape != self.data.shape:
                    raise ValueError("invalid shape of extra attr")
                setattr(self, k, v)
                extra_attrs_to_slice.append(k)
            self._extra_attrs_to_slice = extra_attrs_to_slice


class _MultiSweepDataRef(FrequencyDivisionMultiplexingDataRef):
    """A generic container for FDM frequency sweep data.

    The FDM sweep data are generated by reading around multiple reference
    frequencies ``f_chans`` in the frequency space, with the sweep accomplished
    via the change of the local oscillator frequency at each sweep step
    (referred to as ``sweeps``).

    This class sets the generic structure of the data but the actual
    implementation is in the subclass `MultiSweep`.

    Parameters
    ----------
    f_chans : `astropy.units.Quantity`
        The channel reference frequencies. Typically the center of the sweep.
    f_sweeps : `astropy.units.Quantity`
        The sweep frequencies as offsets with respect to the ``f_chans``.
    frequency : `astropy.units.Quantity`
        The multiplexed frequency grid, built from ``f_chans`` and ``f_sweeps``.
    data : `astropy.nddata.NDData`
        The data.
    kwargs :
        keyword arguments to pass to `astropy.nddata.NDDataRef`.
    """

    _f_sweeps: Any

    def __init__(
        self,
        f_chans=None,
        f_sweeps=None,
        frequency=None,
        data=None,
        **kwargs,
    ):
        if frequency is None:
            if sum([f_chans is None, f_sweeps is None]) == 1:
                raise ValueError("must specify both f_chans and f_sweeps.")
            # create the frequency grid
            if f_chans is not None and f_sweeps is not None:
                frequency = self._make_frequency_grid(f_chans, f_sweeps)
        else:
            frequency = SweepMixin._validate_frequency(frequency)  # noqa: SLF001

        # check lengths of axis
        if (
            data is not None
            and frequency is not None
            and data.shape != frequency.data.shape
        ):
            raise ValueError(
                (
                    f"data of shape {data.shape} does not match the shape of"
                    f" the frequency grid {frequency.data.shape}."
                ),
            )

        super().__init__(f_chans=f_chans, data=data, **kwargs)
        self._f_sweeps = f_sweeps
        self._frequency = frequency

    @staticmethod
    def _make_frequency_grid(f_chans, f_sweeps):
        """Return the frequency array from the `f_chans` and `f_sweeps`."""
        data = f_sweeps[None, :] + f_chans[:, None]
        return NDDataRef(data=data.value, unit=data.unit, uncertainty=None, meta=None)

    @property
    def f_sweeps(self):
        """The sweep frequencies."""
        return self._f_sweeps


class MultiSweep(_MultiSweepDataRef, SweepMixin):
    """A container class for multiplexed frequency sweep data.

    This class is different from the `Sweep` in that the S21 data is measured
    from a multiplexed readout system and is organized in a 2-d array where the
    first axis is for the channels, and the second axis is for the sweep steps.

    Parameters
    ----------
    S21 : `astropy.nddata.NDData`, `astropy.units.Quantity`
        The S21 data, in (or assumed to be in) ADU.
    f_chans : `astropy.units.Quantity`
        The channel reference frequencies.
    sweeps : `astropy.units.Quantity`
        The sweep frequencies.
    frequency : `astropy.units.Quantity`
        The frequency grid.
    kwargs :
        keyword arguments to pass to `astropy.nddata.NDDataRef`.
    """

    def _slice_extra(self, item):
        # this returns extra sliced attributes when slicing
        result = {
            "_frequency": self._frequency[item],
        }
        chan_slice = None
        sweep_slice = None
        if not isinstance(item, tuple):
            # normalize to a tuple
            item = (item,)
        if len(item) == 1:
            (chan_slice,) = item
        elif len(item) == 2:  # noqa: PLR2004
            chan_slice, sweep_slice = item
        else:
            raise ValueError("too many slices.")
        if chan_slice is not None:
            result["_f_chans"] = self._f_chans[chan_slice]
        else:
            result["_f_chans"] = self._f_chans
        if sweep_slice is not None:
            result["_f_sweeps"] = self._f_sweeps[sweep_slice]
        else:
            result["_f_sweeps"] = self._f_sweeps
        return result

    def __init__(self, S21=None, f_chans=None, f_sweeps=None, frequency=None, **kwargs):
        if "data" in kwargs:
            # In cases of slicing, new objects will be initialized with `data`
            # instead of ``S21``. Ensure we grab the `data` argument.
            if S21 is None:
                super().__init__(**kwargs)
                # additional attributes frequency and d21 will be added
                # by the _slice_extra call at the end of the slice
                # automatically
                return
            raise ValueError("data should not be specified.")

        # this is for normal construction
        if S21 is not None:
            S21 = self._validate_S21(S21)
            kwargs["data"] = S21.value
            kwargs["unit"] = S21.unit

        super().__init__(
            f_chans=f_chans,
            f_sweeps=f_sweeps,
            frequency=frequency,
            **kwargs,
        )

    @property
    def unified(self):
        """The associated unified D21 `Sweep` object."""
        return self._unified

    def set_unified(self, sweep):
        """Set the associated unified D21 `Sweep` object.

        Parameters
        ----------
        sweep : `Sweep`
            The `Sweep` object that contains the channel merged data.
        """
        self._unified = sweep

    def make_unified(self, cached=True, *args, **kwargs):
        """Create unified sweep with D21 spectrum.

        Parameters
        ----------
        cached : bool, optional
            If True and D21 exists, it is returned.
        args, kwargs :
            The argument passed to `_make_unified`.
        """
        if not (cached and hasattr(self, "_unified")):
            self.set_unified(self._make_unified(*args, **kwargs))
        return self.unified

    @timeit
    def _make_unified(  # noqa: PLR0913
        self,
        flim=None,
        fstep=None,
        resample=None,
        exclude_edge_samples=5,
        smooth=5,
        method="savgol",
    ):
        """Compute the unified ``D21`` spectrum.

        Parameters
        ----------
        flim: tuple, optional
            If set, the d21 is resampled on to this frequency range.

        fstep: float, optional
            The step size of the frequency grid in Hz.

        resample: float, optional
            Alternative way to specify `fstep`. The frequency step
            used will be smaller than the input data by this factor.

        exclude_edge_samples: int
            Number of samples to exclude at the chan edges.

        smooth: int
            Apply smooth to the IQs for D21 computation.
        method: 'savgol' or 'gradient'
            The method for D21 compuatation.
        """
        if fstep is None and resample is None:
            fstep = u.Quantity(1000.0, u.Hz)
        if sum([fstep is not None, resample is not None]) != 1:
            raise ValueError("only one of fstep or resample can be specified")

        fs = self.frequency
        if flim is None:
            flim = (fs.min(), fs.max())
        fmin, fmax = flim
        if resample is not None:
            fstep = (fs[0, 1] - fs[0, 0]) / resample

        logger.debug(
            (
                f"build D21 with fs=[{fmin}, {fmax}, {fstep}]"
                f" exclude_edge_samples={exclude_edge_samples}"
                f" original fs=[{fs.min()}, {fs.max()}] smooth={smooth} method={method}"
            ),
        )
        fs = (
            np.arange(
                fmin.to_value(u.Hz),
                fmax.to_value(u.Hz),
                cast(u.Quantity, fstep).to_value(u.Hz),
            )
            << u.Hz
        )
        adiqs0 = np.abs(
            self.diqs_df(
                self.S21,
                self.frequency,
                smooth=smooth,
                method=method,
            ),
        )
        adiqs = np.zeros(fs.shape, dtype=np.double) << u.adu / u.Hz  # type: ignore
        adiqscov = np.zeros(fs.shape, dtype=int)
        if exclude_edge_samples > 0:
            if 2 * exclude_edge_samples > fs.shape[-1]:
                raise ValueError("insufficient number of data points")
            es = slice(exclude_edge_samples, -exclude_edge_samples)
        else:
            es = slice(None)

        for i in range(self.frequency.shape[0]):
            m = (fs >= self.frequency[i].min()) & (fs <= self.frequency[i].max())
            tmp = np.interp(
                fs[m],
                self.frequency[i, es],
                adiqs0[i, es],
                left=np.nan,
                right=np.nan,
            )
            cov = ~np.isnan(tmp)
            tmp[~cov] = 0
            tmp[cov] += adiqs[m][cov]
            adiqs[m] += tmp
            adiqscov[m] += cov.astype(dtype=int)
        m = adiqscov > 0
        adiqs[m] /= adiqscov[m]
        adiqs[np.isnan(adiqs)] = 0
        return Sweep(
            S21=None,
            D21=adiqs,
            frequency=fs,
            extra_attrs={
                "d21_cov": adiqscov,
            },
        )

    @staticmethod
    def diqs_df(iqs, fs, smooth=5, method="savgol"):
        """Return the dirrivative of S21 with respect to the frequency."""
        if smooth in (None, 0) and method != "gradient":
            raise ValueError("no-smooth only works for gradient")
        diqs = np.full(
            iqs.shape,
            np.nan,
            dtype=iqs.dtype,
        ) << (
            u.adu / u.Hz
        )  # type: ignore
        if method == "gradient":
            if smooth is not None and smooth > 0:
                iqs = _csmooth(iqs, size=smooth, mode="mirror") << u.adu
            for i in range(iqs.shape[0]):
                diqs[i] = np.gradient(iqs[i], fs[i])
        elif method == "savgol":
            from scipy.signal import savgol_filter

            for i in range(iqs.shape[0]):
                deltaf = (fs[i][1] - fs[i][0]).to_value(u.Hz)
                xx = savgol_filter(
                    iqs[i].real.to_value(u.adu),
                    window_length=smooth,
                    polyorder=2,
                    deriv=1,
                    delta=deltaf,
                )
                yy = savgol_filter(
                    iqs[i].imag.to_value(u.adu),
                    window_length=smooth,
                    polyorder=2,
                    deriv=1,
                    delta=deltaf,
                )
                diqs[i] = make_complex(xx, yy) << (u.adu / u.Hz)  # type: ignore
        return diqs

    def get_sweep(self, chan_id, **kwargs):
        """Return a `Sweep` object for a single channel."""
        s = slice(chan_id, chan_id + 1)
        return Sweep(
            frequency=self.frequency[chan_id],
            S21=self.S21[chan_id],
            D21=np.abs(self.diqs_df(self.S21[s], self.frequency[s], **kwargs)[0]),
        )

    def __str__(self):
        shape = "(empty)" if self.data is None else self.data.shape
        return f"{self.__class__.__name__}{shape}"


def _csmooth(arr, *args, **kwargs):
    arr_r = uniform_filter1d(arr.real, *args, **kwargs)
    arr_i = uniform_filter1d(arr.imag, *args, **kwargs)
    return make_complex(arr_r, arr_i)
