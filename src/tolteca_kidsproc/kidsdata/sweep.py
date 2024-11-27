from functools import cached_property
from typing import ClassVar, Generic, Literal, TypeVar

import astropy.units as u
import numpy as np
from pydantic import Field, model_validator
from scipy.ndimage import uniform_filter1d
from tollan.config.types import FrequencyQuantityField, ImmutableBaseModel
from tollan.utils.log import logger, timeit
from tollan.utils.np import make_complex

from .utils import (
    ExtendedNDDataRef,
    FrequencyDivisionMultiplexingMixin,
    FrequencyQuantityType,
    NDDataProtocol,
    validate_quantity,
)

__all__ = [
    "FrequencySweep",
    "MultiFrequencySweep",
    "Sweep",
    "MultiSweep",
]


QT = TypeVar("QT", bound=u.Quantity)
S21QuantityType = u.Quantity["dimensionless"]
D21QuantityType = u.Quantity["time"]


class FrequencySweepMixin:
    """A mixin class for general frequency sweep data."""

    _slice_attrs_default = (("_frequency", lambda s: s[1:] if len(s) > 1 else s),)
    _frequency: FrequencyQuantityType

    @property
    def frequency(self) -> FrequencyQuantityType:
        """The frequency."""
        return self._frequency

    @classmethod
    def _prepare_init_args(cls, data, frequency, **kwargs):
        nd = frequency.ndim
        if data.shape[-nd:] != frequency.shape:
            raise ValueError(
                f"data of shape {data.shape} is incompatible "
                f"with the frequency {frequency.shape}",
            )
        frequency = validate_quantity(
            frequency,
            "frequency",
            physical_type=FrequencyQuantityType,
        )
        return ExtendedNDDataRef.InitArgs(
            data=data,
            kwargs=kwargs,
            attrs={"_frequency": frequency},
            meta={},
        )


class MultiFrequencySweepMixin(
    FrequencyDivisionMultiplexingMixin,
    FrequencySweepMixin,
):
    """A mixin class for FDM frequency sweep data."""

    _slice_attrs_default = (
        "_frequency",
        ("_f_chans", lambda s: s[0]),
        ("_f_sweep", lambda s: s[1] if len(s) > 1 else None),
    )
    _f_sweep: FrequencyQuantityType

    @staticmethod
    def make_frequency_grid(f_chans, f_sweep):
        """Return the frequency array from the `f_chans` and `f_sweep`."""
        return f_sweep[np.newaxis, :] + f_chans[:, np.newaxis]

    @staticmethod
    def decomp_frequency_grid(
        frequency,
        f_chans=None,
        f_sweep=None,
        f_chans_func=np.median,
    ):
        """Decompose frequency grid to ``f_chans`` and ``f_sweep``.

        if both ``f_chans`` and ``f_sweep`` are not provided, ``f_chans_func`` will be
        used to compute the f_chans from frequency grid.
        """
        if f_chans is None and f_sweep is None:
            f_chans = f_chans_func(frequency, axis=1)
        if f_sweep is None:
            f_sweep = np.unique(frequency - f_chans[:, np.newaxis], axis=0)
            if f_sweep.shape[0] != 1:
                raise ValueError("frequency is not a regular multisweep grid.")
            f_sweep = f_sweep[0]
        if f_chans is None:
            f_chans = np.unique(frequency - f_sweep[np.newaxis, :], axis=1)
            if f_sweep.shape[1] != 1:
                raise ValueError("frequency is not a regular multisweep grid.")
            f_chans = f_chans[:, 0]
        return f_chans, f_sweep

    @property
    def f_sweep(self):
        """The sweep frequency steps."""
        return self._f_sweep

    @property
    def n_steps(self):
        """The number of sweep steps."""
        return self.f_sweep.shape[0]

    @classmethod
    def _prepare_init_args(  # noqa: C901
        cls,
        data,
        f_chans=None,
        f_sweep=None,
        frequency=None,
        **kwargs,
    ):
        if f_chans is None and f_sweep is None and frequency is None:
            raise ValueError("two of f_chans, f_sweep, frequency is required.")
        # check dimensions
        if f_chans is not None:
            f_chans = (
                FrequencyDivisionMultiplexingMixin._prepare_init_args(  # noqa: SLF001
                    data,
                    f_chans,
                ).attrs["_f_chans"]
            )
        if f_sweep is not None:
            if f_sweep.ndim != 1:
                raise ValueError(
                    f"Only 1-d f_sweep is supported, got {f_sweep.shape}.",
                )
            f_sweep = validate_quantity(
                f_sweep,
                "f_sweep",
                physical_type=FrequencyQuantityType,
            )
        if frequency is not None:
            if frequency.ndim != 2:  # noqa: PLR2004
                raise ValueError(
                    f"Only 2-d frequency is supported, got {frequency.shape}.",
                )
            if data.shape != frequency.shape:
                raise ValueError(
                    f"data of shape {data.shape} is incompatible "
                    f"with the frequency {frequency.shape}",
                )
            frequency = validate_quantity(
                frequency,
                "frequency",
                physical_type=FrequencyQuantityType,
            )
        # infer other others if any of the two are provided
        if frequency is None:
            if sum([f_chans is None, f_sweep is None]) == 1:
                raise ValueError("must specify both f_chans and f_sweep.")
            # create the frequency grid
            if f_chans is not None and f_sweep is not None:
                frequency = cls.make_frequency_grid(f_chans, f_sweep)
        if frequency is not None and sum([f_chans is None, f_sweep is None]) > 0:
            # infer f_chans and f_sweep if any of them is none
            f_chans, f_sweep = cls.decomp_frequency_grid(
                frequency,
                f_chans=f_chans,
                f_sweep=f_sweep,
            )
        return ExtendedNDDataRef.InitArgs(
            data=data,
            kwargs=kwargs,
            attrs={
                "_frequency": frequency,
                "_f_chans": f_chans,
                "_f_sweep": f_sweep,
            },
            meta={},
        )


class FrequencySweep(FrequencySweepMixin, ExtendedNDDataRef, Generic[QT]):
    """A generic container for frequency sweep data.

    Parameters
    ----------
    frequency : `astropy.units.Quantity`
        The frequency vector.
    kwargs :
        keyword arguments to pass to `astropy.nddata.NDDataRef`.
    """


class MultiFrequencySweep(
    MultiFrequencySweepMixin,
    ExtendedNDDataRef,
    Generic[QT],
):
    """A generic container for FDM frequency sweep data.

    The FDM sweep data are generated by reading around multiple reference
    frequencies ``f_chans`` in the frequency space, with the sweep accomplished
    via the change of the local oscillator frequency at each sweep step
    (referred to as ``f_sweep``).

    Parameters
    ----------
    f_chans : `astropy.units.Quantity`
        The channel reference frequencies. Typically the center of the sweep.
    f_sweep : `astropy.units.Quantity`
        The sweep frequencies as offsets with respect to the ``f_chans``.
    frequency : `astropy.units.Quantity`
        The multiplexed frequency grid, built from ``f_chans`` and ``f_sweep``.
    kwargs :
        keyword arguments to pass to `astropy.nddata.NDDataRef`.
    """


class IQMixin(Generic[QT]):
    """A mixin class of frequency sweep data of complex type."""

    @property
    def I(self: NDDataProtocol) -> QT:  # noqa: E743
        """The In-phase data."""
        return self.data.real << self.unit

    @property
    def Q(self: NDDataProtocol) -> QT:
        """The quadrature data."""
        return self.data.imag << self.unit

    @property
    def I_unc(self: NDDataProtocol) -> QT:
        """The In-phase uncertainty."""
        return self.uncertainty.quantity.real

    @property
    def Q_unc(self: NDDataProtocol) -> QT:
        """The quadrature uncertainty."""
        return self.uncertainty.quantity.imag


class S21Mixin:

    _slice_attrs_default = ("aS21", "aS21_unc", "aS21_db", "aS21_unc_db")

    @property
    def S21(self: NDDataProtocol) -> S21QuantityType:
        """The S21."""
        return self.quantity

    @property
    def S21_unc(self: NDDataProtocol) -> S21QuantityType:
        """The S21 uncertainty."""
        return self.uncertainty.quantity

    @cached_property
    def aS21(self) -> S21QuantityType:
        """The absolute value of S21."""
        return self.calc_aS21(self.S21)

    @cached_property
    def aS21_unc(self) -> S21QuantityType:
        """The absolute value of S21 uncertainty."""
        return self.calc_aS21(self.S21_unc)

    @cached_property
    def aS21_db(self):
        """The absolute value of S21 in a dB."""
        return self.calc_aS21_db(self.S21)

    @cached_property
    def aS21_unc_db(self):
        """The S21 unc in db."""
        return self.calc_aS21_unc_db(self.S21_unc, self.S21)

    @staticmethod
    def calc_aS21(S21: S21QuantityType) -> S21QuantityType:
        return np.abs(S21)

    @classmethod
    def calc_db(cls, aS21: S21QuantityType) -> S21QuantityType:
        return 20 * np.log10(aS21.value)

    @classmethod
    def calc_aS21_db(cls, S21: S21QuantityType):
        return cls.calc_db(cls.calc_aS21(S21))

    @staticmethod
    def calc_aS21_unc_db(S21_unc: S21QuantityType, S21: S21QuantityType):
        return 20 * np.log10(np.e) * np.abs(S21_unc / S21).value


class Sweep(
    S21Mixin,
    IQMixin[S21QuantityType],
    FrequencySweep[S21QuantityType],
):
    """A container class for S21-frequency sweep data.

    The response of the sweep is expressed as the ``S21`` parameter of the
    2-ports readout circuit as a function of the probing frequency.

    `S21` could also be a n-dim array for `Sweep`, in which case the
    data is for a collection of sweeps that share the same frequency grid.
    The data shall be organized such that the frequency varies along the last
    dimension. Note that this is different from `MultiSweep` in that
    for `MultiSweep`, each sweep has its own frequency grid.

    Parameters
    ----------
    S21 : `astropy.units.Quantity`
        The S21.
    frequency : `astropy.units.Quantity`
        The frequency.
    **kwargs :
        keyword arguments to pass to `astropy.nddata.NDDataRef`.
    """

    _slice_attrs_default = (
        S21Mixin._slice_attrs_default  # noqa: SLF001
        + FrequencySweep._slice_attrs_default  # noqa: SLF001
    )

    def __init__(
        self,
        S21=None,
        frequency=None,
        **kwargs,
    ):
        kwargs.setdefault("data", None)
        super().__init__(S21=S21, frequency=frequency, **kwargs)

    @classmethod
    def _prepare_init_args(cls, data, S21, frequency, **kwargs):
        if data is not None:
            raise ValueError("data should be passed with S21 argument.")
        if S21 is None:
            raise ValueError("S21 is required.")
        if frequency is None:
            raise ValueError("frequency is required.")
        if not isinstance(S21, u.Quantity):
            S21 = S21 << u.dimensionless_unscaled
        S21 = validate_quantity(
            S21,
            data_label="S21",
            physical_type=S21QuantityType,
        )
        return super()._prepare_init_args(data=S21, frequency=frequency, **kwargs)


class MultiSweep(
    S21Mixin,
    IQMixin[S21QuantityType],
    MultiFrequencySweep[S21QuantityType],
):
    """A container class for multiplexed frequency sweep data.

    This class is different from the `Sweep` in that the S21 data is measured
    from a multiplexed readout system and is organized in a 2-d array where the
    first axis is for the channels, and the second axis is for the sweep steps.

    Parameters
    ----------
    S21 : `astropy.units.Quantity`
        The S21 data.
    f_chans : `astropy.units.Quantity`
        The channel reference frequencies.
    f_sweep : `astropy.units.Quantity`
        The sweep frequencies.
    frequency : `astropy.units.Quantity`
        The frequency grid.
    **kwargs :
        keyword arguments to pass to `astropy.nddata.NDDataRef`.
    """

    _slice_attrs_default = (
        S21Mixin._slice_attrs_default  # noqa: SLF001
        + MultiFrequencySweep._slice_attrs_default  # noqa: SLF001
    )

    def __init__(
        self,
        S21=None,
        f_chans=None,
        f_sweep=None,
        frequency=None,
        **kwargs,
    ):
        kwargs.setdefault("data", None)
        super().__init__(
            S21=S21,
            f_chans=f_chans,
            f_sweep=f_sweep,
            frequency=frequency,
            **kwargs,
        )

    @classmethod
    def _prepare_init_args(
        cls,
        data,
        S21,
        f_chans,
        f_sweep,
        frequency,
        **kwargs,
    ):
        if data is not None:
            raise ValueError("data should be passed with S21 argument.")
        if S21 is None:
            raise ValueError("S21 is required.")
        if not isinstance(S21, u.Quantity):
            S21 = S21 << u.dimensionless_unscaled
        S21 = validate_quantity(
            S21,
            data_label="S21",
            physical_type=S21QuantityType,
        )
        return super()._prepare_init_args(
            data=S21,
            f_chans=f_chans,
            f_sweep=f_sweep,
            frequency=frequency,
            **kwargs,
        )


D21SmoothMethod = Literal["savgol", "gradient"]


class D21Analysis(ImmutableBaseModel):
    """Calcualte D21 for sweep data."""

    f_lims: None | tuple[FrequencyQuantityField, FrequencyQuantityField] = Field(
        default=None,
        description="If set, the D21 is resampled to this frequency range.",
    )
    f_step: None | FrequencyQuantityField = Field(
        default=None,
        description="The step size of the frequency grid.",
    )
    resample: int = Field(
        default=1,
        description=(
            "Infer f_step by sampling the original by "
            "this factor, if f_step is not set."
        ),
    )
    exclude_edge: None | int = Field(
        default=None,
        description="Number of samples to exclude at the channel edge.",
    )
    smooth: None | int = Field(
        default=5,
        description="Number of samples used in smoothing the data.",
    )
    method: D21SmoothMethod = Field(
        default="savgol",
        description="The smoothing method",
    )

    _d21_matched_meta_key: ClassVar[str] = "d21"
    _d21_unified_meta_key: ClassVar[str] = "d21_unified"
    _d21_analysis_meta_key: ClassVar[str] = "d21_analysis"
    _d21_coverage_meta_key: ClassVar[str] = "coverage"

    @model_validator(mode="before")
    @classmethod
    def validate_smooth_and_edge(cls, data):
        def _get(k):
            return data.get(k, cls.model_fields[k].default)

        exclude_edge = _get("exclude_edge")
        smooth = _get("smooth")
        if exclude_edge is None:
            exclude_edge = smooth
        # make sure they are non-negative
        smooth = smooth or 0
        if smooth < 0:
            raise ValueError("smooth cannot be negative.")
        exclude_edge = exclude_edge or 0
        if exclude_edge < 0:
            raise ValueError("exclude_edge cannot be negative.")
        method: D21SmoothMethod = _get("method")
        if smooth == 0 and method != "gradient":
            raise ValueError("no-smooth only works for gradient")
        data["smooth"] = smooth
        data["exclude_edge"] = exclude_edge
        return data

    @timeit
    def make_unified(self, swp: Sweep | MultiSweep):
        """Return unified abs(D21) sweep from sweep."""
        fs = swp.frequency
        if fs.ndim == 1:
            # wrap the data as a multi sweep like
            fs = fs[np.newaxis, :]

        f_lims = self.f_lims

        if f_lims is None:
            f_lims = (fs.min(), fs.max())
        f_min, f_max = f_lims

        f_step = self.f_step
        if f_step is None:
            f_step = (fs[0, 1] - fs[0, 0]) / self.resample

        # validate edge
        exclude_edge = self.exclude_edge

        if exclude_edge > 0:
            if 2 * exclude_edge > fs.shape[-1]:
                raise ValueError(
                    f"insufficient number of data points for {exclude_edge=}.",
                )
            edge_slice = slice(exclude_edge, -exclude_edge)
        else:
            edge_slice = slice(None, None)

        smooth = self.smooth
        method = self.method
        logger.debug(
            f"build unified D21 with {f_min=} {f_max=} {f_step=}"
            f" {exclude_edge=} {smooth=} {method=} "
            f"from original f_min={fs.min()} f_max={fs.max()}",
        )

        fs_unified = (
            np.arange(
                f_min.to_value(u.Hz),
                f_max.to_value(u.Hz),
                f_step.to_value(u.Hz),
            )
            << u.Hz
        )

        d21_matched = self.make_matched(swp)
        ad21_matched = np.abs(d21_matched.data)
        ad21_unified = np.zeros(fs_unified.shape, dtype=np.double) << (u.Hz**-1)
        ad21_unified_cov = np.zeros(fs_unified.shape, dtype=int)

        chan_min = np.min(fs, axis=1)
        chan_max = np.max(fs, axis=1)
        for i in range(fs.shape[0]):
            m = (fs_unified >= chan_min[i]) & (fs_unified <= chan_max[i])
            tmp = np.interp(
                fs_unified[m],
                fs[i, edge_slice],
                ad21_matched[i, edge_slice],
                left=np.nan,
                right=np.nan,
            ) << (u.Hz**-1)
            cov = ~np.isnan(tmp)
            tmp[~cov] = 0
            ad21_unified[m] += tmp
            ad21_unified_cov[m] += cov.astype(dtype=int)
        # normalize by cov and clean up
        m = ad21_unified_cov > 0
        ad21_unified[m] /= ad21_unified_cov[m]
        ad21_unified[np.isnan(ad21_unified)] = 0
        ad21_unified = FrequencySweep(
            data=ad21_unified,
            frequency=fs_unified,
            meta={
                self._d21_coverage_meta_key: ad21_unified_cov,
                self._d21_analysis_meta_key: self.model_dump(),
            },
            slice_meta_keys=[self._d21_coverage_meta_key],
        )
        swp.meta[self._d21_unified_meta_key] = ad21_unified
        return ad21_unified

    @timeit
    def make_matched(self, swp: Sweep | MultiSweep):
        """Return complex D21 sweep from sweep with matched frequency grid."""
        fs = swp.frequency.to_value(u.Hz)
        s21 = swp.S21.value
        method = self.method
        smooth = self.smooth
        if method == "gradient":
            if smooth is not None and smooth > 0:
                s21_smooth = uniform_filter1d(s21, size=smooth, mode="mirror", axis=-1)
            else:
                s21_smooth = s21
            d21 = np.gradient(s21_smooth, fs, axis=-1) << (u.Hz**-1)
        elif method == "savgol":
            from scipy.signal import savgol_filter

            deltaf = fs[0][1] - fs[0][0]
            d21_real = savgol_filter(
                s21.real,
                window_length=smooth,
                polyorder=2,
                deriv=1,
                delta=deltaf,
                axis=-1,
            )
            d21_imag = savgol_filter(
                s21.imag,
                window_length=smooth,
                polyorder=2,
                deriv=1,
                delta=deltaf,
                axis=-1,
            )
            d21 = make_complex(d21_real, d21_imag) << (u.Hz**-1)
        swp.add_slicable_meta(self._d21_matched_meta_key, d21)
        swp.meta.update({self._d21_analysis_meta_key: self.model_dump()})
        return d21
