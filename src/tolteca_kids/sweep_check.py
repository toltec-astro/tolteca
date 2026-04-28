"""TolTEC KIDs sweep quality check pipeline step.

Adapts the v2 ``tolteca_kids`` ``SweepCheck`` step for the v3
``xr.DataTree``-based data model.  Input is an ``xr.DataTree`` produced by
:class:`~tolteca_datamodels.toltec.kids.SweepReducer`; data is accessed via
:class:`~tolteca_datamodels.toltec.kids.ReducedSweepView`.

Context is stored in ``dt.attrs["__pipeline_context__"]`` via the
:class:`~tollan.pipeline.Step` base class and retrieved with::

    ctx = SweepCheck.get_context(dt)   # → SweepCheckContext (typed)
    ctx.data.bitmask_chan              # per-channel bitmask array
    ctx.data.mask_chan_bad             # bool bad-channel mask
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag, auto
from typing import TYPE_CHECKING, Literal

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats
import xarray as xr
from astropy.stats import mad_std
from numpy.lib.stride_tricks import sliding_window_view
from pydantic import ConfigDict, Field
from scipy.ndimage import binary_erosion, median_filter
from tollan.config.types import FrequencyQuantityField
from tollan.pipeline import Step, StepConfig, StepContext
from tollan.utils.fmt import BitmaskStats, pformat_mask
from tollan.utils.log import logger, timeit

from tolteca_datamodels.toltec.kids import ReducedSweepView
from tolteca_kidsproc.analysis.d21 import D21Analysis

__all__ = [
    "DespikeMethod",
    "SweepBitMask",
    "SweepCheck",
    "SweepCheckConfig",
    "SweepCheckContext",
    "SweepCheckData",
]


DespikeMethod = Literal["interp_linear"]


class SweepBitMask(IntFlag):
    """Quality bitmask for TolTEC sweep data."""

    range_small = auto()
    """Data have small range."""

    range_large = auto()
    """Data have large range."""

    level_low = auto()
    """Data have low level."""

    level_high = auto()
    """Data have high level."""

    rms_low = auto()
    """Data have low RMS."""

    rms_high = auto()
    """Data have high RMS."""

    skew_high = auto()
    """Data have high skewness."""

    kurtosis_high = auto()
    """Data have high kurtosis."""

    spike = auto()
    """Data point identified as spike."""

    baseline = auto()
    """Data point identified as baseline."""

    tone_amp_zero = auto()
    """Tone amplitude is zero."""

    tone_amp_one = auto()
    """Tone amplitude is one."""

    tone_power_low = auto()
    """Tone driving power low."""

    tone_power_high = auto()
    """Tone driving power high."""


class SweepCheckConfig(StepConfig):
    """Configuration for the sweep quality check step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # despike
    despike: bool = Field(
        default=True,
        description="Remove spikes in data.",
    )
    despike_method: DespikeMethod = Field(
        default="interp_linear",
        description="The despike method.",
    )
    spike_height_frac_min: float = Field(
        default=0.1,
        description=(
            "Height threshold used to identify spikes, measured as "
            "fraction of the channel data range."
        ),
    )
    spike_medfilt_size: int = Field(
        default=5,
        description="Size of median filter used to identify spikes.",
    )

    # channel stat thresholds
    chan_range_db_min: float = Field(
        default=0.1,
        description="Channel data range smaller than this is flagged.",
    )
    chan_range_db_max: float = Field(
        default=np.inf,
        description="Channel data range larger than this is flagged.",
    )
    chan_level_db_min: float = Field(
        default=-np.inf,
        description="Channel data level lower than this is flagged.",
    )
    chan_level_db_max: float = Field(
        default=np.inf,
        description="Channel data level higher than this is flagged.",
    )
    chan_rms_db_min: float = Field(
        default=0.001,
        description="Channel data RMS lower than this is flagged.",
    )
    chan_rms_db_max: float = Field(
        default=0.1,
        description="Channel data RMS higher than this is flagged.",
    )

    # chunk stats
    chunk_size: int = Field(
        default=50,
        description="Chunk size for per-chunk statistics.",
    )
    n_chunks_min: int = Field(
        default=5,
        description="Minimum number of chunks per channel.",
    )
    chunk_skew_max: float = Field(
        default=0.2,
        description="Chunk skew higher than this is flagged.",
    )
    chunk_kurtosis_max: float = Field(
        default=-1.0,
        description="Chunk kurtosis higher than this is flagged.",
    )
    bad_chan_bits: SweepBitMask = Field(
        default=(SweepBitMask.rms_low | SweepBitMask.rms_high),
        description="Bits used to generate channel bad mask.",
    )
    not_baseline_chunk_bits: SweepBitMask = Field(
        default=(SweepBitMask.skew_high | SweepBitMask.kurtosis_high),
        description="Bits used to generate chunk baseline flag.",
    )

    # D21 analysis
    d21_analysis: D21Analysis = Field(
        default_factory=lambda: D21Analysis(
            f_step=1000 << u.Hz,
            smooth=5,
            method="savgol",
        ),
        description="D21 analysis parameters.",
    )


@dataclass(kw_only=True)
class SweepCheckData:
    """Output data from the sweep quality check step.

    Stores per-channel and per-chunk statistics, bitmasks, D21 results,
    and optional noise PSD arrays.  Holds arbitrary types (numpy arrays,
    pandas DataFrames, astropy Quantities) and is stored as-is in
    ``dt.attrs["__pipeline_context__"]``.
    """

    # spike
    mask_spike: npt.NDArray = field(default_factory=lambda: np.empty(0))
    S21_orig: npt.NDArray | None = None
    S21_spike: npt.NDArray | None = None

    # chunking metadata
    chunk_size: int = 0
    n_chunks_per_chan: int = 0
    chunk_windows: npt.NDArray = field(default_factory=lambda: np.empty(0, dtype=int))
    chunk_slices: list[slice] = field(default_factory=list)
    f_chunks: npt.NDArray = field(default_factory=lambda: np.empty(0))

    # bitmasks
    bitmask: npt.NDArray = field(default_factory=lambda: np.empty(0, dtype=int))
    bitmask_chan: npt.NDArray = field(default_factory=lambda: np.empty(0, dtype=int))
    bitmask_chunk: npt.NDArray = field(default_factory=lambda: np.empty(0, dtype=int))
    bitmask_chan_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    bitmask_chunk_stats: pd.DataFrame = field(default_factory=pd.DataFrame)

    mask_chan_bad: npt.NDArray = field(default_factory=lambda: np.empty(0, dtype=bool))
    mask_chunk_baseline: npt.NDArray = field(
        default_factory=lambda: np.empty(0, dtype=bool)
    )
    mask_baseline: npt.NDArray = field(
        default_factory=lambda: np.empty(0, dtype=bool)
    )

    # D21 baseline
    d21_mask_baseline: npt.NDArray = field(
        default_factory=lambda: np.empty(0, dtype=bool)
    )
    d21_chunk_size: int = 0
    d21_n_chunks: int = 0
    d21_chunk_windows: npt.NDArray = field(
        default_factory=lambda: np.empty(0, dtype=int)
    )
    d21_chunk_slices: list[slice] = field(default_factory=list)
    d21_f_chunks: u.Quantity = field(default_factory=lambda: np.empty(0) << u.Hz)
    d21_chunk_baseline: u.Quantity = field(
        default_factory=lambda: np.empty(0) << u.Hz**-1
    )
    d21_chunk_baseline_rms: u.Quantity = field(
        default_factory=lambda: np.empty(0) << u.Hz**-1
    )
    d21_baseline: u.Quantity = field(
        default_factory=lambda: np.empty(0) << u.Hz**-1
    )
    d21_baseline_rms: u.Quantity = field(
        default_factory=lambda: np.empty(0) << u.Hz**-1
    )
    d21_frequency: u.Quantity = field(
        default_factory=lambda: np.empty(0) << u.Hz
    )
    d21: u.Quantity = field(default_factory=lambda: np.empty(0) << u.Hz**-1)
    d21_detrended: u.Quantity = field(
        default_factory=lambda: np.empty(0) << u.Hz**-1
    )

    # per-channel stats
    chan_range: npt.NDArray = field(default_factory=lambda: np.empty(0))
    chan_level: npt.NDArray = field(default_factory=lambda: np.empty(0))
    chan_rms_mean: npt.NDArray = field(default_factory=lambda: np.empty(0))
    chan_rms_std: npt.NDArray = field(default_factory=lambda: np.empty(0))

    chunk_rms_mean: npt.NDArray = field(default_factory=lambda: np.empty(0))
    chunk_rms_std: npt.NDArray = field(default_factory=lambda: np.empty(0))
    chunk_skew: npt.NDArray = field(default_factory=lambda: np.empty(0))
    chunk_kurtosis: npt.NDArray = field(default_factory=lambda: np.empty(0))

    swp_rms_med: float = 0.0
    swp_rms_rms: float = 0.0


class SweepCheckContext(StepContext["SweepCheck", SweepCheckConfig]):
    """Context for the sweep quality check step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: SweepCheckData = Field(default_factory=SweepCheckData)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _SweepArrays:
    """Numpy arrays extracted from a ReducedSweepView for the algorithm."""

    S21: npt.NDArray  # complex [n_chans, n_steps]
    aS21: npt.NDArray  # [n_chans, n_steps]
    aS21_db: npt.NDArray  # [n_chans, n_steps]
    aS21_unc_db: npt.NDArray  # [n_chans, n_steps] — uncertainty in dB
    aS21_unc: npt.NDArray  # [n_chans, n_steps] — uncertainty in linear amplitude
    frequency: u.Quantity  # [n_chans, n_steps] in Hz
    n_chans: int
    n_steps: int


def _extract_sweep_arrays(dt: xr.DataTree) -> _SweepArrays:
    """Extract numpy arrays from reduced sweep DataTree for SweepCheck.

    Parameters
    ----------
    dt : xr.DataTree
        DataTree produced by :class:`~tolteca_datamodels.toltec.kids.SweepReducer`.

    Returns
    -------
    _SweepArrays
        Struct of numpy/Quantity arrays for the SweepCheck algorithm.
    """
    view = ReducedSweepView(dt)

    I_da = view.I
    Q_da = view.Q
    if I_da is None or Q_da is None:
        raise ValueError("ReducedSweepView has no I/Q data.")

    I_np = I_da.values  # [chan, sweep]
    Q_np = Q_da.values  # [chan, sweep]
    S21_np = I_np + 1j * Q_np
    aS21_np = np.abs(S21_np)
    safe_aS21 = np.where(aS21_np > 0, aS21_np, np.finfo(float).tiny)
    aS21_db = 20.0 * np.log10(safe_aS21)

    # Error propagation: sigma_aS21_db = (20/ln10) * sigma_|S21| / |S21|
    unc_I_da = view.unc_I
    unc_Q_da = view.unc_Q
    if unc_I_da is not None and unc_Q_da is not None:
        unc_I_np = unc_I_da.values
        unc_Q_np = unc_Q_da.values
        # sigma_|S21| = sqrt((I*sigma_I)^2 + (Q*sigma_Q)^2) / |S21|
        sigma_abs = np.sqrt((I_np * unc_I_np) ** 2 + (Q_np * unc_Q_np) ** 2)
        sigma_abs /= safe_aS21  # linear amplitude uncertainty = sigma_|S21|
        aS21_unc_db = (20.0 / np.log(10.0)) * sigma_abs / safe_aS21
    else:
        sigma_abs = np.zeros_like(aS21_db)
        aS21_unc_db = np.zeros_like(aS21_db)

    # Absolute frequency: f_lo[chan] + sweep[step]  →  [chan, sweep]
    f_lo_da = view.f_lo
    sweep_da = view.sweep
    n_chans, n_steps = I_np.shape
    if f_lo_da is not None and sweep_da is not None:
        f_lo_np = f_lo_da.values  # [chan]
        sweep_np = sweep_da.values  # [sweep]
        # Both should be in Hz; broadcast to [chan, sweep]
        frequency = (f_lo_np[:, np.newaxis] + sweep_np[np.newaxis, :]) << u.Hz
    else:
        # Fallback: zero-based frequency
        frequency = np.zeros((n_chans, n_steps)) << u.Hz

    return _SweepArrays(
        S21=S21_np,
        aS21=aS21_np,
        aS21_db=aS21_db,
        aS21_unc_db=aS21_unc_db,
        aS21_unc=sigma_abs,
        frequency=frequency,
        n_chans=n_chans,
        n_steps=n_steps,
    )


@dataclass
class _D21SweepAdapter:
    """Minimal adapter making numpy arrays compatible with ``D21Analysis``.

    ``D21Analysis.make_matched`` / ``make_unified`` call ``sweep_view.S21.values``
    and ``sweep_view.frequency.u.to("Hz").values`` / ``sweep_view.frequency.u.quantity``.
    This adapter provides exactly those attributes as ``xr.DataArray`` objects
    with units stored in ``.attrs["units"]``.
    """

    S21: xr.DataArray
    frequency: xr.DataArray


def _make_d21_adapter(arrays: _SweepArrays) -> _D21SweepAdapter:
    """Build a D21Analysis-compatible adapter from sweep arrays."""
    s21_da = xr.DataArray(arrays.S21, dims=["chan", "sweep"])
    freq_np = arrays.frequency.to_value(u.Hz)
    freq_da = xr.DataArray(freq_np, dims=["chan", "sweep"], attrs={"units": "Hz"})
    return _D21SweepAdapter(S21=s21_da, frequency=freq_da)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class SweepCheck(Step[SweepCheckConfig, SweepCheckContext]):
    """Sweep quality check step.

    Operates on the reduced sweep data in an ``xr.DataTree``, performing:

    - Spike detection and optional linear interpolation despiking
    - Per-channel and per-chunk quality statistics (range, level, RMS,
      skewness, kurtosis)
    - D21 analysis for baseline estimation and detrending
    - Bitmask composition and summary logging

    Context is stored in ``dt.attrs["__pipeline_context__"]`` and can be
    retrieved with::

        ctx = SweepCheck.get_context(dt)
        ctx.data.mask_chan_bad   # bad-channel boolean array
        ctx.data.d21_detrended  # detrended unified D21

    Parameters
    ----------
    *args
        Passed to :class:`~tollan.pipeline.Step.__init__` (optional
        :class:`SweepCheckConfig` instance or keyword config fields).
    """

    @classmethod
    @timeit
    def run(cls, data: xr.DataTree, context: SweepCheckContext) -> bool:  # noqa: C901, PLR0915
        """Execute the sweep quality check.

        Parameters
        ----------
        data : xr.DataTree
            DataTree with reduced sweep child node (from SweepReducer).
        context : SweepCheckContext
            Step context — results are written to ``context.data``.

        Returns
        -------
        bool
            ``True`` on success.
        """
        cfg: SweepCheckConfig = context.config
        ctd: SweepCheckData = context.data

        # ----------------------------------------------------------------
        # Extract data arrays from the reduced DataTree
        # ----------------------------------------------------------------
        arrays = _extract_sweep_arrays(data)
        S21 = arrays.S21  # [n_chans, n_steps], complex
        aS21_db = arrays.aS21_db  # [n_chans, n_steps]
        aS21_unc_db = arrays.aS21_unc_db  # [n_chans, n_steps]
        frequency = arrays.frequency  # Quantity [n_chans, n_steps] Hz
        n_chans = arrays.n_chans
        n_steps = arrays.n_steps

        # ----------------------------------------------------------------
        # Spike detection
        # ----------------------------------------------------------------
        mask_spike, ctx_spike = cls.find_spike(
            aS21_db,
            medfilt_size=cfg.spike_medfilt_size,
            y_range_min=cfg.chan_range_db_min,
            height_frac_min=cfg.spike_height_frac_min,
        )
        S21_orig = ctd.S21_orig = S21.copy()

        if cfg.despike:
            despike_method = cfg.despike_method
            with timeit(f"despike with method={despike_method}"):
                if despike_method == "interp_linear":
                    S21_despiked = S21.copy()
                    for ci in range(n_chans):
                        m = mask_spike[ci]
                        if m.any():
                            freq_ci = frequency.to_value(u.Hz)[ci]
                            S21_despiked[ci, m] = np.interp(
                                freq_ci[m],
                                freq_ci[~m],
                                S21[ci, ~m],
                            )
                    S21_spike = S21_orig - S21_despiked
                    S21 = S21_despiked
                    # Recompute aS21_db and aS21_unc_db after despiking
                    safe_aS21 = np.where(
                        np.abs(S21) > 0,
                        np.abs(S21),
                        np.finfo(float).tiny,
                    )
                    aS21_db = 20.0 * np.log10(safe_aS21)
                    # unc_db doesn't change meaningfully after despike
        else:
            S21_spike = None

        ctd.mask_spike = mask_spike
        ctd.S21_spike = S21_spike

        # ----------------------------------------------------------------
        # D21 analysis
        # ----------------------------------------------------------------
        d21_adapter = _make_d21_adapter(
            _SweepArrays(
                S21=S21,
                aS21=np.abs(S21),
                aS21_db=aS21_db,
                aS21_unc_db=aS21_unc_db,
                aS21_unc=arrays.aS21_unc,
                frequency=frequency,
                n_chans=n_chans,
                n_steps=n_steps,
            )
        )
        d21_unified = cfg.d21_analysis.make_unified(d21_adapter)
        d21_data = ctd.d21 = d21_unified.u.quantity
        d21_frequency = ctd.d21_frequency = d21_unified.coords[
            d21_unified.dims[0]
        ].u.quantity

        # ----------------------------------------------------------------
        # Per-channel statistics
        # ----------------------------------------------------------------
        chan_range = ctd.chan_range = ctx_spike["y_range"]
        chan_level = ctd.chan_level = np.mean(aS21_db, axis=1)
        y_rms_no_spike = aS21_unc_db.copy()
        y_rms_no_spike[mask_spike] = np.nan
        chan_rms_mean = ctd.chan_rms_mean = np.nanmean(y_rms_no_spike, axis=1)
        ctd.chan_rms_std = np.nanstd(y_rms_no_spike, axis=1)
        ctd.swp_rms_med = float(np.median(chan_rms_mean))
        ctd.swp_rms_rms = float(mad_std(chan_rms_mean))

        # ----------------------------------------------------------------
        # Chunk statistics
        # ----------------------------------------------------------------
        with timeit("calc chunk statistics"):
            ctx_chan_chunks = cls.make_chunks(
                n_items=n_steps,
                chunk_size=cfg.chunk_size,
                n_chunks_min=cfg.n_chunks_min,
            )
            ctd.chunk_size = ctx_chan_chunks["chunk_size"]
            ctd.n_chunks_per_chan = ctx_chan_chunks["n_chunks"]
            chunk_windows = ctd.chunk_windows = ctx_chan_chunks["chunk_windows"]
            chunk_slices = ctd.chunk_slices = ctx_chan_chunks["chunk_slices"]
            ctd.f_chunks = np.median(
                frequency.to_value(u.Hz)[:, chunk_windows],
                axis=-1,
            )

            chunk_rms_mean = ctd.chunk_rms_mean = np.nanmean(
                y_rms_no_spike[:, chunk_windows],
                axis=-1,
            )
            ctd.chunk_rms_std = np.nanstd(
                y_rms_no_spike[:, chunk_windows],
                axis=-1,
            )

            def _complex_stat(comb_func, stat_func, arr, axis=None):
                return comb_func(
                    stat_func(arr.real, axis=axis),
                    stat_func(arr.imag, axis=axis),
                )

            chunk_skew = ctd.chunk_skew = _complex_stat(
                lambda x, y: np.hypot(x, y) / np.sqrt(2),
                scipy.stats.skew,
                S21[:, chunk_windows],
                axis=-1,
            )
            chunk_kurtosis = ctd.chunk_kurtosis = _complex_stat(
                lambda x, y: np.max([x, y], axis=0),
                scipy.stats.kurtosis,
                S21[:, chunk_windows],
                axis=-1,
            )
            mask_chunk_rms_low = chunk_rms_mean < cfg.chan_rms_db_min
            mask_chunk_rms_high = chunk_rms_mean > cfg.chan_rms_db_max
            mask_chunk_skew_high = chunk_skew > cfg.chunk_skew_max
            mask_chunk_kurtosis_high = chunk_kurtosis > cfg.chunk_kurtosis_max

        # ----------------------------------------------------------------
        # Bitmask composition
        # ----------------------------------------------------------------
        # Channel bitmask
        bitmask_chan = ctd.bitmask_chan = (
            (chan_range < cfg.chan_range_db_min) * SweepBitMask.range_small
            | (chan_range > cfg.chan_range_db_max) * SweepBitMask.range_large
            | (chan_level < cfg.chan_level_db_min) * SweepBitMask.level_low
            | (chan_level > cfg.chan_level_db_max) * SweepBitMask.level_high
            | (chan_rms_mean < cfg.chan_rms_db_min) * SweepBitMask.rms_low
            | (chan_rms_mean > cfg.chan_rms_db_max) * SweepBitMask.rms_high
            | (np.sum(mask_spike, axis=-1) > 0) * SweepBitMask.spike
        )
        # Chunk bitmask
        bitmask_chunk = ctd.bitmask_chunk = (
            (
                bitmask_chan[:, np.newaxis]
                & (
                    SweepBitMask.range_small
                    & SweepBitMask.range_large
                    & SweepBitMask.level_low
                    & SweepBitMask.level_high
                    & SweepBitMask.tone_amp_zero
                    & SweepBitMask.tone_amp_one
                    & SweepBitMask.tone_power_low
                    & SweepBitMask.tone_power_high
                )
            )
            | mask_chunk_rms_low * SweepBitMask.rms_low
            | mask_chunk_rms_high * SweepBitMask.rms_high
            | mask_chunk_skew_high * SweepBitMask.skew_high
            | mask_chunk_kurtosis_high * SweepBitMask.kurtosis_high
        )
        for i, s in enumerate(chunk_slices):
            bitmask_chunk[:, i] |= (
                np.sum(mask_spike[:, s], axis=-1) > 0
            ) * SweepBitMask.spike

        # Baseline chunk mask
        mask_chunk_baseline = ctd.mask_chunk_baseline = (
            bitmask_chunk & cfg.not_baseline_chunk_bits == 0
        )
        bitmask_chunk |= mask_chunk_baseline * SweepBitMask.baseline

        # Aggregate chunk flags back to channel bitmask
        bitmask_chan |= (
            np.any(mask_chunk_skew_high, axis=-1) * SweepBitMask.skew_high
            | np.any(mask_chunk_kurtosis_high, axis=-1) * SweepBitMask.kurtosis_high
            | np.all(mask_chunk_baseline, axis=-1) * SweepBitMask.baseline
        )

        # ----------------------------------------------------------------
        # D21 baseline mask — map chunk baseline back to unified grid
        # ----------------------------------------------------------------
        d21_mask_baseline = np.ones(d21_frequency.shape, dtype=bool)
        freq_Hz = frequency.to_value(u.Hz)
        d21_freq_Hz = d21_frequency.to_value(u.Hz)
        chunk_bounds_in_unified = [
            np.searchsorted(d21_freq_Hz, freq_Hz[:, [s.start, s.stop - 1]])
            for s in chunk_slices
        ]
        for j, bounds in enumerate(chunk_bounds_in_unified):
            for ci, (i0, i1) in enumerate(bounds):
                d21_mask_baseline[i0 : i1 + 2] &= mask_chunk_baseline[ci, j]

        chunk_size_in_unified = (
            chunk_bounds_in_unified[0][0, 1] - chunk_bounds_in_unified[0][0, 0] + 1
            if len(chunk_bounds_in_unified) > 0
            else 1
        )
        if chunk_size_in_unified > 0:
            d21_mask_baseline = binary_erosion(
                d21_mask_baseline,
                structure=np.ones((max(chunk_size_in_unified * 2, 3),)),
            )
        ctd.d21_mask_baseline = d21_mask_baseline

        mask_baseline = ctd.mask_baseline = cls.make_data_mask_from_unified(
            frequency,
            d21_frequency,
            d21_mask_baseline,
        )

        # Final per-point bitmask
        bitmask = ctd.bitmask = np.zeros(freq_Hz.shape, dtype=int)
        for i, s in enumerate(chunk_slices):
            bitmask[:, s] = bitmask_chunk[:, i : i + 1]
        bitmask[:] = bitmask & ~(SweepBitMask.spike | SweepBitMask.baseline)
        bitmask |= (mask_spike * SweepBitMask.spike) | (
            mask_baseline * SweepBitMask.baseline
        )

        mask_chan_bad = ctd.mask_chan_bad = (bitmask_chan & cfg.bad_chan_bits) > 0
        logger.debug(f"channel bad mask {pformat_mask(mask_chan_bad)}")

        bms_chan = BitmaskStats(SweepBitMask, bitmask_chan)
        ctd.bitmask_chan_stats = bms_chan.stats
        logger.debug(f"channel bitmask summary\n{bms_chan.pformat()}")

        bms_chunk = BitmaskStats(SweepBitMask, bitmask_chunk)
        ctd.bitmask_chunk_stats = bms_chunk.stats
        logger.debug(f"chunk bitmask summary\n{bms_chunk.pformat()}")

        # ----------------------------------------------------------------
        # D21 chunk statistics and detrending
        # ----------------------------------------------------------------
        with timeit("calc d21 chunk statistics"):
            ctx_d21_chunks = cls.make_chunks(
                n_items=d21_frequency.shape[0],
                chunk_size=cfg.chunk_size * 100,
                n_chunks_min=cfg.n_chunks_min,
            )
            ctd.d21_chunk_size = ctx_d21_chunks["chunk_size"]
            ctd.d21_n_chunks = ctx_d21_chunks["n_chunks"]
            d21_chunk_windows = ctd.d21_chunk_windows = ctx_d21_chunks["chunk_windows"]
            ctd.d21_chunk_slices = ctx_d21_chunks["chunk_slices"]
            d21_f_chunks = ctd.d21_f_chunks = (
                np.median(d21_freq_Hz[d21_chunk_windows], axis=-1) << u.Hz
            )

            d21_values = d21_data.to_value(u.Hz**-1)
            d21_data_baseline_value = np.copy(d21_values)
            d21_data_baseline_value[(~d21_mask_baseline) | (d21_values == 0)] = np.nan

            _chunk_data = d21_data_baseline_value[d21_chunk_windows].copy()
            _all_nan = np.all(np.isnan(_chunk_data), axis=-1)
            # Avoid RuntimeWarning from nanmedian on all-NaN windows
            _chunk_data[_all_nan, 0] = 0.0
            d21_chunk_baseline_value = np.nanmedian(_chunk_data, axis=-1)
            d21_chunk_baseline_value[_all_nan] = np.nan
            if np.all(np.isnan(d21_chunk_baseline_value)):
                d21_chunk_baseline_value = np.zeros_like(d21_chunk_baseline_value)
            ctd.d21_chunk_baseline = d21_chunk_baseline_value << u.Hz**-1

            _chunk_data_rms = d21_data_baseline_value[d21_chunk_windows].copy()
            _chunk_data_rms[_all_nan, 0] = 0.0
            d21_chunk_baseline_rms = mad_std(
                _chunk_data_rms,
                axis=-1,
                ignore_nan=True,
            )
            d21_chunk_baseline_rms[_all_nan] = np.nan
            if np.all(np.isnan(d21_chunk_baseline_rms)):
                d21_chunk_baseline_rms = np.full_like(d21_chunk_baseline_rms, 0.1)
            m_pos = d21_chunk_baseline_rms > 0
            if m_pos.any():
                min_pos = np.min(d21_chunk_baseline_rms[m_pos])
                d21_chunk_baseline_rms[~m_pos] = min_pos
            ctd.d21_chunk_baseline_rms = d21_chunk_baseline_rms << u.Hz**-1

            m = ~(
                np.isnan(d21_chunk_baseline_value)
                | np.isnan(d21_chunk_baseline_rms)
            )
            d21_f_chunks_Hz = d21_f_chunks.to_value(u.Hz)
            d21_baseline_value = np.interp(
                d21_freq_Hz,
                d21_f_chunks_Hz[m],
                d21_chunk_baseline_value[m],
            )
            ctd.d21_baseline = d21_baseline_value << u.Hz**-1

            d21_baseline_rms_value = np.interp(
                d21_freq_Hz,
                d21_f_chunks_Hz[m],
                d21_chunk_baseline_rms[m],
            )
            ctd.d21_baseline_rms = d21_baseline_rms_value << u.Hz**-1

            d21_detrended_value = d21_values - d21_baseline_value
            d21_detrended_value[d21_values == 0] = 0
            ctd.d21_detrended = d21_detrended_value << u.Hz**-1

        return True

    # ----------------------------------------------------------------
    # Static helpers (ported unchanged from v2)
    # ----------------------------------------------------------------

    @staticmethod
    def make_data_mask_from_unified(
        fs: u.Quantity,
        fs_unified: u.Quantity,
        mask_unified: npt.NDArray,
    ) -> npt.NDArray:
        """Return data mask by nearest-neighbour match to the unified grid.

        Parameters
        ----------
        fs : u.Quantity
            Per-data-point frequencies, shape ``[n_chans, n_steps]``.
        fs_unified : u.Quantity
            Unified frequency grid, shape ``[n_unified]``.
        mask_unified : ndarray
            Boolean mask on the unified grid.

        Returns
        -------
        ndarray
            Boolean mask on the per-data-point grid (same shape as *fs*).
        """
        data_idx = np.searchsorted(
            fs_unified.to_value(u.Hz),
            fs.to_value(u.Hz),
        )
        n_unified = fs_unified.shape[0]
        data_idx[data_idx >= n_unified] = n_unified - 1
        return mask_unified[data_idx]

    @staticmethod
    def make_chunks(
        n_items: int,
        chunk_size: int,
        n_chunks_min: int,
    ) -> dict:
        """Divide *n_items* into ``n_chunks`` evenly-spaced chunks.

        Returns a dict with keys: ``chunk_size``, ``n_chunks``,
        ``chunk_windows``, ``chunk_slices``.
        """
        n_chunks = n_items // chunk_size + (n_items % chunk_size > 0)
        n_chunks = max(n_chunks, n_chunks_min)
        if n_chunks % 2 == 0:
            n_chunks += 1
        # Clamp chunk_size so sliding_window_view doesn't fail
        chunk_size = min(chunk_size, n_items)
        windows = sliding_window_view(np.arange(n_items), chunk_size)
        n_windows = windows.shape[0]
        sw = max(n_windows // max(n_chunks - 1, 1), 1)
        iw = np.r_[
            np.arange(0, n_windows // 2 - sw // 2, sw),
            n_windows // 2,
            np.arange(n_windows - 1, n_windows // 2 + sw // 2, -sw)[::-1],
        ]
        iw = np.unique(np.clip(iw, 0, n_windows - 1))
        chunk_windows = windows[iw]
        chunks = np.split(np.arange(n_items), (iw[1:] + iw[:-1]) // 2)
        chunk_slices = [slice(int(c[0]), int(c[-1]) + 1) for c in chunks]
        logger.debug(f"{n_items=} {n_chunks=} {chunk_size=}")
        chunk_index = np.empty((n_items,), dtype=int)
        for i, s in enumerate(chunk_slices):
            chunk_index[s] = i
        return locals()

    @classmethod
    def find_spike(
        cls,
        y: npt.NDArray,
        medfilt_size: int = 5,
        y_range_min: float = 0.1,
        height_frac_min: float = 0.1,
    ) -> tuple[npt.NDArray, dict]:
        """Identify spikes in 2-D sweep amplitude data.

        Parameters
        ----------
        y : ndarray
            2-D array of shape ``[n_chans, n_steps]`` (e.g. ``aS21_db``).
        medfilt_size : int
            Median filter size along the sweep axis.
        y_range_min : float
            Channels with range smaller than this are not checked.
        height_frac_min : float
            Spike threshold as a fraction of the channel range.

        Returns
        -------
        mask_spike : ndarray, bool
            ``True`` where a spike is found.
        ctx : dict
            Intermediate arrays (``y_med``, ``y_range``, ``s_spike``).
        """
        if y.ndim != 2:  # noqa: PLR2004
            raise ValueError("input data shall be 2-D.")
        y_med = median_filter(y, (1, medfilt_size))
        y_range = np.max(y_med, axis=-1) - np.min(y_med, axis=-1)
        s_spike = (y - y_med) / np.where(
            y_range[:, np.newaxis] > 0, y_range[:, np.newaxis], 1.0
        )
        md_spike0 = np.abs(s_spike) >= height_frac_min
        logger.debug(f"found spike {md_spike0.sum()}/{md_spike0.size}")
        mc_range_small = y_range < y_range_min
        md_spike = md_spike0 & (~mc_range_small[:, np.newaxis])
        logger.debug(f"mask spike {md_spike.sum()}/{md_spike.size}")
        return md_spike, locals()
