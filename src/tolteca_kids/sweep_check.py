from dataclasses import dataclass
from enum import IntFlag, auto
from typing import Literal

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
import scipy.stats
from astropy.stats import mad_std
from numpy.lib.stride_tricks import sliding_window_view
from pydantic import ConfigDict, Field
from scipy.ndimage import binary_erosion, median_filter
from tollan.config.types import FrequencyQuantityField
from tollan.utils.fmt import BitmaskStats, pformat_mask
from tollan.utils.general import rupdate
from tollan.utils.log import logger, timeit
from tollan.utils.plot.plotly import make_range

from tolteca_kidsproc.kidsdata import MultiSweep
from tolteca_kidsproc.kidsdata.sweep import D21Analysis

from .pipeline import Step, StepConfig, StepContext
from .plot import PlotConfig, PlotMixin
from .roach_tone_power import RoachTonePower

__all__ = [
    "DespikeMethod",
    "SweepBitMask",
    "SweepCheck",
    "SweepCheckConfig",
    "SweepCheckContext",
    "SweepCheckData",
]


class SweepBitMask(IntFlag):
    """A bit mask for sweep data."""

    # sweep check flags
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
    """Data point idenfieid as baseline."""

    tone_amp_zero = auto()
    """Tone amplitude is zero."""

    tone_amp_one = auto()
    """Tone amplitude is one."""

    tone_power_low = auto()
    """Tone driving power low."""

    tone_power_high = auto()
    """Tone driving power high."""


DespikeMethod = Literal["interp_linear",]


class SweepCheckConfig(StepConfig):
    """The sweep checking config."""

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
            "Hight threshold used to identify spikes, measured as "
            "fraction to the channel data range."
        ),
    )
    spike_medfilt_size: int = Field(
        default=5,
        description=("Size of median filter used to identify spikes."),
    )

    # chan stats
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
    chan_tone_power_dbm_min: float = Field(
        default=-np.inf,
        description="Channel tone power lower than this is flagged.",
    )
    chan_tone_power_dbm_max: float = Field(
        default=np.inf,
        description="Channel tone power higher than this is flagged.",
    )

    # chunk stats
    chunk_size: int = Field(
        default=50,
        description="Chunk size to use for compute per chunk stats for channel.",
    )
    n_chunks_min: int = Field(
        default=5,
        description="Minimum number of chunks to use per chunk.",
    )
    chunk_skew_max: float = Field(
        default=0.2,
        description="Chunk data kurtosis higher than this is flagged.",
    )
    chunk_kurtosis_max: float = Field(
        default=-1.0,
        description="Chunk data kurtosis higher than this is flagged.",
    )
    bad_chan_bits: SweepBitMask = Field(
        default=(SweepBitMask.rms_low | SweepBitMask.rms_high),
        description=("bits to use to generate channel badmask."),
    )

    not_baseline_chunk_bits: SweepBitMask = Field(
        default=(SweepBitMask.skew_high | SweepBitMask.kurtosis_high),
        description=("bits to use to generate chunk baseline flag."),
    )

    d21_analysis: D21Analysis = Field(
        default={
            # "f_lims": (450, 1050 << u.MHz,
            "f_step": 1000 << u.Hz,
            "smooth": 5,
            "method": "savgol",
        },
        description="D21 analysis paremters.",
    )

    noise_psd: bool = Field(
        default=False,
        description="Calculate noise PSD.",
    )
    noise_psd_nperseg: int = Field(
        default=512,
        description="nperseg for noise PDF calculation.",
    )
    chan_noise_psd_f_lims: tuple[FrequencyQuantityField, FrequencyQuantityField] = (
        Field(
            default=(5 << u.Hz, 20 << u.Hz),
            description="The range to calculate channel PSD noise.",
        )
    )


@dataclass(kw_only=True)
class SweepCheckData:
    """The data class for sweep check."""

    mask_spike: npt.NDArray = ...
    S21_orig: None | npt.NDArray = None
    S21_spike: None | npt.NDArray = None

    chunk_size: int = ...
    n_chunks_per_chan: int = ...
    chunk_windows: npt.NDArray = ...
    chunk_slices: list[slice] = ...
    f_chunks: npt.NDArray = ...

    bitmask: SweepBitMask = ...
    bitmask_chan: SweepBitMask = ...
    bitmask_chunk: SweepBitMask = ...
    bitmask_chan_stats: pd.DataFrame = ...
    bitmask_chunk_stats: pd.DataFrame = ...

    mask_chan_bad: npt.NDArray = ...
    mask_chunk_baseline: npt.NDArray = ...
    mask_baseline: npt.NDArray = ...

    d21_mask_baseline: npt.NDArray = ...

    d21_chunk_size: int = ...
    d21_n_chunks: int = ...
    d21_chunk_windows: npt.NDArray = ...
    d21_chunk_slices: list[slice] = ...
    d21_f_chunks: npt.NDArray = ...
    d21_chunk_baseline: npt.NDArray = ...
    d21_chunk_baseline_rms: npt.NDArray = ...
    d21_baseline: npt.NDArray = ...
    d21_baseline_rms: npt.NDArray = ...
    d21_frequency: npt.NDArray = ...
    d21: npt.NDArray = ...
    d21_detrended: npt.NDArray = ...

    # d21_chunk_skew: npt.NDArray = ...
    # d21_chunk_kurtosis: npt.NDArray = ...
    # d21_bitmask_chunk: SweepBitMask = ...
    # d21_mask_chunk_baseline: npt.NDArray = ...

    chan_range: npt.NDArray = ...
    chan_level: npt.NDArray = ...
    chan_rms_mean: npt.NDArray = ...
    chan_rms_std: npt.NDArray = ...

    roach_tone_power: RoachTonePower = ...

    chunk_rms_mean: npt.NDArray = ...
    chunk_rms_std: npt.NDArray = ...

    chunk_skew: npt.NDArray = ...
    chunk_kurtosis: npt.NDArray = ...

    swp_rms_med: float = ...
    swp_rms_rms: float = ...

    # psd
    f_psd: None | npt.NDArray = None
    f_psd_mask: None | npt.NDArray = None
    I_psd: None | npt.NDArray = None
    Q_psd: None | npt.NDArray = None
    Sphi: None | npt.NDArray = None

    chan_I_psd_med: None | npt.NDArray = None
    chan_I_psd_mad_std: None | npt.NDArray = None
    chan_Q_psd_med: None | npt.NDArray = None
    chan_Q_psd_mad_std: None | npt.NDArray = None
    chan_Sphi_med: None | npt.NDArray = None
    chan_Sphi_mad_std: None | npt.NDArray = None


class SweepCheckContext(StepContext["SweepCheck", SweepCheckConfig]):
    """The context class for sweep check."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: SweepCheckData = Field(default_factory=SweepCheckData)


class SweepCheck(Step[SweepCheckConfig, SweepCheckContext]):
    """Sweep check step.

    This gives a summary of the status of sweep data, and removes
    spikes in it.
    """

    @classmethod
    @timeit
    def run(cls, data: MultiSweep, context):  # noqa: C901, PLR0915
        """Run sweep check."""
        swp = data
        cfg: SweepCheckConfig = context.config
        ctd = context.data
        y = swp.aS21_db
        mask_spike, ctx_spike = cls.find_spike(
            y,
            medfilt_size=cfg.spike_medfilt_size,
            y_range_min=cfg.chan_range_db_min,
            height_frac_min=cfg.spike_height_frac_min,
        )
        S21_orig = ctd.S21_orig = swp.S21.copy()
        if cfg.despike:
            despike_method = cfg.despike_method
            with timeit(f"despike with method={despike_method}"):
                if despike_method == "interp_linear":
                    for ci in range(swp.n_chans):
                        m = mask_spike[ci]
                        swp.S21[ci, m] = np.interp(
                            swp.frequency.value[ci, m],
                            swp.frequency.value[ci, ~m],
                            swp.S21.value[ci, ~m],
                        )
                    swp.reset_cache()
                    # y_nospike = swp.aS21_db
                    S21_spike = S21_orig - swp.S21
        else:
            # y_nospike = y
            S21_spike = None
        ctd.mask_spike = mask_spike
        ctd.S21_spike = S21_spike
        # count spike
        # https://stackoverflow.com/a/24343375
        # chan_n_spikes = np.array(
        #     [
        #         len(
        #             np.diff(
        #                 np.where(
        #                     np.concatenate(
        #                         (
        #                             [md_spike[i, 0]],
        #                             md_spike[i, :-1] != md_spike[i, 1:],
        #                             [True],
        #                         ),
        #                     ),
        #                 )[0],
        #             )[::2],
        #         )
        #         for i in range(swp.n_chans)
        #     ],
        # )
        # make d21 data
        d21_unified = cfg.d21_analysis.make_unified(swp)
        d21_data = ctd.d21 = d21_unified.quantity
        d21_frequency = ctd.d21_frequency = d21_unified.frequency

        # compose bitmasks
        # channel bitmask
        chan_range = ctd.chan_range = ctx_spike["y_range"]
        chan_level = ctd.chan_level = np.mean(swp.aS21_db, axis=1)
        y_rms_no_spike = swp.aS21_unc_db.copy()
        y_rms_no_spike[mask_spike] = np.nan
        chan_rms_mean = ctd.chan_rms_mean = np.nanmean(y_rms_no_spike, axis=1)
        ctd.chan_rms_std = np.nanstd(y_rms_no_spike, axis=1)
        ctd.swp_rms_med = np.median(chan_rms_mean)
        ctd.swp_rms_rms = mad_std(chan_rms_mean)

        # roach tone power
        rtp = ctd.roach_tone_power = RoachTonePower.from_kidsdata(swp)
        logger.debug(rtp.pformat())

        with timeit("calc chunk statistics"):
            # chunking
            ctx_chan_chunks = cls.make_chunks(
                n_items=swp.n_steps,
                chunk_size=cfg.chunk_size,
                n_chunks_min=cfg.n_chunks_min,
            )
            # run windowed statistics
            ctd.chunk_size = ctx_chan_chunks["chunk_size"]
            ctd.n_chunks_per_chan = ctx_chan_chunks["n_chunks"]
            chunk_windows = ctd.chunk_windows = ctx_chan_chunks["chunk_windows"]
            chunk_slices = ctd.chunk_slices = ctx_chan_chunks["chunk_slices"]
            ctd.f_chunks = np.median(swp.frequency[:, chunk_windows], axis=-1)
            chunk_rms_mean = ctd.chunk_rms_mean = np.nanmean(
                y_rms_no_spike[:, chunk_windows],
                axis=-1,
            )
            ctd.chunk_rms_std = ctd.chunk_rms_std = np.nanstd(
                y_rms_no_spike[:, chunk_windows],
                axis=-1,
            )

            def _get_complex_stats(comb_func, stat_func, data, axis=None):
                return comb_func(
                    stat_func(data.real, axis=axis),
                    stat_func(data.imag, axis=axis),
                )

            def _complex_skew(data, axis=None):
                return _get_complex_stats(
                    lambda x, y: np.hypot(x, y) / np.sqrt(2),
                    scipy.stats.skew,
                    data,
                    axis=axis,
                )

            def _complex_kurtosis(data, axis=None):
                return _get_complex_stats(
                    lambda x, y: np.max([x, y], axis=0),
                    scipy.stats.kurtosis,
                    data,
                    axis=axis,
                )

            chunk_skew = ctd.chunk_skew = _complex_skew(
                swp.S21[:, chunk_windows],
                axis=-1,
            )
            chunk_kurtosis = ctd.chunk_kurtosis = _complex_kurtosis(
                swp.S21[:, chunk_windows],
                axis=-1,
            )
            mask_chunk_rms_low = chunk_rms_mean < cfg.chan_rms_db_min
            mask_chunk_rms_high = chunk_rms_mean > cfg.chan_rms_db_max
            mask_chunk_skew_high = chunk_skew > cfg.chunk_skew_max
            mask_chunk_kurtosis_high = chunk_kurtosis > cfg.chunk_kurtosis_max

        # channel bitmask
        bitmask_chan = ctd.bitmask_chan = (
            (chan_range < cfg.chan_range_db_min) * SweepBitMask.range_small
            | (chan_range > cfg.chan_range_db_max) * SweepBitMask.range_large
            | (chan_level < cfg.chan_level_db_min) * SweepBitMask.level_low
            | (chan_level > cfg.chan_level_db_max) * SweepBitMask.level_high
            | (chan_rms_mean < cfg.chan_rms_db_min) * SweepBitMask.rms_low
            | (chan_rms_mean > cfg.chan_rms_db_max) * SweepBitMask.rms_high
            # | (tone_amps <= 0.) * SweepBitMask.tone_amp_zero
            # | (tone_amps >= 1.) * SweepBitMask.tone_amp_one
            # | (tone_powers_dbm < cfg.chan_tone_power_dbm_min)
            #  * SweepBitMask.tone_power_low
            # | (tone_powers_dbm > cfg.chan_tone_power_dbm_max)
            #  * SweepBitMask.tone_power_high
            | (np.sum(mask_spike, axis=-1) > 0) * SweepBitMask.spike
        )
        # chunk bitmask
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
        # spike mask
        for i, s in enumerate(chunk_slices):
            bitmask_chunk[:, i] |= (
                np.sum(mask_spike[:, s], axis=-1) > 0
            ) * SweepBitMask.spike
        # baseline mask
        mask_chunk_baseline = ctd.mask_chunk_baseline = (
            bitmask_chunk & cfg.not_baseline_chunk_bits == 0
        )
        bitmask_chunk |= mask_chunk_baseline * SweepBitMask.baseline

        # set back some aggregates to chan bitmask
        bitmask_chan |= (
            (
                np.any(
                    mask_chunk_skew_high,
                    axis=-1,
                )
            )
            * SweepBitMask.skew_high
            | (
                np.any(
                    mask_chunk_kurtosis_high,
                    axis=-1,
                )
            )
            * SweepBitMask.kurtosis_high
            | (
                np.all(
                    mask_chunk_baseline,
                    axis=-1,
                )
            )
            * SweepBitMask.baseline
        )
        d21_mask_baseline = np.ones(
            d21_frequency.shape,
            dtype=bool,
        )
        chunk_bounds_in_unified = [
            np.searchsorted(
                d21_frequency.to_value(u.Hz),
                swp.frequency.to_value(u.Hz)[:, [s.start, s.stop - 1]],
            )
            for s in chunk_slices
        ]
        for j, bounds in enumerate(chunk_bounds_in_unified):
            for ci, (i0, i1) in enumerate(bounds):
                d21_mask_baseline[i0 : i1 + 2] &= mask_chunk_baseline[ci, j]
        chunk_size_in_unified = (
            chunk_bounds_in_unified[0][0, 1] - chunk_bounds_in_unified[0][0, 0] + 1
        )
        # here we remove any baseline that is less than two chunks short
        d21_mask_baseline = binary_erosion(
            d21_mask_baseline,
            structure=np.ones((chunk_size_in_unified * 2,)),
        )
        ctd.d21_mask_baseline = d21_mask_baseline

        # build per data point baseline by interpolate back from the unified grid
        mask_baseline = ctd.mask_baseline = cls.make_data_mask_from_unified(
            swp.frequency,
            d21_frequency,
            d21_mask_baseline,
        )

        # compose data bitmask from chunk mask, but use per-data spike and baseline mask
        bitmask = ctd.bitmask = np.zeros(swp.frequency.shape, dtype=int)
        for i, s in enumerate(chunk_slices):
            bitmask[:, s] = bitmask_chunk[:, i : i + 1]
        # bitmask[:] = bitmask & ~(SweepBitMask.spike)
        # bitmask |= mask_spike * SweepBitMask.spike
        # ctd.mask_baseline = (bitmask & SweepBitMask.baseline) > 0
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

        # use unified baseline mask to run d21 statistics and detrend
        with timeit("calc d21 chunk statistics"):
            # chunking
            ctx_d21_chunks = cls.make_chunks(
                n_items=d21_frequency.shape[0],
                chunk_size=cfg.chunk_size * 100,
                n_chunks_min=cfg.n_chunks_min,
            )
            # run windowed statistics
            ctd.d21_chunk_size = ctx_d21_chunks["chunk_size"]
            ctd.d21_n_chunks = ctx_d21_chunks["n_chunks"]
            d21_chunk_windows = ctd.d21_chunk_windows = ctx_d21_chunks["chunk_windows"]
            ctd.d21_chunk_slices = ctx_d21_chunks["chunk_slices"]
            d21_f_chunks = ctd.d21_f_chunks = np.median(
                d21_frequency[d21_chunk_windows],
                axis=-1,
            )
            # make masked data
            d21_data_baseline_value = np.copy(d21_data.value)
            d21_data_baseline_value[(~d21_mask_baseline) | (d21_data == 0)] = np.nan
            d21_chunk_baseline_value = np.nanmedian(
                d21_data_baseline_value[d21_chunk_windows],
                axis=-1,
            )
            if np.all(np.isnan(d21_chunk_baseline_value)):
                d21_chunk_baseline_value[:] = 0.0

            ctd.d21_chunk_baseline = d21_chunk_baseline_value << d21_unified.unit
            d21_chunk_baseline_value_rms = mad_std(
                d21_data_baseline_value[d21_chunk_windows],
                axis=-1,
                ignore_nan=True,
            )
            if np.all(np.isnan(d21_chunk_baseline_value_rms)):
                d21_chunk_baseline_value_rms[:] = 0.1
            # make sure the rms values are always larger than 0
            d21_chunk_baseline_value_rms[d21_chunk_baseline_value_rms <= 0] = np.min(
                d21_chunk_baseline_value_rms[d21_chunk_baseline_value_rms > 0],
            )
            ctd.d21_chunk_baseline_rms = (
                d21_chunk_baseline_value_rms << d21_unified.unit
            )
            m = ~(
                np.isnan(d21_chunk_baseline_value)
                | np.isnan(d21_chunk_baseline_value_rms)
            )
            d21_baseline_value = np.interp(
                d21_frequency.to_value(u.Hz),
                d21_f_chunks.to_value(u.Hz)[m],
                d21_chunk_baseline_value[m],
            )
            ctd.d21_baseline = d21_baseline_value << d21_unified.unit
            d21_baseline_value_rms = np.interp(
                d21_frequency.to_value(u.Hz),
                d21_f_chunks.to_value(u.Hz)[m],
                d21_chunk_baseline_value_rms[m],
            )
            ctd.d21_baseline_rms = d21_baseline_value_rms << d21_unified.unit
            d21_detrended_value = d21_data.value - d21_baseline_value
            # this assumes the d21 is already positive, so 0 value is like np.nan
            d21_detrended_value[d21_data.value == 0] = 0
            ctd.d21_detrended = d21_detrended_value << d21_unified.unit

        if cfg.noise_psd:
            with timeit("calc noise psd"):
                assert swp.meta is not None
                I_raw = swp.meta["I_raw"]
                Q_raw = swp.meta["Q_raw"]
                fsmp = swp.meta["fsmp"]
                nperseg = cfg.noise_psd_nperseg
                from scipy.signal import welch

                f_psd, I_psd = welch(I_raw, fs=fsmp, nperseg=nperseg, axis=-1)
                logger.debug(f"psd_freq shape: {f_psd.shape} psd shape: {I_psd.shape}")
                logger.debug(
                    f"psd_freq range: {f_psd[0]}-{f_psd[-1]} Hz"
                    f" step={f_psd[1]-f_psd[0]} Hz",
                )
                _, Q_psd = welch(Q_raw, fs=fsmp, nperseg=nperseg, axis=-1)
                Sphi = (I_psd + Q_psd) / np.mean(I_raw**2 + Q_raw**2, axis=-1)[:, None]
                f_psd_mask = (f_psd >= cfg.chan_noise_psd_f_lims[0].to_value(u.Hz)) & (
                    f_psd <= cfg.chan_noise_psd_f_lims[1].to_value(u.Hz)
                )
                logger.debug(f"psd f_mask: {pformat_mask(f_psd_mask)}")
                chan_I_psd_med = np.median(I_psd[:, f_psd_mask], axis=-1)
                chan_I_psd_mad_std = mad_std(I_psd[:, f_psd_mask], axis=-1)
                chan_Q_psd_med = np.median(Q_psd[:, f_psd_mask], axis=-1)
                chan_Q_psd_mad_std = mad_std(Q_psd[:, f_psd_mask], axis=-1)
                chan_Sphi_med = np.median(Sphi[:, f_psd_mask], axis=-1)
                chan_Sphi_mad_std = mad_std(Sphi[:, f_psd_mask], axis=-1)
                ctd.f_psd = f_psd
                ctd.f_psd_mask = f_psd_mask
                ctd.I_psd = I_psd
                ctd.Q_psd = Q_psd
                ctd.Sphi = Sphi
                ctd.chan_I_psd_med = chan_I_psd_med
                ctd.chan_I_psd_mad_std = chan_I_psd_mad_std
                ctd.chan_Q_psd_med = chan_Q_psd_med
                ctd.chan_Q_psd_mad_std = chan_Q_psd_mad_std
                ctd.chan_Sphi_med = chan_Sphi_med
                ctd.chan_Sphi_mad_std = chan_Sphi_mad_std
        return True

    @staticmethod
    def make_data_mask_from_unified(fs, fs_unified, mask_unified):
        """Return data mask by matching frequency range."""
        data_idx_in_unified = np.searchsorted(
            fs_unified.to_value(u.Hz),
            fs.to_value(u.Hz),
        )
        n_unified = fs_unified.shape[0]
        data_idx_in_unified[data_idx_in_unified >= n_unified] = n_unified - 1
        return mask_unified[data_idx_in_unified]

    @staticmethod
    def make_chunks(n_items, chunk_size, n_chunks_min):
        """Make chunks for n_items."""
        # adjust chunksize
        n_chunks = n_items // chunk_size + (n_items % chunk_size > 0)
        n_chunks = max(n_chunks, n_chunks_min)
        # make sure n_chunks is odd so we always have a center chunk
        if n_chunks % 2 == 0:
            n_chunks += 1
        windows = sliding_window_view(np.arange(n_items), chunk_size)
        # select n_chunks evenly from the window
        n_windows = windows.shape[0]
        sw = n_windows // (n_chunks - 1)
        iw = np.r_[
            np.arange(0, n_windows // 2 - sw // 2, sw),
            n_windows // 2,
            np.arange(n_windows - 1, n_windows // 2 + sw // 2, -sw)[::-1],
        ]
        dw = np.diff(iw)
        chunk_windows = windows[iw]
        chunks = np.split(np.arange(n_items), (iw[1:] + iw[:-1]) // 2)
        chunk_slices = [slice(c[0], c[-1] + 1) for c in chunks]
        logger.debug(
            f"{n_items=} {n_chunks=} {chunk_size=} {iw=} {dw=}",
            # f"chunk_slices={pformat_fancy_index(chunk_slices)}",
        )
        chunk_index = np.empty((n_items,), dtype=int)
        for i, s in enumerate(chunk_slices):
            chunk_index[s] = i
        return locals()

    @classmethod
    def find_spike(
        cls,
        y,
        medfilt_size=SweepCheckConfig.field_defaults["spike_medfilt_size"],
        y_range_min=SweepCheckConfig.field_defaults["chan_range_db_min"],
        height_frac_min=SweepCheckConfig.field_defaults["spike_height_frac_min"],
    ):
        """Identify spikes in data."""
        if y.ndim != 2:  # noqa: PLR2004
            raise ValueError("input data shall be 2-d.")
        y_med = median_filter(y, (1, medfilt_size))
        y_range = np.max(y_med, axis=-1) - np.min(y_med, axis=-1)
        s_spike = (y - y_med) / y_range[:, np.newaxis]
        md_spike0 = np.abs(s_spike) >= height_frac_min
        logger.debug(f"found spike {md_spike0.sum()}/{md_spike0.size}")
        # create spike mask, which is all spikes found in good channel.
        mc_range_small = y_range < y_range_min
        md_spike = md_spike0 & (~mc_range_small[:, np.newaxis])
        logger.debug(f"mask spike {md_spike.sum()}/{md_spike.size}")
        return md_spike, locals()


class SweepCheckPlotConfig(PlotConfig):
    """The sweep check plot config."""


@dataclass(kw_only=True)
class SweepCheckPlotData:
    """The data class for sweep check plot."""

    chan_summary: go.Figure = ...
    S21_f: go.Figure = ...
    S21_f_grid: go.Figure = ...
    I_Q_grid: go.Figure = ...
    roach_tone_power: go.Figure = ...
    noise_psd: go.Figure = ...


class SweepCheckPlotContext(StepContext["SweepCheckPlot", SweepCheckPlotConfig]):
    """The context class for sweep check ploy."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: SweepCheckPlotData = Field(default_factory=SweepCheckPlotData, repr=False)


class SweepCheckPlot(PlotMixin, Step[SweepCheckPlotConfig, SweepCheckPlotContext]):
    """Sweep check plot.

    This step produces visualization for sweep check step.
    """

    @classmethod
    @timeit
    def run(cls, data: MultiSweep, context):
        """Run sweep check."""
        ctx0 = SweepCheck.get_context(data)
        if not ctx0.completed:
            raise ValueError("sweep check step has not run yet.")
        swp = data
        # cfg0 = ctx0.config
        ctd0 = ctx0.data
        # cfg1 = context.config
        ctd1 = context.data

        grid = cls.make_subplot_grid()

        grid.add_subplot(
            row=1,
            col=1,
            fig=cls.make_bitmask_chan_heatmap(
                ctd0.bitmask_chan,
            ),
            row_height=0.5,
        )

        # tone_amp = swp.meta["chan_axis_data"]["amp_tone"]
        # tone_amp_db = 20 * np.log10(tone_amp.max() / tone_amp)
        # tone_amps = ctd0.tone_amps
        tone_amps_db = ctd0.roach_tone_power.tone_amps_db
        tone_power_dbm = ctd0.roach_tone_power.predicted_per_tone.drive_atten.output_dbm
        chan_data_items = [
            ("Channel Level", ctd0.chan_level, {}),
            ("Channel Range", ctd0.chan_range, {}),
            ("Channel RMS", ctd0.chan_rms_mean, {}),
            ("Drive Atten", tone_amps_db, {}),
            ("Drive Power (dbm)", tone_power_dbm, {}),
        ]
        row0 = grid.shape[0] + 1
        for i, (name, value, trace_kw) in enumerate(chan_data_items):
            grid.add_subplot(
                row=row0 + i,
                col=1,
                fig=cls.make_chan_data_heatmap(
                    name,
                    value,
                    trace_kw,
                ),
                row_height=0.75 / len(chan_data_items),
            )
        chunk_data_items = [
            ("Chunk RMS", ctd0.chunk_rms_mean, {}),
            ("Chunk Skew", ctd0.chunk_skew, {"zmin": 0, "zmax": 2}),
            ("Chunk Kurtosis", ctd0.chunk_kurtosis, {"zmin": -2, "zmax": 2}),
            (
                "Chunk Baseline",
                ctd0.mask_chunk_baseline.astype(int),
                {},
            ),
        ]
        row0 = grid.shape[0] + 1
        for i, (name, value, trace_kw) in enumerate(chunk_data_items):
            grid.add_subplot(
                row=row0 + i,
                col=1,
                fig=cls.make_chunk_data_heatmap(
                    name,
                    value,
                    trace_kw,
                ),
                row_height=1 / len(chunk_data_items),
            )

        # Add channel frequency plot at the end
        fig_f_chan = cls.make_tone_freq_figure(swp, marker_color="blue")
        grid.add_subplot(
            row=grid.shape[0] + 1,
            col=1,
            fig=fig_f_chan,
            row_height=0.5,
        )

        fig = ctd1.chan_summary = grid.make_figure(
            shared_xaxes="all",
            vertical_spacing=40 / 1400,
            fig_layout={
                "height": 1400,
            },
        )
        # add a range slider
        fig.update_xaxes(
            rangeslider={
                "autorange": True,
                "range": [0, swp.n_chans],
                "thickness": 0.05,
            },
            row=grid.shape[0],
            col=1,
        )
        # S21-f and D21-f
        ctd1.S21_f = cls.make_s21_f(data, ctx0)

        # grid plots
        n_rows, n_cols = 5, 5
        subplot_kw = {
            "vertical_spacing": 0.02,
            "horizontal_spacing": 0.02,
            "fig_layout": cls.fig_layout_default
            | {
                "width": 1200,
                "height": 1200,
            },
        }
        ctd1.S21_f_grid = cls.make_S21_f_grid(data, n_rows, n_cols, **subplot_kw)
        n_rows, n_cols, n_chans_per_panel = 1, 1, 25
        ctd1.I_Q_grid = cls.make_IQ_grid(
            data,
            n_rows,
            n_cols,
            n_chans_per_panel,
            **subplot_kw,
        )
        ctd1.roach_tone_power = ctd0.roach_tone_power.make_plotly_figure()
        ctd1.noise_psd = cls.make_noise_psd_figure(data, ctx0)
        cls.save_or_show(data, context)
        return True

    @classmethod
    def make_bitmask_chan_heatmap(cls, bitmask_chan, fig=None, panel_kw=None):
        names = []
        data = []
        for name, value in SweepBitMask.__members__.items():
            names.append(name)
            data.append((bitmask_chan & value) > 0)
        data = np.vstack(data).astype(int)

        fig = fig or cls.make_subplots(1, 1)
        panel_kw = panel_kw or {}
        fig.add_heatmap(
            z=data,
            y=names,
            colorscale="rdylgn_r",
            zmin=0,
            zmax=1,
            **panel_kw,
        )
        fig.update_xaxes(
            title="Channel Id",
            **panel_kw,
        )
        fig.update_yaxes(
            **panel_kw,
        )
        fig.update_layout(
            title={
                "text": "Channel Bitmask",
            },
        )
        return fig

    @classmethod
    def make_chan_data_heatmap(
        cls,
        name,
        data,
        trace_kw,
        fig=None,
        panel_kw=None,
    ):
        fig = fig or cls.make_subplots(1, 1)
        panel_kw = panel_kw or {}
        z = data[np.newaxis, :]
        y = [name]
        fig.add_heatmap(
            z=z,
            y=y,
            colorscale="rdylgn_r",
            **trace_kw,
            **panel_kw,
        )
        fig.update_xaxes(
            title="Channel Id",
            **panel_kw,
        )
        fig.update_layout(
            title={
                "text": name,
            },
        )
        return fig

    @classmethod
    def make_chunk_data_heatmap(
        cls,
        name,
        data,
        trace_kw,
        fig=None,
        panel_kw=None,
    ):
        fig = fig or cls.make_subplots(1, 1)
        panel_kw = panel_kw or {}
        z = data.T
        n_chunks = z.shape[0]
        i_center = (n_chunks - 1) // 2
        y = [f"{i}" if i != i_center else f"{name} {i}" for i in range(n_chunks)]
        fig.add_heatmap(
            z=z,
            y=y,
            colorscale="rdylgn_r",
            **trace_kw,
            **panel_kw,
        )
        fig.update_xaxes(
            title="Channel Id",
            **panel_kw,
        )
        fig.update_yaxes(
            tickvals=[i_center],
            **panel_kw,
        )
        fig.update_layout(
            title={
                "text": name,
            },
        )
        return fig

    @classmethod
    def make_s21_f(
        cls,
        data: MultiSweep,
        context: SweepCheckContext,
    ):
        swp = data
        ctd = context.data
        fig = cls.make_subplots(
            n_rows=4,
            n_cols=1,
            shared_xaxes="all",
            vertical_spacing=40 / 1000,
            fig_layout=cls.fig_layout_default
            | {
                "showlegend": False,
                "height": 1000,
            },
            row_heights=[0.5, 0.5, 1, 1],
        )
        d21_panel_kw = {"row": 1, "col": 1}
        d21_mask_panel_kw = {"row": 2, "col": 1}
        s21_panel_kw = {"row": 3, "col": 1}
        s21_mask_panel_kw = {"row": 4, "col": 1}
        # slider_panel_kw = {"row": 3, "col": 1}
        fs = swp.frequency.to_value(u.MHz)
        as21_db = swp.aS21_db
        as21_unc_db = swp.aS21_unc_db
        # ad21 = np.abs(swp.meta["d21"])
        # d21_unified = swp.meta["d21_unified"]
        ds = slice(None, None, 4)

        color_cycle = cls.color_palette.cycle_alternated(1, 0.25, 0.5, 0.25)
        # colors = []
        # for _ in range(fs.shape[0]):
        #     colors.extend([next(color_cycle)] * fs.shape[1])
        # s21_trace = {
        #     "x": fs[:, ds].ravel(),
        #     "y": as21_db[:, ds].ravel(),
        #     "mode": "markers",
        #     "marker": {
        #         "size": 4,
        #         "color": colors,
        #     },
        # }
        # fig.add_scattergl(
        #     **s21_trace,
        #     **s21_panel_kw,
        # )
        #
        mask_baseline = ctd.mask_baseline
        f_chunks = ctd.f_chunks.to_value(u.MHz)
        for ci in range(fs.shape[0]):
            color = next(color_cycle)
            color2 = next(color_cycle)
            color_arr = [color if m else "black" for m in mask_baseline[ci, ds]]
            fig.add_scattergl(
                x=fs[ci, ds],
                y=as21_db[ci, ds],
                error_y={
                    "type": "data",
                    "array": as21_unc_db[ci, ds],
                    "width": 0,
                    "color": color2,
                },
                mode="markers",
                marker={
                    "size": 4,
                    "color": color_arr,
                },
                **s21_panel_kw,
            )
            fig.add_scattergl(
                x=f_chunks[ci],
                y=ctd.chunk_skew[ci],
                mode="lines+markers",
                line={
                    "shape": "hvh",
                },
                marker={
                    "size": 4,
                },
                name="Chunk Skew",
                **s21_mask_panel_kw,
            )
            fig.add_scattergl(
                x=f_chunks[ci],
                y=ctd.chunk_kurtosis[ci],
                mode="lines+markers",
                line={
                    "shape": "hvh",
                },
                marker={
                    "size": 4,
                },
                name="Chunk Kurtosis",
                **s21_mask_panel_kw,
            )
            fig.add_scattergl(
                x=f_chunks[ci],
                y=ctd.mask_chunk_baseline[ci].astype(int) - (3 + 3 * (ci % 2)),
                mode="lines+markers",
                line={
                    "shape": "hvh",
                    "color": "black",
                },
                marker={
                    "size": 4,
                },
                name="Chunk Baseline",
                **s21_mask_panel_kw,
            )

            # fig.add_scatter(
            #     x=fs[ci, ds],
            #     y=ad21[ci, ds],
            #     mode="markers",
            #     marker={
            #         "size": 4,
            #         "color": color,
            #     },
            #     **d21_panel_kw,
            # )
        # add unified
        d21_trace = {
            "x": ctd.d21_frequency.to_value(u.MHz),
            "y": ctd.d21.to_value(u.Hz ** (-1)),
            "mode": "lines",
        }
        fig.add_scatter(
            **d21_trace,
            **d21_panel_kw,
        )
        d21_baseline_trace = {
            "x": ctd.d21_frequency.to_value(u.MHz),
            "y": ctd.d21_baseline.to_value(u.Hz ** (-1)),
            "mode": "lines",
        }
        fig.add_scatter(
            **d21_baseline_trace,
            **d21_panel_kw,
        )
        d21_chunk_baseline_trace = {
            "x": ctd.d21_f_chunks.to_value(u.MHz),
            "y": ctd.d21_chunk_baseline.to_value(u.Hz ** (-1)),
            "mode": "markers",
            "marker": {
                "color": "black",
                "size": 4,
            },
            "error_y": {
                "type": "data",
                "array": ctd.d21_chunk_baseline_rms.to_value(u.Hz ** (-1)),
                "width": 0,
                "color": "black",
            },
        }
        fig.add_scatter(
            **d21_chunk_baseline_trace,
            **d21_panel_kw,
        )

        fig.add_scatter(
            x=ctd.d21_frequency.to_value(u.MHz),
            y=ctd.d21_mask_baseline.astype(int),
            name="d21 baseline mask",
            mode="lines",
            line={
                "shape": "hvh",
            },
            # marker={
            #     "size": 4,
            # },
            showlegend=True,
            **d21_mask_panel_kw,
        )
        # range slider data
        # fig.add_scatter(
        #     **d21_unified_trace,
        #     **slider_panel_kw,
        # )
        fig.update_xaxes(
            # rangeslider={
            #     "autorange": True,
            #     "range": make_range(fs),
            #     "thickness": 0.05,
            # },
            title={
                "text": "Frequency (MHz)",
            },
            **s21_panel_kw,
        )
        fig.update_yaxes(
            title={
                "text": "|S21| (dB)",
            },
            **s21_panel_kw,
        )
        fig.update_yaxes(
            autorange=False,
            range=[-1.0, 5.0],
            title={
                "text": "|D21| (Hz^(-1))",
            },
            **d21_panel_kw,
        )
        return fig

    @classmethod
    def make_S21_f_grid(cls, data: MultiSweep, n_rows, n_cols, **kwargs):
        swp = data

        color_cycle = cls.color_palette.cycles(1, 0.5)

        def _make_s21_f_panel(row, col, panel_id, dummy=False, init=False):
            ci = panel_id

            if dummy:
                x = []
                y = []
                ey = []
                xrange = [0, 1]
                yrange = [0, 1]
                color1 = color2 = None
            else:
                x = swp.frequency[ci].to_value(u.MHz)
                y = swp.aS21_db[ci]
                ey = swp.aS21_unc_db[ci]
                xrange = make_range(x)
                yrange = make_range(y)
                color1, color2 = next(color_cycle)
            trace = {
                "type": "scatter",
                "x": x,
                "y": y,
                "error_y": {
                    "type": "data",
                    "array": ey,
                    "width": 0,
                    "color": color2,
                },
                "name": f"chan {ci}",
                "marker": {
                    "color": color1,
                },
            }
            layout = {
                "xaxis": {
                    "range": xrange,
                },
                "yaxis": {
                    "range": yrange,
                },
            }
            if init:
                rupdate(
                    trace,
                    {
                        "mode": "markers",
                        "marker": {
                            "size": 4,
                        },
                    },
                )
                if row == n_rows and col == 1:
                    rupdate(
                        layout,
                        {
                            "xaxis": {
                                "title": {
                                    "text": "Frequency (MHz)",
                                },
                            },
                            "yaxis": {
                                "title": {
                                    "text": "S21 (dB)",
                                },
                            },
                        },
                    )
                dm = 2
                y_tickvals = np.arange(
                    int(yrange[0] * dm) / dm - 20,
                    int(yrange[1] * dm) / dm + 20,
                    1 / dm,
                )
                rupdate(
                    layout,
                    {
                        "yaxis": {
                            "showgrid": True,
                            "gridcolor": "#dddddd",
                            "ticktext": [
                                f"{v:.0f}" if ((int(v * dm) % dm) == 0) else ""
                                for v in y_tickvals
                            ],
                            "tickvals": y_tickvals.tolist(),
                        },
                    },
                )
            return {
                "data": [trace],
                "layout": layout,
            }

        return cls.make_data_grid_anim(
            name="S21-f",
            n_rows=n_rows,
            n_cols=n_cols,
            n_items=swp.n_chans,
            make_panel_func=_make_s21_f_panel,
            **kwargs,
        )

    @classmethod
    def make_IQ_grid(
        cls,
        data: MultiSweep,
        n_rows,
        n_cols,
        n_chans_per_panel,
        **kwargs,
    ):
        swp = data

        color_cycle = cls.color_palette.cycles(1, 0.5)

        def _make_iq_panel(row, col, panel_id, dummy=False, init=False):
            ci0 = panel_id * n_chans_per_panel
            ci1 = ci0 + n_chans_per_panel

            def _make_trace(ci, init):
                if ci >= swp.n_chans:
                    x = []
                    y = []
                    ex = []
                    ey = []
                    color1, color2 = None, None
                else:
                    x = swp.I[ci].value
                    y = swp.Q[ci].value
                    ex = swp.I_unc[ci].value
                    ey = swp.Q_unc[ci].value
                color1, color2 = next(color_cycle)
                trace = {
                    "type": "scatter",
                    "x": x,
                    "y": y,
                    "error_x": {
                        "type": "data",
                        "array": ex,
                        "width": 0,
                        "color": color2,
                    },
                    "error_y": {
                        "type": "data",
                        "array": ey,
                        "width": 0,
                        "color": color2,
                    },
                    "name": f"chan {ci}",
                    "marker": {
                        "color": color1,
                    },
                }
                if init:
                    rupdate(
                        trace,
                        {
                            "mode": "markers",
                            "marker": {
                                "size": 4,
                            },
                        },
                    )
                return trace

            data = [_make_trace(ci, init=init) for ci in range(ci0, ci1)]
            if dummy:
                xrange = [0, 1]
                yrange = [0, 1]
            else:
                xr = make_range(swp.I[ci0:ci1].value)
                yr = make_range(swp.Q[ci0:ci1].value)
                xd = xr[1] - xr[0]
                yd = yr[1] - yr[0]
                if xd > yd:
                    xrange = xr
                    yrange = xr - np.mean(xr) + np.mean(yr)
                else:
                    xrange = yr - np.mean(yr) + np.mean(xr)
                    yrange = yr
            layout = {
                "xaxis": {
                    "range": xrange,
                },
                "yaxis": {
                    "range": yrange,
                },
            }
            if init:
                if row == n_rows and col == 1:
                    rupdate(
                        layout,
                        {
                            "xaxis": {
                                "title": {
                                    "text": "I (adu)",
                                },
                            },
                            "yaxis": {
                                "title": {
                                    "text": "Q (adu)",
                                },
                            },
                        },
                    )
                rupdate(
                    layout,
                    {
                        "yaxis": {
                            "scaleanchor": "x",
                            "scaleratio": 1,
                        },
                    },
                )
            return {
                "data": data,
                "layout": layout,
            }

        n_items = swp.n_chans // n_chans_per_panel + (
            (swp.n_chans % n_chans_per_panel) > 0
        )
        return cls.make_data_grid_anim(
            name="I-Q",
            n_rows=n_rows,
            n_cols=n_cols,
            n_items=n_items,
            make_panel_func=_make_iq_panel,
            **kwargs,
        )

    @classmethod
    def make_tone_freq_figure(cls, swp: MultiSweep, chan_ids=None, marker_color="blue"):
        """Create a tone frequency scatter plot.

        Parameters
        ----------
        swp : MultiSweep
            MultiSweep object containing tone frequency data in meta.
        chan_ids : array-like, optional
            Channel IDs for x-axis. If None, uses np.arange(swp.n_chans).
        marker_color : str, optional
            Color for markers. Default is "blue".

        Returns
        -------
        fig : go.Figure
            Plotly figure with tone frequency plot.
        """
        f_chan = swp.meta["chan_axis_data"]["f_chan"]

        if chan_ids is None:
            chan_ids = np.arange(swp.n_chans)

        fig = cls.make_subplots(1, 1)
        fig.add_scatter(
            x=chan_ids,
            y=f_chan.to_value(u.MHz)[chan_ids],
            mode="markers",
            marker={"size": 4, "color": marker_color},
            name="Chan Freq",
            row=1,
            col=1,
        )
        fig.update_yaxes(title={"text": "Chan Frequency (MHz)"}, row=1, col=1)
        fig.update_xaxes(title={"text": "Channel ID"}, row=1, col=1)
        return fig

    @classmethod
    def make_noise_psd_figure(cls, data: MultiSweep, context: SweepCheckContext):
        ctx0 = context
        ctd0 = ctx0.data

        # Get channel IDs
        n_chans = data.n_chans
        chan_ids = np.arange(n_chans)

        # Use grid utility similar to run function
        grid = cls.make_subplot_grid()

        # ========== Panel 1: I PSD Waterfall ==========
        fig_i_waterfall = cls.make_subplots(1, 1)
        fig_i_waterfall.add_heatmap(
            x=chan_ids,
            y=ctd0.f_psd,
            z=np.log10(
                ctd0.I_psd.T,
            ),  # Log scale and transpose to get (n_f_psd, n_chans)
            colorscale="Viridis",
            name="I PSD",
            colorbar={"title": "log10(I PSD)"},
            row=1,
            col=1,
        )
        fig_i_waterfall.update_yaxes(title={"text": "Frequency (Hz)"}, row=1, col=1)
        fig_i_waterfall.update_xaxes(title={"text": "Channel ID"}, row=1, col=1)
        grid.add_subplot(
            row=1,
            col=1,
            fig=fig_i_waterfall,
            row_height=1,
        )

        # ========== Panel 2: I PSD per-channel med/mad_std ==========
        fig_i_chan = cls.make_subplots(1, 1)
        fig_i_chan.add_scatter(
            x=chan_ids,
            y=ctd0.chan_I_psd_med,
            error_y={
                "type": "data",
                "array": ctd0.chan_I_psd_mad_std,
                "width": 0,
                "color": "gray",
            },
            mode="markers",
            marker={"size": 4, "color": "blue"},
            name="I PSD Med",
            row=1,
            col=1,
        )
        fig_i_chan.update_yaxes(
            title={"text": "I PSD (adu^2/Hz)"},
            type="log",
            row=1,
            col=1,
        )
        fig_i_chan.update_xaxes(title={"text": "Channel ID"}, row=1, col=1)
        grid.add_subplot(
            row=2,
            col=1,
            fig=fig_i_chan,
            row_height=0.5,
        )

        # ========== Panel 3: Q PSD Waterfall ==========
        fig_q_waterfall = cls.make_subplots(1, 1)
        fig_q_waterfall.add_heatmap(
            x=chan_ids,
            y=ctd0.f_psd,
            z=np.log10(
                ctd0.Q_psd.T,
            ),  # Log scale and transpose to get (n_f_psd, n_chans)
            colorscale="Viridis",
            name="Q PSD",
            colorbar={"title": "log10(Q PSD)"},
            row=1,
            col=1,
        )
        fig_q_waterfall.update_yaxes(title={"text": "Frequency (Hz)"}, row=1, col=1)
        fig_q_waterfall.update_xaxes(title={"text": "Channel ID"}, row=1, col=1)
        grid.add_subplot(
            row=3,
            col=1,
            fig=fig_q_waterfall,
            row_height=1,
        )

        # ========== Panel 4: Q PSD per-channel med/mad_std ==========
        fig_q_chan = cls.make_subplots(1, 1)
        fig_q_chan.add_scatter(
            x=chan_ids,
            y=ctd0.chan_Q_psd_med,
            error_y={
                "type": "data",
                "array": ctd0.chan_Q_psd_mad_std,
                "width": 0,
                "color": "gray",
            },
            mode="markers",
            marker={"size": 4, "color": "green"},
            name="Q PSD Med",
            row=1,
            col=1,
        )
        fig_q_chan.update_yaxes(
            title={"text": "Q PSD (adu^2/Hz)"},
            type="log",
            row=1,
            col=1,
        )
        fig_q_chan.update_xaxes(title={"text": "Channel ID"}, row=1, col=1)
        grid.add_subplot(
            row=4,
            col=1,
            fig=fig_q_chan,
            row_height=0.5,
        )

        # ========== Panel 5: Sphi Waterfall ==========
        fig_sphi_waterfall = cls.make_subplots(1, 1)
        fig_sphi_waterfall.add_heatmap(
            x=chan_ids,
            y=ctd0.f_psd,
            z=np.log10(
                ctd0.Sphi.T,
            ),  # Log scale and transpose to get (n_f_psd, n_chans)
            colorscale="Viridis",
            name="Sphi",
            colorbar={"title": "log10(Sphi)"},
            row=1,
            col=1,
        )
        fig_sphi_waterfall.update_yaxes(title={"text": "Frequency (Hz)"}, row=1, col=1)
        fig_sphi_waterfall.update_xaxes(title={"text": "Channel ID"}, row=1, col=1)
        grid.add_subplot(
            row=5,
            col=1,
            fig=fig_sphi_waterfall,
            row_height=1,
        )

        # ========== Panel 6: Sphi per-channel med/mad_std ==========
        fig_sphi_chan = cls.make_subplots(1, 1)
        fig_sphi_chan.add_scatter(
            x=chan_ids,
            y=ctd0.chan_Sphi_med,
            error_y={
                "type": "data",
                "array": ctd0.chan_Sphi_mad_std,
                "width": 0,
                "color": "gray",
            },
            mode="markers",
            marker={"size": 4, "color": "red"},
            name="Sphi Med",
            row=1,
            col=1,
        )
        fig_sphi_chan.update_yaxes(
            title={"text": "Sphi (1/Hz)"},
            type="log",
            row=1,
            col=1,
        )
        fig_sphi_chan.update_xaxes(title={"text": "Channel ID"}, row=1, col=1)
        grid.add_subplot(
            row=6,
            col=1,
            fig=fig_sphi_chan,
            row_height=0.5,
        )

        # ========== Panel 7: Channel Frequency ==========
        fig_f_chan = cls.make_tone_freq_figure(
            data,
            chan_ids=chan_ids,
            marker_color="purple",
        )
        grid.add_subplot(
            row=7,
            col=1,
            fig=fig_f_chan,
            row_height=0.5,
        )

        # Create the final figure using grid utility
        fig = grid.make_figure(
            shared_xaxes="all",
            vertical_spacing=40 / 1800,
            fig_layout={
                "height": 1800,
                "showlegend": False,
            },
        )

        # Add a range slider on the bottom panel
        fig.update_xaxes(
            rangeslider={
                "autorange": True,
                "range": [0, n_chans],
                "thickness": 0.05,
            },
            row=grid.shape[0],
            col=1,
        )

        return fig
