"""Persisted sweep-check statistics and corrected sweep views."""

from __future__ import annotations

from dataclasses import dataclass

from tollan.accessor import NameMapping, Schema, SubpathMapping
from tollan.accessor.xarray import XarrayAccessorBase, XarrayMapper
from tollan.utils.fmt import BitmaskStats

from .sweep import ReducedSweepView


@dataclass
class SweepCheckSchema(Schema):
    """Sweep-check result datasets, statistics and producing settings."""

    config: NameMapping = NameMapping.attr("tolteca_kids.sweep_check.config")
    sweep: SubpathMapping = SubpathMapping.path("sweep")
    samples_subpath: SubpathMapping = SubpathMapping.path("samples")
    mask_spike: NameMapping = samples_subpath / NameMapping.var("mask_spike")
    bitmask: NameMapping = samples_subpath / NameMapping.var("bitmask")
    mask_baseline: NameMapping = samples_subpath / NameMapping.var("mask_baseline")
    S21_spike: NameMapping = samples_subpath / NameMapping.var("S21_spike")
    S21_orig: NameMapping = samples_subpath / NameMapping.var("S21_orig")
    channels_subpath: SubpathMapping = SubpathMapping.path("channels")
    chan_dim: NameMapping = channels_subpath / NameMapping.dim("chan")
    bitmask_chan: NameMapping = channels_subpath / NameMapping.var("bitmask_chan")
    mask_chan_bad: NameMapping = channels_subpath / NameMapping.var("mask_chan_bad")
    chan_range: NameMapping = channels_subpath / NameMapping.var("chan_range")
    chan_level: NameMapping = channels_subpath / NameMapping.var("chan_level")
    chan_rms_mean: NameMapping = channels_subpath / NameMapping.var("chan_rms_mean")
    chan_rms_std: NameMapping = channels_subpath / NameMapping.var("chan_rms_std")
    chunks_subpath: SubpathMapping = SubpathMapping.path("chunks")
    f_chunks: NameMapping = chunks_subpath / NameMapping.var("f_chunks")
    bitmask_chunk: NameMapping = chunks_subpath / NameMapping.var("bitmask_chunk")
    mask_chunk_baseline: NameMapping = chunks_subpath / NameMapping.var(
        "mask_chunk_baseline"
    )
    chunk_rms_mean: NameMapping = chunks_subpath / NameMapping.var("chunk_rms_mean")
    chunk_rms_std: NameMapping = chunks_subpath / NameMapping.var("chunk_rms_std")
    chunk_skew: NameMapping = chunks_subpath / NameMapping.var("chunk_skew")
    chunk_kurtosis: NameMapping = chunks_subpath / NameMapping.var("chunk_kurtosis")
    chunk_windows: NameMapping = chunks_subpath / NameMapping.var("chunk_windows")
    chunk_bounds: NameMapping = chunks_subpath / NameMapping.var("chunk_bounds")
    d21_subpath: SubpathMapping = SubpathMapping.path("d21")
    d21_mask_baseline: NameMapping = d21_subpath / NameMapping.var("d21_mask_baseline")
    d21_baseline: NameMapping = d21_subpath / NameMapping.var("d21_baseline")
    d21_baseline_rms: NameMapping = d21_subpath / NameMapping.var("d21_baseline_rms")
    d21_frequency: NameMapping = d21_subpath / NameMapping.var("d21_frequency")
    d21: NameMapping = d21_subpath / NameMapping.var("d21")
    d21_detrended: NameMapping = d21_subpath / NameMapping.var("d21_detrended")
    d21_chunks_subpath: SubpathMapping = SubpathMapping.path("d21_chunks")
    d21_f_chunks: NameMapping = d21_chunks_subpath / NameMapping.var("d21_f_chunks")
    d21_chunk_baseline: NameMapping = d21_chunks_subpath / NameMapping.var(
        "d21_chunk_baseline"
    )
    d21_chunk_baseline_rms: NameMapping = d21_chunks_subpath / NameMapping.var(
        "d21_chunk_baseline_rms"
    )
    d21_chunk_windows: NameMapping = d21_chunks_subpath / NameMapping.var(
        "d21_chunk_windows"
    )
    d21_chunk_bounds: NameMapping = d21_chunks_subpath / NameMapping.var(
        "d21_chunk_bounds"
    )
    chunk_size: NameMapping = NameMapping.attr("chunk_size")
    n_chunks_per_chan: NameMapping = NameMapping.attr("n_chunks_per_chan")
    d21_chunk_size: NameMapping = NameMapping.attr("d21_chunk_size")
    d21_n_chunks: NameMapping = NameMapping.attr("d21_n_chunks")
    swp_rms_med: NameMapping = NameMapping.attr("swp_rms_med")
    swp_rms_rms: NameMapping = NameMapping.attr("swp_rms_rms")


class SweepCheckMapper(XarrayMapper[SweepCheckSchema]):
    """Access to sweep-check result arrays and attributes."""


class SweepCheckView(XarrayAccessorBase[SweepCheckMapper]):
    """Quality flags, statistics and corrected data for one sweep block.

    Parameters
    ----------
    data_source : xarray.DataTree
        The sweep-check namespace, obtained with ``SweepCheck.view``.
        Keep the source open while reading lazy arrays.
    mapper : SweepCheckMapper, optional
        Result mapper. Omitted by default, resolving against the supplied tree.

    Notes
    -----
    Numerical properties load the requested arrays. Use ``mapper.get_arr`` on
    ``data_source`` to select channels before loading larger result arrays.
    """

    @property
    def data_source(self):
        """Sweep-check result namespace."""
        return self._data_source

    def _validate(self):
        self.mapper.get_arr(self.data_source, self.mapper.schema.bitmask_chan)
        self.mapper.get_arr(self.data_source, self.mapper.schema.d21)

    @property
    def sweep(self):
        """Corrected sweep with the original uncertainties and frequency axes."""
        return ReducedSweepView(
            self.mapper.get_node(self.data_source, self.mapper.schema.sweep)
        )

    @property
    def config(self):
        """Settings used to compute this result."""
        from .sweep_check import SweepCheckConfig  # noqa: PLC0415

        return SweepCheckConfig.model_validate_json(
            self.mapper.get_value(self.data_source, self.mapper.schema.config)
        )

    @property
    def S21_orig(self):  # noqa: N802 -- scientific notation
        """Original complex sweep samples."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.S21_orig
        ).u.quantity

    @property
    def S21_spike(self):  # noqa: N802 -- scientific notation
        """Removed complex spike signal, or None when despiking was disabled."""
        if self.mapper.schema.S21_spike not in self.mapper:
            return None
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.S21_spike
        ).u.quantity

    @property
    def mask_spike(self):
        """Detected spike samples, indexed by channel and probe."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.mask_spike
        ).values

    @property
    def bitmask(self):
        """Per-sample quality flags, indexed by channel and probe."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.bitmask).values

    @property
    def mask_baseline(self):
        """Samples selected as baseline, indexed by channel and probe."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.mask_baseline
        ).values

    @property
    def bitmask_chan(self):
        """Combined quality flags for each channel."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.bitmask_chan
        ).values

    @property
    def mask_chan_bad(self):
        """Channels rejected by the configured bad-channel bits."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.mask_chan_bad
        ).values

    @property
    def chan_range(self):
        """Median-filtered amplitude range per channel, in dB."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.chan_range
        ).values

    @property
    def chan_level(self):
        """Mean amplitude per channel, in dB."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.chan_level
        ).values

    @property
    def chan_rms_mean(self):
        """Mean amplitude uncertainty per channel, in dB, excluding spikes."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.chan_rms_mean
        ).values

    @property
    def chan_rms_std(self):
        """Amplitude-uncertainty standard deviation per channel, in dB."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.chan_rms_std
        ).values

    @property
    def f_chunks(self):
        """Median frequency of each channel chunk, in Hz."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.f_chunks
        ).u.quantity

    @property
    def bitmask_chunk(self):
        """Quality flags for each channel chunk."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.bitmask_chunk
        ).values

    @property
    def mask_chunk_baseline(self):
        """Chunks selected as baseline for each channel."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.mask_chunk_baseline
        ).values

    @property
    def chunk_rms_mean(self):
        """Mean amplitude uncertainty for each channel chunk, in dB."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.chunk_rms_mean
        ).values

    @property
    def chunk_rms_std(self):
        """Amplitude-uncertainty standard deviation for each chunk, in dB."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.chunk_rms_std
        ).values

    @property
    def chunk_skew(self):
        """Quadrature mean of I/Q skewness in each channel chunk."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.chunk_skew
        ).values

    @property
    def chunk_kurtosis(self):
        """Maximum I/Q excess kurtosis in each channel chunk."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.chunk_kurtosis
        ).values

    @property
    def chunk_windows(self):
        """Probe indices in each overlapping statistics window."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.chunk_windows
        ).values

    @property
    def d21_mask_baseline(self):
        """Baseline selection on the unified frequency grid."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.d21_mask_baseline
        ).values

    @property
    def d21_baseline(self):
        """Interpolated D21 baseline, in inverse Hz."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.d21_baseline
        ).u.quantity

    @property
    def d21_baseline_rms(self):
        """Interpolated D21 baseline RMS, in inverse Hz."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.d21_baseline_rms
        ).u.quantity

    @property
    def d21_frequency(self):
        """Unified D21 frequency grid, in Hz."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.d21_frequency
        ).u.quantity

    @property
    def d21(self):
        """Unified magnitude of the normalized S21 derivative, in inverse Hz."""
        return self.mapper.get_arr(self.data_source, self.mapper.schema.d21).u.quantity

    @property
    def d21_detrended(self):
        """Unified D21 after baseline subtraction, in inverse Hz."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.d21_detrended
        ).u.quantity

    @property
    def d21_f_chunks(self):
        """Median frequencies of D21 chunks, in Hz."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.d21_f_chunks
        ).u.quantity

    @property
    def d21_chunk_baseline(self):
        """Median baseline D21 in each chunk, in inverse Hz."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.d21_chunk_baseline
        ).u.quantity

    @property
    def d21_chunk_baseline_rms(self):
        """Baseline D21 scaled MAD in each chunk, in inverse Hz."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.d21_chunk_baseline_rms
        ).u.quantity

    @property
    def d21_chunk_windows(self):
        """Unified frequency indices in each D21 statistics window."""
        return self.mapper.get_arr(
            self.data_source, self.mapper.schema.d21_chunk_windows
        ).values

    @property
    def chunk_size(self):
        """Number of probe samples per statistics window."""
        return self.mapper.get_value(self.data_source, self.mapper.schema.chunk_size)

    @property
    def n_chunks_per_chan(self):
        """Number of statistics windows per channel."""
        return self.mapper.get_value(
            self.data_source, self.mapper.schema.n_chunks_per_chan
        )

    @property
    def d21_chunk_size(self):
        """Number of unified frequency samples per D21 statistics window."""
        return self.mapper.get_value(
            self.data_source, self.mapper.schema.d21_chunk_size
        )

    @property
    def d21_n_chunks(self):
        """Number of D21 statistics windows."""
        return self.mapper.get_value(self.data_source, self.mapper.schema.d21_n_chunks)

    @property
    def swp_rms_med(self):
        """Median of the per-channel mean amplitude uncertainties, in dB."""
        return self.mapper.get_value(self.data_source, self.mapper.schema.swp_rms_med)

    @property
    def swp_rms_rms(self):
        """Scaled MAD of the per-channel mean amplitude uncertainties, in dB."""
        return self.mapper.get_value(self.data_source, self.mapper.schema.swp_rms_rms)

    @property
    def chunk_slices(self):
        """Nonoverlapping sample bounds for chunk aggregation."""
        bounds = self.mapper.get_arr(
            self.data_source, self.mapper.schema.chunk_bounds
        ).values
        return [slice(int(start), int(stop)) for start, stop in bounds]

    @property
    def d21_chunk_slices(self):
        """Nonoverlapping sample bounds for chunk aggregation."""
        bounds = self.mapper.get_arr(
            self.data_source, self.mapper.schema.d21_chunk_bounds
        ).values
        return [slice(int(start), int(stop)) for start, stop in bounds]

    @property
    def bitmask_chan_stats(self):
        """Counts for each sweep quality flag."""
        from .sweep_check import SweepBitMask  # noqa: PLC0415

        return BitmaskStats(SweepBitMask, self.bitmask_chan).stats

    @property
    def bitmask_chunk_stats(self):
        """Counts for each sweep quality flag."""
        from .sweep_check import SweepBitMask  # noqa: PLC0415

        return BitmaskStats(SweepBitMask, self.bitmask_chunk).stats
