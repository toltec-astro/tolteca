"""TolTEC sweep quality checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag, auto
from typing import Annotated, Literal, cast

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats
import xarray as xr
from astropy.stats import mad_std, sigma_clip
from numpy.lib.stride_tricks import sliding_window_view
from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationInfo,
    field_validator,
)
from scipy.ndimage import binary_erosion, median_filter
from tollan.pipeline import Step, StepConfig, StepContext, StepFailure, StepFailureKind
from tollan.utils.fmt import BitmaskStats, pformat_mask
from tollan.utils.log import logger, timeit

from tolteca_kidsproc.analysis.d21 import D21Analysis

from ._reduction import calculation_settings, select_block
from .roach_tone import READOUT_ELEMENTS, RoachReadoutPowerConfig, readout_power
from .sweep import ReducedSweepView
from .sweep_check_view import SweepCheckMapper, SweepCheckView

__all__ = [
    "DespikeMethod",
    "SweepBitMask",
    "SweepCheck",
    "SweepCheckConfig",
    "SweepCheckContext",
    "SweepCheckView",
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

    unc_nonfinite = auto()
    """Nonfinite original mean I/Q or repeat scatter; per-step diagnostic only."""

    unc_low = auto()
    """Repeat scatter is below the absolute limit; per-step diagnostic only."""

    unc_high = auto()
    """Repeat scatter exceeds the absolute limit; per-step diagnostic only."""

    unc_outlier = auto()
    """Repeat scatter is a sigma-clipped outlier; per-step diagnostic only."""


def _threshold_from_json(value):
    """Restore nonfinite thresholds from their JSON string representations."""
    if isinstance(value, str):
        return {"Infinity": np.inf, "-Infinity": -np.inf, "NaN": np.nan}.get(
            value, value
        )
    return value


def _threshold_to_json(value):
    """Keep unbounded thresholds distinct from omitted values in saved settings."""
    if np.isfinite(value):
        return value
    return "NaN" if np.isnan(value) else "Infinity" if value > 0 else "-Infinity"


_Threshold = Annotated[
    float,
    BeforeValidator(
        _threshold_from_json,
        json_schema_input_type=float | Literal["Infinity", "-Infinity", "NaN"],
    ),
    PlainSerializer(_threshold_to_json, when_used="json"),
]


class SweepCheckConfig(StepConfig):
    """Configuration for the sweep quality check step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    block: int = Field(
        default=-1,
        ge=-1,
        description="Block exposed by the context; all sweep blocks are checked.",
    )
    force: bool = Field(default=False, description="Recompute existing check results.")
    readout_power: RoachReadoutPowerConfig | None = Field(
        default_factory=RoachReadoutPowerConfig,
        description=(
            "Acquired readout-power calibration. Requires recorded probe amplitudes "
            "and attenuation. ADC usage is included when a snapshot is recorded; "
            "None skips these diagnostics."
        ),
    )

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
    chan_range_db_min: _Threshold = Field(
        default=0.1,
        description="Channel data range smaller than this is flagged.",
    )
    chan_range_db_max: _Threshold = Field(
        default=np.inf,
        json_schema_extra={"default": "Infinity"},
        description="Channel data range larger than this is flagged.",
    )
    chan_level_db_min: _Threshold = Field(
        default=-np.inf,
        json_schema_extra={"default": "-Infinity"},
        description="Channel data level lower than this is flagged.",
    )
    chan_level_db_max: _Threshold = Field(
        default=np.inf,
        json_schema_extra={"default": "Infinity"},
        description="Channel data level higher than this is flagged.",
    )
    chan_rms_db_min: _Threshold = Field(
        default=0.001,
        description="Channel data RMS lower than this is flagged.",
    )
    chan_rms_db_max: _Threshold = Field(
        default=0.1,
        description="Channel data RMS higher than this is flagged.",
    )
    step_unc_min: float = Field(
        default=0.0,
        ge=0,
        allow_inf_nan=False,
        description=(
            "Flag finite I or Q repeat scatter below this absolute limit, in "
            "the reduced I/Q units. Diagnostic only."
        ),
    )
    step_unc_max: _Threshold = Field(
        default=np.inf,
        ge=0,
        json_schema_extra={"default": "Infinity"},
        description=(
            "Flag finite I or Q repeat scatter above this absolute limit, in "
            "the reduced I/Q units. Infinity disables the upper limit. Diagnostic only."
        ),
    )
    step_unc_outlier_n_sigma_min: float = Field(
        default=15.0,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Two-sided sigma-clipping threshold for repeat scatter across frequency "
            "steps, separately per channel and quadrature. Uses a median center, "
            "MAD-based scale and up to five iterations. Diagnostic only; does not "
            "reject data or change fit weights."
        ),
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
        strict=False,
        default=(SweepBitMask.rms_low | SweepBitMask.rms_high),
        description="Bits used to generate channel bad mask.",
    )
    not_baseline_chunk_bits: SweepBitMask = Field(
        strict=False,
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

    @field_validator("step_unc_max")
    @classmethod
    def _validate_unc_max(cls, value: float, info: ValidationInfo) -> float:
        minimum = info.data.get("step_unc_min")
        if minimum is not None and minimum > value:
            msg = "step_unc_min must not exceed step_unc_max"
            raise ValueError(msg)
        return value


@dataclass(kw_only=True)
class _SweepCheckData:
    """Sweep quality statistics, flags, and diagnostics."""

    # spike
    mask_spike: npt.NDArray = field(default_factory=lambda: np.empty(0))
    S21_checked: npt.NDArray = field(default_factory=lambda: np.empty(0))
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
    mask_baseline: npt.NDArray = field(default_factory=lambda: np.empty(0, dtype=bool))

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
    d21_baseline: u.Quantity = field(default_factory=lambda: np.empty(0) << u.Hz**-1)
    d21_baseline_rms: u.Quantity = field(
        default_factory=lambda: np.empty(0) << u.Hz**-1
    )
    d21_frequency: u.Quantity = field(default_factory=lambda: np.empty(0) << u.Hz)
    d21: u.Quantity = field(default_factory=lambda: np.empty(0) << u.Hz**-1)
    d21_detrended: u.Quantity = field(default_factory=lambda: np.empty(0) << u.Hz**-1)

    # per-channel stats
    chan_range: npt.NDArray = field(default_factory=lambda: np.empty(0))
    chan_level: npt.NDArray = field(default_factory=lambda: np.empty(0))
    chan_rms_mean: npt.NDArray = field(default_factory=lambda: np.empty(0))
    chan_rms_std: npt.NDArray = field(default_factory=lambda: np.empty(0))
    chan_unc_I_ref: npt.NDArray = field(default_factory=lambda: np.empty(0))  # noqa: N815 -- physical I/Q
    chan_unc_Q_ref: npt.NDArray = field(default_factory=lambda: np.empty(0))  # noqa: N815 -- physical I/Q
    chan_unc_I_std: npt.NDArray = field(default_factory=lambda: np.empty(0))  # noqa: N815 -- physical I/Q
    chan_unc_Q_std: npt.NDArray = field(default_factory=lambda: np.empty(0))  # noqa: N815 -- physical I/Q

    chunk_rms_mean: npt.NDArray = field(default_factory=lambda: np.empty(0))
    chunk_rms_std: npt.NDArray = field(default_factory=lambda: np.empty(0))
    chunk_skew: npt.NDArray = field(default_factory=lambda: np.empty(0))
    chunk_kurtosis: npt.NDArray = field(default_factory=lambda: np.empty(0))

    swp_rms_med: float = 0.0
    swp_rms_rms: float = 0.0
    readout_power: xr.Dataset | None = None


class SweepCheckContext(StepContext["SweepCheck", SweepCheckConfig]):
    """Context for the sweep quality check step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: SweepCheckView | None = Field(default=None, exclude=True)
    block: int = -1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _SweepArrays:
    """Numpy arrays extracted from a ReducedSweepView for the algorithm."""

    S21_unc: npt.NDArray  # complex I/Q uncertainty [n_chans, n_steps]
    unc_I: npt.NDArray  # noqa: N815 -- physical I/Q; [n_chans, n_steps]
    unc_Q: npt.NDArray  # noqa: N815 -- physical I/Q; [n_chans, n_steps]
    S21: npt.NDArray  # complex [n_chans, n_steps]
    aS21: npt.NDArray  # noqa: N815 -- scientific notation; [n_chans, n_steps]
    aS21_db: npt.NDArray  # noqa: N815 -- scientific notation; [n_chans, n_steps]
    aS21_unc_db: npt.NDArray  # noqa: N815 -- scientific notation; [n_chans, n_steps] — uncertainty in dB
    aS21_unc: npt.NDArray  # noqa: N815 -- scientific notation; [n_chans, n_steps] — uncertainty in linear amplitude
    frequency: u.Quantity  # [n_chans, n_steps] in Hz
    n_chans: int
    n_steps: int


def _extract_sweep_arrays(dt: xr.Dataset | xr.DataTree) -> _SweepArrays:
    """Extract numpy arrays from reduced sweep DataTree for SweepCheck.

    Parameters
    ----------
    dt : xarray.Dataset or xarray.DataTree
        Reduced dataset or a block containing it in its ``sweep`` child.

    Returns
    -------
    _SweepArrays
        Struct of numpy/Quantity arrays for the SweepCheck algorithm.
    """
    view = ReducedSweepView(dt)

    I_da = view.I
    Q_da = view.Q
    if I_da is None or Q_da is None:
        msg = "ReducedSweepView has no I/Q data."
        raise ValueError(msg)

    I_np = I_da.to_numpy()  # [chan, sweep]
    Q_np = Q_da.to_numpy()  # [chan, sweep]
    S21_np = I_np + 1j * Q_np
    aS21_np = np.abs(S21_np)
    aS21_db = 20.0 * np.log10(aS21_np)

    # Production v2 uses the magnitude of the complex I/Q uncertainty.
    unc_I_da, unc_Q_da = view.unc_I, view.unc_Q
    if unc_I_da is None or unc_Q_da is None:
        msg = "Sweep check requires I/Q uncertainties"
        raise ValueError(msg)
    unc_I, unc_Q = unc_I_da.to_numpy(), unc_Q_da.to_numpy()
    uncertainty = unc_I + 1j * unc_Q
    sigma_abs = np.abs(uncertainty)
    aS21_unc_db = 20 * np.log10(np.e) * np.abs(uncertainty / S21_np)
    frequency_da = view.frequency
    if frequency_da is None:
        msg = "Sweep check requires absolute sweep frequencies"
        raise ValueError(msg)
    frequency = frequency_da.u.quantity.to(u.Hz)
    n_chans, n_steps = I_np.shape

    return _SweepArrays(
        S21=S21_np,
        S21_unc=uncertainty,
        unc_I=unc_I,
        unc_Q=unc_Q,
        aS21=aS21_np,
        aS21_db=aS21_db,
        aS21_unc_db=aS21_unc_db,
        aS21_unc=sigma_abs,
        frequency=frequency,
        n_chans=n_chans,
        n_steps=n_steps,
    )


def _step_unc_flags(signal, unc_I, unc_Q, *, limits, n_sigma):  # noqa: N803 -- physical I/Q names
    """Flag reduced scatter without rejecting measurements or reading raw repeats."""
    flags = np.zeros(signal.shape, dtype=int)
    nonfinite = ~np.isfinite(signal) | ~np.isfinite(unc_I) | ~np.isfinite(unc_Q)
    flags[nonfinite] |= int(SweepBitMask.unc_nonfinite)
    references, scales = [], []
    for scatter in (unc_I, unc_Q):
        finite = np.isfinite(scatter)
        flags[finite & (scatter < limits[0])] |= int(SweepBitMask.unc_low)
        flags[finite & (scatter > limits[1])] |= int(SweepBitMask.unc_high)
        # Absolute limits do not determine the statistical reference population.
        valid = finite & (scatter >= 0)
        clipped, lower, upper = sigma_clip(
            np.ma.array(scatter, mask=~valid, copy=False),
            sigma=n_sigma,
            maxiters=5,
            cenfunc="median",
            stdfunc="mad_std",
            axis=1,
            masked=True,
            return_bounds=True,
        )
        flags[valid & np.ma.getmaskarray(clipped)] |= int(SweepBitMask.unc_outlier)
        references.append((lower + upper) / 2)
        scales.append((upper - lower) / (2 * n_sigma))
    return flags, *references, *scales


@dataclass
class _D21SweepAdapter:
    """Sweep data for D21 analysis."""

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

    Run after ``SweepReduce``. Pass an acquisition or collection to check every
    reduced sweep block, or a block node to check only that block. On an
    acquisition, the ``block`` config field selects the view exposed by the
    context. Use ``SweepCheck.view`` to read checks from a reopened product.

    Examples
    --------
    For a working acquisition tree ``dt``:

    >>> SweepCheck()(dt)  # doctest: +SKIP
    >>> checked = SweepCheck.view(dt, block=0)  # doctest: +SKIP
    >>> mask_bad = checked.mask_chan_bad  # doctest: +SKIP
    >>> d21 = checked.d21_detrended  # doctest: +SKIP

    Check only one block and retrieve its execution context:

    >>> block = dt["blocks/0"]  # doctest: +SKIP
    >>> SweepCheck()(block)  # doctest: +SKIP
    >>> context = SweepCheck.get_context(block)  # doctest: +SKIP
    """

    @classmethod
    def result_path(cls, *, alias: str | None = None) -> str:
        """Return the result subpath within a block.

        Parameters
        ----------
        alias : str or None, optional
            Invocation alias; None uses the primary check namespace.

        Returns
        -------
        str
            Relative path containing this check's saved datasets.
        """
        return "sweep_check" if alias is None else f"sweep_check_{alias}"

    @classmethod
    def view(
        cls, data: xr.DataTree, *, block: int = -1, alias: str | None = None
    ) -> SweepCheckView:
        """Read saved check results from a block or acquisition.

        Parameters
        ----------
        data : xarray.DataTree
            Working or reopened acquisition, collection, or selected block.
        block : int, optional
            Acquisition block ID. Default -1 selects the last block.
        alias : str or None, optional
            Check invocation label; None selects the primary check.

        Returns
        -------
        SweepCheckView
            Read-only result view, independent of live execution contexts.
        """
        scope = data if "sweep" in data.children else select_block(data, block).tree
        return SweepCheckView(scope[cls.result_path(alias=alias)])

    @classmethod
    def run(cls, data: xr.DataTree, context: SweepCheckContext) -> None:
        """Check reduced sweep blocks and retain their scientific results.

        Parameters
        ----------
        data : xarray.DataTree
            Acquisition, collection, or block with reduced sweep data. The original
            reduction is retained. Results replace the check's block namespace.
        context : SweepCheckContext
            Settings and status. Its data exposes the selected completed block.

        Returns
        -------
        None
            Each block retains check arrays; the context publishes a live view.

        Raises
        ------
        ValueError
            Readout-power diagnostics are enabled but recorded probe properties,
            ROACH identity or attenuator settings are absent. For standalone
            reduced data without acquisition inputs, set readout_power=None.
        """
        cfg = context.config
        if "sweep" in data.children:
            scopes = [data]
            selected = data
            meta = ReducedSweepView(data).meta
            context.block = meta.block if meta is not None else 0
        else:
            selected_block = select_block(data, cfg.block)
            selected = selected_block.tree
            context.block = selected_block.meta.block
            scopes = [b.tree for b in data.toltec_kids.blocks]
        path = cls.result_path(alias=cfg.alias)
        for scope in scopes:
            if "sweep" not in scope.children:
                context.failures.append(
                    StepFailure(
                        kind=StepFailureKind.missing_input,
                        message="Sweep check requires a reduced sweep.",
                        details={"path": scope.path},
                    )
                )
                continue
            if not cfg.force and path in scope.children:
                previous = SweepCheckView(scope[path])
                if (
                    "step_unc_outlier_n_sigma_min" in previous.config.model_fields_set
                    and calculation_settings(previous.config)
                    == calculation_settings(cfg)
                ):
                    continue
            values = cls._compute(scope, cfg)
            result = _make_check_result(values, ReducedSweepView(scope), cfg)
            scope[path] = result
        if path in selected.children:
            context.data = SweepCheckView(selected[path])

    @classmethod
    @timeit
    def _compute(cls, data, cfg):  # noqa: C901, PLR0912, PLR0915 -- retain the production calculation sequence
        """Compute the production-v2 sweep statistics without mutating input data."""
        ctd = _SweepCheckData()

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

        # Independent diagnostics on the original reduced measurements. These
        # bits never enter the v2 selection masks or corrected-signal path.
        (
            bitmask_unc,
            ctd.chan_unc_I_ref,
            ctd.chan_unc_Q_ref,
            ctd.chan_unc_I_std,
            ctd.chan_unc_Q_std,
        ) = _step_unc_flags(
            S21,
            arrays.unc_I,
            arrays.unc_Q,
            limits=(cfg.step_unc_min, cfg.step_unc_max),
            n_sigma=cfg.step_unc_outlier_n_sigma_min,
        )

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
                    aS21_db = 20.0 * np.log10(np.abs(S21))
                    aS21_unc_db = 20 * np.log10(np.e) * np.abs(arrays.S21_unc / S21)
        else:
            S21_spike = None

        ctd.S21_checked = S21
        ctd.mask_spike = mask_spike
        ctd.S21_spike = S21_spike
        if cfg.readout_power is not None:
            ctd.readout_power = _readout_power_result(data, S21, cfg.readout_power)

        # ----------------------------------------------------------------
        # D21 analysis
        # ----------------------------------------------------------------
        d21_adapter = _make_d21_adapter(
            _SweepArrays(
                S21=S21,
                S21_unc=arrays.S21_unc,
                unc_I=arrays.unc_I,
                unc_Q=arrays.unc_Q,
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
        bitmask |= bitmask_unc

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
            d21_chunk_baseline_rms = cast(
                "npt.NDArray",
                mad_std(_chunk_data_rms, axis=-1, ignore_nan=True),
            )
            d21_chunk_baseline_rms[_all_nan] = np.nan
            if np.all(np.isnan(d21_chunk_baseline_rms)):
                d21_chunk_baseline_rms = np.full_like(d21_chunk_baseline_rms, 0.1)
            m_pos = d21_chunk_baseline_rms > 0
            d21_chunk_baseline_rms[d21_chunk_baseline_rms <= 0] = np.min(
                d21_chunk_baseline_rms[m_pos]
            )
            ctd.d21_chunk_baseline_rms = d21_chunk_baseline_rms << u.Hz**-1

            m = ~(np.isnan(d21_chunk_baseline_value) | np.isnan(d21_chunk_baseline_rms))
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

        return ctd

    # ----------------------------------------------------------------
    # Static helpers (ported unchanged from v2)
    # ----------------------------------------------------------------

    @staticmethod
    def make_data_mask_from_unified(
        fs: u.Quantity,
        fs_unified: u.Quantity,
        mask_unified: npt.NDArray,
    ) -> npt.NDArray:
        """Return the unified mask at each frequency's insertion index.

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
        """Select overlapping statistics windows and disjoint sample chunks.

        Parameters
        ----------
        n_items : int
            Number of samples along the channel or unified frequency axis.
        chunk_size : int
            Requested number of samples per statistics window.
        n_chunks_min : int
            Minimum requested number of windows, with a central window included.

        Returns
        -------
        dict
            Window indices in ``chunk_windows``, disjoint sample selections in
            ``chunk_slices``, per-sample chunk assignments in ``chunk_index``,
            and the effective ``chunk_size`` and requested ``n_chunks``.

        Raises
        ------
        ValueError, ZeroDivisionError
            Too few samples for the requested window size and window spacing.
        """
        n_chunks = n_items // chunk_size + (n_items % chunk_size > 0)
        n_chunks = max(n_chunks, n_chunks_min)
        if n_chunks % 2 == 0:
            n_chunks += 1
        windows = sliding_window_view(np.arange(n_items), chunk_size)
        n_windows = windows.shape[0]
        sw = n_windows // (n_chunks - 1)
        iw = np.r_[
            np.arange(0, n_windows // 2 - sw // 2, sw),
            n_windows // 2,
            np.arange(n_windows - 1, n_windows // 2 + sw // 2, -sw)[::-1],
        ]
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
            msg = "input data shall be 2-D."
            raise ValueError(msg)
        y_med = median_filter(y, (1, medfilt_size))
        y_range = np.max(y_med, axis=-1) - np.min(y_med, axis=-1)
        s_spike = (y - y_med) / y_range[:, np.newaxis]
        md_spike0 = np.abs(s_spike) >= height_frac_min
        logger.debug(f"found spike {md_spike0.sum()}/{md_spike0.size}")
        mc_range_small = y_range < y_range_min
        md_spike = md_spike0 & (~mc_range_small[:, np.newaxis])
        logger.debug(f"mask spike {md_spike.sum()}/{md_spike.size}")
        return md_spike, locals()


_RESULT_LAYOUT = {
    "samples": [
        ("mask_spike", "chan,probe"),
        ("bitmask", "chan,probe"),
        ("mask_baseline", "chan,probe"),
        ("S21_spike", "chan,probe"),
        ("S21_orig", "chan,probe"),
    ],
    "channels": [
        ("bitmask_chan", "chan"),
        ("mask_chan_bad", "chan"),
        ("chan_range", "chan"),
        ("chan_level", "chan"),
        ("chan_rms_mean", "chan"),
        ("chan_rms_std", "chan"),
        ("chan_unc_I_ref", "chan"),
        ("chan_unc_Q_ref", "chan"),
        ("chan_unc_I_std", "chan"),
        ("chan_unc_Q_std", "chan"),
    ],
    "chunks": [
        ("f_chunks", "chan,chunk"),
        ("bitmask_chunk", "chan,chunk"),
        ("mask_chunk_baseline", "chan,chunk"),
        ("chunk_rms_mean", "chan,chunk"),
        ("chunk_rms_std", "chan,chunk"),
        ("chunk_skew", "chan,chunk"),
        ("chunk_kurtosis", "chan,chunk"),
        ("chunk_windows", "chunk,window"),
        ("chunk_bounds", "chunk,bound"),
    ],
    "d21": [
        ("d21_mask_baseline", "frequency"),
        ("d21_baseline", "frequency"),
        ("d21_baseline_rms", "frequency"),
        ("d21_frequency", "frequency"),
        ("d21", "frequency"),
        ("d21_detrended", "frequency"),
    ],
    "d21_chunks": [
        ("d21_f_chunks", "chunk"),
        ("d21_chunk_baseline", "chunk"),
        ("d21_chunk_baseline_rms", "chunk"),
        ("d21_chunk_windows", "chunk,window"),
        ("d21_chunk_bounds", "chunk,bound"),
    ],
}

_RESULT_ATTRS = [
    "chunk_size",
    "n_chunks_per_chan",
    "d21_chunk_size",
    "d21_n_chunks",
    "swp_rms_med",
    "swp_rms_rms",
]


def _make_check_result(values, sweep, config):  # noqa: C901, PLR0912 -- explicit result groups and units
    """Build the result tree before replacing a block's prior check."""
    mapper = SweepCheckMapper.from_defaults()
    schema = mapper.schema
    result = xr.DataTree()
    result.attrs[mapper.get_name(schema.config)] = config.model_dump_json()
    for name in _RESULT_ATTRS:
        result.attrs[mapper.get_name(getattr(schema, name))] = getattr(values, name)
    for group, fields in _RESULT_LAYOUT.items():
        variables = {}
        for name, dims in fields:
            if name.endswith("chunk_bounds"):
                value = [
                    (s.start, s.stop)
                    for s in getattr(values, name.replace("bounds", "slices"))
                ]
            else:
                value = getattr(values, name)
            if value is None:
                continue
            attrs = {}
            if isinstance(value, u.Quantity):
                attrs["units"] = str(value.unit)
                value = value.value
            elif name == "f_chunks":
                attrs["units"] = "Hz"
            elif name in {
                "S21_orig",
                "S21_spike",
                "chan_unc_I_ref",
                "chan_unc_Q_ref",
                "chan_unc_I_std",
                "chan_unc_Q_std",
            }:
                attrs["units"] = str(sweep.I.u.unit or u.dimensionless_unscaled)
            elif name == "bitmask":
                attrs.update(
                    flag_masks=[int(flag) for flag in SweepBitMask],
                    flag_meanings=" ".join(flag.name for flag in SweepBitMask),
                    diagnostic_flag_masks=[
                        int(SweepBitMask.unc_nonfinite),
                        int(SweepBitMask.unc_low),
                        int(SweepBitMask.unc_high),
                        int(SweepBitMask.unc_outlier),
                    ],
                )
            variables[mapper.get_name(getattr(schema, name))] = xr.DataArray(
                value, dims=dims.split(","), attrs=attrs
            )
        ds = xr.Dataset(variables)
        if "chan" in ds.dims:
            ds = ds.assign_coords(
                chan=sweep.mapper.get_arr(
                    sweep.data_source, sweep.mapper.schema.chan
                ).to_numpy()
            )
        result[cast("str", mapper.get_name(getattr(schema, group + "_subpath")))] = (
            xr.DataTree(dataset=ds)
        )
    corrected = sweep.data_source.copy(deep=False)
    signal = values.S21_checked
    for source, value in ((sweep.I, signal.real), (sweep.Q, signal.imag)):
        corrected[source.name] = source.copy(data=value)
    result[cast("str", mapper.get_name(schema.sweep))] = xr.DataTree(dataset=corrected)
    if values.readout_power is not None:
        result[cast("str", mapper.get_name(schema.readout_power))] = xr.DataTree(
            dataset=values.readout_power
        )
    return result


def _readout_power_result(block, signal, config):
    """Evaluate the acquired readout using corrected I/Q and recorded settings."""
    sweep = ReducedSweepView(block)
    meta = sweep.meta
    if meta is None or any(
        value is None for value in (meta.roach, meta.atten_drive, meta.atten_sense)
    ):
        msg = "Readout power requires recorded ROACH, drive and sense attenuation"
        raise ValueError(msg)
    probes = sweep.chan_axis_table
    raw = block.toltec_kids.block.raw.toltec_kids
    snapshot = (
        raw.adc_snap.to_numpy() if raw.mapper.schema.adc_snap in raw.mapper else None
    )
    values = readout_power(
        probes["tone_amp"],
        signal.real,
        signal.imag,
        snapshot,
        dac_power_dbm=config.dac_power_dbm,
        gains_db=config.gains(
            meta.roach, atten_drive=meta.atten_drive, atten_sense=meta.atten_sense
        ),
        adc_conversion_offset_db=config.adc_conversion_offset_db,
    )
    variables = {}
    for name, value in values.items():
        if name == "adc_snap_frac":
            dims, unit = (), "1"
        elif name == "gain":
            dims, unit = ("element",), "dB"
        elif name == "tone_amps_db":
            dims, unit = ("chan",), "dB"
        else:
            dims = ("element", "chan") if "per_tone" in name else ("element",)
            unit = "dB(mW)"
        variables[name] = (dims, value, {"units": unit})
    # V2 plots the recorded baseband comb, despite its 'Detector Frequency' label.
    variables["frequency"] = ("chan", probes["f_tone"].to_value(u.Hz), {"units": "Hz"})
    return xr.Dataset(
        variables,
        coords={"element": list(READOUT_ELEMENTS), "chan": np.asarray(probes["chan"])},
        attrs={
            "roach": meta.roach,
            "atten_drive": meta.atten_drive,
            "atten_sense": meta.atten_sense,
            "config": config.model_dump_json(),
            "power_model": "fixed_total_recorded_tone_rows",
        },
    )
