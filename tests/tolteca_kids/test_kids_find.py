"""Tests for tolteca_kids.kids_find — KidsFind step."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import astropy.units as u
from astropy.table import QTable

from tolteca_kids import (
    KidsFind,
    KidsFindConfig,
    KidsFindContext,
    KidsFindData,
    SegmentBitMask,
    SweepCheck,
    SweepCheckConfig,
)
from tolteca_kids.peaks1d import Peaks1D, Peaks1DResult
from tolteca_kids.match1d import Match1D, Match1DResult, calc_shift1d
from tolteca_kidsproc.analysis.d21 import D21Analysis
from tollan.pipeline import PIPELINE_CONTEXT_KEY, get_pipeline_contexts


# ---------------------------------------------------------------------------
# Fixtures shared with sweep_check tests
# ---------------------------------------------------------------------------

_SWEEP_NS = "tolteca_datamodels.toltec.kids.sweep"


def make_reduced_datatree(
    n_chans: int = 6,
    n_steps: int = 100,
    rng_seed: int = 0,
    *,
    add_lorentzian: bool = True,
) -> xr.DataTree:
    """Build a synthetic reduced sweep DataTree with Lorentzian resonances.

    Uses Qr=1000 resonances subtracted from background (DIP shape) so that
    |S21| dips at resonance, which is what the KidsFind D21 algorithm expects.
    """
    rng = np.random.default_rng(rng_seed)

    f_lo = np.linspace(450e6, 456e6, n_chans)  # 450–456 MHz, 1.2 MHz apart
    sweep = np.linspace(-1e6, 1e6, n_steps)  # ±1 MHz sweep

    I = np.ones((n_chans, n_steps)) * 0.8 + rng.normal(0, 0.002, (n_chans, n_steps))
    Q = np.zeros((n_chans, n_steps)) + rng.normal(0, 0.002, (n_chans, n_steps))

    if add_lorentzian:
        for ci in range(n_chans):
            f_abs = f_lo[ci] + sweep
            f0 = f_lo[ci]
            # Qr=1000 → FWHM = 450 kHz ≈ 22 sweep steps out of 100 → spans the
            # center chunk and neighbouring region, creating a bimodal I
            # distribution (U-shape) in the center chunk.
            Qr = 1000
            df = (f_abs - f0) / (f0 / Qr)
            s21_lorentz = 1.0 / (1.0 + 1j * df)
            # SUBTRACT to create a DIP in |S21| (real KIDs behaviour).
            I[ci] -= s21_lorentz.real * 0.5
            Q[ci] -= s21_lorentz.imag * 0.5

    # Very small uncertainty so that SNR is large and peaks are detected
    unc_I = np.abs(rng.normal(0.001, 0.0001, (n_chans, n_steps)))
    unc_Q = np.abs(rng.normal(0.001, 0.0001, (n_chans, n_steps)))

    ds_reduced = xr.Dataset(
        {
            f"{_SWEEP_NS}.I": (["chan", "sweep"], I),
            f"{_SWEEP_NS}.Q": (["chan", "sweep"], Q),
            f"{_SWEEP_NS}.unc_I": (["chan", "sweep"], unc_I),
            f"{_SWEEP_NS}.unc_Q": (["chan", "sweep"], unc_Q),
            "f_lo": (["chan"], f_lo),
        },
        coords={"sweep": sweep, "chan": np.arange(n_chans)},
    )
    return xr.DataTree(
        dataset=xr.Dataset(),
        children={_SWEEP_NS: xr.DataTree(dataset=ds_reduced)},
    )


def _run_sweep_check(dt: xr.DataTree, **kwargs) -> None:
    """Run SweepCheck with coarse D21 step (fast for tests).

    Uses relaxed kurtosis/skew thresholds so that all sweep chunks are
    treated as potential baseline, even when Lorentzian tails affect the
    distribution.  The MAD-std estimator in the chunk baseline code is robust
    against resonance outliers (<15 % of bins), so the estimated noise is
    still representative of the off-resonance noise floor.
    """
    cfg = SweepCheckConfig(
        d21_analysis=D21Analysis(f_step=10_000 << u.Hz, smooth=5, method="savgol"),
        chunk_kurtosis_max=2.0,  # allow Gaussian-ish chunks (kurtosis≈0)
        chunk_skew_max=2.0,  # allow mildly skewed chunks (Lorentzian tail)
        **kwargs,
    )
    SweepCheck(cfg)(dt)


def _make_test_kids_find_config(**kwargs) -> KidsFindConfig:
    """Return a KidsFindConfig with low thresholds for synthetic test data.

    The defaults (d21_snr_min=20, d21_peak_min=0.1 Hz^-1) are designed for
    real TolTEC data.  Synthetic test resonances have SNR ~1.4 and peak
    heights ~5e-7 Hz^-1, so we lower the thresholds here.
    """
    kw = {
        # Very sensitive D21 peakdetect: accept any bump above 0.1×noise
        "d21_detect": Peaks1D(threshold=0, peakdetect_delta_threshold=0.1),
        # Lower quality cuts for synthetic (low-SNR) test data
        "d21_snr_min": 0.5,
        "d21_peak_min": 1e-8 << u.Hz**-1,  # effectively no height cut
        "Qr_min": 100,
        **kwargs,
    }
    return KidsFindConfig(**kw)


def _run_kids_find(dt: xr.DataTree, **kwargs) -> KidsFindContext:
    """Run SweepCheck then KidsFind on dt; return KidsFind context."""
    _run_sweep_check(dt, chunk_size=20, n_chunks_min=3)
    cfg = _make_test_kids_find_config(**kwargs)
    KidsFind(cfg)(dt)
    return KidsFind.get_context(dt)


# ---------------------------------------------------------------------------
# Peaks1D unit tests
# ---------------------------------------------------------------------------


class TestPeaks1D:
    def test_imports(self):
        assert Peaks1D is not None
        assert Peaks1DResult is not None

    def test_default_method(self):
        p = Peaks1D()
        assert p.method == "peakdetect"

    def test_frozen(self):
        """Peaks1D is frozen (immutable)."""
        p = Peaks1D(threshold=3.0)
        with pytest.raises(Exception):
            p.threshold = 5.0  # type: ignore[misc]

    def test_find_peaks_flat(self):
        """Flat signal → no peaks."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 200) * u.MHz
        y = np.ones(200) * 0.5 + rng.normal(0, 0.001, 200)
        ey = np.full(200, 0.01)
        p = Peaks1D(threshold=0)
        result = p(x, y, ey=ey)
        assert isinstance(result, Peaks1DResult)
        # May find 0 or very few noise peaks
        if result.peaks is not None:
            assert len(result.peaks) < 5

    def test_find_peaks_with_peak(self):
        """Clear Gaussian peak is detected."""
        x = np.linspace(0, 10, 300)
        y = np.exp(-0.5 * ((x - 5) / 0.5) ** 2)
        ey = np.full(300, 0.01)
        p = Peaks1D(threshold=0)
        result = p(x * u.MHz, y, ey=ey)
        assert result.peaks is not None
        # Peak should be near x=5
        peak_x = result.peaks["x"].to_value(u.MHz)
        assert np.any(np.abs(peak_x - 5.0) < 1.0)

    def test_raises_for_mismatched_ndim(self):
        """_check_xy raises on ndim mismatch (2-D x vs 1-D y)."""
        x = np.ones((100, 2))
        y = np.ones(100)
        with pytest.raises(ValueError):
            Peaks1D()._check_xy(x, y, None)

    def test_make_mask(self):
        """make_mask returns a boolean mask."""
        rng = np.random.default_rng(0)
        x = np.linspace(0, 10, 200)
        y = np.exp(-0.5 * ((x - 5) / 0.3) ** 2) + rng.normal(0, 0.01, 200)
        ey = np.full(200, 0.05)
        result = Peaks1D(threshold=0)(x * u.MHz, y, ey=ey)
        if result.peaks is not None and len(result.peaks) > 0:
            mask = result.make_mask(np.ones(len(result.peaks), dtype=bool), n_fwhms=3)
            assert mask.dtype == bool
            assert mask.shape == (200,)


# ---------------------------------------------------------------------------
# Match1D unit tests
# ---------------------------------------------------------------------------


class TestMatch1D:
    def test_imports(self):
        assert Match1D is not None
        assert Match1DResult is not None

    def test_default_method(self):
        m = Match1D()
        assert m.method == "dtw_python"

    def test_calc_shift1d_no_shift(self):
        """calc_shift1d returns ~0 when x0 == x1."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * u.MHz
        shift = calc_shift1d(x, x, dx=0.1 * u.MHz)
        assert abs(shift.to_value(u.MHz)) < 0.5

    def test_calc_shift1d_with_known_shift(self):
        """calc_shift1d recovers a known shift."""
        x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * u.MHz
        shift_true = 0.5 * u.MHz
        x1 = x0 + shift_true
        shift = calc_shift1d(x0, x1, dx=0.05 * u.MHz)
        assert abs((shift - shift_true).to_value(u.MHz)) < 0.2

    def test_match1d_basic(self):
        """Match1D matches two similar 1-D vectors."""
        query = np.array([450.0, 451.0, 452.0, 453.0, 454.0]) * u.MHz
        ref = np.array([450.1, 451.1, 452.1, 453.1, 454.1]) * u.MHz
        m = Match1D()
        result = m(
            query,
            ref,
            shift_kw={"dx": 0.05 * u.MHz, "shift_max": 1.0 * u.MHz},
        )
        assert isinstance(result, Match1DResult)
        assert result.matched is not None
        assert "query_matched" in result.data
        assert "ref_matched" in result.data

    def test_match1d_result_columns(self):
        """matched QTable has required columns."""
        query = np.array([450.0, 451.0, 452.0]) * u.MHz
        ref = np.array([450.1, 451.1, 452.1]) * u.MHz
        result = Match1D()(
            query, ref, shift_kw={"dx": 0.05 * u.MHz, "shift_max": 1.0 * u.MHz}
        )
        for col in ["idx_query", "idx_ref", "dist", "dist_shifted", "adist_shifted"]:
            assert col in result.matched.colnames


# ---------------------------------------------------------------------------
# SegmentBitMask
# ---------------------------------------------------------------------------


class TestSegmentBitMask:
    def test_flags_exist(self):
        assert SegmentBitMask.doublet
        assert SegmentBitMask.triplet
        assert SegmentBitMask.manylet
        assert SegmentBitMask.edge
        assert SegmentBitMask.peak_small
        assert SegmentBitMask.snr_low
        assert SegmentBitMask.Qr_small
        assert SegmentBitMask.dark
        assert SegmentBitMask.d21
        assert SegmentBitMask.s21

    def test_blended_composed(self):
        bm = SegmentBitMask.blended
        assert bm & SegmentBitMask.doublet
        assert bm & SegmentBitMask.triplet
        assert bm & SegmentBitMask.manylet


# ---------------------------------------------------------------------------
# KidsFindConfig
# ---------------------------------------------------------------------------


class TestKidsFindConfig:
    def test_defaults(self):
        cfg = KidsFindConfig()
        assert cfg.Qr_min == 1000
        assert cfg.d21_snr_min == 20.0
        assert cfg.match_ref == "chan"

    def test_frozen(self):
        cfg = KidsFindConfig()
        with pytest.raises(Exception):
            cfg.Qr_min = 500  # type: ignore[misc]

    def test_d21_detect_is_peaks1d(self):
        cfg = KidsFindConfig()
        assert isinstance(cfg.d21_detect, Peaks1D)

    def test_match_is_match1d(self):
        cfg = KidsFindConfig()
        assert isinstance(cfg.match, Match1D)


# ---------------------------------------------------------------------------
# KidsFind pipeline mechanics
# ---------------------------------------------------------------------------


class TestKidsFindMechanics:
    @pytest.fixture(scope="class")
    def dt_with_context(self):
        dt = make_reduced_datatree(n_chans=6, n_steps=80)
        _run_kids_find(dt)
        return dt

    def test_has_context_after_run(self, dt_with_context):
        assert KidsFind.has_context(dt_with_context)

    def test_context_is_typed(self, dt_with_context):
        ctx = KidsFind.get_context(dt_with_context)
        assert isinstance(ctx, KidsFindContext)

    def test_completed_flag_set(self, dt_with_context):
        ctx = KidsFind.get_context(dt_with_context)
        assert ctx.completed is True

    def test_data_is_kids_find_data(self, dt_with_context):
        ctx = KidsFind.get_context(dt_with_context)
        assert isinstance(ctx.data, KidsFindData)

    def test_context_key(self):
        assert "KidsFind" in KidsFind.context_key

    def test_sweep_check_context_still_present(self, dt_with_context):
        assert SweepCheck.has_context(dt_with_context)

    def test_return_context_flag(self):
        dt = make_reduced_datatree(n_chans=4, n_steps=60)
        _run_sweep_check(dt, chunk_size=20, n_chunks_min=3)
        step = KidsFind(_make_test_kids_find_config())
        result_dt, ctx = step(dt, return_context=True)
        assert result_dt is dt
        assert isinstance(ctx, KidsFindContext)


# ---------------------------------------------------------------------------
# KidsFindData output fields
# ---------------------------------------------------------------------------


class TestKidsFindOutputs:
    @pytest.fixture(scope="class")
    def ctx(self):
        dt = make_reduced_datatree(n_chans=6, n_steps=80)
        return _run_kids_find(dt)

    def test_d21_peaks_is_result(self, ctx):
        assert isinstance(ctx.data.d21_peaks, Peaks1DResult)

    def test_s21_peaks_is_result(self, ctx):
        assert isinstance(ctx.data.s21_peaks, Peaks1DResult)

    def test_detected_is_qtable(self, ctx):
        assert isinstance(ctx.data.detected, QTable)

    def test_matched_is_match1d_result(self, ctx):
        assert isinstance(ctx.data.matched, Match1DResult)

    def test_chan_matched_is_qtable(self, ctx):
        assert isinstance(ctx.data.chan_matched, QTable)

    def test_detected_has_freq_col(self, ctx):
        assert "f" in ctx.data.detected.colnames

    def test_detected_has_Qr_col(self, ctx):
        assert "Qr" in ctx.data.detected.colnames

    def test_bitmask_d21_shape(self, ctx):
        if ctx.data.d21_peaks.peaks is not None:
            assert ctx.data.bitmask_d21.shape == (len(ctx.data.d21_peaks.peaks),)

    def test_mask_baseline_shape(self, ctx):
        n_chans = 6
        n_steps = 80
        assert ctx.data.mask_baseline.shape == (n_chans, n_steps)


# ---------------------------------------------------------------------------
# make_groups1d
# ---------------------------------------------------------------------------


class TestMakeGroups1D:
    def test_basic_grouping(self):
        # gap between 1-2 and 5-6 is 1.0 MHz; gap between 2-5 is 3.0 MHz
        # use d=1.2 MHz so 1.0-MHz gaps are NOT breaks, 3.0-MHz gap IS a break
        x = np.array([1.0, 2.0, 5.0, 6.0]) * u.MHz
        d = np.array([1.2, 1.2, 1.2, 1.2]) * u.MHz
        grouped, groups, mask = KidsFind.make_groups1d(x, d)
        # [1,2] should be one group, [5,6] another
        assert len(groups) == 2

    def test_single_group(self):
        x = np.array([1.0, 1.5, 2.0]) * u.MHz
        d = np.array([0.8, 0.8, 0.8]) * u.MHz
        grouped, groups, mask = KidsFind.make_groups1d(x, d)
        assert len(groups) == 1

    def test_all_separate(self):
        x = np.array([1.0, 5.0, 10.0]) * u.MHz
        d = np.array([0.1, 0.1, 0.1]) * u.MHz
        grouped, groups, mask = KidsFind.make_groups1d(x, d)
        assert len(groups) == 3

    def test_mask_shape(self):
        x = np.array([1.0, 2.0, 5.0]) * u.MHz
        d = np.array([0.6, 0.6, 0.6]) * u.MHz
        grouped, groups, mask = KidsFind.make_groups1d(x, d)
        assert mask.shape == (3, len(groups))


# ---------------------------------------------------------------------------
# Alias
# ---------------------------------------------------------------------------


class TestAlias:
    def test_alias_has_distinct_context_key(self):
        KidsFindPass1 = KidsFind.alias("pass1")
        KidsFindPass2 = KidsFind.alias("pass2")
        assert KidsFindPass1.context_key != KidsFindPass2.context_key

    def test_alias_runs_independently(self):
        dt = make_reduced_datatree(n_chans=4, n_steps=60)
        _run_sweep_check(dt, chunk_size=20, n_chunks_min=3)
        KidsFindPass1 = KidsFind.alias("pass1")
        KidsFindPass1(_make_test_kids_find_config())(dt)
        assert KidsFindPass1.has_context(dt)
        assert not KidsFind.has_context(dt)
