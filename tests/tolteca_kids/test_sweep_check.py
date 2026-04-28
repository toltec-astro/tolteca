"""Tests for tolteca_kids.sweep_check — SweepCheck step."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import astropy.units as u

from tolteca_kids import (
    SweepBitMask,
    SweepCheck,
    SweepCheckConfig,
    SweepCheckContext,
    SweepCheckData,
)
from tolteca_kidsproc.analysis.d21 import D21Analysis
from tollan.pipeline import get_pipeline_contexts, PIPELINE_CONTEXT_KEY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SWEEP_NS = "tolteca_datamodels.toltec.kids.sweep"


def make_reduced_datatree(
    n_chans: int = 5,
    n_steps: int = 80,
    rng_seed: int = 0,
    *,
    add_lorentzian: bool = True,
    add_spike: bool = False,
) -> xr.DataTree:
    """Build a synthetic reduced sweep DataTree.

    Parameters
    ----------
    n_chans : int
        Number of frequency channels.
    n_steps : int
        Number of sweep steps per channel.
    rng_seed : int
        Random number generator seed.
    add_lorentzian : bool
        If True, add Lorentzian resonance features to S21.
    add_spike : bool
        If True, inject a single spike into channel 0.

    Returns
    -------
    xr.DataTree
        DataTree with reduced sweep child node compatible with ReducedSweepView.
    """
    rng = np.random.default_rng(rng_seed)

    # Narrow band: channels from 450 MHz to 460 MHz (2 MHz apart)
    f_lo = np.linspace(450e6, 460e6, n_chans)  # Hz
    sweep = np.linspace(-1e6, 1e6, n_steps)  # Hz offsets

    # Build S21 with small noise; optionally add Lorentzian dips
    I = np.ones((n_chans, n_steps)) * 0.8 + rng.normal(0, 0.01, (n_chans, n_steps))
    Q = np.zeros((n_chans, n_steps)) + rng.normal(0, 0.01, (n_chans, n_steps))

    if add_lorentzian:
        for ci in range(n_chans):
            f_abs = f_lo[ci] + sweep  # [n_steps]
            f0 = f_lo[ci]  # resonance at center
            Qr = 1e4
            df = (f_abs - f0) / (f0 / Qr)
            # Simple Lorentzian dip in |S21|
            s21_lorentz = 1.0 / (1.0 + 1j * df)
            I[ci] += s21_lorentz.real * 0.3
            Q[ci] += s21_lorentz.imag * 0.3

    if add_spike:
        I[0, n_steps // 2] += 5.0  # large spike in channel 0

    unc_I = np.abs(rng.normal(0.005, 0.001, (n_chans, n_steps)))
    unc_Q = np.abs(rng.normal(0.005, 0.001, (n_chans, n_steps)))

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


def _run_sweep_check(dt: xr.DataTree, **kwargs) -> SweepCheckContext:
    """Run SweepCheck with fine D21 step (avoid giant grid in tests)."""
    cfg = SweepCheckConfig(
        d21_analysis=D21Analysis(
            f_step=10_000 << u.Hz,  # 10 kHz — coarse enough for tests
            smooth=5,
            method="savgol",
        ),
        **kwargs,
    )
    step = SweepCheck(cfg)
    step(dt)
    return SweepCheck.get_context(dt)


# ---------------------------------------------------------------------------
# Import / instantiation
# ---------------------------------------------------------------------------


class TestImports:
    def test_bitmask_flags(self):
        """SweepBitMask has the expected flags."""
        assert SweepBitMask.range_small
        assert SweepBitMask.rms_low
        assert SweepBitMask.spike

    def test_config_defaults(self):
        cfg = SweepCheckConfig()
        assert cfg.enabled is True
        assert cfg.despike is True
        assert cfg.chunk_size == 50

    def test_step_construction_no_args(self):
        step = SweepCheck()
        assert step.config.enabled is True

    def test_step_construction_kwargs(self):
        step = SweepCheck(chunk_size=20)
        assert step.config.chunk_size == 20

    def test_step_construction_config_object(self):
        cfg = SweepCheckConfig(chunk_size=30)
        step = SweepCheck(cfg)
        assert step.config.chunk_size == 30

    def test_context_key(self):
        assert "SweepCheck" in SweepCheck.context_key


# ---------------------------------------------------------------------------
# Core pipeline mechanics
# ---------------------------------------------------------------------------


class TestPipelineMechanics:
    def test_context_stored_in_attrs(self):
        dt = make_reduced_datatree()
        _run_sweep_check(dt, chunk_size=20, n_chunks_min=3)
        assert PIPELINE_CONTEXT_KEY in dt.attrs
        pctx = get_pipeline_contexts(dt)
        assert SweepCheck.context_key in pctx

    def test_has_context_before_run(self):
        dt = make_reduced_datatree()
        assert not SweepCheck.has_context(dt)

    def test_has_context_after_run(self):
        dt = make_reduced_datatree()
        _run_sweep_check(dt)
        assert SweepCheck.has_context(dt)

    def test_context_is_typed(self):
        dt = make_reduced_datatree()
        _run_sweep_check(dt)
        ctx = SweepCheck.get_context(dt)
        assert isinstance(ctx, SweepCheckContext)

    def test_completed_flag_set(self):
        dt = make_reduced_datatree()
        _run_sweep_check(dt)
        ctx = SweepCheck.get_context(dt)
        assert ctx.completed is True

    def test_data_is_sweep_check_data(self):
        dt = make_reduced_datatree()
        _run_sweep_check(dt)
        ctx = SweepCheck.get_context(dt)
        assert isinstance(ctx.data, SweepCheckData)

    def test_config_stored_in_context(self):
        dt = make_reduced_datatree()
        _run_sweep_check(dt, chunk_size=22, n_chunks_min=3)
        ctx = SweepCheck.get_context(dt)
        assert ctx.config.chunk_size == 22

    def test_second_run_replaces_context(self):
        dt = make_reduced_datatree()
        _run_sweep_check(dt, chunk_size=20, n_chunks_min=3)
        _run_sweep_check(dt, chunk_size=25, n_chunks_min=3)
        ctx = SweepCheck.get_context(dt)
        assert ctx.config.chunk_size == 25

    def test_return_context_flag(self):
        dt = make_reduced_datatree()
        cfg = SweepCheckConfig(
            d21_analysis=D21Analysis(f_step=10_000 << u.Hz, smooth=5)
        )
        step = SweepCheck(cfg)
        result_dt, ctx = step(dt, return_context=True)
        assert result_dt is dt
        assert isinstance(ctx, SweepCheckContext)
        assert ctx.completed is True


# ---------------------------------------------------------------------------
# Output arrays
# ---------------------------------------------------------------------------


class TestOutputArrays:
    @pytest.fixture(scope="class")
    def ctx(self):
        dt = make_reduced_datatree(n_chans=5, n_steps=80)
        return _run_sweep_check(dt, chunk_size=20, n_chunks_min=3)

    def test_mask_chan_bad_shape(self, ctx):
        assert ctx.data.mask_chan_bad.shape == (5,)

    def test_mask_chan_bad_dtype(self, ctx):
        assert ctx.data.mask_chan_bad.dtype == bool

    def test_bitmask_chan_shape(self, ctx):
        assert ctx.data.bitmask_chan.shape == (5,)

    def test_bitmask_chan_stats_is_dataframe(self, ctx):
        import pandas as pd
        assert isinstance(ctx.data.bitmask_chan_stats, pd.DataFrame)

    def test_d21_is_quantity(self, ctx):
        assert hasattr(ctx.data.d21, "unit")
        assert ctx.data.d21.unit.is_equivalent(u.Hz**-1)

    def test_d21_frequency_is_quantity(self, ctx):
        assert hasattr(ctx.data.d21_frequency, "unit")
        assert ctx.data.d21_frequency.unit.is_equivalent(u.Hz)

    def test_d21_detrended_shape_matches_d21(self, ctx):
        assert ctx.data.d21_detrended.shape == ctx.data.d21.shape

    def test_mask_spike_shape(self, ctx):
        assert ctx.data.mask_spike.shape == (5, 80)

    def test_chunk_windows_present(self, ctx):
        assert ctx.data.chunk_windows.ndim == 2  # [n_chunks, chunk_size]

    def test_chan_range_shape(self, ctx):
        assert ctx.data.chan_range.shape == (5,)

    def test_chan_rms_mean_shape(self, ctx):
        assert ctx.data.chan_rms_mean.shape == (5,)


# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------


class TestSpikeDetection:
    def test_find_spike_no_spike(self):
        """Clean data → no spikes."""
        y = np.ones((3, 50)) * 0.5 + np.random.default_rng(0).normal(0, 0.001, (3, 50))
        mask, ctx = SweepCheck.find_spike(y)
        # Some spikes may be found due to noise, but fraction should be low
        assert mask.sum() / mask.size < 0.05

    def test_find_spike_detects_injected_spike(self):
        """Injected spike is detected in a channel with sufficient range."""
        rng = np.random.default_rng(1)
        y = np.ones((3, 50)) * 0.5 + rng.normal(0, 0.01, (3, 50))
        # Add a large spike to channel 0 at position 25
        y[0, 25] = y[0, 25] + 5.0  # clearly above noise
        # Modify y so channel 0 has a clear range (median filter won't fully kill it)
        # by adding a small gradient to pass y_range_min threshold
        y[0] += np.linspace(0, 1.0, 50)  # range = 1.0 dB
        mask, _ = SweepCheck.find_spike(y, y_range_min=0.01)
        assert mask[0, 25]

    def test_find_spike_raises_for_1d(self):
        with pytest.raises(ValueError, match="2-D"):
            SweepCheck.find_spike(np.ones(10))

    def test_spike_flag_in_bitmask(self):
        """Injected spike → SweepBitMask.spike set on that channel."""
        dt = make_reduced_datatree(add_spike=True)
        _run_sweep_check(dt, chunk_size=20, n_chunks_min=3)
        ctx = SweepCheck.get_context(dt)
        # Channel 0 has the spike
        assert ctx.data.bitmask_chan[0] & SweepBitMask.spike

    def test_despike_stores_s21_orig(self):
        dt = make_reduced_datatree(add_spike=True)
        ctx = _run_sweep_check(dt, chunk_size=20, n_chunks_min=3, despike=True)
        assert ctx.data.S21_orig is not None

    def test_despike_false_no_s21_spike(self):
        dt = make_reduced_datatree()
        ctx = _run_sweep_check(dt, chunk_size=20, n_chunks_min=3, despike=False)
        assert ctx.data.S21_spike is None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


class TestChunking:
    def test_make_chunks_basic(self):
        result = SweepCheck.make_chunks(n_items=100, chunk_size=20, n_chunks_min=3)
        assert "chunk_size" in result
        assert "chunk_windows" in result
        assert "chunk_slices" in result
        assert len(result["chunk_slices"]) >= 3

    def test_make_chunks_covers_all_items(self):
        n = 80
        result = SweepCheck.make_chunks(n_items=n, chunk_size=15, n_chunks_min=3)
        # All indices covered by slices
        covered = set()
        for s in result["chunk_slices"]:
            covered.update(range(s.start, s.stop))
        assert covered == set(range(n))

    def test_chunk_windows_shape(self):
        result = SweepCheck.make_chunks(n_items=100, chunk_size=10, n_chunks_min=5)
        windows = result["chunk_windows"]
        assert windows.ndim == 2
        assert windows.shape[1] == min(10, 100)


# ---------------------------------------------------------------------------
# Bitmask helpers
# ---------------------------------------------------------------------------


class TestBitmaskHelpers:
    def test_sweep_bit_mask_compound(self):
        bm = SweepBitMask.rms_low | SweepBitMask.spike
        assert bm & SweepBitMask.rms_low
        assert bm & SweepBitMask.spike
        assert not (bm & SweepBitMask.range_small)

    def test_bad_chan_bits_default(self):
        cfg = SweepCheckConfig()
        # Default: rms_low | rms_high flags bad channels
        assert cfg.bad_chan_bits & SweepBitMask.rms_low
        assert cfg.bad_chan_bits & SweepBitMask.rms_high


# ---------------------------------------------------------------------------
# Alias
# ---------------------------------------------------------------------------


class TestAlias:
    def test_aliased_step_has_independent_context(self):
        SweepCheckPass1 = SweepCheck.alias("pass1")
        SweepCheckPass2 = SweepCheck.alias("pass2")
        assert SweepCheckPass1.context_key != SweepCheckPass2.context_key
        assert "SweepCheck" in SweepCheckPass1.context_key
        assert "pass1" in SweepCheckPass1.context_key

    def test_alias_runs_independently(self):
        dt = make_reduced_datatree(n_chans=3, n_steps=50)
        SweepCheckPass1 = SweepCheck.alias("pass1")
        cfg = SweepCheckConfig(
            chunk_size=15,
            n_chunks_min=3,
            d21_analysis=D21Analysis(f_step=10_000 << u.Hz, smooth=5),
        )
        SweepCheckPass1(cfg)(dt)
        assert SweepCheckPass1.has_context(dt)
        assert not SweepCheck.has_context(dt)
