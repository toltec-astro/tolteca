"""Tests for tolteca_kids.pipeline — KidsPipeline end-to-end zarr runner."""

from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path

import xarray as xr

# SweepCheck produces divide-by-zero warnings on pure-noise test data (no signal).
# This is expected behaviour with synthetic fixtures — suppress for this test module.
pytestmark = pytest.mark.filterwarnings(
    "ignore:divide by zero:RuntimeWarning",
    "ignore:invalid value:RuntimeWarning",
)

from tolteca_kids.pipeline import (
    KidsPipeline,
    KidsPipelineConfig,
    KidsPipelineResult,
    SWEEP_CHECK_GROUP,
    KIDS_FIND_GROUP,
    has_kids_reduction,
    read_sweep_check,
    read_kids_find,
    _build_datatree_from_zarr,
    _quality_score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_CHAN = 6
N_STEPS = 60
LO_CENTER_HZ = 500e6


def _make_zarr_dataset(
    n_chan: int = N_CHAN,
    n_steps: int = N_STEPS,
    lo_center_hz: float = LO_CENTER_HZ,
    rng_seed: int = 0,
) -> xr.Dataset:
    """Build a minimal zarr-schema dataset (no Lorentzians — fast)."""
    rng = np.random.default_rng(rng_seed)
    lo_sweep_range = 2e6  # 2 MHz sweep range
    lo_freq = np.linspace(
        lo_center_hz - lo_sweep_range / 2,
        lo_center_hz + lo_sweep_range / 2,
        n_steps,
    )
    tone_freq = np.linspace(-800e3, 800e3, n_chan)  # offsets around LO centre

    I = rng.normal(0.5, 0.05, (n_chan, n_steps)).astype(np.float32)
    Q = rng.normal(0.0, 0.05, (n_chan, n_steps)).astype(np.float32)

    return xr.Dataset(
        {
            "I": xr.DataArray(I, dims=["chan", "sample"]),
            "Q": xr.DataArray(Q, dims=["chan", "sample"]),
            "tone_freq": xr.DataArray(tone_freq, dims=["chan"]),
        },
        coords={"lo_freq": xr.DataArray(lo_freq, dims=["sample"])},
        attrs={"lo_center_freq_hz": lo_center_hz},
    )


def _write_zarr(ds: xr.Dataset, path: Path) -> None:
    ds.to_zarr(str(path), mode="w")


@pytest.fixture
def zarr_store(tmp_path: Path) -> Path:
    """A temporary zarr store with zarr-schema data."""
    p = tmp_path / "nw0.zarr"
    ds = _make_zarr_dataset()
    _write_zarr(ds, p)
    return p


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestBuildDatatreeFromZarr:
    def test_returns_datatree(self):
        ds = _make_zarr_dataset()
        dt = _build_datatree_from_zarr(ds)
        assert isinstance(dt, xr.DataTree)

    def test_has_sweep_child(self):
        ds = _make_zarr_dataset()
        dt = _build_datatree_from_zarr(ds)
        from tolteca_kids.pipeline import _SWEEP_NS
        assert _SWEEP_NS in dt.children

    def test_sweep_child_has_I_Q(self):
        ds = _make_zarr_dataset()
        dt = _build_datatree_from_zarr(ds)
        from tolteca_kids.pipeline import _SWEEP_NS
        ds_child = dt[_SWEEP_NS].dataset
        assert f"{_SWEEP_NS}.I" in ds_child.data_vars
        assert f"{_SWEEP_NS}.Q" in ds_child.data_vars

    def test_chan_sweep_dims(self):
        ds = _make_zarr_dataset(n_chan=4, n_steps=20)
        dt = _build_datatree_from_zarr(ds)
        from tolteca_kids.pipeline import _SWEEP_NS
        I_da = dt[_SWEEP_NS].dataset[f"{_SWEEP_NS}.I"]
        assert I_da.dims == ("chan", "sweep")
        assert I_da.shape == (4, 20)

    def test_f_lo_coord_shape(self):
        ds = _make_zarr_dataset(n_chan=5, n_steps=10)
        dt = _build_datatree_from_zarr(ds)
        from tolteca_kids.pipeline import _SWEEP_NS
        f_lo = dt[_SWEEP_NS].dataset.coords["f_lo"]
        assert f_lo.shape == (5,)

    def test_reduced_sweep_view_ok(self):
        """ReducedSweepView must be constructable from the DataTree."""
        from tolteca_datamodels.toltec.kids import ReducedSweepView
        ds = _make_zarr_dataset()
        dt = _build_datatree_from_zarr(ds)
        view = ReducedSweepView(dt)
        assert view.I is not None
        assert view.Q is not None


class TestQualityScore:
    def test_zero_when_no_good_chans(self):
        assert _quality_score(5, 5, 10) == 0.0

    def test_one_when_all_detected(self):
        assert _quality_score(10, 0, 10) == 1.0

    def test_clamp_above_one(self):
        # More detections than good channels → clamped to 1
        assert _quality_score(5, 0, 100) == 1.0

    def test_fractional(self):
        score = _quality_score(10, 2, 4)  # 4/8 = 0.5
        assert abs(score - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# has_kids_reduction / read helpers (before pipeline run)
# ---------------------------------------------------------------------------


class TestHasKidsReduction:
    def test_false_on_raw_zarr(self, zarr_store: Path):
        assert has_kids_reduction(zarr_store) is False

    def test_true_after_pipeline(self, zarr_store: Path):
        KidsPipeline().run(zarr_store)
        assert has_kids_reduction(zarr_store) is True

    def test_false_on_nonexistent(self, tmp_path: Path):
        assert has_kids_reduction(tmp_path / "no_such.zarr") is False


class TestReadHelpers:
    def test_read_sweep_check_none_before_run(self, zarr_store: Path):
        assert read_sweep_check(zarr_store) is None

    def test_read_kids_find_none_before_run(self, zarr_store: Path):
        assert read_kids_find(zarr_store) is None

    def test_read_sweep_check_after_run(self, zarr_store: Path):
        KidsPipeline().run(zarr_store)
        sc = read_sweep_check(zarr_store)
        assert sc is not None
        assert sc.bitmask_chan.dtype == np.int32
        assert sc.mask_chan_bad.dtype == bool
        assert sc.bitmask_chan.shape == (N_CHAN,)
        assert sc.mask_chan_bad.shape == (N_CHAN,)

    def test_read_kids_find_after_run(self, zarr_store: Path):
        KidsPipeline().run(zarr_store)
        kf = read_kids_find(zarr_store)
        assert kf is not None
        assert kf.f_tone_hz.dtype == np.float64
        assert kf.Qr.dtype == np.float64
        assert kf.bitmask_det.dtype == np.int32
        assert isinstance(kf.meta, dict)


# ---------------------------------------------------------------------------
# KidsPipeline.run — result structure
# ---------------------------------------------------------------------------


class TestKidsPipelineResult:
    @pytest.fixture(autouse=True)
    def _run(self, zarr_store: Path):
        self.result = KidsPipeline().run(zarr_store)
        self.zarr_store = zarr_store

    def test_returns_result(self):
        assert isinstance(self.result, KidsPipelineResult)

    def test_n_chans_correct(self):
        assert self.result.n_chans == N_CHAN

    def test_n_bad_chans_nonnegative(self):
        assert self.result.n_bad_chans >= 0

    def test_n_bad_chans_le_n_chans(self):
        assert self.result.n_bad_chans <= self.result.n_chans

    def test_n_detected_nonnegative(self):
        assert self.result.n_detected >= 0

    def test_quality_score_range(self):
        assert 0.0 <= self.result.quality_score <= 1.0

    def test_zarr_path_set(self):
        assert self.result.zarr_path == self.zarr_store

    def test_not_skipped(self):
        assert self.result.skipped is False


# ---------------------------------------------------------------------------
# Zarr groups written correctly
# ---------------------------------------------------------------------------


class TestZarrGroupsWritten:
    @pytest.fixture(autouse=True)
    def _run(self, zarr_store: Path):
        KidsPipeline().run(zarr_store)
        self.zarr_store = zarr_store

    def test_sweep_check_group_exists(self):
        ds = xr.open_zarr(str(self.zarr_store), group=SWEEP_CHECK_GROUP)
        assert "bitmask_chan" in ds.data_vars
        assert "mask_chan_bad" in ds.data_vars

    def test_bitmask_chan_shape(self):
        ds = xr.open_zarr(str(self.zarr_store), group=SWEEP_CHECK_GROUP)
        assert ds["bitmask_chan"].shape == (N_CHAN,)

    def test_mask_chan_bad_shape(self):
        ds = xr.open_zarr(str(self.zarr_store), group=SWEEP_CHECK_GROUP)
        assert ds["mask_chan_bad"].shape == (N_CHAN,)

    def test_kids_find_group_exists(self):
        ds = xr.open_zarr(str(self.zarr_store), group=KIDS_FIND_GROUP)
        assert "f_tone_hz" in ds.data_vars
        assert "Qr" in ds.data_vars
        assert "bitmask_det" in ds.data_vars

    def test_kids_find_meta_json(self):
        ds = xr.open_zarr(str(self.zarr_store), group=KIDS_FIND_GROUP)
        meta = json.loads(ds.attrs["meta"])
        assert "n_detected" in meta
        assert "n_bad_chans" in meta
        assert "tolteca_kids_version" in meta

    def test_original_data_untouched(self):
        """Root zarr groups I, Q, tone_freq must still be present."""
        ds = xr.open_zarr(str(self.zarr_store))
        assert "I" in ds.data_vars
        assert "Q" in ds.data_vars
        assert "tone_freq" in ds.data_vars


# ---------------------------------------------------------------------------
# skip_if_done
# ---------------------------------------------------------------------------


class TestSkipIfDone:
    def test_skip_on_second_run(self, zarr_store: Path):
        KidsPipeline().run(zarr_store)
        r2 = KidsPipeline().run(zarr_store)
        assert r2.skipped is True
        assert r2.skip_reason == "already_done"

    def test_force_rerun(self, zarr_store: Path):
        KidsPipeline().run(zarr_store)
        cfg = KidsPipelineConfig(skip_if_done=False)
        r2 = KidsPipeline(cfg).run(zarr_store)
        assert r2.skipped is False
