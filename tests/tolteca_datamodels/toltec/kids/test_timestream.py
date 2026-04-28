"""Tests for TolTEC timestream reducer and PSD analysis."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tolteca_datamodels.toltec.kids import ReducedTimestreamView, TimestreamReducer

# Namespace constant (must match module __name__)
NAMESPACE = "tolteca_datamodels.toltec.kids.timestream"


def get_reduced_ds(dt: xr.DataTree) -> xr.Dataset:
    """Extract the reduced timestream child dataset from a DataTree."""
    return dt.children[NAMESPACE].dataset


class TestTimestreamReducerDataTree:
    """Test that TimestreamReducer returns xr.DataTree."""

    def test_returns_datatree(self):
        """Test that reducer returns a DataTree."""
        ds = xr.Dataset(
            {
                "I": (["time"], np.random.randn(1000)),
                "Q": (["time"], np.random.randn(1000)),
            },
        )
        ds["time"] = np.arange(1000) / 100.0
        ds.attrs["f_smp"] = 100.0

        reducer = TimestreamReducer(psd_nperseg=64)
        dt = reducer(ds)
        assert isinstance(dt, xr.DataTree)

    def test_root_has_raw_data(self):
        """Test that root node contains the original raw data."""
        ds = xr.Dataset(
            {
                "I": (["time"], np.ones(1000)),
                "Q": (["time"], np.zeros(1000)),
            },
        )
        ds["time"] = np.arange(1000) / 100.0
        ds.attrs["f_smp"] = 100.0

        reducer = TimestreamReducer(psd_nperseg=64)
        dt = reducer(ds)
        assert "I" in dt.dataset
        assert "Q" in dt.dataset

    def test_child_node_exists(self):
        """Test that the reduced timestream child node exists."""
        ds = xr.Dataset(
            {
                "I": (["time"], np.random.randn(1000)),
                "Q": (["time"], np.random.randn(1000)),
            },
        )
        ds["time"] = np.arange(1000) / 100.0
        ds.attrs["f_smp"] = 100.0

        reducer = TimestreamReducer(psd_nperseg=64)
        dt = reducer(ds)
        assert NAMESPACE in dt.children


class TestTimestreamReducer:
    """Test suite for TimestreamReducer with PSD analysis."""

    def test_single_channel_basic(self):
        """Test basic PSD computation for single-channel timestream."""
        # Create synthetic timestream data
        n_samples = 10000
        fsmp = 1000.0  # Hz
        time = np.arange(n_samples) / fsmp

        # Signal: 10 Hz sine + noise
        signal = np.sin(2 * np.pi * 10 * time)
        noise = 0.1 * np.random.randn(n_samples)

        ds = xr.Dataset(
            {
                "I": (["time"], signal + noise),
                "Q": (["time"], noise),
            },
        )
        ds["time"] = time
        ds.attrs["f_smp"] = fsmp

        # Create reducer
        reducer = TimestreamReducer(
            psd_nperseg=1024,
            psd_stat_freq_range=(5.0, 50.0),
        )

        # Reduce
        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)

        # Check PSD data exists
        assert f"{NAMESPACE}.f_psd" in child_ds
        assert f"{NAMESPACE}.I_psd" in child_ds
        assert f"{NAMESPACE}.Q_psd" in child_ds

        # Check PSD frequency axis
        f_psd = child_ds[f"{NAMESPACE}.f_psd"].values
        assert len(f_psd) == 1024 // 2 + 1  # Welch returns nperseg // 2 + 1 points
        assert f_psd[0] == 0.0
        assert f_psd[-1] == pytest.approx(fsmp / 2, rel=0.01)  # Nyquist

        # Check PSD summary statistics
        assert f"{NAMESPACE}.I_psd_median" in child_ds
        assert f"{NAMESPACE}.Q_psd_median" in child_ds
        assert f"{NAMESPACE}.I_psd_mad_std" in child_ds
        assert f"{NAMESPACE}.Q_psd_mad_std" in child_ds

        # Check config cached in child dataset attrs
        assert f"{NAMESPACE}.reducer_config" in child_ds.attrs

    def test_multi_channel_basic(self):
        """Test PSD computation for multi-channel timestream."""
        n_chans = 100
        n_samples = 5000
        fsmp = 500.0

        time = np.arange(n_samples) / fsmp

        # Create multi-channel data with varying noise levels
        I_data = (
            np.random.randn(n_chans, n_samples)
            * np.linspace(0.5, 1.5, n_chans)[:, None]
        )
        Q_data = (
            np.random.randn(n_chans, n_samples)
            * np.linspace(0.5, 1.5, n_chans)[:, None]
        )

        ds = xr.Dataset(
            {
                "I": (["chan", "time"], I_data),
                "Q": (["chan", "time"], Q_data),
            },
        )
        ds["time"] = time
        ds["chan"] = np.arange(n_chans)
        ds.attrs["f_smp"] = fsmp

        # Create reducer
        reducer = TimestreamReducer(
            psd_nperseg=512,
            psd_stat_freq_range=(10.0, 100.0),
        )

        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)

        # Check PSD dimensions
        I_psd = child_ds[f"{NAMESPACE}.I_psd"]
        assert I_psd.dims == ("chan", "f_psd")
        assert I_psd.shape[0] == n_chans
        assert I_psd.shape[1] == 512 // 2 + 1

        # Check summary statistics shape
        I_psd_median = child_ds[f"{NAMESPACE}.I_psd_median"]
        assert I_psd_median.dims == ("chan",)
        assert len(I_psd_median) == n_chans

    def test_with_r_x_components(self):
        """Test PSD computation including r and x components."""
        n_samples = 8000
        fsmp = 1000.0
        time = np.arange(n_samples) / fsmp

        ds = xr.Dataset(
            {
                "I": (["time"], np.random.randn(n_samples)),
                "Q": (["time"], np.random.randn(n_samples)),
                "r": (["time"], np.random.randn(n_samples)),
                "x": (["time"], np.random.randn(n_samples)),
            },
        )
        ds["time"] = time
        ds.attrs["f_smp"] = fsmp

        reducer = TimestreamReducer(
            psd_nperseg=1024,
            compute_r_x=True,
            psd_stat_freq_range=(10.0, 100.0),
        )

        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)

        # Check r and x PSDs exist
        assert f"{NAMESPACE}.r_psd" in child_ds
        assert f"{NAMESPACE}.x_psd" in child_ds
        assert f"{NAMESPACE}.r_psd_median" in child_ds
        assert f"{NAMESPACE}.x_psd_median" in child_ds
        assert f"{NAMESPACE}.r_psd_mad_std" in child_ds
        assert f"{NAMESPACE}.x_psd_mad_std" in child_ds

    def test_psd_disabled(self):
        """Test reducer with PSD computation disabled."""
        n_samples = 5000
        fsmp = 500.0

        ds = xr.Dataset(
            {
                "I": (["time"], np.random.randn(n_samples)),
                "Q": (["time"], np.random.randn(n_samples)),
            },
        )
        ds["time"] = np.arange(n_samples) / fsmp
        ds.attrs["f_smp"] = fsmp

        reducer = TimestreamReducer(compute_psd=False)
        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)

        # Check that PSD data doesn't exist in child dataset
        assert f"{NAMESPACE}.f_psd" not in child_ds
        assert f"{NAMESPACE}.I_psd" not in child_ds

        # Config should still be cached in child dataset attrs
        assert f"{NAMESPACE}.reducer_config" in child_ds.attrs

    def test_caching(self):
        """Test that results are cached when DataTree is passed back."""
        n_samples = 5000
        ds = xr.Dataset(
            {
                "I": (["time"], np.random.randn(n_samples)),
                "Q": (["time"], np.random.randn(n_samples)),
            },
        )
        ds["time"] = np.arange(n_samples) / 500.0
        ds.attrs["f_smp"] = 500.0

        reducer = TimestreamReducer(psd_nperseg=512)

        # First call
        dt1 = reducer(ds)

        # Second call with DataTree should use cache
        dt2 = reducer(dt1)

        assert get_reduced_ds(dt1)[f"{NAMESPACE}.I_psd"].equals(
            get_reduced_ds(dt2)[f"{NAMESPACE}.I_psd"],
        )

        # Force recomputation
        dt3 = reducer(dt1, force=True)
        assert isinstance(dt3, xr.DataTree)
        assert (
            get_reduced_ds(dt3)[f"{NAMESPACE}.I_psd"].shape
            == get_reduced_ds(dt1)[f"{NAMESPACE}.I_psd"].shape
        )

    def test_missing_data_error(self):
        """Test error handling for missing I or Q data."""
        ds = xr.Dataset({"I": (["time"], np.random.randn(1000))})
        ds["time"] = np.arange(1000) / 100.0

        reducer = TimestreamReducer()

        with pytest.raises(ValueError, match="must contain 'I' and 'Q'"):
            reducer(ds)

    def test_missing_time_dimension_error(self):
        """Test error handling for missing time dimension."""
        ds = xr.Dataset(
            {
                "I": (["sample"], np.random.randn(1000)),
                "Q": (["sample"], np.random.randn(1000)),
            },
        )
        ds.attrs["f_smp"] = 100.0

        reducer = TimestreamReducer()

        with pytest.raises(ValueError, match="Could not find time dimension"):
            reducer(ds)

    def test_psd_parameters(self):
        """Test various PSD parameter combinations."""
        n_samples = 10000
        ds = xr.Dataset(
            {
                "I": (["time"], np.random.randn(n_samples)),
                "Q": (["time"], np.random.randn(n_samples)),
            },
        )
        ds["time"] = np.arange(n_samples) / 1000.0
        ds.attrs["f_smp"] = 1000.0

        # Test different window functions
        for window in ["hann", "hamming", "blackman"]:
            reducer = TimestreamReducer(psd_window=window, psd_nperseg=1024)
            dt = reducer(ds)
            assert f"{NAMESPACE}.I_psd" in get_reduced_ds(dt)

        # Test different nperseg values
        for nperseg in [256, 512, 2048]:
            reducer = TimestreamReducer(psd_nperseg=nperseg)
            dt = reducer(ds)
            f_psd = get_reduced_ds(dt)[f"{NAMESPACE}.f_psd"].values
            assert len(f_psd) == nperseg // 2 + 1

    def test_stat_freq_range_filtering(self):
        """Test that stat_freq_range correctly filters PSD for statistics."""
        n_samples = 10000
        fsmp = 1000.0
        time = np.arange(n_samples) / fsmp

        # Create signal with known frequency content
        signal = (
            np.sin(2 * np.pi * 5 * time)  # 5 Hz
            + np.sin(2 * np.pi * 50 * time)  # 50 Hz (in range)
            + np.sin(2 * np.pi * 200 * time)  # 200 Hz (out of range)
        )

        ds = xr.Dataset(
            {
                "I": (["time"], signal),
                "Q": (["time"], 0.1 * np.random.randn(n_samples)),
            },
        )
        ds["time"] = time
        ds.attrs["f_smp"] = fsmp

        # Compute PSD with frequency range 20-100 Hz
        reducer = TimestreamReducer(
            psd_nperseg=2048,
            psd_stat_freq_range=(20.0, 100.0),
        )
        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)

        # Should have statistics
        assert f"{NAMESPACE}.I_psd_median" in child_ds

        # The 50 Hz signal should dominate in the stat range
        f_psd = child_ds[f"{NAMESPACE}.f_psd"].values
        I_psd = child_ds[f"{NAMESPACE}.I_psd"].values

        # Find peak in 20-100 Hz range
        mask = (f_psd >= 20) & (f_psd <= 100)
        peak_idx = np.argmax(I_psd[mask])
        peak_freq = f_psd[mask][peak_idx]

        # Should be close to 50 Hz
        assert 45 < peak_freq < 55


class TestReducedTimestreamView:
    """Test ReducedTimestreamView with DataTree input."""

    def _make_dt(self, n_samples: int = 5000, fsmp: float = 500.0) -> xr.DataTree:
        ds = xr.Dataset(
            {
                "I": (["time"], np.random.randn(n_samples)),
                "Q": (["time"], np.random.randn(n_samples)),
            },
        )
        ds["time"] = np.arange(n_samples) / fsmp
        ds.attrs["f_smp"] = fsmp
        reducer = TimestreamReducer(
            psd_nperseg=256,
            psd_stat_freq_range=(10.0, 100.0),
        )
        return reducer(ds)

    def test_view_from_datatree(self):
        """Test that ReducedTimestreamView resolves DataTree child node."""
        dt = self._make_dt()
        view = ReducedTimestreamView(dt)
        assert view.f_psd is not None
        assert view.I_psd is not None
        assert view.Q_psd is not None

    def test_view_data_matches_child(self):
        """Test that view data matches what's in the child dataset."""
        dt = self._make_dt()
        view = ReducedTimestreamView(dt)
        child_ds = get_reduced_ds(dt)

        np.testing.assert_array_equal(
            view.f_psd.values,
            child_ds[f"{NAMESPACE}.f_psd"].values,
        )
        np.testing.assert_array_equal(
            view.I_psd.values,
            child_ds[f"{NAMESPACE}.I_psd"].values,
        )

    def test_view_psd_median(self):
        """Test PSD median properties."""
        dt = self._make_dt()
        view = ReducedTimestreamView(dt)
        assert view.I_psd_median is not None
        assert view.Q_psd_median is not None

    def test_view_from_flat_dataset(self):
        """Test that ReducedTimestreamView also works on a flat Dataset."""
        dt = self._make_dt()
        child_ds = get_reduced_ds(dt)

        view = ReducedTimestreamView(child_ds)
        assert view.f_psd is not None
        assert view.I_psd is not None
