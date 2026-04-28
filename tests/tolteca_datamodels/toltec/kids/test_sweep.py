"""Edge case tests for SweepReducer."""

from __future__ import annotations

import json

import numpy as np
import xarray as xr

from tolteca_datamodels.toltec.kids import ReducedSweepView, SweepReducer

# Namespace constant (must match module __name__)
NAMESPACE = "tolteca_datamodels.toltec.kids.sweep"


def get_var_name(field: str) -> str:
    """Get namespaced variable name."""
    return f"{NAMESPACE}.{field}"


def get_reduced_ds(dt: xr.DataTree) -> xr.Dataset:
    """Extract the reduced sweep child dataset from a DataTree."""
    return dt.children[NAMESPACE].dataset


def create_minimal_sweep_dataset(
    n_channels: int = 10,
    n_times: int = 100,
    n_sweeps: int = 10,
    samples_per_sweep: int = 10,
    add_nan: bool = False,
) -> xr.Dataset:
    """Create a minimal valid sweep dataset for testing.

    Parameters
    ----------
    n_channels : int
        Number of channels
    n_times : int
        Total number of time samples
    n_sweeps : int
        Number of sweep steps
    samples_per_sweep : int
        Samples per sweep step
    add_nan : bool
        If True, add NaN values to test NaN handling
    """
    # Create data
    rng = np.random.default_rng(42)
    I = rng.standard_normal((n_times, n_channels))
    Q = rng.standard_normal((n_times, n_channels))

    if add_nan:
        # Add some NaN values
        I[5:10, 0] = np.nan
        Q[15:20, 1] = np.nan

    # Create sweep frequency pattern
    # LO frequency should be absolute (positive), sweep offset relative to center
    f_lo_center = 675e6  # Hz (typical TolTEC LO center)
    f_sweep_offset = np.linspace(-5e5, 5e5, n_sweeps)  # Hz offset
    f_lo_steps = f_lo_center + f_sweep_offset
    f_lo = np.repeat(f_lo_steps, samples_per_sweep)[:n_times]

    # Create dataset
    ds = xr.Dataset(
        {
            "Data.Toltec.Is": (["time", "iqlen"], I),
            "Data.Toltec.Qs": (["time", "iqlen"], Q),
            "Data.Toltec.LoFreq": (["time"], f_lo),  # Use correct field name
            "Header.Toltec.LoCenterFreq": f_lo_center,
            "Header.Toltec.NumSweepSteps": n_sweeps,
            "Header.Toltec.NumSamplesPerSweepStep": samples_per_sweep,
        },
        coords={
            "time": np.arange(n_times),
            "iqlen": np.arange(n_channels),
        },
    )

    return ds


class TestSweepReducerDataTree:
    """Test that SweepReducer returns xr.DataTree."""

    def test_returns_datatree(self):
        """Test that reducer returns a DataTree."""
        ds = create_minimal_sweep_dataset()
        reducer = SweepReducer()
        dt = reducer(ds)
        assert isinstance(dt, xr.DataTree)

    def test_root_has_raw_data(self):
        """Test that root node contains original raw data."""
        ds = create_minimal_sweep_dataset()
        reducer = SweepReducer()
        dt = reducer(ds)
        assert "Data.Toltec.Is" in dt.dataset
        assert "Data.Toltec.LoFreq" in dt.dataset

    def test_child_node_exists(self):
        """Test that the reduced sweep child node exists."""
        ds = create_minimal_sweep_dataset()
        reducer = SweepReducer()
        dt = reducer(ds)
        assert NAMESPACE in dt.children

    def test_child_has_reduced_data(self):
        """Test that the child node contains reduced I/Q data."""
        ds = create_minimal_sweep_dataset()
        reducer = SweepReducer()
        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)
        assert get_var_name("I") in child_ds
        assert get_var_name("Q") in child_ds


class TestSweepReducerEdgeCases:
    """Test SweepReducer with edge cases."""

    def test_single_sweep_step(self):
        """Test with only one sweep step."""
        ds = create_minimal_sweep_dataset(
            n_channels=5,
            n_times=10,
            n_sweeps=1,
            samples_per_sweep=10,
        )

        reducer = SweepReducer()
        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)

        # Should have 1 sweep
        assert child_ds.sizes["sweep"] == 1
        assert get_var_name("I") in child_ds
        assert get_var_name("Q") in child_ds

    def test_single_channel(self):
        """Test with only one channel."""
        ds = create_minimal_sweep_dataset(
            n_channels=1,
            n_times=100,
            n_sweeps=10,
            samples_per_sweep=10,
        )

        reducer = SweepReducer()
        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)

        # Should have 1 channel
        assert child_ds.sizes["iqlen"] == 1
        assert child_ds[get_var_name("I")].shape == (1, 10)

    def test_uneven_samples_per_sweep(self):
        """Test with uneven number of samples per sweep step."""
        # Create dataset with irregular sampling
        n_channels = 10
        n_sweeps = 5
        samples = [10, 12, 8, 15, 11]  # Different samples per sweep

        I_blocks = []
        Q_blocks = []
        f_lo_blocks = []

        rng = np.random.default_rng(42)
        f_lo_center = 675e6  # Hz
        f_sweep_offset = np.linspace(-5e5, 5e5, n_sweeps)

        for i, n_samp in enumerate(samples):
            I_blocks.append(rng.standard_normal((n_samp, n_channels)))
            Q_blocks.append(rng.standard_normal((n_samp, n_channels)))
            f_lo_blocks.append(np.full(n_samp, f_lo_center + f_sweep_offset[i]))

        I = np.vstack(I_blocks)
        Q = np.vstack(Q_blocks)
        f_lo = np.concatenate(f_lo_blocks)
        n_samples = len(f_lo)

        ds = xr.Dataset(
            {
                "Data.Toltec.Is": (["time", "iqlen"], I),
                "Data.Toltec.Qs": (["time", "iqlen"], Q),
                "Data.Toltec.LoFreq": (["time"], f_lo),  # Use correct field name
                "Header.Toltec.LoCenterFreq": 150e6,
            },
            coords={
                "time": np.arange(n_samples),
                "iqlen": np.arange(n_channels),
            },
        )

        reducer = SweepReducer()
        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)

        # Should detect 5 sweeps
        assert child_ds.sizes["sweep"] == n_sweeps
        # Check that samples per sweep are stored
        assert "samples_per_sweep" in child_ds.attrs

    def test_nan_handling_with_uncertainty(self):
        """Test that NaN values are handled correctly in uncertainty calculation."""
        ds = create_minimal_sweep_dataset(
            n_channels=5,
            n_times=100,
            n_sweeps=10,
            samples_per_sweep=10,
            add_nan=True,
        )

        reducer = SweepReducer(compute_uncertainty=True)
        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)

        # Reduced data should not have NaN (nanmean/nanstd used)
        I_reduced = child_ds[get_var_name("I")].values
        Q_reduced = child_ds[get_var_name("Q")].values

        # Allow NaN only if ALL samples for a sweep were NaN
        # (which shouldn't happen in this test)
        assert not np.all(np.isnan(I_reduced))
        assert not np.all(np.isnan(Q_reduced))

        # Uncertainty should be computed
        assert get_var_name("unc_I") in child_ds
        assert get_var_name("unc_Q") in child_ds

    def test_no_uncertainty_computation(self):
        """Test with uncertainty computation disabled."""
        ds = create_minimal_sweep_dataset(
            n_channels=10,
            n_times=100,
            n_sweeps=10,
            samples_per_sweep=10,
        )

        reducer = SweepReducer(compute_uncertainty=False)
        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)

        # Should have I and Q but no uncertainties
        assert get_var_name("I") in child_ds
        assert get_var_name("Q") in child_ds
        assert get_var_name("unc_I") not in child_ds
        assert get_var_name("unc_Q") not in child_ds

    def test_multi_block_detection(self):
        """Test automatic multi-block detection."""
        # Create data with 2 blocks (frequency decreases once)
        n_channels = 10
        n_sweeps_per_block = 10
        samples_per_sweep = 10

        rng = np.random.default_rng(42)
        f_lo_center = 675e6  # Hz
        f_sweep_offset = np.linspace(-5e5, 5e5, n_sweeps_per_block)

        # Block 1: sweep up
        f_lo_1 = np.repeat(f_lo_center + f_sweep_offset, samples_per_sweep)
        I_1 = rng.standard_normal((len(f_lo_1), n_channels))
        Q_1 = rng.standard_normal((len(f_lo_1), n_channels))

        # Block 2: sweep up again (frequency resets)
        f_lo_2 = np.repeat(f_lo_center + f_sweep_offset, samples_per_sweep)
        I_2 = rng.standard_normal((len(f_lo_2), n_channels))
        Q_2 = rng.standard_normal((len(f_lo_2), n_channels))

        # Concatenate
        I = np.vstack([I_1, I_2])
        Q = np.vstack([Q_1, Q_2])
        f_lo = np.concatenate([f_lo_1, f_lo_2])
        n_samples = len(f_lo)

        ds = xr.Dataset(
            {
                "Data.Toltec.Is": (["time", "iqlen"], I),
                "Data.Toltec.Qs": (["time", "iqlen"], Q),
                "Data.Toltec.LoFreq": (["time"], f_lo),  # Use correct field name
                "Header.Toltec.LoCenterFreq": 150e6,
            },
            coords={
                "time": np.arange(n_samples),
                "iqlen": np.arange(n_channels),
            },
        )

        reducer = SweepReducer(detect_blocks=True)
        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)

        # Should detect 2 blocks
        assert "block" in child_ds.dims
        assert child_ds.sizes["block"] == 2
        assert child_ds.attrs["is_multi_block"] is True
        assert child_ds.attrs["n_blocks"] == 2

        # Data should have block dimension
        assert child_ds[get_var_name("I")].dims == ("block", "iqlen", "sweep")

    def test_no_block_detection(self):
        """Test with block detection disabled."""
        # Create data that would normally trigger block detection
        ds = create_minimal_sweep_dataset(
            n_channels=10,
            n_times=100,
            n_sweeps=10,
            samples_per_sweep=10,
        )

        reducer = SweepReducer(detect_blocks=False)
        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)

        # Should not have block dimension
        assert "block" not in child_ds.dims
        assert child_ds.attrs.get("is_multi_block", False) is False

    def test_explicit_sweep_axis(self):
        """Test with explicitly specified sweep axis."""
        ds = create_minimal_sweep_dataset(
            n_channels=10,
            n_times=100,
            n_sweeps=10,
            samples_per_sweep=10,
        )

        # Explicitly specify sweep axis (normally auto-detected)
        reducer = SweepReducer(sweep_axis="Data.Toltec.LoFreq")
        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)

        assert child_ds.sizes["sweep"] == 10
        assert get_var_name("I") in child_ds

    def test_caching_behavior(self):
        """Test that caching works correctly when DataTree is passed back."""
        ds = create_minimal_sweep_dataset(
            n_channels=10,
            n_times=100,
            n_sweeps=10,
            samples_per_sweep=10,
        )

        reducer = SweepReducer()

        # First call - should compute
        dt1 = reducer(ds)

        # Second call with DataTree - should use cache (no recomputation)
        dt2 = reducer(dt1)

        # Results should be identical
        np.testing.assert_array_equal(
            get_reduced_ds(dt1)[get_var_name("I")].values,
            get_reduced_ds(dt2)[get_var_name("I")].values,
        )

        # Force recomputation
        dt3 = reducer(dt1, force=True)
        assert isinstance(dt3, xr.DataTree)
        assert get_var_name("I") in get_reduced_ds(dt3)

    def test_extreme_values(self):
        """Test with extreme ADU values."""
        ds = create_minimal_sweep_dataset(
            n_channels=10,
            n_times=100,
            n_sweeps=10,
            samples_per_sweep=10,
        )

        # Set extreme values (typical TolTEC range)
        ds["Data.Toltec.Is"].values[:] = np.random.uniform(
            -3e6, 3e6, ds["Data.Toltec.Is"].shape,
        )
        ds["Data.Toltec.Qs"].values[:] = np.random.uniform(
            -3e6, 3e6, ds["Data.Toltec.Qs"].shape,
        )

        reducer = SweepReducer()
        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)

        # Should complete without errors
        assert get_var_name("I") in child_ds
        assert get_var_name("Q") in child_ds

        # Values should be in reasonable range after averaging
        I_mean = child_ds[get_var_name("I")].values
        assert np.abs(I_mean).max() < 3.5e6  # Allow some headroom

    def test_missing_metadata_fields(self):
        """Test with missing optional metadata fields."""
        # Create minimal dataset without NumSweepSteps metadata
        ds = create_minimal_sweep_dataset(
            n_channels=10,
            n_times=100,
            n_sweeps=10,
            samples_per_sweep=10,
        )

        # Remove optional metadata
        del ds["Header.Toltec.NumSweepSteps"]
        del ds["Header.Toltec.NumSamplesPerSweepStep"]

        reducer = SweepReducer()
        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)

        # Should still work (reducer auto-detects from data)
        assert child_ds.sizes["sweep"] == 10
        assert get_var_name("I") in child_ds

    def test_config_stored_in_attrs(self):
        """Test that reducer config is stored in child dataset attributes."""
        ds = create_minimal_sweep_dataset(
            n_channels=10,
            n_times=100,
            n_sweeps=10,
            samples_per_sweep=10,
        )

        reducer = SweepReducer(
            detect_blocks=True,
            compute_uncertainty=True,
        )
        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)

        # Config should be in child dataset attrs
        assert f"{NAMESPACE}.reducer_config" in child_ds.attrs
        config = child_ds.attrs[f"{NAMESPACE}.reducer_config"]

        # Should be parseable JSON
        config_dict = json.loads(config)
        assert config_dict["detect_blocks"] is True
        assert config_dict["compute_uncertainty"] is True

    def test_preserved_metadata(self):
        """Test that original metadata is preserved in reduced dataset."""
        ds = create_minimal_sweep_dataset(
            n_channels=10,
            n_times=100,
            n_sweeps=10,
            samples_per_sweep=10,
        )

        # Add some metadata
        ds["Header.Toltec.ObsNum"] = 12345
        ds["Header.Toltec.ObsType"] = "vnasweep"

        reducer = SweepReducer()
        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)

        # Metadata should be preserved in reduced dataset
        assert "Header.Toltec.ObsNum" in child_ds
        assert child_ds["Header.Toltec.ObsNum"].item() == 12345
        assert "Header.Toltec.ObsType" in child_ds


class TestReducedSweepView:
    """Test ReducedSweepView with DataTree input."""

    def test_view_from_datatree(self):
        """Test that ReducedSweepView resolves DataTree child node."""
        ds = create_minimal_sweep_dataset(n_channels=5, n_times=100, n_sweeps=10)
        reducer = SweepReducer(compute_uncertainty=True)
        dt = reducer(ds)

        view = ReducedSweepView(dt)
        assert view.I is not None
        assert view.Q is not None
        assert view.unc_I is not None
        assert view.unc_Q is not None

    def test_view_data_matches_child(self):
        """Test that view data matches what's in the child dataset."""
        ds = create_minimal_sweep_dataset(n_channels=5, n_times=100, n_sweeps=10)
        reducer = SweepReducer()
        dt = reducer(ds)

        view = ReducedSweepView(dt)
        child_ds = get_reduced_ds(dt)

        np.testing.assert_array_equal(
            view.I.values,
            child_ds[get_var_name("I")].values,
        )

    def test_view_from_flat_dataset(self):
        """Test that ReducedSweepView also works on a flat Dataset."""
        ds = create_minimal_sweep_dataset(n_channels=5, n_times=100, n_sweeps=10)
        reducer = SweepReducer()
        dt = reducer(ds)
        child_ds = get_reduced_ds(dt)

        # Should work with flat Dataset too
        view = ReducedSweepView(child_ds)
        assert view.I is not None

    def test_view_sweep_property(self):
        """Test sweep frequency coordinate access."""
        ds = create_minimal_sweep_dataset(n_channels=5, n_times=100, n_sweeps=10)
        reducer = SweepReducer()
        dt = reducer(ds)

        view = ReducedSweepView(dt)
        assert view.sweep is not None
        assert view.sweep.dims == ("sweep",)

    def test_view_has_blocks(self):
        """Test has_blocks / n_blocks on multi-block DataTree."""
        n_channels = 5
        n_sweeps_per_block = 5
        samples_per_sweep = 5
        rng = np.random.default_rng(0)
        f_lo_center = 675e6
        f_sweep_offset = np.linspace(-5e5, 5e5, n_sweeps_per_block)

        f_lo_1 = np.repeat(f_lo_center + f_sweep_offset, samples_per_sweep)
        f_lo_2 = np.repeat(f_lo_center + f_sweep_offset, samples_per_sweep)
        I = rng.standard_normal((len(f_lo_1) + len(f_lo_2), n_channels))
        Q = rng.standard_normal((len(f_lo_1) + len(f_lo_2), n_channels))
        f_lo = np.concatenate([f_lo_1, f_lo_2])

        ds = xr.Dataset(
            {
                "Data.Toltec.Is": (["time", "iqlen"], I),
                "Data.Toltec.Qs": (["time", "iqlen"], Q),
                "Data.Toltec.LoFreq": (["time"], f_lo),
                "Header.Toltec.LoCenterFreq": f_lo_center,
            },
            coords={"time": np.arange(len(f_lo)), "iqlen": np.arange(n_channels)},
        )

        reducer = SweepReducer(detect_blocks=True)
        dt = reducer(ds)
        view = ReducedSweepView(dt)

        assert view.has_blocks is True
        assert view.n_blocks == 2
