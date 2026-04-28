"""Performance benchmarks for SweepReducer.

These benchmarks measure the performance of SweepReducer under various conditions.
Run with: pytest test_sweep_reducer_performance.py --benchmark-only
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tolteca_datamodels.toltec.kids import SweepReducer

# Skip if pytest-benchmark not available
pytest.importorskip("pytest_benchmark")

NAMESPACE = "tolteca_datamodels.toltec.kids.sweep"


def get_reduced_ds(dt: xr.DataTree) -> xr.Dataset:
    """Extract the reduced sweep child dataset from a DataTree."""
    return dt.children[NAMESPACE].dataset


def create_sweep_dataset(
    n_channels: int = 1000,
    n_times: int = 5000,
    n_sweeps: int = 500,
) -> xr.Dataset:
    """Create a realistic TolTEC sweep dataset for benchmarking."""
    samples_per_sweep = n_times // n_sweeps

    # Create realistic data
    rng = np.random.default_rng(42)
    I = rng.standard_normal((n_times, n_channels)) * 1e6  # ADU values
    Q = rng.standard_normal((n_times, n_channels)) * 1e6

    # Realistic LO frequency pattern
    f_lo_center = 675e6  # Hz
    f_sweep_offset = np.linspace(-5e5, 5e5, n_sweeps)
    f_lo = np.repeat(f_lo_center + f_sweep_offset, samples_per_sweep)[:n_times]

    ds = xr.Dataset(
        {
            "Data.Toltec.Is": (["time", "iqlen"], I),
            "Data.Toltec.Qs": (["time", "iqlen"], Q),
            "Data.Toltec.LoFreq": (["time"], f_lo),
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


class TestSweepReducerPerformance:
    """Performance benchmarks for SweepReducer."""

    def test_small_dataset_performance(self, benchmark):
        """Benchmark reduction of small dataset (100 channels, 1000 samples)."""
        ds = create_sweep_dataset(n_channels=100, n_times=1000, n_sweeps=100)
        reducer = SweepReducer()

        result = benchmark(reducer, ds)

        # Verify result
        child_ds = get_reduced_ds(result)
        assert child_ds.sizes["sweep"] == 100
        assert child_ds.sizes["iqlen"] == 100

    def test_medium_dataset_performance(self, benchmark):
        """Benchmark reduction of medium dataset (500 channels, 2500 samples)."""
        ds = create_sweep_dataset(n_channels=500, n_times=2500, n_sweeps=250)
        reducer = SweepReducer()

        result = benchmark(reducer, ds)

        # Verify result
        child_ds = get_reduced_ds(result)
        assert child_ds.sizes["sweep"] == 250
        assert child_ds.sizes["iqlen"] == 500

    def test_large_dataset_performance(self, benchmark):
        """Benchmark reduction of large dataset (1000 channels, 5000 samples)."""
        ds = create_sweep_dataset(n_channels=1000, n_times=5000, n_sweeps=500)
        reducer = SweepReducer()

        result = benchmark(reducer, ds)

        # Verify result
        child_ds = get_reduced_ds(result)
        assert child_ds.sizes["sweep"] == 500
        assert child_ds.sizes["iqlen"] == 1000

    def test_with_uncertainty_computation(self, benchmark):
        """Benchmark with uncertainty computation enabled."""
        ds = create_sweep_dataset(n_channels=1000, n_times=5000, n_sweeps=500)
        reducer = SweepReducer(compute_uncertainty=True)

        result = benchmark(reducer, ds)

        # Verify uncertainty was computed
        child_ds = get_reduced_ds(result)
        assert f"{NAMESPACE}.unc_I" in child_ds
        assert f"{NAMESPACE}.unc_Q" in child_ds

    def test_without_uncertainty_computation(self, benchmark):
        """Benchmark with uncertainty computation disabled."""
        ds = create_sweep_dataset(n_channels=1000, n_times=5000, n_sweeps=500)
        reducer = SweepReducer(compute_uncertainty=False)

        result = benchmark(reducer, ds)

        # Verify no uncertainty
        child_ds = get_reduced_ds(result)
        assert f"{NAMESPACE}.unc_I" not in child_ds
        assert f"{NAMESPACE}.unc_Q" not in child_ds

    def test_multi_block_detection_performance(self, benchmark):
        """Benchmark with multi-block detection."""
        # Create dataset with 2 blocks
        n_channels = 1000
        n_sweeps_per_block = 250
        samples_per_sweep = 10

        rng = np.random.default_rng(42)
        f_lo_center = 675e6
        f_sweep_offset = np.linspace(-5e5, 5e5, n_sweeps_per_block)

        # Block 1
        f_lo_1 = np.repeat(f_lo_center + f_sweep_offset, samples_per_sweep)
        I_1 = rng.standard_normal((len(f_lo_1), n_channels)) * 1e6
        Q_1 = rng.standard_normal((len(f_lo_1), n_channels)) * 1e6

        # Block 2
        f_lo_2 = np.repeat(f_lo_center + f_sweep_offset, samples_per_sweep)
        I_2 = rng.standard_normal((len(f_lo_2), n_channels)) * 1e6
        Q_2 = rng.standard_normal((len(f_lo_2), n_channels)) * 1e6

        # Concatenate
        I = np.vstack([I_1, I_2])
        Q = np.vstack([Q_1, Q_2])
        f_lo = np.concatenate([f_lo_1, f_lo_2])

        ds = xr.Dataset(
            {
                "Data.Toltec.Is": (["time", "iqlen"], I),
                "Data.Toltec.Qs": (["time", "iqlen"], Q),
                "Data.Toltec.LoFreq": (["time"], f_lo),
                "Header.Toltec.LoCenterFreq": f_lo_center,
            },
            coords={
                "time": np.arange(len(f_lo)),
                "iqlen": np.arange(n_channels),
            },
        )

        reducer = SweepReducer(detect_blocks=True)

        result = benchmark(reducer, ds)

        # Verify multi-block
        child_ds = get_reduced_ds(result)
        assert "block" in child_ds.dims
        assert child_ds.sizes["block"] == 2

    def test_caching_overhead(self, benchmark):
        """Benchmark the overhead of caching mechanism (DataTree passed back)."""
        ds = create_sweep_dataset(n_channels=1000, n_times=5000, n_sweeps=500)
        reducer = SweepReducer()

        # First call to produce DataTree
        dt = reducer(ds)

        # Benchmark cached call (passing DataTree triggers cache check)
        result = benchmark(reducer, dt)

        assert get_reduced_ds(result).sizes["sweep"] == 500


class TestRealDataPerformance:
    """Performance benchmarks with real TolTEC data."""

    def test_real_vnasweep_performance(self, benchmark, real_vnasweep_file):
        """Benchmark reduction of a real VNA sweep file."""
        import xarray as xr

        ds = xr.open_dataset(real_vnasweep_file)
        reducer = SweepReducer()

        result = benchmark(reducer, ds)

        child_ds = get_reduced_ds(result)
        assert child_ds.sizes["iqlen"] > 0
        assert child_ds.sizes["sweep"] > 0
