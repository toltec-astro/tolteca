"""Repetition-averaging and acquisition reuse benchmarks.

Run with pytest test_sweep_performance.py --benchmark-only.
"""

import numpy as np
import pytest
import xarray as xr

from tests.kids_helpers import apply_step
from tolteca_kids import SweepReduce
from tolteca_kidsproc.reduction import reduce_sweep_samples

pytest.importorskip("pytest_benchmark")


@pytest.mark.parametrize(("n_chans", "n_steps"), [(100, 100), (500, 250), (1000, 500)])
@pytest.mark.parametrize("uncertainty", [True, False])
def test_repetition_averaging(benchmark, n_chans, n_steps, uncertainty):
    values = np.random.default_rng(42).normal(size=(n_steps * 10, n_chans)) * 1e6
    i_data = xr.DataArray(values, dims=("time", "chan"))
    q_data = -i_data
    slices = [slice(i * 10, (i + 1) * 10) for i in range(n_steps)]
    result = benchmark(
        reduce_sweep_samples,
        i_data,
        q_data,
        sample_dim="time",
        sample_slices=slices,
        compute_uncertainty=uncertainty,
    )
    assert result.I.shape == (n_chans, n_steps)
    assert ("unc_I" in result) == uncertainty


@pytest.mark.parametrize("force", [True, False])
def test_real_tune_reduction_and_reuse(benchmark, data_lmt_path, force):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    reducer = SweepReduce()
    with xr.open_dataset(path) as source:
        tree = apply_step(reducer, source)
        result = benchmark(apply_step, reducer, tree, force=force)
        assert result is tree
        assert result.toltec_kids.blocks.n_blocks == 2
