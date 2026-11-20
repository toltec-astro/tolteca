"""Explicit sweep repetition intervals, array geometry and units."""

import numpy as np
import pytest
import xarray as xr

from tolteca_kidsproc.reduction import reduce_sweep_samples


@pytest.mark.parametrize("shape", [(16,), (16, 1), (16, 4)])
@pytest.mark.parametrize("sample_first", [True, False])
@pytest.mark.parametrize("compute_uncertainty", [True, False])
def test_explicit_repetition_intervals(shape, sample_first, compute_uncertainty):
    """Unequal repetition counts, dimension ordering, units and coordinates."""
    values = np.random.default_rng(42).normal(size=shape) * 3e6
    dims = ("tick",) if len(shape) == 1 else ("tick", "detector")
    i_data = xr.DataArray(values, dims=dims).u.set("adu")
    if len(shape) > 1:
        i_data = i_data.assign_coords(detector=np.arange(shape[1]) * 3)
    if not sample_first:
        i_data = i_data.transpose(*reversed(dims))
    q_data = -i_data
    intervals = [slice(0, 3), slice(3, 10), slice(10, 16)]
    result = reduce_sweep_samples(
        i_data,
        q_data,
        sample_dim="tick",
        sample_slices=intervals,
        compute_uncertainty=compute_uncertainty,
    )
    assert result.I.dims == (*[d for d in i_data.dims if d != "tick"], "sweep")
    for name in ("I", "Q"):
        source = i_data if name == "I" else q_data
        reference = np.stack(
            [
                np.mean(source.isel(tick=part).values, axis=source.get_axis_num("tick"))
                for part in intervals
            ],
            axis=-1,
        )
        np.testing.assert_array_equal(result[name], reference)
        assert result[name].u.unit_str == "adu"
        if compute_uncertainty:
            expected = np.stack(
                [
                    np.std(
                        source.isel(tick=part).values, axis=source.get_axis_num("tick")
                    )
                    for part in intervals
                ],
                axis=-1,
            )
            np.testing.assert_array_equal(result[f"unc_{name}"], expected)
        else:
            assert f"unc_{name}" not in result
    if "detector" in i_data.coords:
        xr.testing.assert_identical(result.detector, i_data.detector)
    assert not result.attrs


def test_nan_propagation_and_one_sample():
    i_data = xr.DataArray([1.0, np.nan, 3.0], dims="sample")
    result = reduce_sweep_samples(
        i_data, -i_data, sample_dim="sample", sample_slices=[slice(0, 2), slice(2, 3)]
    )
    np.testing.assert_array_equal(result.I, [np.nan, 3.0])
    np.testing.assert_array_equal(result.unc_I, [np.nan, 0.0])
    assert result.I.u.unit is None


@pytest.mark.parametrize(
    "intervals", [[], [slice(1, 1)], [slice(-1, 2)], [slice(0, 5)], [slice(0, 2, 2)]]
)
def test_invalid_intervals(intervals):
    i_data = xr.DataArray([1.0, 2.0, 3.0], dims="sample")
    with pytest.raises(ValueError, match="Sample slices"):
        reduce_sweep_samples(
            i_data, i_data, sample_dim="sample", sample_slices=intervals
        )


def test_array_dimensions_must_match():
    i_data = xr.DataArray(np.ones((3, 2)), dims=("tick", "chan"))
    for q_data, dim in ((i_data.T, "tick"), (i_data, "unknown")):
        with pytest.raises(ValueError, match="I/Q must share"):
            reduce_sweep_samples(
                i_data, q_data, sample_dim=dim, sample_slices=[slice(0, 3)]
            )


def test_reduce_raw_sweep_mock_data():
    """Test raw sweep reduction with mock data."""

    n_chans = 100
    n_sweeps = 20
    n_samples_per_step = 10
    f_lo_center = 150e6

    # Create mock raw time-series data
    f_sweep = np.linspace(-2e6, 2e6, n_sweeps)
    f_lo = f_sweep + f_lo_center

    # Repeat f_lo for each sample
    f_lo_repeated = np.repeat(f_lo, n_samples_per_step)
    n_samples = len(f_lo_repeated)

    # Create I/Q data with deterministic signal
    # Base signal varying across channels
    chan_phase = np.linspace(0, 2 * np.pi, n_chans)
    base_i = np.sin(chan_phase)
    base_q = np.cos(chan_phase)

    # Add deterministic variation across samples
    I_raw = np.zeros((n_samples, n_chans))
    Q_raw = np.zeros((n_samples, n_chans))
    for i in range(n_samples):
        sample_phase = 2 * np.pi * i / n_samples_per_step
        I_raw[i, :] = base_i + 0.1 * np.sin(sample_phase + chan_phase)
        Q_raw[i, :] = base_q + 0.1 * np.cos(sample_phase + chan_phase)

    # Create raw dataset
    ds_raw = xr.Dataset(
        {
            "Data.Toltec.Is": (["time", "iqlen"], I_raw),
            "Data.Toltec.Qs": (["time", "iqlen"], Q_raw),
            "Data.Toltec.LoFreq": (["time"], f_lo_repeated),
        },
        attrs={"Header.Toltec.LoCenterFreq": f_lo_center},
    )

    # Numerical inputs provide repetition bounds and frequency coordinates.
    ds_reduced = reduce_sweep_samples(
        ds_raw["Data.Toltec.Is"],
        ds_raw["Data.Toltec.Qs"],
        sample_dim="time",
        sample_slices=[
            slice(i * n_samples_per_step, (i + 1) * n_samples_per_step)
            for i in range(n_sweeps)
        ],
    ).assign_coords(sweep=f_sweep)

    # Check output structure
    assert "I" in ds_reduced
    assert "Q" in ds_reduced
    assert "unc_I" in ds_reduced
    assert "unc_Q" in ds_reduced
    assert "sweep" in ds_reduced.dims
    assert "sweep" in ds_reduced.coords

    # Check dimensions
    assert ds_reduced.sizes["iqlen"] == n_chans
    assert ds_reduced.sizes["sweep"] == n_sweeps

    # Check that mean values are computed
    assert not np.isnan(ds_reduced["I"].values).all()
    assert not np.isnan(ds_reduced["Q"].values).all()

    # Check that uncertainties are positive
    assert (ds_reduced["unc_I"].values >= 0).all()
    assert (ds_reduced["unc_Q"].values >= 0).all()

    # Check sweep coordinate
    np.testing.assert_allclose(ds_reduced.coords["sweep"].values, f_sweep, rtol=1e-5)
