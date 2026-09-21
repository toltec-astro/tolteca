"""Reducer mappings and lazy-input read boundaries."""

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from tolteca_datamodels.toltec.kids import ReducedSweepView, SweepReducer


@pytest.fixture
def sweep_input():
    values = np.arange(24, dtype=float).reshape(12, 2)
    return xr.Dataset(
        {
            "Data.Toltec.Is": (("time", "chan"), values, {"units": "adu"}),
            "Data.Toltec.Qs": (("time", "chan"), -values, {"units": "adu"}),
            "Data.Toltec.LoFreq": ("time", np.repeat([99.0, 100.0, 101.0], 4)),
            "Header.Toltec.LoCenterFreq": 100.0,
        }
    )


def test_mapped_aliases_keep_values_and_units(sweep_input):
    expected = ReducedSweepView(SweepReducer()(sweep_input))
    generic = sweep_input.rename(
        {"Data.Toltec.Is": "I", "Data.Toltec.Qs": "Q", "Data.Toltec.LoFreq": "f_lo"}
    ).drop_vars("Header.Toltec.LoCenterFreq")
    generic.attrs["f_center"] = 100.0
    original = generic.copy(deep=True)
    actual = ReducedSweepView(SweepReducer()(generic))
    for name in ("I", "Q", "unc_I", "unc_Q", "sweep", "f_lo"):
        xr.testing.assert_identical(getattr(actual, name), getattr(expected, name))
    array_I = actual.I
    assert array_I is not None
    assert array_I.u.unit_str == "adu"
    xr.testing.assert_identical(generic, original)


def test_lo_vector_computed_once_for_multiple_blocks(sweep_input):
    calls = []

    @dask.delayed
    def read_lo():
        calls.append(True)
        return np.tile(np.repeat([99.0, 100.0, 101.0], 2), 2)

    sweep_input["Data.Toltec.LoFreq"] = (
        "time",
        da.from_delayed(read_lo(), shape=(12,), dtype=float),
    )
    with dask.config.set(scheduler="synchronous"):
        view = ReducedSweepView(SweepReducer()(sweep_input))
    assert len(calls) == 1
    assert view.n_blocks == 2
    array_I = view.I
    assert array_I is not None
    assert array_I.shape == (2, 2, 3)


def test_mapped_block_dimension_needs_no_coordinate(sweep_input):
    reduced = ReducedSweepView(SweepReducer()(sweep_input)).data_source
    with_blocks = reduced.expand_dims(block=2).drop_vars("block", errors="ignore")
    view = ReducedSweepView(with_blocks)
    assert view.has_blocks
    assert view.n_blocks == 2


def test_center_header_location_and_missing_center(sweep_input):
    expected = ReducedSweepView(SweepReducer()(sweep_input)).sweep
    attr_source = sweep_input.drop_vars("Header.Toltec.LoCenterFreq")
    attr_source.attrs["Header.Toltec.LoCenterFreq"] = 100.0
    xr.testing.assert_identical(
        ReducedSweepView(SweepReducer()(attr_source)).sweep, expected
    )
    without_center = sweep_input.drop_vars("Header.Toltec.LoCenterFreq")
    xr.testing.assert_identical(
        ReducedSweepView(SweepReducer()(without_center)).sweep, expected
    )


def test_ambiguous_center_uses_mapper_validation(sweep_input):
    sweep_input.attrs["Header.Toltec.LoCenterFreq"] = 100.0
    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        SweepReducer()(sweep_input)
