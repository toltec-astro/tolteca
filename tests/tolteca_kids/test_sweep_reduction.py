"""Repetition averaging and block-scoped sweep reduction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError
from tollan.accessor.xarray import ensure_datatree

from tests.kids_helpers import apply_step
from tolteca_kids import BlockCreate, ReducedSweepView, SweepReduce

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("field", ["detect_blocks", "sweep_axis", "time_axis"])
def test_acquisition_layout_is_not_a_reducer_setting(field):
    with pytest.raises(ValidationError, match="Extra inputs"):
        SweepReduce(**{field: False})


def test_native_metadata_remains_required():
    with pytest.raises((ValueError, KeyError)):
        apply_step(
            SweepReduce(),
            xr.Dataset({"I": ("time", [1.0, 2.0]), "Q": ("time", [2.0, 3.0])}),
        )


@pytest.mark.parametrize(
    "kind", ["dataset", "native_tree", "acquisition", "collection"]
)
def test_acquisition_entry_retains_raw_and_reuses_results(data_lmt_path, kind):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    with xr.open_dataset(path) as raw:
        source = raw.toltec_kids.select_channels([3, 0])
        if kind == "dataset":
            data = source
        elif kind == "native_tree":
            data = xr.DataTree(dataset=source)
        elif kind == "collection":
            data = BlockCreate()(ensure_datatree(source)).toltec_kids.blocks.detach()
        else:
            data = BlockCreate()(ensure_datatree(source))
        reducer = SweepReduce()
        result = apply_step(reducer, data)
        if isinstance(data, xr.DataTree):
            assert result is data
        assert not result.dataset.variables
        blocks = result.toltec_kids.blocks
        assert blocks.n_blocks == 2
        first = blocks[1].sweep
        expected = first.I.copy(deep=True)
        assert apply_step(reducer, result) is result
        assert blocks[1].sweep.I.data is first.I.data
        apply_step(reducer, result, force=True)
        assert blocks[1].sweep.I.data is not first.I.data
        xr.testing.assert_identical(blocks[1].sweep.I, expected)
        np.testing.assert_array_equal(first.meta.chan_axis_table["chan"], [3, 0])
        np.testing.assert_array_equal(first.chan_axis_table["chan"], [3, 0])
        np.testing.assert_array_equal(
            first.chan_axis_table["tone_amp"],
            first.meta.chan_axis_table["tone_amp"],
        )
        assert first.meta.block == 1
        assert "Header.Toltec.ObsNum" in blocks[1].raw
        with pytest.raises(ValueError, match="Select a block"):
            _ = ReducedSweepView(result).data_source
        no_unc = SweepReduce(compute_uncertainty=False)
        apply_step(no_unc, blocks[1].tree)
        assert no_unc.get_context(blocks[1].tree).config == no_unc.config
        assert blocks[1].sweep.unc_I is None
        assert blocks[0].sweep.unc_I is not None


def test_parent_namespace_within_a_caller_owned_tree(data_lmt_path):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    with xr.open_dataset(path) as raw:
        source = raw.toltec_kids.select_channels([0, 1])
        tree = xr.DataTree.from_dict({"other": xr.Dataset({"value": 42})})
        tree["measurement"] = (
            BlockCreate()(ensure_datatree(source)).toltec_kids.blocks[1].detach()
        )
        result = apply_step(SweepReduce(), tree, parent_namespace="measurement")
        assert result is tree
        assert set(tree.children) == {"measurement", "other"}
        assert tree["other/value"].item() == 42
        assert tree["measurement"].toltec_kids.sweep.meta.block == 1


def _check_raw_sweep_reduction(filepath: Path) -> None:
    """Assert correct reduction of a raw sweep file (shared by vna/targ tests)."""
    from tolteca_datamodels.toltec import open_toltec_kids
    from tolteca_kids import SweepReduce

    with open_toltec_kids(filepath) as ds_raw:
        # Verify raw data structure
        assert "Data.Toltec.Is" in ds_raw
        assert "Data.Toltec.Qs" in ds_raw
        assert "Data.Toltec.LoFreq" in ds_raw

        # Reduce the sweep (returns DataTree)
        dt = apply_step(SweepReduce(), ds_raw)
        assert isinstance(dt, xr.DataTree)
        block = dt.toltec_kids.blocks[0]
        ds_reduced = block.sweep.data_source

        # Check output structure
        assert "I" in ds_reduced
        assert "Q" in ds_reduced
        assert "unc_I" in ds_reduced
        assert "unc_Q" in ds_reduced
        assert "sweep" in ds_reduced.dims
        assert "sweep" in ds_reduced.coords

        # Check data types
        for field in ("I", "Q", "unc_I", "unc_Q"):
            assert ds_reduced[field].dtype in [np.float32, np.float64]

        # Verify no NaN values
        assert not np.any(np.isnan(ds_reduced["I"].values))
        assert not np.any(np.isnan(ds_reduced["Q"].values))
        assert not np.any(np.isnan(ds_reduced["unc_I"].values))
        assert not np.any(np.isnan(ds_reduced["unc_Q"].values))

        # Verify uncertainties are non-negative
        assert np.all(ds_reduced["unc_I"] >= 0)
        assert np.all(ds_reduced["unc_Q"] >= 0)

        # Verify key metadata attrs
        assert "f_center" in ds_reduced.attrs
        assert block.meta.n_times > 0
        assert "n_samples" in block.probe_axis_table.colnames

        # Check that reduction actually reduced sample count
        n_samples_original = block.meta.n_times
        n_sweeps = ds_reduced.sizes["sweep"]
        assert n_sweeps < n_samples_original


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep"])
def test_real_sweep_reduction(data_lmt_path: Path, suffix: str) -> None:
    """Reduce reference VNA and target sweeps without changing their values."""
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob(f"*1456*_{suffix}.nc"))
    _check_raw_sweep_reduction(path)


def test_real_tune_multi_block_data(data_lmt_path: Path) -> None:
    """Test multi-block detection and processing with a real tune file."""
    from tolteca_datamodels.toltec import open_toltec_kids
    from tolteca_kids import SweepReduce

    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    with open_toltec_kids(path) as ds_raw:
        # Verify raw data structure
        assert "Data.Toltec.Is" in ds_raw
        assert "Data.Toltec.Qs" in ds_raw
        assert "Data.Toltec.LoFreq" in ds_raw

        # Detect expected block count from LO frequency monotonicity
        f_lo = ds_raw["Data.Toltec.LoFreq"].values
        if f_lo.ndim > 1:
            f_lo = f_lo[0]
        expected_n_blocks = len(np.where(np.diff(f_lo) < 0)[0]) + 1

        tree = apply_step(SweepReduce(), ds_raw)
        blocks = tree.toltec_kids.blocks
        assert blocks.n_blocks == expected_n_blocks
        for block in blocks:
            reduced = block.sweep.data_source
            assert "block" not in reduced.dims
            for field in ("I", "Q", "unc_I", "unc_Q"):
                assert field in reduced
                assert not np.any(np.isnan(reduced[field].values))
