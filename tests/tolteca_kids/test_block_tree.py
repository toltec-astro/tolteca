"""Block scope analysis and persistence preserve data without duplicated raw arrays."""

import json

import numpy as np
import pytest
import xarray as xr
from scipy.signal import welch
from tollan.accessor.xarray import ensure_datatree
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tests.kids_helpers import apply_step
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_kids import (
    BlockCreate,
    SweepReduce,
    TimestreamReduce,
    ToltecBlocksView,
)
from tolteca_kidsproc.analysis import D21Analysis


def _tune(data_lmt_path):
    return next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))


def _assert_leaf_arrays(tree):
    for node in tree.subtree:
        if node.children:
            assert not node.dataset.variables, node.path


def test_reopened_index_is_eager_but_axes_and_payload_stay_lazy(
    data_lmt_path, tmp_path
):
    target = tmp_path / "acquisition.zarr"
    with open_toltec_kids(_tune(data_lmt_path)) as ds:
        BlockCreate()(
            ensure_datatree(ds.toltec_kids.select_channels([0, 1]))
        ).toltec_kids.to_zarr(target)
    with xr.open_datatree(target, engine="zarr", chunks={}) as tree:
        # An unselected invalid block must not prevent inspection of the index.
        tree["blocks/0"].attrs["meta"] = "invalid until selected"
        view = tree.toltec_kids.blocks
        assert all(v.variable._in_memory for v in view.data_source.values())
        assert not tree["blocks/probe/f_lo"].variable._in_memory
        assert not tree["raw/Data.Toltec.Is"].variable._in_memory
        assert view.block_index_table["block"].tolist() == [0, 1]
        assert view.block_index_table is view.meta.block_index_table
        assert not hasattr(view, "block_table")
        assert not tree["blocks/probe/f_lo"].variable._in_memory
        assert not tree["raw/Data.Toltec.Is"].variable._in_memory
        assert view[1].meta.block == 1
        assert view[1] is view[1]
        assert not view[1].raw["Data.Toltec.Is"].variable._in_memory
        assert view[1].meta.n_chans == 2
        assert tree.toltec_kids.data_kind & ToltecDataKind.sweep
        assert tree["blocks"].toltec_kids.meta.n_blocks == 2
        assert (
            tree["blocks/1"].toltec_kids.meta is tree["blocks/1"].toltec_kids.block.meta
        )
        _assert_leaf_arrays(tree)


@pytest.mark.parametrize("chunks", [None, {}])
def test_block_reduction_d21_and_compact_export(data_lmt_path, tmp_path, chunks):
    reducer, d21 = SweepReduce(), D21Analysis()
    full, single = tmp_path / "full.zarr", tmp_path / "single.zarr"
    with open_toltec_kids(_tune(data_lmt_path), chunks=chunks) as ds:
        source = ds.toltec_kids.select_channels([0, 1, 2])
        tree = BlockCreate()(ensure_datatree(source))
        meta = tree.toltec_kids.blocks[1].meta.model_dump_json()
        with pytest.raises(ValueError, match="select or reduce"):
            _ = tree.toltec_kids.sweep
        result = apply_step(reducer, tree, parent_namespace="blocks/1")
        result = d21(
            result["blocks/1"].kids.sweep,
            output=result,
            parent_namespace="blocks/1",
        )
        assert result is tree
        assert not result["blocks/0"].children
        assert apply_step(reducer, result["blocks/1"]) is result["blocks/1"]
        assert (
            d21(result["blocks/1"].kids.sweep, output=result["blocks/1"])
            is result["blocks/1"]
        )
        assert not any(
            "d21" in str(name) for name in result["blocks/1/sweep"].data_vars
        )
        reference = apply_step(reducer, source).toltec_kids.blocks[1].sweep
        actual = result["blocks/1"].toltec_kids.sweep
        assert (
            actual.meta.block,
            actual.meta.chan_axis_table["chan"].tolist(),
        ) == (1, [0, 1, 2])
        for name in ("I", "Q", "unc_I", "unc_Q", "frequency", "f_lo"):
            np.testing.assert_array_equal(
                getattr(actual, name), getattr(reference, name)
            )
        expected = reference.I
        assert (
            result["blocks/1/sweep"].toltec_kids.data_kind
            == ToltecDataKind.reduced_sweep
        )
        derivative_node = result["blocks/1"].kids.d21.data_source
        assert derivative_node.toltec_kids.data_kind == ToltecDataKind.d21
        recomputed = apply_step(
            SweepReduce(compute_uncertainty=False), result["blocks/1"]
        )
        # Sibling analyses persist; force recomputation after changing their input.
        d21(recomputed.kids.sweep, output=recomputed, force=True)
        derivative = result["blocks/1"].toltec_kids.d21.d21.load()
        result.toltec_kids.to_zarr(full)
        assert "sweep" in result["blocks/1"].children
        assert "raw" not in result["blocks/1"].children
        result["blocks/1"].toltec_kids.to_zarr(single)
    with xr.open_datatree(full, engine="zarr") as reopened:
        _assert_leaf_arrays(reopened)
        block = reopened.toltec_kids.blocks[1]
        assert "raw" not in block.tree.children
        assert "chan" not in block.tree.children
        assert json.loads(block.meta.model_dump_json()) == json.loads(meta)
        np.testing.assert_array_equal(block.sweep.I, expected)
        assert not any(
            str(name).startswith("Header.")
            for name in block.sweep.data_source.data_vars
        )
        xr.testing.assert_equal(block.tree.toltec_kids.d21.d21, derivative)
        assert block.raw.sizes["time"] == block.meta.n_times
    with xr.open_datatree(single, engine="zarr") as reopened:
        _assert_leaf_arrays(reopened)
        block = reopened.toltec_kids.block
        assert json.loads(block.meta.model_dump_json()) == json.loads(meta)
        assert np.all(block.chan_axis_table["block"] == 1)
        assert np.all(block.probe_axis_table["block"] == 1)
        assert block.raw.sizes["time"] == block.meta.n_times
        xr.testing.assert_equal(reopened.toltec_kids.d21.d21, derivative)
        fresh = d21(reopened.kids.sweep, output=reopened, force=True)
        xr.testing.assert_equal(fresh.toltec_kids.d21.d21, derivative)


def test_timestream_block_psd_without_raw_duplication(data_lmt_path, tmp_path):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*_09_27_57.nc"))
    reducer = TimestreamReduce(psd={"nperseg": 128})
    with open_toltec_kids(path) as source:
        ds = source.isel(time=slice(0, 512)).toltec_kids.select_channels([0, 1, 2])
        block = BlockCreate()(ensure_datatree(ds)).toltec_kids.blocks[0]
        result = apply_step(reducer, block.tree)
        assert apply_step(reducer, result) is result
        freq, reference = welch(
            block.raw["Data.Toltec.Is"].values.T,
            fs=block.meta.f_smp.to_value("Hz"),
            nperseg=128,
        )
        actual = result.toltec_kids.timestream
        np.testing.assert_array_equal(actual.f_psd, freq)
        np.testing.assert_allclose(actual.I_psd, reference)
        assert not any(
            "time" in var.dims for var in result.kids.psd.psd.data_vars.values()
        )
        target = tmp_path / "timestream.zarr"
        result.toltec_kids.to_zarr(target, include_raw=False)
    with xr.open_datatree(target, engine="zarr") as reopened:
        _assert_leaf_arrays(reopened)
        assert "raw" not in reopened.children
        np.testing.assert_allclose(reopened.toltec_kids.timestream.I_psd, reference)


def test_corrupt_axis_bounds_fail_before_payload_reads(data_lmt_path):
    with open_toltec_kids(_tune(data_lmt_path)) as ds:
        tree = BlockCreate()(ensure_datatree(ds))
        tree["blocks/index"]["chan_end"] = xr.DataArray([-1, -1], dims="block")
        with pytest.raises(ValueError, match="row bounds"):
            ToltecBlocksView(tree)


def test_block_export_rejects_unscoped_results(data_lmt_path, tmp_path):
    with open_toltec_kids(_tune(data_lmt_path)) as ds:
        tree = xr.DataTree(dataset=ds, children={"unscoped": xr.DataTree()})
        with pytest.raises(ValueError, match="outside the block layout"):
            tree.toltec_kids.to_zarr(tmp_path / "unscoped.zarr")


@pytest.mark.parametrize("chunks", [None, {}])
def test_tree_identity_and_detach_preserve_lazy_source(
    data_lmt_path, monkeypatch, chunks
):
    with open_toltec_kids(_tune(data_lmt_path), chunks=chunks) as ds:
        tree = BlockCreate()(ensure_datatree(ds))
        blocks = tree.toltec_kids.blocks
        assert ds.toltec_kids.block_index is ds.toltec_kids.block_index
        assert blocks is tree.toltec_kids.blocks
        assert blocks.tree is tree["blocks"]
        assert BlockCreate()(ensure_datatree(tree)) is tree
        assert BlockCreate()(ensure_datatree(tree["blocks/1"])) is blocks[1].tree

        def no_reads(*args):
            pytest.fail("Detaching must not read source arrays")

        monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", no_reads)
        for standalone in (
            blocks[1].detach(),
            tree["blocks/1"].toltec_kids.detach(),
        ):
            assert standalone.parent is None
            assert set(standalone.children) == {"raw", "chan", "probe"}
            assert standalone.toltec_kids.meta.block == 1
            assert not standalone.toltec_kids.block.raw[
                "Data.Toltec.Is"
            ].variable._in_memory
            assert (
                standalone.toltec_kids.block.raw.sizes["time"] == blocks[1].meta.n_times
            )
        assert blocks[1].tree.parent is blocks.tree
        assert not blocks[1].tree.children
        assert "Data.Toltec.Is" in ds
        assert not ds["Data.Toltec.Is"].variable._in_memory


def test_detached_collection_retains_raw_selections(data_lmt_path, tmp_path):
    with open_toltec_kids(_tune(data_lmt_path)) as ds:
        source = ds.toltec_kids.select_channels([0, 1])
        blocks = BlockCreate()(ensure_datatree(source)).toltec_kids.blocks
        collection = blocks.detach()
        assert collection.parent is None
        assert collection.toltec_kids.meta.n_blocks == 2
        for original, selected in zip(
            blocks, collection.toltec_kids.blocks, strict=True
        ):
            assert original.meta.model_dump_json() == selected.meta.model_dump_json()
            xr.testing.assert_equal(
                original.raw["Data.Toltec.Is"], selected.raw["Data.Toltec.Is"]
            )
        target = tmp_path / "collection.zarr"
        collection.toltec_kids.to_zarr(target)
    with xr.open_datatree(target, engine="zarr") as reopened:
        assert reopened.toltec_kids.meta.n_blocks == 2
        assert reopened.toltec_kids.blocks[1].raw.sizes["time"] == 1770
        assert len(reopened.toltec_kids.blocks[1].chan_axis_table) == 2


def test_detach_preserves_analysis_children(data_lmt_path):
    with open_toltec_kids(_tune(data_lmt_path)) as ds:
        tree = BlockCreate()(ensure_datatree(ds.toltec_kids.select_channels([0, 1])))
        result = apply_step(SweepReduce(), tree, parent_namespace="blocks/1")
        result = D21Analysis()(
            result["blocks/1"].kids.sweep,
            output=result,
            parent_namespace="blocks/1",
        )
        derivative = result["blocks/1"].toltec_kids.d21.d21
        detached = result.toltec_kids.detach()
        assert detached.parent is None
        assert set(detached["blocks/1"].children) == set(result["blocks/1"].children)
        assert "raw" in detached.children
        assert "sweep" in detached["blocks/1"].children
        assert "raw" not in detached["blocks/1"].children
        xr.testing.assert_equal(detached["blocks/1"].toltec_kids.d21.d21, derivative)
        detached["blocks/1"].attrs["detached_only"] = True
        assert "detached_only" not in result["blocks/1"].attrs
