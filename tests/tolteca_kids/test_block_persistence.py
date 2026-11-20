"""Processing-tree persistence and borrowed detector-data lifetime."""

import json

import numpy as np
import pytest
import xarray as xr
from tollan.accessor.xarray import ensure_datatree
from tollan.pipeline import PIPELINE_CONTEXT_KEY
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_kids import BlockCreate, ToltecBlocksView


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
@pytest.mark.parametrize("engine", ["netcdf4", "zarr"])
def test_axis_tree_roundtrip_without_detector_reads(
    data_lmt_path, tmp_path, monkeypatch, suffix, engine
):
    pattern = f"*1456*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob(pattern)
    read = NetCDF4ArrayWrapper.__getitem__

    def guarded(array, key):
        assert array.variable_name not in {
            "Data.Toltec.Is",
            "Data.Toltec.Qs",
            "Header.Toltec.AdcSnapData",
        }
        return read(array, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", guarded)
    with open_toltec_kids(path) as source:
        prepared = BlockCreate()(ensure_datatree(source))
        view = prepared.toltec_kids.blocks
        assert view is prepared.toltec_kids.blocks
        assert view.n_blocks == (2 if suffix == "tune" else 1)
        assert (
            len(view.chan_axis_table) == view.n_blocks * source.toltec_kids.meta.n_chans
        )
        assert all(block.meta._block_view is block for block in view)
        expected_index = view.block_index_table
        store = BlockCreate()(ensure_datatree(source)).toltec_kids.detach(
            include_raw=False
        )
        # Direct xarray export needs portable attributes; omit live step state.
        store.attrs.pop(PIPELINE_CONTEXT_KEY, None)
        target = tmp_path / "axes"
        if engine == "zarr":
            store.to_zarr(target, consolidated=False)
        else:
            store.to_netcdf(target)
        expected = [
            (
                meta.model_dump_json(round_trip=True),
                meta.chan_axis_table,
                meta.probe_axis_table,
            )
            for meta in [block.meta for block in view]
        ]
        assert not source["Data.Toltec.Is"].variable._in_memory
    # Persisted metadata and axes do not depend on the original file lifetime.
    with xr.open_datatree(
        target, engine=engine, **({"consolidated": False} if engine == "zarr" else {})
    ) as tree:
        reopened = tree.toltec_kids.blocks
        assert tree.toltec_kids.meta.n_blocks_max >= reopened.n_blocks
        assert reopened.meta.block_index is reopened.data_source
        assert reopened.block_index_table is reopened.meta.block_index_table
        assert set(reopened.block_index_table.colnames) == set(expected_index.colnames)
        for name in expected_index.colnames:
            np.testing.assert_array_equal(
                reopened.block_index_table[name], expected_index[name]
            )
            assert reopened.block_index_table[name].dtype == expected_index[name].dtype
        assert reopened.meta.chan_axis_table is reopened.chan_axis_table
        assert reopened.meta.probe_axis_table is reopened.probe_axis_table
        for meta, (encoded, chan, probe) in zip(
            [block.meta for block in reopened], expected, strict=True
        ):
            assert json.loads(meta.model_dump_json(round_trip=True)) == json.loads(
                encoded
            )
            for actual, reference in [
                (meta.chan_axis_table, chan),
                (meta.probe_axis_table, probe),
            ]:
                assert actual.colnames == reference.colnames
                for name in reference.colnames:
                    np.testing.assert_array_equal(actual[name], reference[name])
                    assert getattr(actual[name], "unit", None) == getattr(
                        reference[name], "unit", None
                    )
                    assert actual[name].dtype == reference[name].dtype
        with pytest.raises(ValueError, match="not present"):
            _ = reopened[0].raw


def test_tune_source_slices_and_shared_column_storage(data_lmt_path):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    with open_toltec_kids(path) as source:
        prepared = BlockCreate()(ensure_datatree(source))
        view = prepared.toltec_kids.blocks
        block = view[1].raw
        meta = view[1].meta
        assert block.sizes["time"] == meta.n_times == 1770
        assert not block["Data.Toltec.Is"].variable._in_memory
        assert meta.probe_axis_table["sample_start"][0] == 1770
        assert meta.probe_axis_table["sample_end"][-1] == 3540
        np.testing.assert_array_equal(
            block["Header.Toltec.ToneFreq"],
            source["Header.Toltec.ToneFreq"].isel(numSweeps=1),
        )
        np.testing.assert_array_equal(
            block["Data.Toltec.Is"],
            source["Data.Toltec.Is"].isel(time=slice(1770, 3540)),
        )
        selected = view.chan_axis.select_block(1).data_source
        assert np.shares_memory(
            selected["f_tone"].data, view.chan_axis.data_source["f_tone"].data
        )
        for axis in (meta.chan_axis_table, meta.probe_axis_table):
            assert np.all(axis["block"] == 1)
        for invalid in (-1, 2):
            with pytest.raises(ValueError, match="out of range"):
                _ = view[invalid].meta
            with pytest.raises(ValueError, match="out of range"):
                _ = view[invalid].raw


def test_channel_count_invariant_on_reopened_metadata(data_lmt_path):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    with open_toltec_kids(path) as source:
        tree = (
            BlockCreate()(ensure_datatree(source))
            .toltec_kids.detach(include_raw=False)
            .copy(deep=True)
        )
    changed = json.loads(tree["blocks/1"].attrs["meta"])
    changed["n_chans"] += 1
    tree["blocks/1"].attrs["meta"] = json.dumps(changed)
    with pytest.raises(ValueError, match="cannot change"):
        _ = ToltecBlocksView(tree)[1]


# netCDF4's native string writer uses ndarray.shape assignment on NumPy 2.5.
@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array:DeprecationWarning"
)
@pytest.mark.parametrize("engine", ["netcdf4", "zarr"])
def test_full_tree_reopens_raw_block_data(target_sweep_path, tmp_path, engine):
    target = tmp_path / "raw_blocks"
    with open_toltec_kids(target_sweep_path) as source:
        raw = source.toltec_kids.select_channels([0, 5, 10]).isel(time=slice(0, 37))
        tree = BlockCreate()(ensure_datatree(raw))
        view = tree.toltec_kids.blocks
        expected = view[0].raw["Data.Toltec.Is"].load()
        if engine == "zarr":
            tree.toltec_kids.to_zarr(target, consolidated=False)
        else:
            tree.attrs.pop(PIPELINE_CONTEXT_KEY, None)
            tree.to_netcdf(target)
    options = {"consolidated": False} if engine == "zarr" else {}
    with xr.open_datatree(target, engine=engine, **options) as tree:
        reopened = tree.toltec_kids.blocks
        assert reopened[0].meta.n_chans == 3
        assert reopened[0].meta.probe_axis_table["n_samples"][-1] == 7
        xr.testing.assert_equal(reopened[0].raw["Data.Toltec.Is"], expected)


@pytest.mark.parametrize("engine", ["netcdf4", "zarr"])
def test_empty_tune_axis_tree(data_lmt_path, tmp_path, engine):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    target = tmp_path / "empty_axes"
    with open_toltec_kids(path) as source:
        empty = BlockCreate()(
            ensure_datatree(source.isel(time=slice(0, 0)))
        ).toltec_kids.blocks
        assert empty.n_blocks == 0
        assert list(empty) == []
        assert len(empty.chan_axis_table) == len(empty.probe_axis_table) == 0
        assert {"block", "step", "n_samples", "f_lo"} <= set(
            empty.probe_axis_table.colnames
        )
        tree = BlockCreate()(ensure_datatree(empty.tree.parent)).toltec_kids.detach(
            include_raw=False
        )
        if engine == "zarr":
            tree.toltec_kids.to_zarr(target, consolidated=False)
        else:
            tree.attrs.pop(PIPELINE_CONTEXT_KEY, None)
            tree.to_netcdf(target)
    options = {"consolidated": False} if engine == "zarr" else {}
    with xr.open_datatree(target, engine=engine, **options) as tree:
        assert [item.meta for item in tree.toltec_kids.blocks] == []
        assert len(tree.toltec_kids.blocks.probe_axis_table) == 0
