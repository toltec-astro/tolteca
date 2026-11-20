"""Identification publishes block structure without creating processing nodes."""

import pickle

import numpy as np
import pytest
import xarray as xr
from tollan.accessor.xarray import ensure_datatree
from tollan.pipeline import PIPELINE_CONTEXT_KEY
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.kids.core import ToltecKidsIOMapper
from tolteca_datamodels.toltec.metadata import (
    ToltecRawObsBlockMetadata,
    ToltecRawObsBlocksMetadata,
    ToltecRawObsMetadata,
)
from tolteca_kids import BlockCreate, BlockIdentify


def test_identification_keeps_native_layout_and_reuses_index(
    target_sweep_path, monkeypatch
):
    read = NetCDF4ArrayWrapper.__getitem__

    def guarded(array, key):
        assert array.variable_name not in {
            "Data.Toltec.Is",
            "Data.Toltec.Qs",
            "Header.Toltec.AdcSnapData",
        }
        return read(array, key)

    def no_creation(*args, **kwargs):
        pytest.fail("Identification and metadata access must not create blocks")

    monkeypatch.setattr(BlockCreate, "run", no_creation)
    with open_toltec_kids(target_sweep_path) as source:
        monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", guarded)
        tree = ensure_datatree(source)
        buffers = {name: var._data for name, var in tree.dataset.variables.items()}
        tree["unrelated"] = xr.DataTree(dataset=xr.Dataset({"value": 7}))
        returned, context = BlockIdentify()(tree, return_context=True)
        assert returned is tree
        assert context.completed
        index = context.data
        assert index is tree.toltec_kids.block_index
        assert type(index.meta) is ToltecRawObsBlocksMetadata
        assert isinstance(index.meta, ToltecRawObsMetadata)
        assert not isinstance(index.meta, ToltecRawObsBlockMetadata)
        assert index.meta.n_blocks == 1
        for name in ToltecRawObsMetadata.model_fields:
            assert getattr(index.meta, name) == getattr(tree.toltec_kids.meta, name)
        assert not hasattr(index, "blocks")
        assert index.index.sizes["block"] == 1
        assert index.chan.sizes["chan_entry"] == index.meta.n_chans
        assert index.probe.sizes["probe"] > 0
        meta = index.meta
        assert meta.block_index is index.index
        assert meta.block_index_table["block"].tolist() == [0]
        channels = meta.chan_axis_table
        probes = meta.probe_axis_table
        assert channels is meta.chan_axis_table
        assert probes is meta.probe_axis_table
        assert len(channels) == index.chan.sizes["chan_entry"]
        assert len(probes) == index.probe.sizes["probe"]
        np.testing.assert_array_equal(channels["block"], index.chan.block)
        np.testing.assert_array_equal(probes["sample_end"], index.probe.sample_end)
        assert np.shares_memory(channels["f_tone"].value, index.chan.f_tone.data)
        assert set(tree.children) == {"unrelated"}
        assert set(tree.dataset.variables) == set(buffers)
        assert all(
            tree.dataset[name].variable._data is buf for name, buf in buffers.items()
        )
        assert PIPELINE_CONTEXT_KEY not in source.attrs
        assert not source["Data.Toltec.Is"].variable._in_memory

        def no_discovery(*args, **kwargs):
            pytest.fail("The existing index must be reused")

        monkeypatch.setattr(ToltecKidsIOMapper, "get_blocks", no_discovery)
        BlockIdentify()(tree)
        assert BlockIdentify.get_context(tree).data is index
        restored = BlockIdentify.load_context(BlockIdentify.dump_context(context))
        assert restored.completed and restored.data is None


def test_collection_metadata_tables_are_lazy_and_excluded_from_json(
    target_sweep_path, monkeypatch
):
    from tolteca_datamodels.toltec import axes, metadata

    convert = axes.dataset_to_qtable
    calls = []

    def tracked(*args, **kwargs):
        calls.append(kwargs["dim"])
        return convert(*args, **kwargs)

    monkeypatch.setattr(axes, "dataset_to_qtable", tracked)
    monkeypatch.setattr(metadata, "dataset_to_qtable", tracked)
    with open_toltec_kids(target_sweep_path) as source:
        meta = source.toltec_kids.block_index.meta
        encoded = meta.model_dump_json(round_trip=True)
        assert calls == []
        assert meta.block_index is not None
        assert meta.block_index.sizes["block"] == meta.n_blocks
        assert calls == []
        index_table = meta.block_index_table
        assert index_table is meta.block_index_table
        assert index_table is not None
        assert index_table.colnames == list(meta.block_index.variables)
        for name in index_table.colnames:
            np.testing.assert_array_equal(index_table[name], meta.block_index[name])
            assert index_table[name].dtype == meta.block_index[name].dtype
            assert np.shares_memory(index_table[name], meta.block_index[name].data)
        assert calls == ["block"]
        assert meta.chan_axis_table is meta.chan_axis_table
        assert meta.probe_axis_table is meta.probe_axis_table
        assert calls == ["block", "chan_entry", "probe"]
        assert meta.model_dump_json(round_trip=True) == encoded
        restored = ToltecRawObsBlocksMetadata.model_validate_json(encoded)
        assert restored.model_dump() == meta.model_dump()
        assert restored.block_index is None
        assert restored.block_index_table is None
        assert restored.chan_axis_table is None
        assert restored.probe_axis_table is None


def test_creation_consumes_checkpointed_identification(target_sweep_path, monkeypatch):
    with open_toltec_kids(target_sweep_path) as source:
        tree = ensure_datatree(source)
        BlockIdentify()(tree)
        checkpoint = pickle.dumps(tree)
    restored = pickle.loads(checkpoint)  # noqa: S301 -- locally created checkpoint
    try:
        identified = BlockIdentify.get_context(restored)
        index = identified.data
        assert index is not None
        assert index.meta.chan_axis_table is not None
        assert index.meta.block_index is index.index
        assert len(index.meta.chan_axis_table) == index.chan.sizes["chan_entry"]

        def no_identification(*args, **kwargs):
            pytest.fail("Creation must consume the published index")

        monkeypatch.setattr(BlockIdentify, "__call__", no_identification)
        monkeypatch.setattr(ToltecKidsIOMapper, "get_blocks", no_identification)
        BlockCreate()(restored)
        assert BlockIdentify.get_context(restored) is identified
        assert BlockCreate.get_context(restored).completed
        xr.testing.assert_identical(restored["blocks/index"].to_dataset(), index.index)
        xr.testing.assert_identical(restored["blocks/chan"].to_dataset(), index.chan)
        xr.testing.assert_identical(restored["blocks/probe"].to_dataset(), index.probe)
        assert restored.toltec_kids.blocks[0].meta.obsnum == index.meta.obsnum
        assert isinstance(
            restored.toltec_kids.blocks[0].meta, ToltecRawObsBlocksMetadata
        )
        assert restored.toltec_kids.blocks.meta.model_dump() == index.meta.model_dump()
        blocks = restored.toltec_kids.blocks
        assert blocks.meta.block_index is blocks.data_source
        assert blocks.meta.block_index_table is blocks.block_index_table
        assert blocks.meta.chan_axis_table is blocks.chan_axis_table
        assert blocks.meta.probe_axis_table is blocks.probe_axis_table
        assert blocks[0].meta.block_index is blocks.meta.block_index
    finally:
        restored.close()


def test_applied_waveform_settings_follow_each_acquired_tune_block(
    data_lmt_path, tmp_path, monkeypatch
):
    import shutil

    import netCDF4

    source = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    target = tmp_path / source.name
    shutil.copyfile(source, target)
    with netCDF4.Dataset(target, "a") as raw:
        dim = raw["Header.Toltec.ToneFreq"].dimensions[0]
        for name, values in {
            "DriveAttenByBlock": [10, 7.25],
            "SenseAttenByBlock": [8, 9],
            "DacNormalization": [0.01, 0.02],
            "DacWaveformValid": [1, 1],
        }.items():
            raw.createVariable(f"Header.Toltec.{name}", "f8", (dim,))[:] = values
        raw.createDimension("dacSampleLen", 4)
        for name in ("DacWaveformI", "DacWaveformQ"):
            variable = raw.createVariable(
                f"Header.Toltec.{name}", "i2", (dim, "dacSampleLen")
            )
            variable[:] = 0
    read = NetCDF4ArrayWrapper.__getitem__

    def guarded(array, key):
        assert "DacWaveformI" not in array.variable_name
        assert "DacWaveformQ" not in array.variable_name
        return read(array, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", guarded)
    with open_toltec_kids(target) as acquired:
        tree = ensure_datatree(acquired)
        BlockIdentify()(tree)
        BlockCreate()(tree)
        blocks = list(tree.toltec_kids.blocks)
        assert [b.meta.atten_drive for b in blocks] == [10, 7.25]
        assert [b.meta.atten_sense for b in blocks] == [8, 9]
        assert [b.meta.dac_normalization for b in blocks] == [0.01, 0.02]
    with netCDF4.Dataset(target, "a") as raw:
        raw["Header.Toltec.DacWaveformValid"][1] = 0
    with (
        open_toltec_kids(target) as acquired,
        pytest.raises(ValueError, match="no valid applied DAC waveform"),
    ):
        BlockIdentify()(ensure_datatree(acquired))
