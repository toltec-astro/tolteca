"""Block creation consumes identified structure and publishes live views."""

import pytest
import xarray as xr
from tollan.accessor.xarray import ensure_datatree
from tollan.config import RuntimeContext
from tollan.pipeline import PIPELINE_CONTEXT_KEY
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.kids.core import ToltecKidsIOMapper
from tolteca_datamodels.toltec.metadata import ToltecSweepMetadata
from tolteca_kids import BlockCreate, BlockIdentify, SweepReduce
from tolteca_kids.accessor import ToltecKidsTreeAccessor, ToltecKidsTreeMapper
from tolteca_kids.core import Kids
from tolteca_kids.noop import NoOp


def test_block_creation_supplies_specialized_sweep_metadata(target_sweep_path):
    with open_toltec_kids(target_sweep_path) as source:
        acquisition = source.toltec_kids.meta
        tree = BlockCreate()(ensure_datatree(source))
        metadata = tree.toltec_kids.blocks[0].meta
        assert isinstance(metadata, ToltecSweepMetadata)
        assert metadata.obsnum == acquisition.obsnum
        assert metadata.roach == acquisition.roach
        assert metadata.n_chans == acquisition.n_chans


@pytest.mark.parametrize("prepared", [False, True])
def test_tree_reader_reuses_native_mapping(target_sweep_path, monkeypatch, prepared):
    with open_toltec_kids(target_sweep_path) as source:
        native = source.toltec_kids
        metadata = native.meta
        tree = ensure_datatree(source)
        if prepared:
            BlockCreate()(tree)

        def unexpected(*args, **kwargs):
            pytest.fail("An existing native mapping must not be resolved again")

        monkeypatch.setattr(ToltecKidsIOMapper, "from_data_source", unexpected)
        reader = ToltecKidsTreeAccessor(tree, mapper=native.mapper)
        assert reader.tree is tree
        assert reader._data_source is tree
        assert "_obj" not in reader.__dict__
        assert reader.mapper is native.mapper
        assert reader.meta == metadata
        assert not reader.data_source["Data.Toltec.Is"].variable._in_memory


def test_acquisition_mapping_survives_block_creation(target_sweep_path):
    with open_toltec_kids(target_sweep_path) as source:
        tree = ensure_datatree(source)
        reader = tree.toltec_kids
        acquisition = reader.data_source
        signal = acquisition["Data.Toltec.Is"].variable._data
        native_mapper = reader.mapper
        meta = reader.meta
        tree_mapper = ToltecKidsTreeMapper.from_data_source(tree)
        assert tree_mapper.get_node(tree, tree_mapper.schema.acquisition) is tree

        BlockCreate()(tree)
        assert reader.data_source is acquisition
        assert reader.mapper is native_mapper
        assert reader.meta == meta
        assert reader.data_source["Data.Toltec.Is"].variable._data is signal
        assert not acquisition["Data.Toltec.Is"].variable._in_memory

        copied = tree.copy(deep=False)
        tree_mapper = ToltecKidsTreeMapper.from_data_source(copied)
        assert (
            tree_mapper.get_node(copied, tree_mapper.schema.acquisition)
            is copied["raw"]
        )
        xr.testing.assert_identical(
            copied.toltec_kids.data_source, copied["raw"].dataset
        )
        assert copied.toltec_kids.meta == meta


def test_discovery_preserves_source_and_publishes_collection(
    target_sweep_path, monkeypatch
):
    read = NetCDF4ArrayWrapper.__getitem__

    def guarded(array, key):
        assert array.variable_name not in {"Data.Toltec.Is", "Data.Toltec.Qs"}
        return read(array, key)

    with open_toltec_kids(target_sweep_path) as source:
        monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", guarded)
        tree = ensure_datatree(source)
        tree["unrelated"] = xr.DataTree(dataset=xr.Dataset({"value": 7}))
        returned, context = BlockCreate()(tree, return_context=True)
        assert returned is tree
        assert context.completed
        assert context.data is tree.toltec_kids.blocks
        assert context.data.n_blocks == 1
        assert context.data[0].raw.sizes["time"] == source.sizes["time"]
        assert tree["unrelated"]["value"].item() == 7
        assert not tree.dataset.variables
        assert PIPELINE_CONTEXT_KEY not in tree["raw"].attrs
        assert PIPELINE_CONTEXT_KEY not in source.attrs
        assert not source["Data.Toltec.Is"].variable._in_memory
        restored = BlockCreate.load_context(BlockCreate.dump_context(context))
        assert restored.completed
        assert restored.data is None


@pytest.mark.parametrize("scope", ["/", "blocks", "blocks/0"])
def test_reopened_structure_needs_no_discovery_or_live_context(
    target_sweep_path, tmp_path, monkeypatch, scope
):
    target = tmp_path / "blocks.zarr"
    with open_toltec_kids(target_sweep_path) as source:
        tree = BlockCreate()(ensure_datatree(source))
        tree.toltec_kids.to_zarr(target, include_raw=False)

    def no_discovery(*args, **kwargs):
        pytest.fail("Existing block structures must not be rediscovered")

    monkeypatch.setattr(ToltecKidsIOMapper, "get_blocks", no_discovery)
    with xr.open_datatree(target, engine="zarr") as reopened:
        node = reopened if scope == "/" else reopened[scope]
        assert PIPELINE_CONTEXT_KEY not in node.attrs
        BlockIdentify()(node)
        assert BlockIdentify.get_context(node).completed
        assert BlockIdentify.get_context(node).data is None
        returned, context = BlockCreate()(node, return_context=True)
        assert returned is node
        assert context.completed
        expected = (
            node.toltec_kids.block if scope == "blocks/0" else node.toltec_kids.blocks
        )
        assert context.data is expected


def test_collision_records_failure_without_restructuring(target_sweep_path):
    with open_toltec_kids(target_sweep_path) as source:
        tree = ensure_datatree(source)
        BlockIdentify()(tree)
        assert BlockIdentify.get_context(tree).completed
        tree["raw"] = xr.DataTree(dataset=xr.Dataset({"unrelated": 5}))
        with pytest.raises(ValueError, match="conflicts"):
            BlockCreate()(tree)
        context = BlockCreate.get_context(tree)
        assert not context.completed
        assert context.data is None
        assert context.failures[0].kind == "exception"
        assert "blocks" not in tree.children
        assert "Data.Toltec.Is" in tree.dataset
        assert tree["raw"]["unrelated"].item() == 5


def test_reducers_reuse_discovery_and_noop_skips_it(target_sweep_path):
    kids = Kids(RuntimeContext())
    assert kids.sweep_pipeline.steps[0] is kids.sweep_reduce
    assert kids.timestream_pipeline.steps == [kids.timestream_reduce]
    with open_toltec_kids(target_sweep_path) as source:
        tree = ensure_datatree(source.toltec_kids.select_channels([0, 5]))
        kids.noop_pipeline(tree)
        assert NoOp.get_context(tree).completed
        assert not BlockCreate.has_context(tree)
        assert not BlockIdentify.has_context(tree)
        assert "blocks" not in tree.children
        SweepReduce()(tree)
        discovered = BlockCreate.get_context(tree)
        identified = BlockIdentify.get_context(tree)
        assert discovered.completed
        SweepReduce()(tree)
        assert BlockCreate.get_context(tree) is discovered
        assert BlockIdentify.get_context(tree) is identified
        assert SweepReduce.get_context(tree).completed
