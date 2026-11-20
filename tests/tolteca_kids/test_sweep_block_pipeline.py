"""Block-scoped sweep execution, publication and downstream analysis."""

import numpy as np
import pytest
import xarray as xr
from tollan.accessor.xarray import ensure_datatree
from tollan.pipeline import PIPELINE_CONTEXT_KEY, SequentialPipeline

from tolteca_kids import BlockCreate, BlockIdentify
from tolteca_kids.dataprod_output import DataProdOutput
from tolteca_kids.product import (
    open_sweep_product,
    read_product_provenance,
    read_sweep_context,
    write_sweep_product,
)
from tolteca_kids.sweep_reduce import SweepReduce
from tolteca_kidsproc.analysis import D21Analysis


@pytest.fixture
def tune(data_lmt_path):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    with xr.open_dataset(path) as native:
        yield native.toltec_kids.select_channels([5, 0])


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune"])
def test_transform_pipeline_publishes_without_file_runner(
    data_lmt_path, tmp_path, suffix
):
    source = next((data_lmt_path / "toltec/tcs/toltec0").glob(f"*1456*_{suffix}.nc"))
    pipeline = SequentialPipeline(
        steps=[
            BlockIdentify(),
            BlockCreate(),
            SweepReduce(),
            DataProdOutput(path=tmp_path, namespace="transformed", enabled=True),
        ]
    )
    with xr.open_dataset(source) as native:
        tree = ensure_datatree(native.toltec_kids.select_channels([5, 0]))
        assert pipeline(tree) is tree
        output = DataProdOutput.get_context(tree)
        assert output.completed
        assert output.data is not None
        assert (
            output.data.parent
            == tmp_path
            / "transformed"
            / tree.toltec_kids.meta.uid_raw_obs.removeprefix("tcs-")
            / "dp_raw_obs_proc"
        )
        expected_meta = tree.toltec_kids.meta.model_dump_json()
        expected_raw = tree["raw"].dataset.load()
        expected_axes = {
            name: tree[f"blocks/{name}"].dataset.load()
            for name in ("index", "chan", "probe")
        }
        expected_sweeps = {
            block.meta.block: block.sweep.data_source.load()
            for block in tree.toltec_kids.blocks
        }
        contexts = {
            step: step.get_context(tree)
            for step in (BlockIdentify, BlockCreate, SweepReduce)
        }

    path = output.data
    assert path is not None
    provenance = read_product_provenance(path).inputs["toltec0"]
    assert provenance.source is None
    assert provenance.config is None
    with open_sweep_product(path) as saved:
        assert all(PIPELINE_CONTEXT_KEY not in node.attrs for node in saved.subtree)
        assert saved.toltec_kids.meta.model_dump_json() == expected_meta
        xr.testing.assert_identical(saved["raw"].dataset, expected_raw)
        for name, expected in expected_axes.items():
            xr.testing.assert_identical(saved[f"blocks/{name}"].dataset, expected)
        for block, expected in expected_sweeps.items():
            xr.testing.assert_identical(
                saved.toltec_kids.blocks[block].sweep.data_source, expected
            )
        for step, context in contexts.items():
            restored = step.load_context(saved["tolteca_kids"].attrs[step.context_key])
            assert restored.model_dump() == context.model_dump()
            assert restored.completed
            assert restored.data is None
            assert not step.has_context(saved)
        assert DataProdOutput.context_key not in saved["tolteca_kids"].attrs


@pytest.mark.parametrize(
    "scope",
    ["native", "probed_native", "acquisition", "collection", "block", "attached_block"],
)
@pytest.mark.parametrize("include_raw", [True, False])
def test_scope_identity_and_detached_block_product(tune, tmp_path, scope, include_raw):  # noqa: C901 -- exercise each supported input scope
    if scope in {"native", "probed_native"}:
        tree = ensure_datatree(tune)
        if scope == "probed_native":
            BlockCreate()(tree)
            blocks = tree.toltec_kids.blocks
            prior_block = blocks[1]
    elif scope == "acquisition":
        tree = BlockCreate()(ensure_datatree(tune))
    elif scope == "collection":
        tree = BlockCreate()(ensure_datatree(tune)).toltec_kids.blocks.detach()
    elif scope == "attached_block":
        tree = BlockCreate()(ensure_datatree(tune))["blocks/1"]
    else:
        tree = BlockCreate()(ensure_datatree(tune)).toltec_kids.blocks[1].detach()
    tree["unrelated"] = xr.DataTree(dataset=xr.Dataset({"value": 7}))
    result, context = SweepReduce()(tree, return_context=True)
    assert result is tree
    assert context.block == 1
    assert context.data.meta.block == 1
    assert context.data.I.dims == ("iqlen", "sweep")
    if scope == "probed_native":
        assert tree.toltec_kids.blocks is blocks
        assert prior_block.tree is tree["blocks/1"]
        xr.testing.assert_identical(prior_block.sweep.I, context.data.I)
    assert tree["unrelated/value"].item() == 7
    for node in tree.subtree:
        if node.children:
            assert not node.dataset.variables, node.path
        if node is not tree:
            assert PIPELINE_CONTEXT_KEY not in node.attrs
    expected = context.data.I.load()
    product = write_sweep_product(
        tree if include_raw else tree.toltec_kids.detach(include_raw=False),
        tmp_path / scope,
    )
    with open_sweep_product(product) as saved:
        view = SweepReduce.view(saved, block=1)
        xr.testing.assert_identical(view.I, expected)
        np.testing.assert_array_equal(view.meta.chan_axis_table["chan"], [5, 0])
        assert len(view.meta.probe_axis_table) > 0
        assert read_sweep_context(saved).block == 1
        assert not SweepReduce.has_context(saved)
        if scope in {"block", "attached_block"}:
            selected = saved
        else:
            assert saved.toltec_kids.blocks.n_blocks == 2
            assert SweepReduce.view(saved, block=0).meta.block == 0
            selected = saved.toltec_kids.blocks[1].tree
        assert selected.toltec_kids.model_params is not None
        assert saved["unrelated/value"].item() == 7
        _check_raw_payload(tree, saved, selected, scope, include_raw)
        # Analysis consumes the same block scope in either representation.
        D21Analysis()(selected.kids.sweep, output=selected)
        assert selected.kids.d21.d21.size > 0


def _check_raw_payload(tree, saved, selected, scope, include_raw):
    assert any("raw" in node.children for node in saved.subtree) is include_raw
    if not include_raw:
        return
    expected = (
        tree.toltec_kids.block.raw
        if scope in {"block", "attached_block"}
        else tree.toltec_kids.blocks[1].raw
    )
    # Attached raw slices are Datasets; stored leaves expose DatasetViews.
    xr.testing.assert_identical(selected.toltec_kids.block.raw.copy(), expected.copy())


def test_alias_metadata_and_results_survive_product(tune, tmp_path):
    tree = BlockCreate()(ensure_datatree(tune))
    SweepReduce(block=1)(tree)
    SweepReduce(block=0, alias="first", force=True)(tree)
    product = write_sweep_product(tree, tmp_path / "aliases")
    with open_sweep_product(product) as saved:
        primary = read_sweep_context(saved)
        aliased = read_sweep_context(saved, alias="first")
        assert (primary.block, aliased.block) == (1, 0)
        assert aliased.make_step().config.alias == "first"
        assert primary.data is aliased.data is None
        for block in (0, 1):
            original = SweepReduce.view(tree, block=block, alias="first")
            reopened = SweepReduce.view(saved, block=block, alias="first")
            xr.testing.assert_identical(original.I, reopened.I)
            assert "sweep_first" in saved[f"blocks/{block}"].children


def test_configured_reducer_settings_round_trip(tune, tmp_path):
    from tolteca_kids import SweepReduce

    tree = BlockCreate()(ensure_datatree(tune))
    reducer = SweepReduce(compute_uncertainty=False)
    _, context = reducer(tree, return_context=True)
    assert context.config == reducer.config
    assert context.data.unc_I is None
    assert context.data.unc_I is None
    path = write_sweep_product(tree, tmp_path / "without_uncertainty")
    with open_sweep_product(path) as saved:
        restored = read_sweep_context(saved)
        assert restored.config == reducer.config
        assert SweepReduce.view(saved, block=restored.block).unc_I is None
