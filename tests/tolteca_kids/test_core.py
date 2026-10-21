"""Reducer-only pipeline, result views, and durable product tests."""

import json
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr
import zarr
from pydantic import Field, ValidationError
from tollan.accessor.xarray import ensure_datatree
from tollan.config import RuntimeContext
from tollan.pipeline import PIPELINE_CONTEXT_KEY, SequentialPipeline
from typer.testing import CliRunner

import tolteca_kids.sweep_reduce as sweep_reduce_module
from tests.kids_helpers import apply_step
from tolteca_datamodels.data_prod import DataProdIndex
from tolteca_datamodels.toltec.file import (
    get_info_from_source,
    guess_info_from_sources,
    read_info_from_sources,
)
from tolteca_kids import BlockCreate
from tolteca_kids.cli import app
from tolteca_kids.config import DataProdOutputConfig, SweepReduceConfig
from tolteca_kids.core import Kids, KidsConfig
from tolteca_kids.dataprod_output import DataProdOutput
from tolteca_kids.noop import NoOp
from tolteca_kids.product import (
    open_sweep_product,
    read_product_provenance,
    read_sweep_context,
    write_sweep_product,
)
from tolteca_kids.runner import create_kids, run_source
from tolteca_kids.sweep_reduce import SweepReduce


@pytest.fixture
def raw_sweep(target_sweep_path):
    with xr.open_dataset(target_sweep_path) as native:
        return native.toltec_kids.select_channels([8, 9]).load()


def test_pipeline_owns_steps_and_preserves_working_tree(raw_sweep):
    kids = Kids(RuntimeContext())
    assert kids.sweep_reduce is kids.sweep_reduce
    assert kids.sweep_pipeline.steps == [
        kids.sweep_reduce,
        kids.sweep_check,
        kids.sweep_check_plot,
        kids.kids_find,
        kids.tone_prepare,
    ]
    original = raw_sweep.copy(deep=True)
    working = ensure_datatree(raw_sweep)
    working["unrelated"] = xr.DataTree(dataset=xr.Dataset({"a": 3}))
    kids.block_identify(working)
    kids.block_create(working)
    block = working["blocks/0"]
    assert kids.sweep_pipeline(block) is block
    view = SweepReduce.view(working, block=0)
    expected = apply_step(
        SweepReduce(),
        BlockCreate()(ensure_datatree(raw_sweep)).toltec_kids.blocks[0].tree,
    ).toltec_kids.sweep
    xr.testing.assert_identical(view.I, expected.I)
    xr.testing.assert_identical(view.unc_I, expected.unc_I)
    xr.testing.assert_identical(raw_sweep, original)
    assert "unrelated" in working.children
    assert not SweepReduce.has_context(working)
    assert SweepReduce.get_context(block).completed
    assert SweepReduce.get_context(block).config == kids.config.sweep_reduce
    assert block.attrs[PIPELINE_CONTEXT_KEY][SweepReduce] is SweepReduce.get_context(
        block
    )
    assert SweepReduce.get_context(block).data.I.data is view.I.data


def test_standalone_step_and_repeat_preserve_results(raw_sweep):
    working = ensure_datatree(raw_sweep)
    step = SweepReduce()
    result, context = step(working, return_context=True)
    assert result is working
    assert context.completed
    first_I = SweepReduce.view(working, block=0).I
    assert first_I is not None
    first = first_I.data
    step(working)
    second_I = SweepReduce.view(working, block=0).I
    assert second_I is not None
    assert second_I.data is first
    assert SweepReduce.get_context(working).model_dump() == context.model_dump()


def test_aliases_keep_separate_result_nodes(raw_sweep):
    working = ensure_datatree(raw_sweep)
    other = "second_pass"
    pipeline = SequentialPipeline(
        steps=[
            SweepReduce(),
            SweepReduce(force=True, alias=other),
        ]
    )
    pipeline(working)
    assert SweepReduce.view(working, block=0).unc_I is not None
    assert SweepReduce.view(working, block=0, alias=other).unc_I is not None
    assert SweepReduce.get_context(working, alias=other).completed
    assert SweepReduce.get_context(working, alias=other).config.force
    assert not SweepReduce.get_context(working).config.force


def test_alias_context_recreates_the_right_step_after_serialization(raw_sweep):
    working = ensure_datatree(raw_sweep)
    _, original = SweepReduce()(working, return_context=True)
    alias = "repeat"
    _, aliased = SweepReduce(alias=alias)(working, return_context=True)
    assert type(original.make_step()) is SweepReduce
    assert type(aliased.make_step()) is SweepReduce
    assert type(SweepReduce.get_context(working).make_step()) is SweepReduce
    assert (
        type(SweepReduce.get_context(working, alias=alias).make_step()) is SweepReduce
    )
    restored = SweepReduce.load_context(SweepReduce.dump_context(aliased))
    assert type(restored.make_step()) is SweepReduce
    assert restored.model_dump() == aliased.model_dump()
    assert restored.make_step().config.alias == alias
    assert original.config.alias is None


@pytest.mark.parametrize("primary_first", [False, True])
def test_alias_reuses_its_result_without_recomputing(
    raw_sweep, monkeypatch, primary_first
):
    working = ensure_datatree(raw_sweep)
    working["unrelated"] = xr.DataTree(dataset=xr.Dataset({"a": 3}))
    calls = []
    reduce = sweep_reduce_module.reduce_sweep_samples

    def count(reducer, *args, **kwargs):
        calls.append(args)
        return reduce(reducer, *args, **kwargs)

    monkeypatch.setattr(sweep_reduce_module, "reduce_sweep_samples", count)
    primary = None
    if primary_first:
        SweepReduce()(working)
        primary = SweepReduce.view(working, block=0).data_source.copy(deep=True)
    alias = "cached"
    SweepReduce(alias=alias)(working)
    first = SweepReduce.view(working, block=0, alias=alias).I
    assert first is not None
    assert len(calls) == 1
    SweepReduce(alias=alias)(working)
    repeated = SweepReduce.view(working, block=0, alias=alias).I
    assert repeated is not None
    assert repeated.data is first.data
    assert len(calls) == 1
    SweepReduce(force=True, alias=alias)(working)
    forced = SweepReduce.view(working, block=0, alias=alias).I
    assert forced is not None
    assert forced.data is not first.data
    xr.testing.assert_identical(forced, first)
    assert len(calls) == 2
    assert working["unrelated"]["a"].item() == 3
    if primary_first:
        xr.testing.assert_identical(
            SweepReduce.view(working, block=0).data_source, primary
        )
    else:
        assert "sweep" not in working["blocks/0"]


def test_alias_keeps_its_cache_when_primary_is_replaced(raw_sweep, monkeypatch):
    working = ensure_datatree(raw_sweep)
    alias = "cached"
    SweepReduce(alias=alias)(working)
    aliased = SweepReduce.view(working, block=0, alias=alias).I
    assert aliased is not None
    SweepReduce(force=True)(working)
    primary = SweepReduce.view(working, block=0).I
    assert primary is not None
    assert primary.data is not aliased.data

    def unexpected(*args, **kwargs):
        pytest.fail("unforced repeat computed an already available result")

    monkeypatch.setattr(sweep_reduce_module, "reduce_sweep_samples", unexpected)
    SweepReduce(alias=alias)(working)
    repeated = SweepReduce.view(working, block=0, alias=alias).I
    assert repeated is not None
    assert repeated.data is aliased.data


def test_alias_cache_still_checks_numerical_configuration(raw_sweep):
    working = BlockCreate()(ensure_datatree(raw_sweep))
    alias = "different_config"
    without_uncertainty = apply_step(
        SweepReduce(compute_uncertainty=False),
        working.toltec_kids.blocks[0].detach(),
    )
    working[f"blocks/0/{SweepReduce.result_path(alias=alias)}"] = without_uncertainty[
        "sweep"
    ]
    assert SweepReduce.view(working, block=0, alias=alias).unc_I is None
    SweepReduce(alias=alias)(working)
    assert SweepReduce.view(working, block=0, alias=alias).unc_I is not None


def test_output_alias_context_recreates_the_right_step(tmp_path, raw_sweep):
    working = ensure_datatree(raw_sweep)
    original = DataProdOutput(path=tmp_path).create_context(working)
    alias = "second"
    aliased = DataProdOutput(path=tmp_path, alias=alias).create_context(working)
    assert type(original.make_step()) is DataProdOutput
    assert type(aliased.make_step()) is DataProdOutput
    assert DataProdOutput.get_context(working) is original
    assert DataProdOutput.get_context(working, alias=alias) is aliased
    restored = DataProdOutput.load_context(DataProdOutput.dump_context(aliased))
    assert restored.model_dump() == aliased.model_dump()
    assert restored.make_step().config.alias == alias
    assert original.config.alias is None
    assert type(DataProdOutput.get_context(working).make_step()) is DataProdOutput
    assert (
        type(DataProdOutput.get_context(working, alias=alias).make_step())
        is DataProdOutput
    )


def test_pipeline_passes_reducer_settings(raw_sweep):
    kids = Kids(
        RuntimeContext(
            {
                "kids": {
                    "sweep_reduce": {
                        "compute_uncertainty": False,
                    }
                }
            }
        )
    )
    working = ensure_datatree(raw_sweep)
    kids.block_create(working)
    kids.sweep_pipeline(working["blocks/0"])
    assert SweepReduce.view(working, block=0).unc_I is None
    assert SweepReduce.view(working, block=0).unc_Q is None


def test_disabled_pipeline_does_not_reduce(raw_sweep, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("disabled stage invoked the reducer")

    monkeypatch.setattr(SweepReduce, "__call__", unexpected)
    kids = Kids(RuntimeContext({"kids": {"sweep_reduce": {"enabled": False}}}))
    working = ensure_datatree(raw_sweep)
    kids.block_create(working)
    block = working["blocks/0"]
    assert kids.sweep_pipeline(block) is block
    assert not SweepReduce.has_context(block)


def test_invalid_block_keeps_incomplete_context_without_arrays(raw_sweep, tmp_path):
    working = ensure_datatree(raw_sweep)
    with pytest.raises(IndexError, match="Block 1 out of range"):
        SweepReduce(block=1)(working)
    assert SweepReduce.has_context(working)
    context = SweepReduce.get_context(working)
    assert not context.completed
    assert context.config.block == 1
    assert context.data is None
    assert "sweep" not in working["blocks/0"]
    with pytest.raises(ValueError, match="not completed"):
        write_sweep_product(working, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()


def test_reduction_context_is_attached_during_run(raw_sweep, monkeypatch):
    working = ensure_datatree(raw_sweep)
    run = SweepReduce.run
    seen = []

    def inspect(cls, data, context):
        assert cls.has_context(data)
        assert cls.get_context(data) is context
        assert not context.completed
        seen.append(context)
        result = run(data, context)
        assert cls.get_context(data) is context
        assert not context.completed
        return result

    monkeypatch.setattr(SweepReduce, "run", classmethod(inspect))
    _, context = SweepReduce()(working, return_context=True)
    assert seen == [context]
    assert seen[0] is context
    assert context.completed
    assert SweepReduce.get_context(working).model_dump() == context.model_dump()
    assert working.attrs[PIPELINE_CONTEXT_KEY][SweepReduce] is context
    assert PIPELINE_CONTEXT_KEY not in raw_sweep.attrs


def test_failed_rerun_replaces_success_context_and_cannot_publish(raw_sweep, tmp_path):
    working = ensure_datatree(raw_sweep)
    SweepReduce(block=0)(working)
    first = SweepReduce.get_context(working)
    first_I = SweepReduce.view(working, block=0).I
    assert first_I is not None
    values = first_I.data
    with pytest.raises(IndexError, match="Block 1 out of range"):
        SweepReduce(block=1)(working)
    current = SweepReduce.get_context(working)
    assert current is not first
    assert current.config.block == 1
    assert not current.completed
    assert first.completed
    after_failure = SweepReduce.view(working, block=0).I
    assert after_failure is not None
    assert after_failure.data is values
    with pytest.raises(ValueError, match="not completed"):
        write_sweep_product(working, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    SweepReduce(block=0)(working)
    assert SweepReduce.get_context(working).completed
    after_retry = SweepReduce.view(working, block=0).I
    assert after_retry is not None
    assert after_retry.data is values


def test_failed_alias_keeps_the_primary_result_and_context(raw_sweep):
    working = ensure_datatree(raw_sweep)
    SweepReduce()(working)
    before = SweepReduce.get_context(working)
    result = SweepReduce.view(working, block=0).data_source.copy(deep=True)
    alias = "invalid_block"
    with pytest.raises(IndexError, match="Block 1 out of range"):
        SweepReduce(block=1, alias=alias)(working)
    assert SweepReduce.has_context(working, alias=alias)
    assert not SweepReduce.get_context(working, alias=alias).completed
    assert SweepReduce.get_context(working) == before
    xr.testing.assert_identical(SweepReduce.view(working, block=0).data_source, result)


def test_conditional_recipe_and_handler_isolation(tmp_path, target_sweep_path):
    recipe = tmp_path / "10_kids.yaml"
    recipe.write_text("""
- source: {kids: {sweep_reduce: {force: false}}}
- enable_if: roach == 0 & file_suffix in ['targsweep', 'tune']
  source: {kids: {sweep_reduce: {force: true}}}
""")
    rc = RuntimeContext.from_cli(config_paths=[tmp_path])
    guessed = get_info_from_source(target_sweep_path)
    assert guessed is not None
    info = next(read_info_from_sources([target_sweep_path]).itertuples(index=False))
    first = Kids(rc.with_config(validation_context=cast("Any", info)._asdict()))
    second = Kids(
        rc.with_config(validation_context={"roach": 8, "file_suffix": "targsweep"})
    )
    assert first.config.sweep_reduce.force
    assert not second.config.sweep_reduce.force
    assert info.roach == 0
    assert info.file_suffix == "targsweep"
    assert rc.runtime_info.validation_context == {}
    with first.temporary_config(
        overrides={"sweep_reduce": {"enabled": False}}
    ) as third:
        assert not third.config.sweep_reduce.enabled
    assert first.config.sweep_reduce.enabled


@pytest.mark.parametrize(
    "config",
    [
        {"sweep_reduc": {}},
        {"kids_find": {}},
        {"sweep_reduce": {"detect_blocks": False}},
        {"sweep_reduce": {"compute_uncertanty": False}},
    ],
)
def test_unknown_config_is_rejected(config):
    with pytest.raises(ValidationError):
        Kids(RuntimeContext({"kids": config}))


def test_config_roundtrip():
    config = KidsConfig()
    assert KidsConfig.model_validate_json(config.model_dump_json()) == config


def test_export_reopen_and_exclusive_destination(tmp_path, raw_sweep):
    raw_sweep = raw_sweep.assign_coords(time=np.arange(raw_sweep.sizes["time"]))
    working = ensure_datatree(raw_sweep)
    kids = Kids(RuntimeContext())
    kids.block_create(working)
    kids.sweep_pipeline(working["blocks/0"])
    path = write_sweep_product(
        working,
        tmp_path / "product",
        output_config=DataProdOutputConfig(
            run={"name": {"generator": "counter", "value": 0, "key": "test"}}
        ),
    )
    with open_sweep_product(path) as reopened:
        actual = SweepReduce.view(reopened, block=0)
        expected = SweepReduce.view(working, block=0)
        expected_dataset = expected.data_source
        xr.testing.assert_identical(actual.data_source, expected_dataset)
        assert (
            read_sweep_context(reopened).model_dump()
            == SweepReduce.get_context(working["blocks/0"]).model_dump()
        )
        assert not SweepReduce.has_context(reopened)
        assert not reopened.dataset.data_vars
        assert "Data.Toltec.Is" not in actual.data_source
        assert "time" not in actual.data_source.dims
        signal = actual.I
        assert signal is not None
        np.testing.assert_array_equal(signal.iqlen, [8, 9])
    with pytest.raises(FileExistsError):
        write_sweep_product(
            working,
            tmp_path / "product",
            output_config=DataProdOutputConfig(
                run={"name": {"generator": "counter", "value": 0, "key": "other"}}
            ),
        )


def test_incomplete_product_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_sweep_product(tmp_path)


def test_product_has_consolidated_metadata_before_completion(tmp_path, raw_sweep):
    working = ensure_datatree(raw_sweep)
    SweepReduce()(working)
    path = write_sweep_product(working, tmp_path / "product")
    assert (path / "array.zarr" / ".zmetadata").is_file()
    with xr.open_datatree(
        path / "array.zarr", engine="zarr", consolidated=True
    ) as tree:
        assert set(tree.children) == {"toltec0"}
        assert not tree.dataset.variables
        assert tree["toltec0"].toltec_kids.meta.uid_raw_obs_file == (
            working.toltec_kids.meta.uid_raw_obs_file
        )
        expected = SweepReduce.view(working, block=0)
        snapshot = expected.data_source
        xr.testing.assert_identical(
            SweepReduce.view(tree["toltec0"], block=0).data_source, snapshot
        )
    assert read_product_provenance(path).inputs["toltec0"].config is None


def test_failed_consolidation_does_not_publish_completion(
    tmp_path, raw_sweep, monkeypatch
):
    working = ensure_datatree(raw_sweep)
    SweepReduce()(working)

    def fail(*args, **kwargs):
        msg = "simulated consolidation failure"
        raise OSError(msg)

    monkeypatch.setattr(zarr, "consolidate_metadata", fail)
    path = tmp_path / "product"
    with pytest.raises(OSError, match="consolidation failure"):
        write_sweep_product(working, path)
    assert not list(path.rglob("array.zarr"))
    with pytest.raises(FileNotFoundError):
        open_sweep_product(path)


def test_reader_uses_consolidated_metadata_without_child_metadata_reads(
    tmp_path, raw_sweep, monkeypatch
):
    from zarr.storage import LocalStore

    working = ensure_datatree(raw_sweep)
    SweepReduce()(working)
    path = write_sweep_product(working, tmp_path / "product")
    keys = []
    get = LocalStore.get

    async def record(store, key, *args, **kwargs):
        keys.append(key)
        return await get(store, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", record)
    with open_sweep_product(path) as tree:
        assert read_sweep_context(tree).completed
    assert ".zmetadata" in keys
    assert not any(
        key.startswith("toltec0/")
        and key.endswith((".zattrs", ".zarray", ".zgroup", "zarr.json"))
        for key in keys
    )


def test_reader_uses_native_fallback_without_consolidated_metadata(tmp_path, raw_sweep):
    working = ensure_datatree(raw_sweep)
    SweepReduce()(working)
    path = write_sweep_product(working, tmp_path / "product")
    # Keep the test's original index for inspection, but make it unavailable.
    (path / "array.zarr" / ".zmetadata").rename(tmp_path / "metadata.saved")
    expected = SweepReduce.view(working, block=0)
    snapshot = expected.data_source
    with pytest.warns(RuntimeWarning, match="non-consolidated metadata"):
        tree = open_sweep_product(path)
    with tree:
        xr.testing.assert_identical(
            SweepReduce.view(tree, block=0).data_source, snapshot
        )


def test_export_excludes_live_context_without_mutating_working_tree(
    tmp_path, raw_sweep
):
    raw_sweep.attrs["user_label"] = "interactive test"
    working = ensure_datatree(raw_sweep)
    NoOp()(working)
    live_context = NoOp.get_context(working)
    SweepReduce()(working)
    reduced = SweepReduce.view(working, block=0).data_source
    live_store = working.attrs[PIPELINE_CONTEXT_KEY]
    assert PIPELINE_CONTEXT_KEY not in reduced.attrs
    expected = reduced.copy(deep=False)
    context = SweepReduce.get_context(working)
    path = write_sweep_product(working, tmp_path / "product")
    with open_sweep_product(path) as reopened:
        saved = SweepReduce.view(reopened, block=0)
        xr.testing.assert_identical(saved.data_source, expected)
        serialized = reopened["tolteca_kids"].attrs[SweepReduce.context_key]
        assert isinstance(serialized, str)
        assert "step_cls" not in json.loads(serialized)
        restored = read_sweep_context(reopened)
        assert restored.model_dump() == context.model_dump()
        assert restored.completed
        assert restored.data is None
        assert context.data is not None
        assert not SweepReduce.has_context(reopened)
        assert reopened.attrs["user_label"] == "interactive test"
        assert PIPELINE_CONTEXT_KEY not in saved.data_source.attrs
    assert SweepReduce.context_key not in reduced.attrs
    assert working.attrs[PIPELINE_CONTEXT_KEY] is live_store
    assert NoOp.get_context(working) is live_context
    assert PIPELINE_CONTEXT_KEY not in raw_sweep.attrs


def test_export_does_not_silently_discard_unknown_opaque_metadata(tmp_path, raw_sweep):
    raw_sweep.attrs["unsupported_resource"] = object()
    working = ensure_datatree(raw_sweep)
    SweepReduce()(working)
    path = tmp_path / "product"
    with pytest.raises(TypeError, match="Invalid attribute"):
        write_sweep_product(working, path)
    assert not list(path.rglob("array.zarr"))
    assert "unsupported_resource" in working.attrs


def test_resolved_workflow_config_replay(tmp_path, target_sweep_path, monkeypatch):
    source = target_sweep_path
    info = next(iter(read_info_from_sources([source]).itertuples(index=False)))
    runtime = RuntimeContext(
        [
            {
                "source": {
                    "kids": {
                        "output": {
                            "enabled": True,
                            "path": str(tmp_path),
                            "namespace": "first",
                        }
                    }
                }
            },
            {
                "enable_if": "roach == 0",
                "source": {"kids": {"sweep_reduce": {"block": 0}}},
            },
        ]
    )
    path = run_source(cast("str", info.source), runtime)
    assert path is not None
    provenance = read_product_provenance(path).inputs["toltec0"]
    assert provenance.uid_raw_obs_file == info.uid_raw_obs_file
    assert provenance.source is not None
    assert provenance.source.path == target_sweep_path
    assert provenance.config is not None
    assert provenance.config.sweep_reduce.block == 0
    index = DataProdIndex.read(path)
    # Immutable product index is a publication-time snapshot. Completion is
    # recorded on the namespace invocation after all output publications finish.
    from tolteca_datamodels.data_prod_store import DataProdStore

    completed = (
        DataProdStore(tmp_path, namespace="first").read_namespace().runs[index.run.uid]
    )
    assert completed.status == "completed"
    offset = completed.time_completed.utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 0
    payload = index.provenance["inputs"]["toltec0"]
    assert isinstance(payload["source"], dict)
    assert "input" not in payload
    assert "data_kind" not in payload
    assert payload["config"]["sweep_reduce"]["compute_uncertainty"] is True
    snapshot = tmp_path / "replay.yaml"
    snapshot.write_text(
        json.dumps(
            Kids.prepare_runtime_config_data(
                provenance.config.model_dump(mode="json", round_trip=True)
            )
        )
    )

    class LaterSweepConfig(SweepReduceConfig):
        compute_uncertainty: bool = False

    class LaterKidsConfig(KidsConfig):
        sweep_reduce: LaterSweepConfig = Field(default_factory=LaterSweepConfig)

    monkeypatch.setattr(Kids, "config_model_cls", LaterKidsConfig)
    assert not Kids(RuntimeContext()).config.sweep_reduce.compute_uncertainty

    result = CliRunner().invoke(
        app,
        [
            "--config_paths",
            str(snapshot),
            "--kids.output.namespace",
            "replayed",
            "--",
            str(source),
        ],
    )
    assert result.exit_code == 0, result.exception
    with (
        open_sweep_product(path) as first,
        open_sweep_product(
            tmp_path / "replayed/145606-0-1/dp_raw_obs_proc"
        ) as replayed,
    ):
        xr.testing.assert_identical(
            SweepReduce.view(first, block=0).data_source,
            SweepReduce.view(replayed, block=0).data_source,
        )


@pytest.mark.parametrize("missing", [False, True])
def test_invalid_provenance_rejected_before_array_open(
    tmp_path, raw_sweep, monkeypatch, missing
):
    working = ensure_datatree(raw_sweep)
    SweepReduce()(working)
    path = write_sweep_product(working, tmp_path / "product")
    index = DataProdIndex.read(path)
    provenance = None if missing else dict(index.provenance, format="unknown")
    index.model_copy(update={"provenance": provenance}).write(path, overwrite=True)

    def unexpected(*args, **kwargs):
        pytest.fail("invalid provenance attempted to open arrays")

    monkeypatch.setattr(xr, "open_datatree", unexpected)
    with pytest.raises(ValidationError):
        open_sweep_product(path)


def test_workflow_and_later_step_settings_have_separate_snapshots(
    tmp_path, target_sweep_path
):
    from tolteca_kids.runner import run_data

    with xr.open_dataset(target_sweep_path) as raw:
        working = run_data(raw, target_sweep_path, RuntimeContext())
        SweepReduce(compute_uncertainty=False, force=True)(working)
        path = write_sweep_product(working, tmp_path)
    request = read_product_provenance(path).inputs["toltec0"].config
    assert request.sweep_reduce.compute_uncertainty
    with open_sweep_product(path) as saved:
        context = read_sweep_context(saved)
        assert not context.config.compute_uncertainty
        assert SweepReduce.view(saved, block=context.block).unc_I is None


@pytest.mark.parametrize("unit", [None, "adu", "V"])
def test_preserve_signal_units(tmp_path, raw_sweep, unit):
    for name in ("Data.Toltec.Is", "Data.Toltec.Qs"):
        if unit is not None:
            raw_sweep[name].attrs["units"] = unit
    working = ensure_datatree(raw_sweep)
    SweepReduce()(working)
    path = write_sweep_product(working, tmp_path / "product")
    with open_sweep_product(path) as reopened:
        view = SweepReduce.view(reopened, block=0)
        for name in ("I", "Q", "unc_I", "unc_Q"):
            array = getattr(view, name)
            assert array.u.unit_str == unit
            if unit is None:
                assert array.u.quantity is array
            else:
                assert str(array.u.quantity.unit) == unit


def test_cli_config_and_reduction(tmp_path, target_sweep_path, monkeypatch):
    source = tmp_path / "source.nc"
    source.symlink_to(target_sweep_path)
    runner = CliRunner()
    with monkeypatch.context() as patch:

        def unexpected(*args, **kwargs):
            pytest.fail("--show_config invoked reduction")

        patch.setattr(SweepReduce, "__call__", unexpected)
        result = runner.invoke(
            app,
            [
                str(source),
                "--show_config",
                "--kids.sweep_reduce.block",
                "0",
            ],
        )
        assert result.exit_code == 0, result.output
        assert '"block": 0' in result.output
    product = tmp_path / "product"
    result = runner.invoke(
        app,
        [
            str(source),
            "--kids.output.enabled",
            "true",
            "--kids.output.path",
            str(product),
        ],
    )
    assert result.exit_code == 0, result.exception
    with open_sweep_product(product / "scratch/145606-0-1/dp_raw_obs_proc") as tree:
        assert read_sweep_context(tree).completed
    result = runner.invoke(
        app,
        [
            str(source),
            "--kids.output.enabled",
            "true",
            "--kids.output.path",
            str(product),
        ],
    )
    assert result.exit_code == 0, result.exception
    indexes = list(
        (product / "scratch/145606-0-1/dp_raw_obs_proc").glob("*/index.yaml")
    )
    assert len(indexes) == 2


def test_prepare_does_not_read_signal(target_sweep_path):
    import dask.array as da
    from dask import delayed

    raw_sweep = xr.load_dataset(target_sweep_path)

    @delayed
    def unexpected():
        pytest.fail("metadata preparation read I/Q arrays")

    raw_sweep["Data.Toltec.Is"].data = da.from_delayed(
        unexpected(), shape=raw_sweep["Data.Toltec.Is"].shape, dtype=float
    )
    raw_sweep["Data.Toltec.Qs"].data = da.from_delayed(
        unexpected(), shape=raw_sweep["Data.Toltec.Qs"].shape, dtype=float
    )
    table = guess_info_from_sources([target_sweep_path])
    with table.toltec_file.open(io_objs=[raw_sweep]) as opened:
        info = next(opened.drop(columns=["io_obj"]).itertuples(index=False))
    kids = Kids(
        RuntimeContext().with_config(validation_context=cast("Any", info)._asdict())
    )
    assert info.roach == 0
    assert kids.config.sweep_reduce.enabled


def test_pipeline_output_config_does_not_create_products(tmp_path):
    products = tmp_path / "products"
    runtime = RuntimeContext({"kids": {"output": {"path": str(products)}}})
    kids = Kids(runtime)
    assert kids.config.output.path == products
    assert kids.config.output.enabled is False
    assert not products.exists()


@pytest.mark.parametrize("name", ["data_lmt_path", "dataprod_toltec_path", "select"])
def test_kids_config_does_not_accept_file_access_settings(name):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        KidsConfig.model_validate({name: "unused"})


def test_cli_product_root_is_used_by_enabled_output(tmp_path, target_sweep_path):
    source = tmp_path / "source.nc"
    source.symlink_to(target_sweep_path)
    products = tmp_path / "products"
    result = CliRunner().invoke(
        app,
        [
            "--dataprod_toltec_path",
            str(products),
            "--kids.output.enabled",
            "true",
            "--",
            str(source),
        ],
    )
    assert result.exit_code == 0, result.exception
    with open_sweep_product(products / "scratch/145606-0-1/dp_raw_obs_proc") as tree:
        assert read_sweep_context(tree).completed


def test_create_kids_output_settings_and_input_context_are_isolated(tmp_path):
    base = tmp_path / "base"
    conditional = tmp_path / "conditional"
    runtime = RuntimeContext(
        [
            {"source": {"kids": {"output": {"path": str(base)}}}},
            {
                "enable_if": "roach == 7",
                "source": {"kids": {"output": {"path": str(conditional)}}},
            },
        ]
    )
    first = create_kids(runtime, context_input={"roach": 0})
    second = create_kids(runtime, context_input={"roach": 7})
    assert first.config.output.path == base
    assert second.config.output.path == conditional
    assert first.config.output.path == base
    assert runtime.config_dict["kids"] == {"output": {"path": str(base)}}
    assert "roach" not in runtime.runtime_info.validation_context
    assert not base.exists()
    assert not conditional.exists()


def test_python_runner_uses_pipeline_output_path(tmp_path, target_sweep_path):
    source = tmp_path / "source.nc"
    source.symlink_to(target_sweep_path)
    products = tmp_path / "products"
    info = next(iter(read_info_from_sources([source]).itertuples(index=False)))
    runtime = RuntimeContext(
        {
            "kids": {"output": {"enabled": True, "path": str(products)}},
        }
    )
    assert (
        run_source(cast("str", info.source), runtime).parent
        == products / "scratch/145606-0-1/dp_raw_obs_proc"
    )
    assert runtime.config_dict["kids"] == {
        "output": {"enabled": True, "path": str(products)}
    }
    with open_sweep_product(products / "scratch/145606-0-1/dp_raw_obs_proc") as tree:
        assert read_sweep_context(tree).completed


@pytest.mark.parametrize("suffix", ["targsweep", "vnasweep", "tune"])
def test_reference_file_through_runner(data_lmt_path, tmp_path, suffix):
    paths = sorted(
        (data_lmt_path / "toltec" / "tcs" / "toltec0").glob(f"*1456*_{suffix}.nc")
    )
    if not paths:
        pytest.skip(f"No {suffix} fixture")
    source = paths[0]
    info = next(iter(read_info_from_sources([source]).itertuples(index=False)))
    runtime = RuntimeContext(
        {
            "kids": {
                "output": {
                    "enabled": True,
                    "path": str(tmp_path),
                    "namespace": "product",
                }
            }
        }
    )
    path = run_source(cast("str", info.source), runtime)
    assert path is not None
    with open_sweep_product(path) as reopened, xr.open_dataset(source) as raw:
        expected = apply_step(SweepReduce(), raw).toltec_kids.blocks
        assert reopened.toltec_kids.blocks.n_blocks == expected.n_blocks
        for block in reopened.toltec_kids.blocks:
            actual = SweepReduce.view(reopened, block=block.meta.block)
            reference = expected[block.meta.block].sweep
            for name in ("I", "Q", "unc_I", "unc_Q", "frequency"):
                np.testing.assert_array_equal(
                    getattr(actual, name), getattr(reference, name)
                )
        assert read_sweep_context(reopened).block == expected.n_blocks - 1


def test_output_step_and_metadata_view(tmp_path, target_sweep_path, monkeypatch):
    from tolteca_kids import dataprod_output

    raw_sweep = xr.load_dataset(target_sweep_path)
    table = read_info_from_sources([target_sweep_path]).assign(test_label="run")
    info = next(table.itertuples(index=False))
    kids = Kids(
        RuntimeContext(
            {
                "kids": {
                    "output": {
                        "enabled": True,
                        "path": str(tmp_path),
                        "namespace": "ql",
                    }
                }
            }
        )
    )
    working = ensure_datatree(raw_sweep)
    working["tolteca_kids"] = xr.DataTree()
    working["tolteca_kids"].attrs["source"] = info.source
    # Product placement follows acquisition metadata, not arbitrary row columns.
    working["tolteca_kids"].attrs["input_row"] = dict(
        cast("Any", info)._asdict(), obsnum=999
    )
    seen = []

    def inspect(data, *args, **kwargs):
        context = DataProdOutput.get_context(data)
        assert not context.completed
        assert context.data is None
        seen.append(context)
        from tolteca_kids.product import write_kids_product

        return write_kids_product(data, *args, **kwargs)

    monkeypatch.setattr(dataprod_output, "write_kids_product", inspect)
    kids.block_create(working)
    kids.sweep_pipeline(working["blocks/0"])
    kids.output(working)
    context = DataProdOutput.get_context(working)
    assert context.completed
    assert context.data is not None
    assert context.data.parent == tmp_path / "ql/145606-0-1/dp_raw_obs_proc"
    assert context.config == kids.config.output
    assert seen == [context]
    assert seen[0].completed
    provenance = read_product_provenance(context.data).inputs["toltec0"]
    assert provenance.config == kids.config
    assert provenance.uid_raw_obs_file == info.uid_raw_obs_file
    assert provenance.source is not None
    assert provenance.source.path == target_sweep_path
    assert "input_row" not in provenance.model_dump()
    assert "tolteca_kids" not in raw_sweep
    assert PIPELINE_CONTEXT_KEY not in raw_sweep.attrs
    kids.output(working)
    repeated = DataProdOutput.get_context(working)
    assert repeated.completed
    assert repeated.data != context.data
    assert context.completed
    assert read_product_provenance(context.data).inputs["toltec0"] == provenance


def test_output_missing_input_keeps_incomplete_context(tmp_path):
    working = xr.DataTree()
    DataProdOutput(path=tmp_path)(working)
    assert DataProdOutput.has_context(working)
    context = DataProdOutput.get_context(working)
    assert not context.completed
    assert context.data is None
    assert context.failures[0].kind == "missing_input"
    assert list(tmp_path.iterdir()) == []


def test_output_disabled_and_standalone(tmp_path, target_sweep_path):
    source = tmp_path / "source.nc"
    source.symlink_to(target_sweep_path)
    info = next(iter(read_info_from_sources([source]).itertuples(index=False)))
    assert run_source(cast("str", info.source), RuntimeContext()) is None
    assert list(tmp_path.iterdir()) == [source]
    with xr.open_dataset(source) as raw:
        working = ensure_datatree(raw)
        SweepReduce()(working)
        working["tolteca_kids"] = xr.DataTree()
        working["tolteca_kids"].attrs["source"] = info.source
        step = DataProdOutput(path=tmp_path, namespace="standalone")
        _, context = step(working, return_context=True)
    assert context.completed
    assert context.data.parent == tmp_path / "standalone/145606-0-1/dp_raw_obs_proc"


def test_cli_multiple_inputs_with_per_input_config(tmp_path, data_lmt_path):
    sources = sorted((data_lmt_path / "toltec/tcs/toltec0").glob("*1456*sweep.nc"))
    assert len(sources) == 2
    config = tmp_path / "recipe.yaml"
    config.write_text("""
- source: {kids: {output: {namespace: scratch}}}
- enable_if: obsnum == 145606 & atten_drive == 7.25
  source: {kids: {sweep_reduce: {force: true}}}
""")
    output = tmp_path / "products"
    result = CliRunner().invoke(
        app,
        [
            *map(str, sources),
            "--config_paths",
            str(config),
            "--kids.output.enabled",
            "true",
            "--kids.output.path",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.exception
    products = sorted((output / "scratch").glob("*/dp_raw_obs_proc"))
    assert len(products) == 2
    for obsnum, product in zip((145605, 145606), products, strict=True):
        with open_sweep_product(product) as tree:
            view = SweepReduce.view(tree, block=0)
            assert read_sweep_context(tree).config.force == (obsnum == 145606)
            assert view.unc_I is not None
            assert view.unc_Q is not None


def test_cli_explicit_disabled_output_overrides_shorthand(tmp_path, target_sweep_path):
    source = tmp_path / "source.nc"
    source.symlink_to(target_sweep_path)
    output = tmp_path / "products"
    result = CliRunner().invoke(
        app,
        [
            str(source),
            "--kids.output.enabled",
            "true",
            "--kids.output.path",
            str(output),
            "--kids.output.enabled",
            "false",
        ],
    )
    assert result.exit_code == 0, result.exception
    assert not output.exists()


@pytest.mark.parametrize("field", ["namespace", "tag"])
@pytest.mark.parametrize("value", ["../escape", "/absolute", "", ".."])
def test_output_components_cannot_escape_root(tmp_path, field, value):
    message = "relative path" if field == "namespace" else "one path component"
    with pytest.raises(ValueError, match=message):
        DataProdOutput(path=tmp_path, **{field: value})
    assert list(tmp_path.iterdir()) == []
