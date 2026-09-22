"""Reducer-only pipeline, result views, and durable product tests."""

import json

import numpy as np
import pytest
import xarray as xr
import zarr
from pydantic import ValidationError
from tollan.accessor.xarray import ensure_datatree
from tollan.config import RuntimeContext
from tollan.pipeline import PIPELINE_CONTEXT_KEY, SequentialPipeline
from typer.testing import CliRunner

from tolteca_datamodels.file import SourceInfoModel
from tolteca_datamodels.toltec.filestore import build_input_catalog
from tolteca_datamodels.toltec.kids import SweepReducer
from tolteca_datamodels.toltec.kids.source import identify_source
from tolteca_kids.cli import app
from tolteca_kids.core import Kids, KidsConfig
from tolteca_kids.dataprod_output import DataProdOutput
from tolteca_kids.noop import NoOp
from tolteca_kids.product import (
    open_sweep_product,
    read_sweep_manifest,
    write_sweep_product,
)
from tolteca_kids.runner import create_kids, run_source
from tolteca_kids.sweep_reduce import SweepReduce, SweepReduceView


@pytest.fixture
def raw_sweep():
    values = np.arange(24, dtype=float).reshape(12, 2)
    return xr.Dataset(
        {
            "Data.Toltec.Is": (("time", "chan"), values),
            "Data.Toltec.Qs": (("time", "chan"), -values),
            "Data.Toltec.LoFreq": ("time", np.repeat([99.0, 100.0, 101.0], 4)),
            "Header.Toltec.LoCenterFreq": 100.0,
            "Header.Toltec.RoachIndex": 0,
            "Header.Toltec.ObsNum": 1,
            "Header.Toltec.SubObsNum": 0,
            "Header.Toltec.ScanNum": 0,
            "Header.Toltec.ObsType": 3,
        }
    )


def test_pipeline_owns_steps_and_preserves_working_tree(raw_sweep):
    kids = Kids(RuntimeContext())
    assert kids.sweep_reduce is kids.sweep_reduce
    assert kids.sweep_pipeline.steps == [kids.sweep_reduce, kids.output]
    original = raw_sweep.copy(deep=True)
    working = ensure_datatree(raw_sweep)
    working["unrelated"] = xr.DataTree(dataset=xr.Dataset({"a": 3}))
    assert kids.sweep_pipeline(working) is working
    view = SweepReduceView(working)
    expected = SweepReducer()(raw_sweep).toltec_kids.sweep
    xr.testing.assert_identical(view.I, expected.I)
    xr.testing.assert_identical(view.unc_I, expected.unc_I)
    xr.testing.assert_identical(raw_sweep, original)
    assert "unrelated" in working.children
    assert view.context.completed
    assert view.context.config == kids.config.sweep_reduce
    assert PIPELINE_CONTEXT_KEY not in working.attrs
    assert isinstance(
        working[SweepReduce.result_path()].attrs[SweepReduce.context_key], dict
    )


def test_standalone_step_and_repeat_preserve_results(raw_sweep):
    working = ensure_datatree(raw_sweep)
    step = SweepReduce()
    result, context = step(working, return_context=True)
    assert result is working
    assert context.completed
    first_I = SweepReduce.view(working).I
    assert first_I is not None
    first = first_I.data
    step(working)
    second_I = SweepReduce.view(working).I
    assert second_I is not None
    assert second_I.data is first
    assert SweepReduce.get_context(working) == context


def test_aliases_keep_separate_result_nodes(raw_sweep):
    working = ensure_datatree(raw_sweep)
    other = SweepReduce.alias("second_pass")
    pipeline = SequentialPipeline(
        steps=[
            SweepReduce(),
            other(force=True),
        ]
    )
    pipeline(working)
    assert SweepReduce.view(working).unc_I is not None
    assert other.view(working).unc_I is not None
    assert other.get_context(working).completed
    assert other.get_context(working).config.force
    assert not SweepReduce.get_context(working).config.force


def test_alias_context_recreates_the_right_step_after_serialization(raw_sweep):
    working = ensure_datatree(raw_sweep)
    _, original = SweepReduce()(working, return_context=True)
    alias = SweepReduce.alias("repeat")
    _, aliased = alias()(working, return_context=True)
    assert type(original.make_step()) is SweepReduce
    assert type(aliased.make_step()) is alias
    assert type(SweepReduce.view(working).context.make_step()) is SweepReduce
    assert type(alias.view(working).context.make_step()) is alias
    restored = alias.context_cls.model_validate_json(aliased.model_dump_json())
    assert type(restored.make_step()) is alias
    assert restored.model_dump() == aliased.model_dump()


@pytest.mark.parametrize("primary_first", [False, True])
def test_alias_reuses_its_result_without_recomputing(
    raw_sweep, monkeypatch, primary_first
):
    working = ensure_datatree(raw_sweep)
    working["unrelated"] = xr.DataTree(dataset=xr.Dataset({"a": 3}))
    calls = []
    reduce = SweepReducer._reduce

    def count(reducer, data):
        calls.append(data)
        return reduce(reducer, data)

    monkeypatch.setattr(SweepReducer, "_reduce", count)
    primary = None
    if primary_first:
        SweepReduce()(working)
        primary = SweepReduce.view(working).data_source.copy(deep=True)
    alias = SweepReduce.alias("cached")
    alias()(working)
    first = alias.view(working).I
    assert first is not None
    assert len(calls) == 1
    alias()(working)
    repeated = alias.view(working).I
    assert repeated is not None
    assert repeated.data is first.data
    assert len(calls) == 1
    alias(force=True)(working)
    forced = alias.view(working).I
    assert forced is not None
    assert forced.data is not first.data
    xr.testing.assert_identical(forced, first)
    assert len(calls) == 2
    assert working["unrelated"]["a"].item() == 3
    if primary_first:
        xr.testing.assert_identical(SweepReduce.view(working).data_source, primary)
    else:
        assert SweepReduce.result_path() not in working


def test_alias_keeps_its_cache_when_primary_is_replaced(raw_sweep, monkeypatch):
    working = ensure_datatree(raw_sweep)
    alias = SweepReduce.alias("cached")
    alias()(working)
    aliased = alias.view(working).I
    assert aliased is not None
    SweepReduce(force=True)(working)
    primary = SweepReduce.view(working).I
    assert primary is not None
    assert primary.data is not aliased.data

    def unexpected(*args):
        pytest.fail("unforced repeat computed an already available result")

    monkeypatch.setattr(SweepReducer, "_reduce", unexpected)
    alias()(working)
    repeated = alias.view(working).I
    assert repeated is not None
    assert repeated.data is aliased.data


def test_alias_cache_still_checks_numerical_configuration(raw_sweep):
    working = ensure_datatree(raw_sweep)
    alias = SweepReduce.alias("different_config")
    without_uncertainty = SweepReducer(compute_uncertainty=False)(raw_sweep)
    working[alias.result_path()] = without_uncertainty[SweepReducer.__module__]
    assert alias.view(working).unc_I is None
    alias()(working)
    assert alias.view(working).unc_I is not None


def test_output_alias_context_recreates_the_right_step(tmp_path, raw_sweep):
    working = ensure_datatree(raw_sweep)
    original = DataProdOutput(path=tmp_path).create_context(working)
    alias = DataProdOutput.alias("second")
    aliased = alias(path=tmp_path).create_context(working)
    assert type(original.make_step()) is DataProdOutput
    assert type(aliased.make_step()) is alias
    assert DataProdOutput.get_context(working) is original
    assert alias.get_context(working) is aliased
    DataProdOutput.set_context(working, original)
    alias.set_context(working, aliased)
    assert type(DataProdOutput.view(working).context.make_step()) is DataProdOutput
    assert type(alias.view(working).context.make_step()) is alias


def test_pipeline_reducer_settings_are_fixed(raw_sweep):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SweepReduce(reducer={"compute_uncertainty": False})
    working = ensure_datatree(raw_sweep)
    SweepReduce()(working)
    assert SweepReduce.view(working).unc_I is not None
    assert SweepReduce.view(working).unc_Q is not None


def test_disabled_pipeline_does_not_reduce(raw_sweep, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("disabled stage invoked the reducer")

    monkeypatch.setattr(SweepReducer, "__call__", unexpected)
    kids = Kids(RuntimeContext({"kids": {"sweep_reduce": {"enabled": False}}}))
    working = ensure_datatree(raw_sweep)
    assert kids.sweep_pipeline(working) is working
    assert not SweepReduce.has_context(working)


def test_invalid_block_keeps_incomplete_context_without_arrays(raw_sweep, tmp_path):
    working = ensure_datatree(raw_sweep)
    with pytest.raises(IndexError, match="Block 1 out of range"):
        SweepReduce(block=1)(working)
    assert SweepReduce.has_context(working)
    context = SweepReduce.get_context(working)
    assert not context.completed
    assert context.config.block == 1
    assert SweepReduce.view(working).I is None
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
    assert SweepReduce.get_context(working) == context
    assert isinstance(
        working[SweepReduce.result_path()].attrs[SweepReduce.context_key], dict
    )
    assert PIPELINE_CONTEXT_KEY not in raw_sweep.attrs


def test_failed_rerun_replaces_success_context_and_cannot_publish(raw_sweep, tmp_path):
    working = ensure_datatree(raw_sweep)
    SweepReduce(block=0)(working)
    first = SweepReduce.get_context(working)
    first_I = SweepReduce.view(working).I
    assert first_I is not None
    values = first_I.data
    with pytest.raises(IndexError, match="Block 1 out of range"):
        SweepReduce(block=1)(working)
    current = SweepReduce.get_context(working)
    assert current is not first
    assert current.config.block == 1
    assert not current.completed
    assert first.completed
    after_failure = SweepReduce.view(working).I
    assert after_failure is not None
    assert after_failure.data is values
    with pytest.raises(ValueError, match="not completed"):
        write_sweep_product(working, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    SweepReduce(block=0)(working)
    assert SweepReduce.get_context(working).completed
    after_retry = SweepReduce.view(working).I
    assert after_retry is not None
    assert after_retry.data is values


def test_failed_alias_keeps_the_primary_result_and_context(raw_sweep):
    working = ensure_datatree(raw_sweep)
    SweepReduce()(working)
    before = SweepReduce.get_context(working)
    result = SweepReduce.view(working).data_source.copy(deep=True)
    alias = SweepReduce.alias("invalid_block")
    with pytest.raises(IndexError, match="Block 1 out of range"):
        alias(block=1)(working)
    assert alias.has_context(working)
    assert not alias.get_context(working).completed
    assert SweepReduce.get_context(working) == before
    xr.testing.assert_identical(SweepReduce.view(working).data_source, result)


def test_conditional_recipe_and_handler_isolation(tmp_path, raw_sweep):
    raw_sweep = raw_sweep.assign(
        {
            "Header.Toltec.Filename": (
                "/data_lmt/toltec/tcs/toltec0/"
                "toltec0_000001_000_0000_2025_10_31_01_02_03_targsweep.nc"
            )
        }
    )
    recipe = tmp_path / "10_kids.yaml"
    recipe.write_text("""
- source: {kids: {sweep_reduce: {force: false}}}
- enable_if: roach == 0 & file_suffix in ['targsweep', 'tune']
  source: {kids: {sweep_reduce: {force: true}}}
""")
    rc = RuntimeContext.from_cli(config_paths=[tmp_path])
    info = identify_source(
        raw_sweep, SourceInfoModel(source=str(tmp_path / "input.nc"))
    )
    first = Kids(
        rc.with_config(
            validation_context=info.model_dump(
                mode="json",
                context={"flat": True},
                exclude={"filepath_info", "filepath_info_orig"},
            )
        )
    )
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
        {"sweep_reduce": {"reducer": {"compute_uncertanty": False}}},
    ],
)
def test_unknown_config_is_rejected(config):
    with pytest.raises(ValidationError):
        Kids(RuntimeContext({"kids": config}))


def test_config_roundtrip():
    config = KidsConfig()
    assert KidsConfig.model_validate_json(config.model_dump_json()) == config


def test_export_reopen_and_exclusive_destination(tmp_path, raw_sweep):
    raw_sweep = raw_sweep.assign_coords(time=np.arange(12), chan=[8, 9])
    working = ensure_datatree(raw_sweep)
    Kids(RuntimeContext()).sweep_pipeline(working)
    path = write_sweep_product(working, tmp_path / "product")
    with open_sweep_product(path) as reopened:
        actual = SweepReduceView(reopened)
        expected = SweepReduceView(working)
        xr.testing.assert_identical(actual.data_source, expected.data_source)
        assert actual.context == expected.context
        assert not reopened.dataset.data_vars
        assert "Data.Toltec.Is" not in actual.data_source
        assert "time" not in actual.data_source.dims
        signal = actual.I
        assert signal is not None
        np.testing.assert_array_equal(signal.chan, [8, 9])
    with pytest.raises(FileExistsError):
        write_sweep_product(working, path)


def test_incomplete_product_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_sweep_product(tmp_path)


def test_product_has_consolidated_metadata_before_completion(tmp_path, raw_sweep):
    working = ensure_datatree(raw_sweep)
    SweepReduce()(working)
    path = write_sweep_product(working, tmp_path / "product")
    assert (path / "arrays.zarr" / ".zmetadata").is_file()
    with xr.open_datatree(
        path / "arrays.zarr", engine="zarr", consolidated=True
    ) as tree:
        xr.testing.assert_identical(
            SweepReduce.view(tree).data_source, SweepReduce.view(working).data_source
        )
    assert read_sweep_manifest(path).arrays == "arrays.zarr"


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
    assert not (path / "manifest.json").exists()
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
        assert SweepReduce.view(tree).context.completed
    assert ".zmetadata" in keys
    assert not any(
        key.startswith(SweepReduce.result_path())
        and key.endswith((".zattrs", ".zarray", ".zgroup", "zarr.json"))
        for key in keys
    )


def test_reader_uses_native_fallback_without_consolidated_metadata(tmp_path, raw_sweep):
    working = ensure_datatree(raw_sweep)
    SweepReduce()(working)
    path = write_sweep_product(working, tmp_path / "product")
    # Keep the test's original index for inspection, but make it unavailable.
    (path / "arrays.zarr" / ".zmetadata").rename(tmp_path / "metadata.saved")
    with (
        pytest.warns(RuntimeWarning, match="non-consolidated metadata"),
        open_sweep_product(path) as tree,
    ):
        xr.testing.assert_identical(
            SweepReduce.view(tree).data_source, SweepReduce.view(working).data_source
        )


def test_export_excludes_live_context_without_mutating_working_tree(
    tmp_path, raw_sweep
):
    raw_sweep.attrs["user_label"] = "interactive test"
    working = ensure_datatree(raw_sweep)
    NoOp()(working)
    live_context = NoOp.get_context(working)
    SweepReduce()(working)
    reduced = SweepReduce.view(working).data_source
    live_store = reduced.attrs[PIPELINE_CONTEXT_KEY]
    assert live_store is working.attrs[PIPELINE_CONTEXT_KEY]
    expected = reduced.copy(deep=False)
    expected.attrs.pop(PIPELINE_CONTEXT_KEY)
    path = write_sweep_product(working, tmp_path / "product")
    with open_sweep_product(path) as reopened:
        saved = SweepReduce.view(reopened)
        xr.testing.assert_identical(saved.data_source, expected)
        assert saved.context.completed
        assert saved.data_source.attrs["user_label"] == "interactive test"
        assert PIPELINE_CONTEXT_KEY not in saved.data_source.attrs
    assert reduced.attrs[PIPELINE_CONTEXT_KEY] is live_store
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
    assert not (path / "manifest.json").exists()
    assert "unsupported_resource" in SweepReduce.view(working).data_source.attrs


def test_manifest_and_effective_config_replay(tmp_path, raw_sweep):
    source = tmp_path / "toltec0_000001_000_0000_targsweep.nc"
    raw_sweep.to_netcdf(source, engine="scipy")
    info = build_input_catalog([source]).toltec_file.to_info_list()[0]
    runtime = RuntimeContext(
        [
            {
                "source": {
                    "kids": {
                        "output": {
                            "enabled": True,
                            "path": str(tmp_path),
                            "name": "first",
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
    path = run_source(info, runtime)
    assert path is not None
    manifest = read_sweep_manifest(path)
    assert manifest.input == info
    assert manifest.config is not None
    assert manifest.config.sweep_reduce.block == 0
    offset = manifest.time_created.utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 0
    payload = json.loads((path / "manifest.json").read_text())
    assert isinstance(payload["input"]["source"], dict)
    assert isinstance(payload["input"]["data_kind"], int)
    assert "file_suffix" not in payload["input"]
    assert "filepath_info_orig" not in payload["input"]
    snapshot = tmp_path / "replay.yaml"
    snapshot.write_text(
        json.dumps(
            Kids.prepare_runtime_config_data(
                manifest.config.model_dump(mode="json", round_trip=True)
            )
        )
    )
    result = CliRunner().invoke(
        app,
        [
            "--config_paths",
            str(snapshot),
            "--kids.output.name",
            "replayed",
            "--",
            str(source),
        ],
    )
    assert result.exit_code == 0, result.exception
    with (
        open_sweep_product(path) as first,
        open_sweep_product(tmp_path / "replayed") as replayed,
    ):
        xr.testing.assert_identical(
            SweepReduce.view(first).data_source,
            SweepReduce.view(replayed).data_source,
        )


@pytest.mark.parametrize(
    ("key", "value"), [("format", "unknown"), ("arrays", "../outside.zarr")]
)
def test_invalid_manifest_rejected_before_array_open(
    tmp_path, raw_sweep, monkeypatch, key, value
):
    working = ensure_datatree(raw_sweep)
    SweepReduce()(working)
    path = write_sweep_product(working, tmp_path / "product")
    manifest = path / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload[key] = value
    manifest.write_text(json.dumps(payload))

    def unexpected(*args, **kwargs):
        pytest.fail("invalid manifest attempted to open arrays")

    monkeypatch.setattr(xr, "open_datatree", unexpected)
    with pytest.raises(ValidationError):
        open_sweep_product(path)


def test_manifest_config_must_match_saved_context(tmp_path, raw_sweep):
    source = tmp_path / "source.nc"
    raw_sweep.to_netcdf(source, engine="scipy")
    info = build_input_catalog([source]).toltec_file.to_info_list()[0]
    runtime = RuntimeContext(
        {
            "kids": {
                "output": {"enabled": True, "path": str(tmp_path), "name": "product"}
            }
        }
    )
    path = run_source(info, runtime)
    assert path is not None
    manifest = path / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["config"]["sweep_reduce"]["block"] = 0
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="configuration disagree"):
        open_sweep_product(path)


@pytest.mark.parametrize("unit", [None, "adu", "V"])
def test_preserve_signal_units(tmp_path, raw_sweep, unit):
    for name in ("Data.Toltec.Is", "Data.Toltec.Qs"):
        if unit is not None:
            raw_sweep[name].attrs["units"] = unit
    working = ensure_datatree(raw_sweep)
    SweepReduce()(working)
    path = write_sweep_product(working, tmp_path / "product")
    with open_sweep_product(path) as reopened:
        view = SweepReduce.view(reopened)
        for name in ("I", "Q", "unc_I", "unc_Q"):
            array = getattr(view, name)
            assert array.u.unit_str == unit
            if unit is None:
                assert array.u.quantity is array
            else:
                assert str(array.u.quantity.unit) == unit


def test_cli_config_and_reduction(tmp_path, raw_sweep, monkeypatch):
    source = tmp_path / "source.nc"
    raw_sweep.to_netcdf(source, engine="scipy")
    runner = CliRunner()
    with monkeypatch.context() as patch:

        def unexpected(*args, **kwargs):
            pytest.fail("--show_config invoked reduction")

        patch.setattr(SweepReducer, "__call__", unexpected)
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
    with open_sweep_product(product / "source") as tree:
        assert SweepReduce.get_context(tree).completed
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
    assert result.exit_code != 0


def test_prepare_does_not_read_signal(tmp_path, raw_sweep):
    import dask.array as da
    from dask import delayed

    @delayed
    def unexpected():
        pytest.fail("metadata preparation read I/Q arrays")

    raw_sweep["Data.Toltec.Is"].data = da.from_delayed(
        unexpected(), shape=(12, 2), dtype=float
    )
    raw_sweep["Data.Toltec.Qs"].data = da.from_delayed(
        unexpected(), shape=(12, 2), dtype=float
    )
    info = identify_source(
        raw_sweep, SourceInfoModel(source=str(tmp_path / "source.nc"))
    )
    kids = Kids(
        RuntimeContext().with_config(
            validation_context=info.model_dump(
                mode="json",
                context={"flat": True},
                exclude={"filepath_info", "filepath_info_orig"},
            )
        )
    )
    assert info.roach == 0
    assert kids.config.sweep_reduce.enabled


def test_cli_product_root_is_used_by_enabled_output(tmp_path, raw_sweep):
    source = tmp_path / "source.nc"
    raw_sweep.to_netcdf(source, engine="scipy")
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
    with open_sweep_product(products / "source") as tree:
        assert SweepReduce.get_context(tree).completed


def test_create_kids_defaults_and_input_context_are_isolated(tmp_path):
    shared = tmp_path / "shared"
    conditional = tmp_path / "conditional"
    runtime = RuntimeContext(
        [
            {"source": {"files": {"dataprod_toltec_path": str(shared)}}},
            {
                "enable_if": "roach == 7",
                "source": {"kids": {"output": {"path": str(conditional)}}},
            },
        ]
    )
    first = create_kids(runtime, context_input={"roach": 0})
    second = create_kids(runtime, context_input={"roach": 7})
    assert first.config.output.path == shared
    assert second.config.output.path == conditional
    assert first.config.output.path == shared
    assert "kids" not in runtime.config_dict
    assert "roach" not in runtime.runtime_info.validation_context
    assert not shared.exists()
    assert not conditional.exists()


def test_python_runner_uses_shared_product_root(tmp_path, raw_sweep):
    source = tmp_path / "source.nc"
    raw_sweep.to_netcdf(source, engine="scipy")
    products = tmp_path / "products"
    info = build_input_catalog([source]).toltec_file.to_info_list()[0]
    runtime = RuntimeContext(
        {
            "files": {"dataprod_toltec_path": str(products)},
            "kids": {"output": {"enabled": True}},
        }
    )
    assert run_source(info, runtime) == products / "source"
    assert runtime.config_dict["kids"] == {"output": {"enabled": True}}
    with open_sweep_product(products / "source") as tree:
        assert SweepReduce.get_context(tree).completed


@pytest.mark.parametrize("suffix", ["targsweep", "vnasweep", "tune"])
def test_reference_file_through_runner(data_lmt_path, tmp_path, suffix):
    paths = sorted(
        (data_lmt_path / "toltec" / "tcs" / "toltec0").glob(f"*_{suffix}.nc")
    )
    if not paths:
        pytest.skip(f"No {suffix} fixture")
    source = paths[0]
    info = build_input_catalog([source]).toltec_file.to_info_list()[0]
    runtime = RuntimeContext(
        {
            "kids": {
                "output": {"enabled": True, "path": str(tmp_path), "name": "product"}
            }
        }
    )
    path = run_source(info, runtime)
    assert path is not None
    with open_sweep_product(path) as reopened, xr.open_dataset(source) as raw:
        actual = SweepReduceView(reopened)
        expected = SweepReducer()(raw).toltec_kids.sweep
        xr.testing.assert_identical(actual.I, expected.I)
        xr.testing.assert_identical(actual.unc_I, expected.unc_I)
        assert actual.context.block_index == expected.n_blocks - 1


def test_output_step_and_metadata_view(tmp_path, raw_sweep, monkeypatch):
    from tolteca_kids import dataprod_output

    info = identify_source(raw_sweep, SourceInfoModel(source="source.nc"))
    kids = Kids(
        RuntimeContext(
            {
                "kids": {
                    "output": {
                        "enabled": True,
                        "path": str(tmp_path),
                        "name": "{obsnum:06d}-{roach}",
                    }
                }
            }
        )
    )
    working = ensure_datatree(raw_sweep)
    working["tolteca_kids"] = xr.DataTree()
    working["tolteca_kids"].attrs["input"] = info.model_dump(
        mode="json",
        round_trip=True,
    )
    seen = []

    def inspect(data, *args, **kwargs):
        context = DataProdOutput.get_context(data)
        assert not context.completed
        assert context.product_path is None
        seen.append(context)
        return write_sweep_product(data, *args, **kwargs)

    monkeypatch.setattr(dataprod_output, "write_sweep_product", inspect)
    kids.sweep_pipeline(working)
    context = DataProdOutput.view(working).context
    assert context.completed
    assert context.product_path is not None
    assert context.product_path == tmp_path / "000001-0"
    assert context.config == kids.config.output
    assert seen == [context]
    assert seen[0].completed
    manifest = read_sweep_manifest(context.product_path)
    assert manifest.config == kids.config
    assert manifest.input == info
    assert "tolteca_kids" not in raw_sweep
    assert PIPELINE_CONTEXT_KEY not in raw_sweep.attrs
    with pytest.raises(FileExistsError):
        kids.output(working)
    failed = DataProdOutput.get_context(working)
    assert not failed.completed
    assert failed.product_path is None
    assert context.completed
    assert read_sweep_manifest(context.product_path) == manifest


def test_output_missing_input_keeps_incomplete_context(tmp_path):
    working = xr.DataTree()
    with pytest.raises(KeyError):
        DataProdOutput(path=tmp_path)(working)
    assert DataProdOutput.has_context(working)
    context = DataProdOutput.get_context(working)
    assert not context.completed
    assert context.product_path is None
    assert list(tmp_path.iterdir()) == []


def test_output_disabled_and_standalone(tmp_path, raw_sweep):
    source = tmp_path / "source.nc"
    raw_sweep.to_netcdf(source, engine="scipy")
    info = build_input_catalog([source]).toltec_file.to_info_list()[0]
    assert run_source(info, RuntimeContext()) is None
    assert list(tmp_path.iterdir()) == [source]
    working = ensure_datatree(raw_sweep)
    SweepReduce()(working)
    working["tolteca_kids"] = xr.DataTree()
    working["tolteca_kids"].attrs["input"] = info.model_dump(
        mode="json",
        round_trip=True,
    )
    step = DataProdOutput(path=tmp_path, name="standalone")
    _, context = step(working, return_context=True)
    assert context.completed
    assert context.product_path == tmp_path / "standalone"


def test_cli_multiple_inputs_with_per_input_config(tmp_path, raw_sweep):
    sources = []
    for roach in (0, 8):
        source = tmp_path / f"toltec{roach}_000001_000_0000_targsweep.nc"
        data = raw_sweep.assign({"Header.Toltec.RoachIndex": roach})
        data.to_netcdf(source, engine="scipy")
        sources.append(str(source))
    config = tmp_path / "recipe.yaml"
    config.write_text("""
- source: {kids: {output: {name: '{stem}'}}}
- enable_if: roach == 8
  source: {kids: {sweep_reduce: {force: true}}}
""")
    output = tmp_path / "products"
    result = CliRunner().invoke(
        app,
        [
            *sources,
            "--config_paths",
            str(config),
            "--kids.output.enabled",
            "true",
            "--kids.output.path",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.exception
    products = sorted(output.iterdir())
    assert len(products) == 2
    for roach, product in zip((0, 8), products, strict=True):
        with open_sweep_product(product) as tree:
            view = SweepReduce.view(tree)
            assert view.context.config.force == (roach == 8)
            assert view.unc_I is not None
            assert view.unc_Q is not None


def test_cli_explicit_disabled_output_overrides_shorthand(tmp_path, raw_sweep):
    source = tmp_path / "source.nc"
    raw_sweep.to_netcdf(source, engine="scipy")
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


@pytest.mark.parametrize("name", ["../escape", "/absolute", "", ".."])
def test_output_name_cannot_escape_root(tmp_path, raw_sweep, name):
    working = ensure_datatree(raw_sweep)
    working["tolteca_kids"] = xr.DataTree()
    working["tolteca_kids"].attrs["input"] = {"source": "source.nc"}
    with pytest.raises(ValueError, match="single directory component"):
        DataProdOutput(path=tmp_path, name=name)(working)
    assert list(tmp_path.iterdir()) == []
