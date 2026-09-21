"""Reducer-only pipeline, result views, and durable product tests."""

import json

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError
from tollan.accessor.xarray import ensure_datatree
from tollan.config import RuntimeContext
from tollan.pipeline import PIPELINE_CONTEXT_KEY, SequentialPipeline
from typer.testing import CliRunner

from tolteca_datamodels.toltec.file import SourceInfoModel
from tolteca_datamodels.toltec.filestore import build_input_catalog
from tolteca_datamodels.toltec.kids import SweepReducer
from tolteca_datamodels.toltec.kids.source import identify_source
from tolteca_kids.cli import app
from tolteca_kids.core import Kids, KidsConfig
from tolteca_kids.dataprod_output import DataProdOutput
from tolteca_kids.product import open_sweep_product, write_sweep_product
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


def test_invalid_block_does_not_publish_result(raw_sweep):
    working = ensure_datatree(raw_sweep)
    with pytest.raises(IndexError, match="Block 1 out of range"):
        SweepReduce(block=1)(working)
    assert not SweepReduce.has_context(working)
    assert list(working.children) == []


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
    path = write_sweep_product(working, tmp_path / "product", input_info={})
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
        write_sweep_product(working, path, input_info={})


def test_incomplete_product_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_sweep_product(tmp_path)


@pytest.mark.parametrize("unit", [None, "adu", "V"])
def test_preserve_signal_units(tmp_path, raw_sweep, unit):
    for name in ("Data.Toltec.Is", "Data.Toltec.Qs"):
        if unit is not None:
            raw_sweep[name].attrs["units"] = unit
    working = ensure_datatree(raw_sweep)
    SweepReduce()(working)
    path = write_sweep_product(working, tmp_path / "product", input_info={})
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


def test_output_step_and_metadata_view(tmp_path, raw_sweep):
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
        context={"flat": True},
        exclude={"filepath_info", "filepath_info_orig"},
    )
    kids.sweep_pipeline(working)
    context = DataProdOutput.view(working).context
    assert context.completed
    assert context.product_path is not None
    assert context.product_path == tmp_path / "000001-0"
    assert context.config == kids.config.output
    manifest = json.loads((context.product_path / "manifest.json").read_text())
    assert manifest["output_config"] == kids.config.output.model_dump(mode="json")
    assert manifest["input"]["data_kind"] == "TargetSweep"
    assert "tolteca_kids" not in raw_sweep
    assert PIPELINE_CONTEXT_KEY not in raw_sweep.attrs


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
        context={"flat": True},
        exclude={"filepath_info", "filepath_info_orig"},
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
