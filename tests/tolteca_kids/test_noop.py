"""CLI selection and configuration through a real no-op pipeline."""

from io import StringIO

import pytest
import xarray as xr
from tollan.accessor.xarray import ensure_datatree
from tollan.config import RuntimeContext
from tollan.pipeline import StepFailureKind
from tollan.utils.fileloc import fileloc
from typer.testing import CliRunner
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper, NetCDF4DataStore

from tolteca_datamodels.toltec.file import get_info_from_source
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_kids import SweepReduce
from tolteca_kids.cli import app
from tolteca_kids.core import Kids
from tolteca_kids.dataprod_output import DataProdOutput
from tolteca_kids.noop import NoOp
from tolteca_kids.runner import run_source


def test_noop_live_context_and_array_identity():
    raw = xr.Dataset({"signal": ("sample", [1, 2, 3])})
    working = ensure_datatree(raw)
    original = raw.copy(deep=True)
    assert Kids(RuntimeContext()).noop_pipeline(working) is working
    assert NoOp.get_context(working).completed
    assert working["signal"].data is raw["signal"].data
    xr.testing.assert_identical(raw, original)


def test_telescope_noop_retains_its_source_kind(data_lmt_path, monkeypatch):
    source = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    info = get_info_from_source(source)
    assert info is not None
    contexts = []
    reads = []
    read_array = NetCDF4ArrayWrapper._getitem

    def run(cls, data, context):
        current = data["tolteca_kids"].attrs["input_row"]
        assert current["data_kind_toltec"] == ToltecDataKind.lmt_tel.value
        assert current["interface"] == "tel"
        assert current["obsnum"] == 145609
        assert current["master"] == "tcs"
        assert current["file_prefix"] == info.file_prefix
        contexts.append(context)

    def read(array, key):
        if array.variable_name.startswith("Data."):
            assert array.variable_name == "Data.TelescopeBackend.TelTime"
            assert key in ((0,), (array.shape[0] - 1,))
            reads.append(key)
        return read_array(array, key)

    monkeypatch.setattr(NoOp, "run", classmethod(run))
    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", read)
    assert run_source(info.source, RuntimeContext(), dry_run=True) is None
    assert len(contexts) == 1
    assert contexts[0].completed
    assert len(reads) == 2


@pytest.mark.parametrize("action", ["--show_inputs", "--dry_run"])
@pytest.mark.parametrize(
    "selection",
    [
        "interface == 'tel' & master == 'tcs' & obsnum == 145609",
        "bits_any_set(data_kind_toltec, lmt_tel | lmt_tel2) & obsnum == 145609",
    ],
)
def test_cli_selects_renamed_telescope_by_header(
    data_lmt_path, tmp_path, monkeypatch, action, selection
):
    import json

    source = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    downloaded = tmp_path / "downloaded.nc"
    downloaded.symlink_to(source)
    closed = []
    time_reads = []
    close_store = NetCDF4DataStore.close
    getitem = NetCDF4ArrayWrapper._getitem

    def close(data, **kwargs):
        close_store(data, **kwargs)
        closed.append(data)

    def read(array, key):
        if array.variable_name.startswith("Data."):
            assert array.variable_name == "Data.TelescopeBackend.TelTime"
            assert key in ((0,), (array.shape[0] - 1,))
            time_reads.append(key)
        return getitem(array, key)

    monkeypatch.setattr(NetCDF4DataStore, "close", close)
    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", read)
    result = CliRunner().invoke(
        app,
        [
            str(downloaded),
            "--select",
            selection,
            action,
        ],
    )
    assert result.exit_code == 0, result.output
    if action == "--show_inputs":
        [row] = json.loads(result.stdout)
        assert row["source"] == downloaded.as_uri()
        assert row["data_kind_toltec"] == ToltecDataKind.lmt_tel.value
        assert "data_kind_str" not in row
        assert row["filepath_info"] is None
        assert row["file_prefix"] is None
    else:
        assert f"{downloaded}: no-op completed" in result.stdout
    # guess_or_read identifies the renamed source during discovery; preparation
    # reopens it for execution and retains the already populated metadata.
    assert len(closed) == 2
    assert len(time_reads) == 2


@pytest.mark.parametrize(
    "selector", ["tcs-145609/0", "tcs-145609/toltec(0,5)", "tcs-145609/(0,5)[-1]"]
)
def test_cli_noop_selection_config_logging_and_no_products(
    data_lmt_path, tmp_path, monkeypatch, selector
):
    from tollan.utils import typer as typer_utils
    from tollan.utils.log import reset_logger

    log_output = StringIO()

    def capture_logs(**kwargs):
        reset_logger(sink=log_output, **kwargs)

    # The CLI resets sinks; capture its selected level with the real logger.
    monkeypatch.setattr(typer_utils, "reset_logger", capture_logs)
    getitem = NetCDF4ArrayWrapper._getitem

    def read(array, key):
        assert not array.variable_name.startswith("Data.")
        return getitem(array, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", read)
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("""
- source: {kids: {sweep_reduce: {block: 1}}}
- enable_if: atten_drive == 7.25 & scannum == 2
  source: {kids: {sweep_reduce: {block: 2}}}
""")
    configs = []
    contexts = []
    original_pipeline = Kids.noop_pipeline.fget
    original_run = NoOp.run

    def pipeline(kids):
        configs.append(kids.config)
        return original_pipeline(kids)

    def run(cls, data, context):
        contexts.append(context)
        assert NoOp.get_context(data) is context
        return original_run(data, context)

    def unexpected(*args, **kwargs):
        pytest.fail("dry-run invoked computation or product output")

    from tolteca_kids import BlockCreate, BlockIdentify

    monkeypatch.setattr(Kids, "noop_pipeline", property(pipeline))
    monkeypatch.setattr(NoOp, "run", classmethod(run))
    monkeypatch.setattr(BlockCreate, "__call__", unexpected)
    monkeypatch.setattr(BlockIdentify, "__call__", unexpected)
    monkeypatch.setattr(SweepReduce, "__call__", unexpected)
    monkeypatch.setattr(DataProdOutput, "run", unexpected)
    result = CliRunner().invoke(
        app,
        [
            "--log_level",
            "DEBUG",
            "--data_lmt_path",
            str(data_lmt_path),
            "--config_paths",
            str(recipe),
            "--dry_run",
            "--kids.output.enabled",
            "true",
            "--kids.output.path",
            str(tmp_path / "products"),
            "--kids.sweep_reduce.force",
            "true",
            "--",
            selector,
        ],
    )
    assert result.exit_code == 0, result.output
    assert [config.sweep_reduce.block for config in configs] == [1, 2]
    assert all(config.sweep_reduce.force for config in configs)
    assert len(contexts) == 2
    assert all(context.completed for context in contexts)
    assert not (tmp_path / "products").exists()
    logs = log_output.getvalue() + result.output
    for message in (
        "ObsSpec",
        "Resolving configuration with input context",
        "Effective KIDs configuration",
        "No-op: skipping computation and product output",
        "no-op completed",
    ):
        assert message in logs


@pytest.mark.parametrize("repeated", [False, True])
def test_cli_paths_load_ordered_runtime_sources(
    data_lmt_path, tmp_path, monkeypatch, repeated
):
    from tolteca_kids import runner

    source = (
        data_lmt_path / "toltec/tcs/toltec0/"
        "toltec0_145609_000_0002_2025_10_31_09_27_57.nc"
    )
    first = tmp_path / "env.d"
    first.mkdir()
    second = tmp_path / "second.env"
    (first / "00_base.env").write_text("FIRST=one\nSHARED=first\n")
    second.write_text("SECOND=two\nSHARED=second\n")
    config_dir = tmp_path / "config.d"
    config_dir.mkdir()
    (config_dir / "00_base.yaml").write_text(
        "CONFIG_BASE: true\nkids: {sweep_reduce: {block: 0}}\n"
    )
    config_override = tmp_path / "override.yaml"
    config_override.write_text(
        "CONFIG_OVERRIDE: true\nkids: {sweep_reduce: {block: 1}}\n"
    )
    snapshots = []
    run_input = runner.run_input

    def capture_runtime(data, row, runtime, **kwargs):
        snapshots.append(runtime.config_dict)
        return run_input(data, row, runtime, **kwargs)

    monkeypatch.setattr(runner, "run_input", capture_runtime)
    env_args = ["--env_paths", str(first)]
    if repeated:
        env_args.append("--env_paths")
    env_args.append(str(second))
    config_args = ["--config_paths", str(config_dir)]
    if repeated:
        config_args.append("--config_paths")
    config_args.append(str(config_override))
    result = CliRunner().invoke(
        app,
        [
            *env_args,
            *config_args,
            "--dry_run",
            "--kids.sweep_reduce.block",
            "2",
            "--",
            str(source),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no-op completed" in result.stdout
    assert len(snapshots) == 1
    assert snapshots[0]["_env"]["FIRST"] == "one"
    assert snapshots[0]["_env"]["SECOND"] == "two"
    assert snapshots[0]["_env"]["SHARED"] == "second"
    assert snapshots[0]["CONFIG_BASE"] is True
    assert snapshots[0]["CONFIG_OVERRIDE"] is True
    assert snapshots[0]["kids"]["sweep_reduce"]["block"] == 2


def test_cli_help_uses_python_option_names():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in (
        "config_paths",
        "env_paths",
        "data_lmt_path",
        "show_inputs",
        "show_config",
        "dry_run",
    ):
        assert f"--{name}" in result.stdout
        assert f"--{name.replace('_', '-')}" not in result.stdout
    assert "--env-file" not in result.stdout


@pytest.mark.parametrize("action", ["--show_inputs", "--show_config"])
def test_cli_header_selection_and_conditional_config(
    data_lmt_path, tmp_path, monkeypatch, action
):
    import json

    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("""
- enable_if: atten_drive == 7.25 & obsnum == 145606
  source: {kids: {sweep_reduce: {force: true}}}
""")
    getitem = NetCDF4ArrayWrapper._getitem

    def read(array, key):
        assert not array.variable_name.startswith("Data.")
        return getitem(array, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", read)
    result = CliRunner().invoke(
        app,
        [
            "--data_lmt_path",
            str(data_lmt_path),
            "--config_paths",
            str(recipe),
            "--select",
            "atten_drive == 7.25 & obsnum == 145606",
            action,
            "--",
            "{145605,145606}/0",
        ],
    )
    assert result.exit_code == 0, result.exception
    if action == "--show_inputs":
        [row] = json.loads(result.stdout)
        assert row["obsnum"] == 145606
        assert row["atten_drive"] == 7.25
    else:
        source, payload = result.stdout.split("\n", 1)
        assert "145606" in source
        assert json.loads(payload)["kids"]["sweep_reduce"]["force"]


def test_noop_failure_is_recorded_and_propagates(monkeypatch):
    def fail(cls, data, context):
        msg = "test step failure"
        raise ValueError(msg)

    monkeypatch.setattr(NoOp, "run", classmethod(fail))
    working = xr.DataTree()
    with pytest.raises(ValueError, match="test step failure"):
        Kids(RuntimeContext()).noop_pipeline(working)
    context = NoOp.get_context(working)
    assert not context.completed
    assert len(context.failures) == 1
    assert context.failures[0].kind is StepFailureKind.exception
    assert context.failures[0].message == "test step failure"
    assert context.failures[0].exception_type == "ValueError"


@pytest.mark.parametrize(
    "kind",
    [
        ToltecDataKind.vnasweep,
        ToltecDataKind.targsweep,
        ToltecDataKind.tune,
        ToltecDataKind.raw_timestream,
    ],
)
def test_reference_noop_reads_only_headers_and_closes_files(
    data_lmt_path, tmp_path, monkeypatch, kind
):
    records = [
        get_info_from_source(path)
        for path in sorted((data_lmt_path / "toltec/tcs/toltec0").glob("*.nc"))
    ]
    matching = [info for info in records if info.data_kind_toltec == kind.value]
    if not matching:
        pytest.skip(f"No {kind.name} reference fixture")
    info = matching[0]
    source = fileloc(info.source).path
    original_stat = source.stat()
    reads = []
    closed = []
    contexts = []
    read_array = NetCDF4ArrayWrapper._getitem
    close_store = NetCDF4DataStore.close
    noop_run = NoOp.run

    def read(self, key):
        # Guard at the disk-read boundary, including reads triggered indirectly.
        assert not self.variable_name.startswith("Data."), self.variable_name
        reads.append(self.variable_name)
        return read_array(self, key)

    def close(self, **kwargs):
        close_store(self, **kwargs)
        closed.append(self)

    def run(cls, data, context):
        assert cls.get_context(data) is context
        assert not context.completed
        assert data["tolteca_kids"].attrs["input_row"]["data_kind_toltec"] == kind.value
        contexts.append(context)
        return noop_run(data, context)

    def unexpected(*args, **kwargs):
        pytest.fail("no-op invoked reduction or product output")

    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", read)
    monkeypatch.setattr(NetCDF4DataStore, "close", close)
    monkeypatch.setattr(NoOp, "run", classmethod(run))
    monkeypatch.setattr(SweepReduce, "__call__", unexpected)
    monkeypatch.setattr(DataProdOutput, "run", unexpected)
    selector = f"tcs-{info.obsnum}-{info.subobsnum}-{info.scannum}/{info.roach}"
    output = tmp_path / "products"
    result = CliRunner().invoke(
        app,
        [
            selector,
            str(source),
            "--data_lmt_path",
            str(data_lmt_path),
            "--select",
            f"data_kind_toltec == {kind.name}.value",
            "--dry_run",
            "--kids.output.enabled",
            "true",
            "--kids.output.path",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.count("no-op completed") == 1
    assert len(contexts) == 1 and contexts[0].completed
    assert reads and all(name.startswith("Header.") for name in reads)
    # Input preparation retains its handle for execution.
    assert len(closed) == 1
    assert not output.exists()
    assert source.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert source.stat().st_size == original_stat.st_size


@pytest.mark.parametrize("failure", ["step", "config"])
def test_cli_noop_failure_closes_source(timestream_path, monkeypatch, failure):
    source = timestream_path
    closed = []
    contexts = []
    close_store = NetCDF4DataStore.close

    def close(self, **kwargs):
        close_store(self, **kwargs)
        closed.append(self)

    def fail(cls, data, context):
        contexts.append(context)
        msg = "test no-op failure"
        raise ValueError(msg)

    monkeypatch.setattr(NetCDF4DataStore, "close", close)
    monkeypatch.setattr(NoOp, "run", classmethod(fail))
    args = ["--kids.sweep_reduce.block", "invalid"] if failure == "config" else []
    result = CliRunner().invoke(app, [str(source), "--dry_run", *args])
    assert result.exit_code != 0
    assert "no-op completed" not in result.stdout
    assert len(closed) == 1
    if failure == "step":
        assert len(contexts) == 1 and not contexts[0].completed
        assert isinstance(result.exception, ValueError)
        assert str(result.exception) == "test no-op failure"
    else:
        assert contexts == []


@pytest.mark.parametrize("inspection", ["--show_inputs", "--show_config"])
def test_inspection_exits_before_noop(timestream_path, monkeypatch, inspection):
    source = timestream_path

    def unexpected(*args, **kwargs):
        pytest.fail("inspection executed a pipeline step")

    monkeypatch.setattr(NoOp, "run", unexpected)
    result = CliRunner().invoke(app, [str(source), "--dry_run", inspection])
    assert result.exit_code == 0, result.output
    assert source.name in result.stdout
    assert "no-op completed" not in result.stdout


def test_empty_selection_does_not_start_noop(timestream_path, monkeypatch):
    source = timestream_path

    def unexpected(*args, **kwargs):
        pytest.fail("empty selection executed a pipeline step")

    monkeypatch.setattr(NoOp, "run", unexpected)
    result = CliRunner().invoke(
        app,
        [str(source), "--select", "roach == 7", "--dry_run"],
    )
    assert result.exit_code == 2
    assert "No inputs selected" in result.output
