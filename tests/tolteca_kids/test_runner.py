"""KIDs execution with caller-owned xarray data and input tables."""

import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError
from tollan.accessor.xarray import ensure_datatree
from tollan.config import RuntimeContext
from typer.testing import CliRunner
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tests.kids_helpers import apply_step
from tolteca_datamodels.toltec.file import (
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_kids import BlockCreate, cli, runner
from tolteca_kids.dataprod_output import DataProdOutput
from tolteca_kids.noop import NoOp
from tolteca_kids.product import open_sweep_product, read_product_provenance
from tolteca_kids.runner import run_data
from tolteca_kids.sweep_reduce import SweepReduce


def test_borrowed_dataset_noop_uses_headers_without_reopening(
    timestream_path, monkeypatch
):
    # Deliberately conflicting guesses and an unavailable source location.
    source = f"https://example.invalid/{timestream_path.name}"
    closed = []
    close = xr.Dataset.close
    getitem = NetCDF4ArrayWrapper._getitem

    def record_close(data):
        closed.append(data)
        close(data)

    def unexpected(*args, **kwargs):
        pytest.fail("Borrowed no-op reopened a source or ran reduction/output")

    def read(array, key):
        assert not array.variable_name.startswith("Data.")
        return getitem(array, key)

    with xr.open_dataset(timestream_path) as raw:
        monkeypatch.setattr(xr.Dataset, "close", record_close)
        monkeypatch.setattr(xr, "open_dataset", unexpected)
        monkeypatch.setattr(SweepReduce, "__call__", unexpected)
        monkeypatch.setattr(DataProdOutput, "run", unexpected)
        monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", read)
        attrs = raw.attrs.copy()
        working = run_data(raw, source, RuntimeContext(), dry_run=True)
        current = working["tolteca_kids"].attrs["input_row"]
        assert {"io_obj", "accessor_obj"}.isdisjoint(current)
        assert current["source"] == source
        assert (current["master"], current["obsnum"], current["roach"]) == (
            "tcs",
            145609,
            0,
        )
        assert NoOp.get_context(working).completed
        assert raw.attrs == attrs
        assert not closed
    assert len(closed) == 1 and closed[0] is raw


@pytest.mark.parametrize("dry_run", [False, True])
def test_open_table_runs_without_reopening_and_preserves_tree(
    target_sweep_path, monkeypatch, dry_run
):
    table = guess_info_from_sources([target_sweep_path])
    with table.toltec_file.open() as opened:
        records = list(opened.itertuples(index=False))
        objects = opened.io_obj
        assert objects is not None
        raw = objects.iloc[0]
        working = ensure_datatree(raw)
        working["unrelated"] = xr.DataTree(dataset=xr.Dataset({"value": 4}))
        working["tolteca_kids"] = xr.DataTree()
        working["tolteca_kids"].attrs["note"] = "caller-owned"
        working["tolteca_kids/unrelated"] = xr.DataTree(
            dataset=xr.Dataset({"value": 5})
        )

        def unexpected(*args, **kwargs):
            pytest.fail("run_data opened or closed a caller-owned source")

        with monkeypatch.context() as borrowed:
            borrowed.setattr(xr, "open_dataset", unexpected)
            borrowed.setattr(xr.Dataset, "close", unexpected)
            borrowed.setattr(xr.DataTree, "close", unexpected)
            result = run_data(
                working, records[0].source, RuntimeContext(), dry_run=dry_run
            )
            assert result is working
            assert working["unrelated/value"].item() == 4
            assert working["tolteca_kids/unrelated/value"].item() == 5
            assert working["tolteca_kids"].attrs["note"] == "caller-owned"
            if dry_run:
                assert NoOp.get_context(working).completed
                assert SweepReduce.result_path() not in working
            else:
                view = SweepReduce.view(working, block=0)
                expected = apply_step(
                    SweepReduce(),
                    BlockCreate()(ensure_datatree(raw)).toltec_kids.blocks[0].tree,
                ).toltec_kids.sweep
                assert SweepReduce.get_context(working["blocks/0"]).completed
                for name in ("I", "Q", "unc_I", "unc_Q", "frequency"):
                    xr.testing.assert_identical(
                        getattr(view, name), getattr(expected, name)
                    )


@pytest.mark.parametrize("failure", ["config", "step"])
def test_borrowed_data_remains_open_on_failure(timestream_path, monkeypatch, failure):
    source = timestream_path
    runtime = RuntimeContext(
        {"kids": {"sweep_reduce": {"block": "invalid"}}} if failure == "config" else {}
    )
    with xr.open_dataset(timestream_path) as raw:
        working = ensure_datatree(raw)

        def fail(cls, data, context):
            msg = "test step failure"
            raise RuntimeError(msg)

        def unexpected(*args, **kwargs):
            pytest.fail("run_data closed caller-owned data on failure")

        with monkeypatch.context() as borrowed:
            borrowed.setattr(xr.Dataset, "close", unexpected)
            borrowed.setattr(xr.DataTree, "close", unexpected)
            borrowed.setattr(NoOp, "run", classmethod(fail))
            error = {
                "config": ValidationError,
                "step": RuntimeError,
            }[failure]
            with pytest.raises(error):
                run_data(working, source, runtime, dry_run=True)
            if failure == "step":
                assert not NoOp.get_context(working).completed
            assert raw["Header.Toltec.ObsNum"].item() == 145609


def test_run_data_publishes_configured_product(target_sweep_path, tmp_path):
    source = target_sweep_path
    runtime = RuntimeContext(
        {"kids": {"output": {"enabled": True, "path": tmp_path, "namespace": "result"}}}
    )
    with xr.open_dataset(target_sweep_path) as raw:
        working = run_data(raw, target_sweep_path, runtime)
        context = DataProdOutput.get_context(working)
        assert context.completed
        assert context.data is not None
        assert context.data.parent == tmp_path / "result/145606-0-1/dp_raw_obs_proc"
        provenance = read_product_provenance(context.data).inputs["toltec0"]
        assert "145606" in provenance.uid_raw_obs_file
        assert provenance.config is not None and provenance.config.output.enabled
        with open_sweep_product(context.data) as reopened:
            xr.testing.assert_identical(
                SweepReduce.view(reopened, block=0).I,
                SweepReduce.view(working, block=0).I,
            )


def test_batch_noop_retains_each_input_and_table_order(
    data_lmt_path, target_sweep_path, tmp_path, monkeypatch
):
    sources = [
        next((data_lmt_path / "toltec/tcs/toltec7").glob("*156755*_tune.nc")),
        target_sweep_path,
        next((data_lmt_path / "toltec/tcs/toltec11").glob("*156755*_tune.nc")),
    ]
    table = guess_info_from_sources(sources)
    output = tmp_path / "products"
    runtime = RuntimeContext({"kids": {"output": {"enabled": True, "path": output}}})

    def unexpected(*args, **kwargs):
        pytest.fail("A batch no-op reserved a product or changed resource ownership")

    with table.toltec_file.open() as opened:
        with monkeypatch.context() as borrowed:
            borrowed.setattr(runner.DataProdStore, "publication", unexpected)
            borrowed.setattr(xr, "open_dataset", unexpected)
            borrowed.setattr(xr.Dataset, "close", unexpected)
            borrowed.setattr(xr.DataTree, "close", unexpected)
            results = list(runner.run_inputs(opened, runtime, dry_run=True))
        assert [uid for uid, _ in results] == opened.uid_raw_obs.tolist()
        assert len(results) == len(sources)
        for (_, working), source in zip(results, sources, strict=True):
            assert NoOp.get_context(working).completed
            assert working["tolteca_kids"].attrs["input_row"]["source"] == source
            assert {"io_obj", "accessor_obj"}.isdisjoint(
                working["tolteca_kids"].attrs["input_row"]
            )
            assert "interfaces" not in working
            assert not DataProdOutput.has_context(working)
    assert not output.exists()


@pytest.mark.parametrize("option", ["--dry_run", "--show_config"])
def test_cli_consumes_native_rows(timestream_path, monkeypatch, option):
    table = guess_info_from_sources([timestream_path])
    expected = next(table.itertuples(index=False))
    calls = []
    run_input = runner.run_input

    def unexpected(*args, **kwargs):
        pytest.fail("A row consumer converted the table to dictionaries")

    def run(data, row, runtime, *, dry_run, defer_output=False):
        calls.append(row.source)
        assert dry_run
        assert isinstance(data, xr.Dataset)
        assert row.source_loc is expected.source_loc
        assert {"io_obj", "accessor_obj"}.isdisjoint(row._fields)
        assert not any(name.startswith("_") for name in row._fields)
        return run_input(data, row, runtime, dry_run=dry_run, defer_output=defer_output)

    def create(runtime, *, context_input):
        calls.append(context_input["source"])
        assert context_input["data_kind_toltec"] == expected.data_kind_toltec
        assert isinstance(context_input["file_ut"], pd.Timestamp)
        assert context_input["source_loc"] is expected.source_loc
        assert {"io_obj", "accessor_obj"}.isdisjoint(context_input)
        assert not any(name.startswith("_") for name in context_input)
        return runner.create_kids(runtime, context_input=context_input)

    monkeypatch.setattr(cli.InputDiscovery, "discover", lambda *args, **kwargs: table)
    monkeypatch.setattr(runner, "run_input", run)
    monkeypatch.setattr(cli, "create_kids", create)
    monkeypatch.setattr(pd.DataFrame, "to_dict", unexpected)
    result = CliRunner().invoke(cli.app, [option, "--", str(timestream_path)])
    assert result.exit_code == 0, result.exception
    assert calls == [expected.source]


def test_runner_uses_same_native_row_for_config_and_steps(timestream_path, monkeypatch):
    create = runner.create_kids
    contexts = []

    def capture(runtime, *, context_input):
        contexts.append(context_input)
        kids = create(runtime, context_input=context_input)
        assert kids.config.sweep_reduce.force
        return kids

    def unexpected(*args, **kwargs):
        pytest.fail("The runner converted a table to dictionaries")

    monkeypatch.setattr(runner, "create_kids", capture)
    monkeypatch.setattr(pd.DataFrame, "to_dict", unexpected)
    with xr.open_dataset(timestream_path) as raw:
        runtime = RuntimeContext(
            [
                {
                    "enable_if": (
                        f"data_kind_toltec == {ToltecDataKind.raw_timestream.value}"
                        " & file_ut.dt.year == 2025 & n_blocks.isna()"
                    ),
                    "source": {"kids": {"sweep_reduce": {"force": True}}},
                }
            ]
        )
        working = run_data(raw, timestream_path, runtime, dry_run=True)
    row = working["tolteca_kids"].attrs["input_row"]
    [context] = contexts
    assert isinstance(row, dict)
    assert {"io_obj", "accessor_obj"}.isdisjoint(context)
    assert {"io_obj", "accessor_obj"}.isdisjoint(row)
    assert row["source"] is timestream_path
    assert isinstance(row["file_ut"], pd.Timestamp)
    assert pd.api.types.is_integer(row["data_kind_toltec"])
    assert pd.isna(row["n_blocks"])
    assert context.keys() == row.keys()
    assert all(context[name] is value for name, value in row.items())
