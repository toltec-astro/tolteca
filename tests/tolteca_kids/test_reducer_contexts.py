"""Live result publication, retained views and file-backed checkpoints."""

import pickle

import numpy as np
import pytest
import xarray as xr
from tollan.accessor.xarray import ensure_datatree
from tollan.config import RuntimeContext
from tollan.pipeline import PIPELINE_CONTEXT_KEY, SequentialPipeline
from typer.testing import CliRunner

from tolteca_datamodels.data_prod import DataProdIndex
from tolteca_datamodels.data_prod_store import DataProdStore
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_kids.cli import app
from tolteca_kids.dataprod_output import DataProdOutput
from tolteca_kids.noop import NoOp
from tolteca_kids.runner import run_data
from tolteca_kids.sweep_reduce import SweepReduce
from tolteca_kids.timestream_reduce import TimestreamReduce


def test_sweep_payload_shares_arrays_and_survives_rerun(target_sweep_path):
    with xr.open_dataset(target_sweep_path) as raw:
        working = ensure_datatree(raw)
        SweepReduce()(working)
        first = SweepReduce.get_context(working)
        view = first.data
        assert view is not None
        signal = view.I
        assert signal is not None
        assert signal.data is SweepReduce.view(working, block=0).I.data
        SweepReduce(force=True)(working)
        latest = SweepReduce.get_context(working)
        assert latest is not first
        assert latest.data is not view
        assert view.I.data is signal.data
        assert latest.data.I.data is not signal.data
        xr.testing.assert_identical(latest.data.I, signal)
        assert first.completed
        restored = SweepReduce.load_context(SweepReduce.dump_context(first))
        assert restored.data is None
        assert restored.model_dump() == first.model_dump()
        assert SweepReduce.get_context(working) is latest
        assert PIPELINE_CONTEXT_KEY not in view.data_source.attrs


def test_timestream_payload_retains_spectra_and_config(timestream_path):
    with open_toltec_kids(timestream_path) as native:
        raw = native.isel(time=slice(0, 256)).toltec_kids.select_channels([0, 5])
        working = ensure_datatree(raw)
        TimestreamReduce(psd={"nperseg": 64})(working)
        first = TimestreamReduce.get_context(working)
        view = first.data
        assert view is not None
        spectrum = view.I_psd
        TimestreamReduce(psd={"nperseg": 128})(working)
        latest = TimestreamReduce.get_context(working)
        assert latest.data.I_psd.sizes["f_psd"] == 65
        assert view.I_psd.data is spectrum.data
        assert view.I_psd.sizes["f_psd"] == 33
        assert view.psd.psd_analysis.nperseg == 64
        assert latest.data.psd.psd_analysis.nperseg == 128
        for node in working.subtree:
            if node is not working:
                assert PIPELINE_CONTEXT_KEY not in node.attrs


@pytest.mark.parametrize("kind", ["sweep", "timestream", "noop"])
def test_file_backed_checkpoint_after_source_close(
    target_sweep_path, timestream_path, kind
):
    source = target_sweep_path if kind == "sweep" else timestream_path
    step = {"sweep": SweepReduce, "timestream": TimestreamReduce, "noop": NoOp}[kind]
    with xr.open_dataset(source) as native:
        data = (
            native
            if kind == "sweep"
            else native.isel(time=slice(0, 128)).toltec_kids.select_channels([0, 5])
        )
        working = run_data(data, source, RuntimeContext(), dry_run=kind == "noop")
        scope = working if kind == "noop" else working["blocks/0"]
        before = step.get_context(scope)
        expected = None if before.data is None else before.data.I.values.copy()
        step(alias="repeat")(scope)
        checkpoint = pickle.dumps(working)
    restored = pickle.loads(checkpoint)  # noqa: S301 -- locally created checkpoint
    try:
        scope = restored if kind == "noop" else restored["blocks/0"]
        context = step.get_context(scope)
        assert context.completed
        assert context.make_step().config == before.config
        aliased = step.get_context(scope, alias="repeat")
        assert aliased.completed
        assert aliased.make_step().config.alias == "repeat"
        assert isinstance(restored["tolteca_kids"].attrs["input_row"], dict)
        if expected is None:
            assert context.data is None
        else:
            np.testing.assert_array_equal(context.data.I, expected)
            np.testing.assert_array_equal(aliased.data.I, expected)
        raw = (
            restored["raw"].dataset if "raw" in restored.children else restored.dataset
        )
        # Native samples remain accessible after the original handle closed.
        assert np.isfinite(raw["Data.Toltec.Is"].isel(time=0).values).all()
    finally:
        restored.close()


def test_missing_reduction_records_failure_and_allows_following_step(tmp_path):
    working = xr.DataTree()
    pipeline = SequentialPipeline(
        steps=[DataProdOutput(path=tmp_path, enabled=True), NoOp()]
    )
    assert pipeline(working) is working
    failed = DataProdOutput.get_context(working)
    assert not failed.completed
    assert failed.data is None
    assert [failure.kind for failure in failed.failures] == ["missing_input"]
    assert NoOp.get_context(working).completed
    assert not list(tmp_path.iterdir())


def test_lazy_spectral_failure_is_recorded_by_output(timestream_path, tmp_path):
    with open_toltec_kids(timestream_path) as native:
        raw = native.isel(time=slice(0, 128)).toltec_kids.select_channels([0])
        working = ensure_datatree(raw)
        config = RuntimeContext(
            {
                "kids": {
                    "timestream_reduce": {"psd": {"window": "invalid-window"}},
                    "output": {
                        "enabled": True,
                        "path": str(tmp_path),
                        "namespace": "failed",
                    },
                }
            }
        )
        with pytest.raises(ValueError, match="Invalid window name"):
            run_data(working, timestream_path, config)
        reduced = TimestreamReduce.get_context(working["blocks/0"])
        assert reduced.completed
        assert reduced.data is not None
        output = DataProdOutput.get_context(working)
        assert output.data is None
        assert not output.completed
        assert [failure.kind for failure in output.failures] == ["exception"]
        assert output.failures[0].exception_type.endswith("ValueError")
        assert not list((tmp_path / "failed").rglob("array.zarr"))


def test_cli_reports_missing_publication_and_continues_inputs(
    target_sweep_path, tmp_path
):
    sources = sorted(target_sweep_path.parent.glob("*1456*sweep.nc"))
    assert len(sources) == 2
    result = CliRunner().invoke(
        app,
        [
            "--kids.sweep_reduce.enabled",
            "false",
            "--kids.output.enabled",
            "true",
            "--kids.output.path",
            str(tmp_path / "products"),
            "--",
            *map(str, sources),
        ],
    )
    assert result.exit_code == 1
    assert result.stderr.count("output failed:") == 2
    assert "no completed reduction" in result.stderr
    assert "completed without output" not in result.stdout
    indexes = list((tmp_path / "products").rglob("index.yaml"))
    assert len(indexes) == 2
    groups = set()
    for path in indexes:
        index = DataProdIndex.read(path)
        groups.add(index.uid)
        assert not index.products
        assert {item.meta["format"] for item in index.data_items} == {"data_prod_index"}
        assert index.tags["main"].current is None
    namespace = DataProdStore(tmp_path / "products").read_namespace()
    [run] = namespace.runs.values()
    assert run.status == "failed"
    assert run.time_completed is not None
    assert set(run.publications) == groups
    assert all(request.status == "failed" for request in run.publications.values())
    assert all(request.tag == "main" for request in run.publications.values())
    assert not list((tmp_path / "products").rglob("array.zarr"))
