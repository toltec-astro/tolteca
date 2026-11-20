"""Recorded calibration drives native solving, persistence and pinned replay."""

import os
import shutil
import subprocess
from contextlib import ExitStack

import numpy as np
import pytest
import xarray as xr
from tollan.config import RuntimeContext
from tollan.utils.fileloc import fileloc

from tolteca_datamodels.data_prod import (
    DataProdIndex,
    DataProdItem,
    DataProdSource,
    DataProdStore,
    Location,
    WorkflowRun,
)
from tolteca_datamodels.raw_obs_context import read_run_inputs
from tolteca_datamodels.toltec.file import guess_info_from_sources
from tolteca_kids.context import _prepare_context_inputs
from tolteca_kids.core import Kids
from tolteca_kids.dataprod_output import DataProdOutput
from tolteca_kids.product import make_kids_product_index
from tolteca_kids.runner import run_sources


@pytest.fixture(scope="module")
def reference(data_lmt_path, tmp_path_factory):
    executable = os.environ.get("KIDSCPP_EXECUTABLE")
    if not executable:
        pytest.skip("Set KIDSCPP_EXECUTABLE for calibrated native solving")
    data_lmt_path = data_lmt_path.resolve()
    root = tmp_path_factory.mktemp("context-native")
    tune = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    timestream = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_0002_*.nc"))
    report = root / "fit.ecsv"
    subprocess.run(  # noqa: S603 -- explicit native executable and literal argv
        [executable, "fit", str(tune), "--blocks", "1", "--output", str(report)],
        check=True,
        capture_output=True,
    )
    expected = root / "expected.nc"
    subprocess.run(  # noqa: S603 -- explicit native executable and literal argv
        [
            executable,
            "solve",
            str(timestream),
            "--blocks",
            "0",
            "--model_params",
            str(report),
            "--output",
            str(expected),
        ],
        check=True,
        capture_output=True,
    )
    files = [
        *sorted((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*.nc")),
        *sorted((data_lmt_path / "toltec/tcs/toltec1").glob("*145609*.nc")),
        *sorted((data_lmt_path / "tel").glob("*145609*.nc")),
    ]
    return {
        "executable": executable,
        "root": data_lmt_path.resolve(),
        "files": files,
        "report": report,
        "expected": expected,
    }


def _model(store, reference, name):
    with ExitStack() as records, WorkflowRun.create() as run:
        records.callback(store.write_run, run)
        index = make_kids_product_index("tcs", 145609, 0, 1, run=run)
        with store.materialize(index, run_name=name) as output:
            shutil.copyfile(reference["report"], output.base_path / "model.ecsv")
            output.locations["self"] = Location(root=".", location_type="local")
            output.product.data_items.append(
                DataProdItem(
                    meta={
                        "type": "artifact",
                        "format": "ecsv",
                        "producer": "kidscpp.fit",
                        "name": "model",
                        "interface": "toltec0",
                        "scope": "blocks/1",
                        "run_uid": str(run.uid),
                    },
                    source=DataProdSource.model_validate(
                        {"location": "self", "path": "model.ecsv"}
                    ),
                )
            )
    return DataProdIndex.read(output.base_path)


def _config(reference, root, model):
    return {
        "kids": {
            "input": {
                "select": "scannum == 2 and interface == 'toltec0'",
                "raise_on_errors": True,
            },
            "context": {
                "locations": {
                    "data_lmt": {
                        "root": reference["root"].as_uri(),
                        "location_type": "local",
                    }
                },
                "models": [
                    {"index": str(model.base_path.parent / "index.yaml"), "tag": "main"}
                ],
            },
            "timestream_solve": {
                "enabled": True,
                "backend": "kidscpp",
                "executable": reference["executable"],
            },
            "timestream_reduce": {"psd": None},
            "output": {
                "enabled": True,
                "path": str(root),
                "run_name": {"generator": "counter", "key": "solve"},
            },
        }
    }


def _run(files, config):
    outputs = []
    for _uid, working in run_sources(files, RuntimeContext(config)):
        context = DataProdOutput.get_context(working)
        assert context.completed, context.failures
        outputs.append(context.data)
    assert len(outputs) == 1
    return DataProdIndex.read(outputs[0])


def test_context_consumption_and_replay_preserve_full_inventory(reference, tmp_path):
    store = DataProdStore(tmp_path / "products")
    first_model = _model(store, reference, "fit-first")
    config = _config(reference, tmp_path / "products", first_model)
    table = guess_info_from_sources(reference["files"])
    with table.toltec_file.open() as opened:
        table.toltec_file.read_metadata()
        kids = Kids(RuntimeContext(config))
        prepared = _prepare_context_inputs(
            opened, kids.input.select(opened), kids.config.context
        )
    raw = prepared.raw_indexes["tcs-145609-0-2"]
    interfaces = {item.meta["interface"] for item in raw.product.data_items}
    assert {"toltec0", "tel"} == interfaces
    assert [item.interface for item in prepared.contexts[0].data_items] == ["toltec0"]
    raw_paths = [store.write_index(index) for index in prepared.batch.items]
    before = {path: path.read_bytes() for path in raw_paths}
    first = _run(reference["files"], config)
    assert {path: path.read_bytes() for path in raw_paths} == before
    _check_native(first, reference)
    first_context = read_run_inputs(first.run)
    assert (
        first_context[0].data_items[0].cal_items[0].data_items[0].model.product
        == first_model.product.reference
    )
    assert store.read_run(first.run.uid).status == "completed"
    second_model = _model(store, reference, "fit-second")
    assert (
        DataProdIndex.read(first_model.base_path.parent).product.tags["main"].current
        == second_model.product.reference
    )
    config["kids"]["context"]["raw_indexes"] = [str(path) for path in raw_paths]
    config["kids"]["context"]["locations"]["data_lmt"]["root"] = str(reference["root"])
    config["kids"]["context"]["replay"] = str(
        store.namespace_path / ".runs/by_uid" / str(first.run.uid) / "run.yaml"
    )
    replay = _run([], config)
    assert replay.run.uid != first.run.uid
    assert read_run_inputs(replay.run) == first_context
    _check_native(replay, reference)
    assert {path: path.read_bytes() for path in raw_paths} == before
    with xr.open_datatree(replay.base_path / "array.zarr", engine="zarr") as tree:
        solved = tree["toltec0/blocks/0/solved"].dataset
        assert (
            fileloc(solved.attrs["calibration_source"]).path
            == first_model.base_path / "model.ecsv"
        )
    config["kids"]["context"]["models"] = [
        {"index": str(second_model.base_path / "index.yaml")}
    ]
    with pytest.raises(ValueError, match="found 0"):
        list(run_sources([], RuntimeContext(config)))
    assert (
        DataProdIndex.read(replay.base_path.parent).product.tags["main"].current
        == replay.product.reference
    )


def _check_native(index, reference):
    with (
        xr.open_datatree(index.base_path / "array.zarr", engine="zarr") as tree,
        xr.open_dataset(reference["expected"]) as expected,
    ):
        assert list(tree.children) == ["toltec0"]
        actual = tree["toltec0/blocks/0/solved"].dataset
        for name in expected.data_vars:
            np.testing.assert_array_equal(actual[name], expected[name])
        assert actual.attrs["iq_normalization"] == expected.attrs["iq_normalization"]
        view = tree["toltec0"].toltec_kids.blocks[0].timestream
        np.testing.assert_array_equal(view.r, expected.r.T)
        np.testing.assert_array_equal(view.x, expected.x.T)


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array:DeprecationWarning"
)
def test_selected_stubs_survive_workflow_output_and_replay(reference, tmp_path):
    from tolteca_reduce.stubs import materialize_model_stubs

    store = DataProdStore(tmp_path / "products")
    model = _model(store, reference, "fit")
    config = _config(reference, tmp_path / "products", model)
    table = guess_info_from_sources(reference["files"])
    with table.toltec_file.open() as opened:
        table.toltec_file.read_metadata()
        kids = Kids(RuntimeContext(config))
        prepared = _prepare_context_inputs(
            opened, kids.input.select(opened), kids.config.context
        )
    indexes = {
        i.product.reference: i
        for i in (*prepared.raw_indexes.values(), *prepared.model_indexes.values())
    }
    with WorkflowRun.create() as producing:
        context, stub_index = materialize_model_stubs(
            prepared.contexts[0], indexes=indexes, store=store, run=producing
        )
    store.write_run(producing)
    stub_index.run = producing
    prepared.contexts = (context,)
    prepared.artifact_indexes = {stub_index.product.reference: stub_index}
    with WorkflowRun.create(
        context={
            "inputs": prepared.inputs,
            "input_indexes": prepared.snapshot_indexes(),
        }
    ) as selection:
        pass
    config["kids"]["context"]["raw_indexes"] = [
        str(store.write_index(i)) for i in prepared.batch.items
    ]
    config["kids"]["context"]["replay"] = str(store.write_run(selection))
    first = _run([], config)
    _check_native(first, reference)
    assert read_run_inputs(first.run)[0].stubs == context.stubs
    config["kids"]["context"]["replay"] = str(
        store.namespace_path / ".runs/by_uid" / str(first.run.uid) / "run.yaml"
    )
    replay = _run([], config)
    _check_native(replay, reference)
    assert read_run_inputs(replay.run)[0].stubs == context.stubs
    # The second execution uses the first execution's exact artifact inventory.
    assert replay.run.context["input_indexes"] == first.run.context["input_indexes"]
    assert all(s.artifact.resolve_path(stub_index).is_file() for s in context.stubs)


def test_unresolved_or_ambiguous_models_fail_before_execution(reference, tmp_path):
    store = DataProdStore(tmp_path / "products")
    first = _model(store, reference, "fit-first")
    second = _model(store, reference, "fit-second")
    config = _config(reference, tmp_path / "output", first)
    config["kids"]["context"]["models"] = []
    with pytest.raises(ValueError, match="Expected one exact model"):
        list(run_sources(reference["files"], RuntimeContext(config)))
    config["kids"]["context"]["models"] = [
        {"index": str(index.base_path / "index.yaml")} for index in (first, second)
    ]
    with pytest.raises(ValueError, match="found 2"):
        list(run_sources(reference["files"], RuntimeContext(config)))
    assert not (tmp_path / "output").exists()


def test_native_failure_retains_context_and_previous_tag(reference, tmp_path):
    store = DataProdStore(tmp_path / "products")
    model = _model(store, reference, "fit")
    config = _config(reference, tmp_path / "products", model)
    first = _run(reference["files"], config)
    config["kids"]["timestream_solve"]["executable"] = "/usr/bin/false"
    with pytest.raises(subprocess.CalledProcessError):
        list(run_sources(reference["files"], RuntimeContext(config)))
    saved = store.read_namespace_index()
    failed = [run for run in saved.runs.values() if run.status == "failed"]
    assert len(failed) == 1
    assert read_run_inputs(failed[0]) == read_run_inputs(first.run)
    assert (
        DataProdIndex.read(first.base_path.parent).product.tags["main"].current
        == first.product.reference
    )


def test_python_inspection_publishes_and_reopens_native_equivalent_results(
    reference, tmp_path
):
    """Real workflow uses lazy Python solving, persisted windows and a saved Viz."""
    from tolteca_kids.timestream_inspect_viz import TimestreamInspectionViz

    store = DataProdStore(tmp_path / "products")
    model = _model(store, reference, "fit")
    config = _config(reference, tmp_path / "products", model)
    config["kids"]["timestream_solve"] = {"enabled": True, "chunk_samples": 1024}
    config["kids"]["timestream_inspect"] = {"window_seconds": 2, "interval_seconds": 20}
    result = _run(reference["files"], config)
    with (
        xr.open_datatree(result.base_path / "array.zarr", engine="zarr") as tree,
        xr.open_dataset(reference["expected"]) as expected,
    ):
        actual = tree["toltec0/blocks/0/solved"].dataset
        for name in ("r", "x"):
            np.testing.assert_allclose(
                actual[name], expected[name], rtol=2e-13, atol=1e-18
            )
        np.testing.assert_array_equal(actual.flags, expected.flags)
        np.testing.assert_array_equal(actual.sample_index, expected.sample_index)
        np.testing.assert_array_equal(actual.chan, expected.chan)
        assert actual.attrs["iq_normalization"] == expected.attrs["iq_normalization"]
        inspection = tree["toltec0/blocks/0/timestream_inspect"]
        assert len(inspection["windows"].children) == 4
        assert set(inspection["trend"].data_vars) == {"I", "Q", "r", "x"}
        figure = TimestreamInspectionViz()(
            inspection, mode="native", signal="x", channels=[3], window=0
        )
        assert figure.data[0].type == "scattergl"
        np.testing.assert_array_equal(
            figure.data[0].y, inspection["windows/0"].dataset.x.isel(chan=3)
        )


@pytest.mark.parametrize(
    "setting",
    ["atten_drive", "atten_sense", "f_lo_center", "tone_amp", "f_tone", "tone_mask"],
)
def test_detector_settings_must_match_recorded_sweep(reference, setting):
    from tollan.accessor.xarray import ensure_datatree

    from tolteca_datamodels.toltec import open_toltec_kids
    from tolteca_kids import BlockCreate
    from tolteca_kids.timestream_solve import _validate_detector_state

    tune = next(path for path in reference["files"] if path.name.endswith("_tune.nc"))
    source = next(
        path
        for path in reference["files"]
        if path.name.startswith("toltec0_") and "_0002_" in path.name
    )
    with open_toltec_kids(tune) as a, open_toltec_kids(source) as b:
        sweep = BlockCreate()(ensure_datatree(a)).toltec_kids.blocks[1]
        timestream = BlockCreate()(ensure_datatree(b)).toltec_kids.blocks[0]
        _validate_detector_state(sweep, timestream)
        # Change only an isolated view used as invalid evidence, never the files.
        from copy import deepcopy
        from types import SimpleNamespace

        meta = timestream.meta.model_copy(deep=True)
        tones = deepcopy(timestream.chan_axis_table)
        if setting in type(meta).model_fields:
            meta = meta.model_copy(update={setting: getattr(meta, setting) + 1})
        elif setting == "tone_mask":
            tones[setting][0] = not tones[setting][0]
        else:
            tones[setting][0] *= 1.01
        changed = SimpleNamespace(meta=meta, chan_axis_table=tones)
        with pytest.raises(ValueError, match=setting):
            _validate_detector_state(sweep, changed)
