"""Native fit validation preserves acquired-channel model identity."""

import os
import shutil
import subprocess
from contextlib import ExitStack

import numpy as np
import pytest
from astropy.table import Table
from tollan.config import RuntimeContext
from tollan.utils.fileloc import fileloc

from tolteca_datamodels.data_prod import (
    DataProdIndex,
    DataProdItem,
    DataProdRef,
    DataProdSource,
    DataProdStore,
    Location,
)
from tolteca_datamodels.data_prod import WorkflowRun as ExecutionRun
from tolteca_datamodels.raw_obs_context import SweepCalDataItem, SweepCalItem
from tolteca_kids import (
    prepare_model_binding,
)
from tolteca_kids.dataprod_output import DataProdOutput
from tolteca_kids.model_fit import ModelFit, open_fit
from tolteca_kids.model_fit_artifacts import KidscppFitFiles, ModelFitTable
from tolteca_kids.product import make_kids_product_index
from tolteca_kids.runner import run_sources


@pytest.fixture(scope="module", params=[0, 1])
def native_fit(data_lmt_path, tmp_path_factory, request):
    executable = os.environ.get("KIDSCPP_EXECUTABLE")
    if not executable:
        pytest.skip("Set KIDSCPP_EXECUTABLE for native fit/install integration")
    source = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    report = tmp_path_factory.mktemp("native-fit") / "fit.ecsv"
    result = subprocess.run(  # noqa: S603 -- explicitly selected native executable
        [
            executable,
            "fit",
            str(source),
            "--blocks",
            str(request.param),
            "--output",
            str(report),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return source, report


@pytest.mark.parametrize(
    "change",
    [
        "obs",
        "block",
        "channels",
        "frequency",
        "accumlen",
        "dtype",
        "units",
        "incomplete",
    ],
)
def test_invalid_fit_does_not_replace_installed_files(native_fit, tmp_path, change):
    source, report = native_fit
    table = Table.read(report, format="ascii.ecsv")
    if change == "obs":
        table.meta["Header.Toltec.ObsNum"] += 1
    elif change == "block":
        table.meta["block"] = 2
    elif change == "channels":
        table = table[::-1]
    elif change == "frequency":
        table["f_in"][0] += 100
    elif change == "accumlen":
        table.meta["Header.Toltec.AccumLen"] *= 2
    elif change == "dtype":
        table["flag"] = table["flag"].astype(float)
    elif change == "units":
        table["fr"].unit = "kHz"
    elif change == "incomplete":
        table.meta["acquisition_status"]["blocks"][table.meta["block"]]["state"] = (
            "ongoing"
        )
    invalid = tmp_path / "invalid.ecsv"
    table.write(invalid, format="ascii.ecsv")
    network = tmp_path / "etc/toltec0"
    network.mkdir(parents=True)
    sentinel = network / "targ_freqs.dat"
    sentinel.write_bytes(b"previous installed result\n")

    with pytest.raises(ValueError), open_fit(source, invalid):
        pytest.fail("Invalid native model binding accepted")
    assert sentinel.read_bytes() == b"previous installed result\n"
    assert list(network.iterdir()) == [sentinel]


def test_unusable_fit_row_keeps_nan_and_companions(native_fit, tmp_path):
    source, report = native_fit
    native = Table.read(report, format="ascii.ecsv")
    native["flag"][0] = 0
    native["Qr"][0] = np.nan
    flagged = tmp_path / "flagged.ecsv"
    native.write(flagged, format="ascii.ecsv")
    with pytest.raises(ValueError, match="invalid values"), open_fit(source, flagged):
        pytest.fail("Unexpected unflagged NaN coefficient")
    native["flag"][0] |= 8192
    native.write(flagged, format="ascii.ecsv", overwrite=True)
    with open_fit(source, flagged) as (model, _):
        assert np.isnan(model["Qr"][0])
        assert model["flag"][0] == native["flag"][0]


def test_published_model_binds_actual_sweep_and_rejects_false_scope(
    native_fit, tmp_path
):
    source, report = native_fit
    native = Table.read(report, format="ascii.ecsv")
    block = native.meta["block"]
    sweep = SweepCalItem(
        type="sweep",
        raw_obs=DataProdRef(uid="tcs-145609-0-1"),
        data_items=(SweepCalDataItem(interface="toltec0", block=block),),
    )
    index = make_kids_product_index("tcs", 145609, 0, 1)
    store = DataProdStore(tmp_path)
    with (
        ExitStack() as _run_records,
        ExecutionRun.create() as invocation,
        store.materialize(index.model_copy(update={"run": invocation})) as out,
    ):
        _run_records.callback(store.write_run, invocation)
        assert out.base_path is not None
        shutil.copyfile(report, out.base_path / "model.ecsv")
        out.locations["self"] = Location(root=".", location_type="local")
        out.product.data_items.append(
            DataProdItem(
                meta={
                    "type": "artifact",
                    "format": "ecsv",
                    "interface": "toltec0",
                    "scope": f"blocks/{block}",
                    "producer": "kidscpp.fit",
                    "name": "model",
                    "run_uid": str(invocation.uid),
                },
                source=DataProdSource.model_validate(
                    {"location": "self", "path": "model.ecsv"}
                ),
            )
        )
    published = DataProdIndex.read(out.base_path)
    before = published.model_dump_json()
    selected = prepare_model_binding(
        published, sweep=sweep, source=source, producer="kidscpp.fit", name="model"
    )
    assert selected.raw_obs == sweep.raw_obs
    assert selected.data_items[0].model is not None
    assert selected.data_items[0].model.product == published.product.reference
    assert published.model_dump_json() == before
    assert sweep.data_items[0].model is None
    # Renaming an artifact's scope cannot make another native block match.
    [artifact] = [
        i for i in published.product.data_items if i.meta.get("name") == "model"
    ]
    other = 1 - block
    artifact.meta["scope"] = f"blocks/{other}"
    wrong = SweepCalItem(
        type="sweep",
        raw_obs=sweep.raw_obs,
        data_items=(SweepCalDataItem(interface="toltec0", block=other),),
    )
    with pytest.raises(ValueError, match="recorded calibration sweep"):
        prepare_model_binding(
            published, sweep=wrong, source=source, producer="kidscpp.fit", name="model"
        )


@pytest.mark.parametrize("retain_processed", [False, True])
def test_file_workflow_publishes_each_fitted_block_and_preserves_main_on_failure(
    data_lmt_path, tmp_path, retain_processed
):
    executable = os.environ.get("KIDSCPP_EXECUTABLE")
    if not executable:
        pytest.skip("Set KIDSCPP_EXECUTABLE for native model workflow integration")
    source = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    config = {
        "kids": {
            "model_fit": {
                "backend": "kidscpp",
                "executable": executable,
                "retain_processed": retain_processed,
            },
            "output": {
                "enabled": True,
                "path": tmp_path,
                "namespace": "models",
                "tag": "main",
                "run_name": {"generator": "counter", "key": "fit"},
            },
        },
    }
    for _, working in run_sources([source], RuntimeContext(config)):
        output = DataProdOutput.get_context(working)
        assert output.completed
        assert output.data is not None
        index = DataProdIndex.read(output.data)
        models = [i for i in index.product.data_items if i.meta.get("name") == "model"]
        assert {i.meta["scope"] for i in models} == {"blocks/0", "blocks/1"}
        processed = [
            i for i in index.product.data_items if i.meta.get("name") == "processed"
        ]
        assert {i.meta["scope"] for i in processed} == (
            {"blocks/0", "blocks/1"} if retain_processed else set()
        )
        assert all(i.meta["format"] == "nc" for i in processed)
        for block in working.toltec_kids.blocks:
            context = ModelFit.get_context(block.tree)
            diagnostics = context.artifacts[KidscppFitFiles].data
            assert set(diagnostics) == {
                "blocks.ecsv",
                "channels.ecsv",
                "command.json",
                "native.log",
            } | ({"processed.nc"} if retain_processed else set())
            directory = (
                output.data
                / "artifacts/toltec0"
                / f"blocks/{block.meta.block}"
                / "tolteca_kids.model_fit:ModelFit"
            )
            assert (directory / "kids_model.ecsv").read_bytes() == (
                context.artifacts[ModelFitTable].data
            )
            assert {p.name for p in directory.iterdir()} == {
                "kids_model.ecsv",
                "kidscpp",
            }
            for filename, content in diagnostics.items():
                diagnostic_path = directory / "kidscpp" / filename
                assert diagnostic_path.read_bytes() == content
                [item] = [
                    i
                    for i in index.product.data_items
                    if i.source.path == diagnostic_path.relative_to(output.data)
                ]
                assert item.meta["scope"] == f"blocks/{block.meta.block}"
                assert item.meta["producer"] == "tolteca_kids.model_fit:ModelFit"
                assert item.meta["interface"] == "toltec0"
        for artifact in models:
            assert artifact.meta["interface"] == "toltec0"
            assert artifact.source.path.as_posix() == (
                f"artifacts/toltec0/{artifact.meta['scope']}/"
                "tolteca_kids.model_fit:ModelFit/kids_model.ecsv"
            )
            with open_fit(source, fileloc(index.get_source_uri(artifact.source)).path):
                pass
        group_path = output.data.parent / "index.yaml"
        group_before = group_path.read_bytes()
        artifacts_before = list(tmp_path.rglob("kids_model.ecsv"))
    config["kids"]["model_fit"]["executable"] = str(tmp_path / "missing-kidscpp")
    with pytest.raises(FileNotFoundError, match="missing-kidscpp"):
        list(run_sources([source], RuntimeContext(config)))
    assert group_path.read_bytes() == group_before
    assert list(tmp_path.rglob("kids_model.ecsv")) == artifacts_before
