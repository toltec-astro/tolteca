"""Two independent drive-fit workflows and their registered product lineage."""

import json

import numpy as np
import pytest
import xarray as xr
from astropy.table import QTable
from tollan.config import RuntimeContext
from tollan.pipeline import StepFailureKind
from typer.testing import CliRunner

from tolteca_datamodels.data_prod import DataProdIndex
from tolteca_datamodels.data_prod_store import DataProdStore, DataProdTagBusyError
from tolteca_kids.drive_fit import (
    DriveFit,
    DriveFitOutputStep,
    DriveFitPlotStep,
    DriveFitStep,
    DriveFitViz,
    make_drive_fit_product_index,
    open_drive_fit_product,
    prepare_drive_fit_inputs,
    read_drive_fit_context,
)
from tolteca_kids.drive_fit.artifacts import DriveFitTableArtifact
from tolteca_kids.drive_fit_commit import (
    DriveFitCommit,
    DriveFitCommitPlotStep,
    DriveFitCommitStep,
    DriveFitCommitViz,
    compute_ampcor,
    open_drive_fit_commit_product,
    prepare_drive_fit_commit_inputs,
)
from tolteca_kids.drive_fit_commit.cli import app as commit_app
from tolteca_kids.match1d import Match1D


@pytest.fixture(scope="module")
def sequence(drive_fit_calibration_sequence):
    return drive_fit_calibration_sequence


@pytest.fixture(scope="module")
def published(sequence, tmp_path_factory):
    root = tmp_path_factory.mktemp("drive-fit-products")
    runtime = RuntimeContext(
        {
            "drive_fit": {
                "fit": {"channels": [0, 100, 300, 684]},
                "plot": {"enabled": True},
                "output": {"enabled": True, "path": root, "run": {"name": "fit"}},
            }
        }
    )
    working = DriveFit(runtime)(sequence)
    return DriveFitOutputStep.get_context(working).data, working


def test_typed_fit_artifacts_and_context_roundtrip(published):
    path, working = published
    live = working["interfaces/toltec0"]
    context = DriveFitStep.get_context(live)
    assert context.completed and context.data is not None
    assert context.artifacts[DriveFitTableArtifact].data is not None
    index = DataProdIndex.read(path)
    assert index.meta["data_prod_type"] == "dp_drivefit"
    assert len(index.assocs) == 31
    assert sum(a.is_primary for a in index.assocs) == 1
    assert {i.meta["format"] for i in index.data_items} == {
        "zarr",
        "ecsv",
        "html",
        "js",
        "data_prod_index",
    }
    assert all(
        i.meta.get("interface") == "toltec0"
        for i in index.data_items
        if i.meta["format"] != "data_prod_index"
    )
    with open_drive_fit_product(path) as saved:
        restored = read_drive_fit_context(saved)
        assert restored.completed and restored.data is None and not restored.artifacts
        assert restored.config == context.config
        xr.testing.assert_identical(DriveFitStep.view(saved).data, context.data.data)
    # Serialization must not remove the live context or artifacts.
    assert DriveFitStep.get_context(live) is context
    assert DriveFitPlotStep.get_context(live).artifacts[DriveFitViz].data is not None


def test_sequence_identity_uses_members_not_count(sequence):
    first = make_drive_fit_product_index(prepare_drive_fit_inputs(sequence[:5]))
    second = make_drive_fit_product_index(
        prepare_drive_fit_inputs([sequence[0], *sequence[2:6]])
    )
    reordered = make_drive_fit_product_index(
        prepare_drive_fit_inputs(reversed(sequence[:5]))
    )
    assert first.uid != second.uid
    assert first == reordered


def test_commit_uses_shared_matcher_and_saved_results(
    published, sequence, tmp_path, monkeypatch
):
    reference, _ = published
    calls = []
    original = Match1D.__call__

    def record(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Match1D, "__call__", record)
    runtime = RuntimeContext(
        {
            "drive_fit_commit": {
                "plot": {"enabled": True},
                "output": {
                    "enabled": True,
                    "path": tmp_path,
                    "run": {"name": "commit"},
                },
            }
        }
    )
    working = DriveFitCommit(runtime)([sequence[-1]], reference=reference)
    tree = working["interfaces/toltec0"]
    result = DriveFitCommitStep.view(tree)
    assert len(calls) == 1
    assert calls[0]["shift_kw"]["shift_max"].to_value("MHz") == 10
    assert result.data.sizes["chan"] == 685
    assert result.n_success > 0
    assert result.data.amp_best.max().item() == 1
    assert np.isfinite(result.data.amp_best.values).all()
    assert result.data.attrs["atten_drive_global"] % 0.25 == 0
    path = DriveFitOutputStep.get_context(working).data
    index = DataProdIndex.read(path)
    assert index.meta["data_prod_type"] == "dp_drive_fit_commit"
    assert {(a.uid, a.is_primary) for a in index.assocs} == {
        ("ics-17068-30-0", True),
        (DataProdIndex.read(reference).uid, False),
    }
    assert {i.meta["format"] for i in index.data_items} == {
        "zarr",
        "ecsv",
        "ascii",
        "text",
        "html",
        "data_prod_index",
    }
    table_path = next(path.rglob("adrv_commit.ecsv"))
    table = QTable.read(table_path)
    np.testing.assert_array_equal(table["amp_best"], result.table["amp_best"])
    amp_path = next(path.rglob("default_targ_amps.dat"))
    np.testing.assert_array_equal(np.loadtxt(amp_path), result.data.amp_best)
    assert (
        float(next(path.rglob("atten_drive.dat")).read_text())
        == result.data.attrs["atten_drive_global"]
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Reopening and plotting must not read raw files or match again")

    monkeypatch.setattr(Match1D, "__call__", forbidden)
    monkeypatch.setattr("tolteca_kids.drive_fit_commit.step._read_sweep", forbidden)
    with open_drive_fit_commit_product(path) as saved:
        xr.testing.assert_identical(DriveFitCommitStep.view(saved).data, result.data)
        DriveFitCommitPlotStep()(saved)
        fig = (
            DriveFitCommitPlotStep.get_context(saved).artifacts[DriveFitCommitViz].data
        )
        np.testing.assert_array_equal(
            fig.data[3].y,
            result.data.power_error_db.values[~result.data.amp_flag.values],
        )


def test_commit_cli_and_ambiguous_targets(published, sequence, tmp_path):
    reference, _ = published
    with pytest.raises(ValueError, match="exactly one target"):
        prepare_drive_fit_commit_inputs(sequence[:2], reference)
    result = CliRunner().invoke(
        commit_app,
        [
            "--drive_fit_commit.reference",
            str(reference),
            "--drive_fit_commit.plot.enabled",
            "true",
            "--drive_fit_commit.output.enabled",
            "true",
            "--drive_fit_commit.output.path",
            str(tmp_path),
            "--",
            str(sequence[-1]),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    [index] = list(tmp_path.glob("scratch/*/*/*/index.yaml"))
    assert DataProdIndex.read(index).meta["data_prod_type"] == "dp_drive_fit_commit"
    show = CliRunner().invoke(
        commit_app, ["--show_config", "--drive_fit_commit.commit.adrv_max", "25"]
    )
    assert show.exit_code == 0
    assert json.loads(show.output)["commit"]["adrv_max"] == 25


def test_missing_results_record_plot_failures():
    for step in (DriveFitPlotStep(), DriveFitCommitPlotStep()):
        tree = step(xr.DataTree())
        context = step.get_context(tree)
        assert not context.completed
        assert context.failures[0].kind == StepFailureKind.missing_input
        assert not context.artifacts


def test_ampcor_preserves_v2_fill_cutoff_and_relative_shape():
    result = compute_ampcor(
        [-1.0, 10.0, 20.0, np.nan],
        [0, 0, 0, 1],
        np.array([1.0, 0.5, 0.25, 0.5]) ** 2 / 1.5625,
        output_frequency=[1e6, 2e6, 3e6, 4e6],
        adrv_max=30,
        fill_percentile=50,
        reference_percentile=10,
        reference_step=0.25,
    )
    np.testing.assert_array_equal(result["m_good"], [True, True, True, False])
    np.testing.assert_array_equal(result["adrv_filled"], [2.0, 10.0, 20.0, 10.0])
    np.testing.assert_allclose(
        result["ampcor"],
        [1.0, 0.5 * 10 ** (-8 / 20), 0.25 * 10 ** (-18 / 20), 0.5 * 10 ** (-8 / 20)],
    )
    assert 0 <= result["adrv_exact"] - result["adrv_ref_total"] < 0.25
    with pytest.raises(ValueError, match="No valid drive"):
        compute_ampcor(
            [10.0],
            [1],
            [1.0],
            output_frequency=[1e6],
            adrv_max=30,
            fill_percentile=50,
            reference_percentile=10,
            reference_step=0.25,
        )


def test_busy_tag_prevents_fitting_and_failed_export_retains_prior_run(
    sequence, tmp_path, monkeypatch
):
    output = {"enabled": True, "path": tmp_path, "run": {"name": "first"}}
    runtime = RuntimeContext(
        {"drive_fit": {"fit": {"channels": [0]}, "output": output}}
    )
    working = DriveFit(runtime)(sequence)
    path = DriveFitOutputStep.get_context(working).data
    current = (path / "index.yaml").read_bytes()
    proposal = make_drive_fit_product_index(working)
    store = DataProdStore(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("A busy tag must stop before numerical fitting")

    with monkeypatch.context() as patch:
        patch.setattr("tolteca_kids.drive_fit.computation.fit_channel", forbidden)
        with (
            store.publication(proposal, primary_uid="ics-17068-0-0"),
            pytest.raises(DataProdTagBusyError),
        ):
            DriveFit(runtime)(sequence)

    def fail(*args, **kwargs):
        msg = "table export failed"
        raise OSError(msg)

    monkeypatch.setattr(DriveFitTableArtifact, "write", fail)
    with pytest.raises(OSError, match="table export failed"):
        DriveFit(
            runtime.with_config(
                overrides={"drive_fit": {"output": {"run": {"name": "failed"}}}}
            )
        )(sequence)
    assert (
        DataProdIndex.read(path.parent).tags["main"].current.uid
        == DataProdIndex.read(path).uid
    )
    assert (path / "index.yaml").is_file()
    assert not (path.parent / "failed").exists()
    assert any(run.status == "failed" for run in store.read_namespace().runs.values())
    # A new tag reservation can update auditing, but not the committed payload.
    assert (path / "index.yaml").read_bytes() == current


def test_tlaloc_output_reuses_published_settings_and_preserves_tone_order(
    published, sequence, tmp_path
):
    from tolteca_kids.tlaloc_output import TlalocOutput

    reference, _ = published
    etc = tmp_path / "tlaloc/etc"
    network = etc / "toltec0"
    network.mkdir(parents=True)
    from tolteca_kids.drive_fit.computation import _read_sweep

    target = _read_sweep(str(sequence[-1]), None)
    frequencies = QTable(
        {"f_centered": target.f_chan - target.meta.f_lo_center},
        meta={"f_lo_center": target.meta.f_lo_center},
    )
    frequencies.write(network / "targ_freqs.dat", format="ascii.ecsv")
    (network / "random_phases.dat").write_text("retain phases")
    (network / "default_targ_amps.dat").write_text("old amplitudes")
    (network / "atten_drive.dat").write_text("old attenuation")
    runtime = RuntimeContext(
        {
            "drive_fit_commit": {
                "output": {"enabled": True, "path": tmp_path / "products"},
                "tlaloc_output": {"enabled": True, "path": etc},
            }
        }
    )
    working = DriveFitCommit(runtime)([sequence[-1]], reference=reference)
    path = DriveFitOutputStep.get_context(working).data
    tree = working["interfaces/toltec0"]
    context = TlalocOutput.get_context(tree)
    assert context.completed and len(context.files) == 2
    assert (network / "random_phases.dat").read_text() == "retain phases"
    assert (network / "default_targ_amps.dat").read_bytes() == next(
        path.rglob("default_targ_amps.dat")
    ).read_bytes()
    assert (network / "atten_drive.dat").read_bytes() == next(
        path.rglob("atten_drive.dat")
    ).read_bytes()
    replayed = tmp_path / "replayed-etc/toltec0"
    replayed.mkdir(parents=True)
    frequencies.write(replayed / "targ_freqs.dat", format="ascii.ecsv")
    with open_drive_fit_commit_product(path) as saved:
        TlalocOutput(enabled=True, path=tmp_path / "replayed-etc")(saved)
    assert (tmp_path / "replayed-etc/toltec0/default_targ_amps.dat").read_bytes() == (
        network / "default_targ_amps.dat"
    ).read_bytes()


def test_invalid_tlaloc_settings_do_not_touch_destination(tmp_path):
    from pydantic import ValidationError

    from tolteca_kids.tlaloc_output import TlalocOutput, TlalocOutputConfig

    with pytest.raises(ValidationError, match="explicit path"):
        TlalocOutputConfig(enabled=True)
    tree = xr.DataTree.from_dict(
        {
            "tlaloc": xr.Dataset(
                {
                    "amp_tone": ("chan", [1.0, np.nan]),
                    "f_chan": ("chan", [501e6, 499e6]),
                },
                attrs={"roach": 0, "atten_drive": 10.0, "f_lo_center": 500e6},
            )
        }
    )
    with pytest.raises(ValueError, match="Invalid Tlaloc"):
        TlalocOutput(enabled=True, path=tmp_path / "absent")(tree)
    assert not (tmp_path / "absent").exists()


def test_incomplete_plot_prevents_publication(sequence, tmp_path):
    runtime = RuntimeContext(
        {
            "drive_fit": {
                "fit": {"channels": [0]},
                "plot": {"enabled": True, "fit_alias": "missing"},
                "output": {"enabled": True, "path": tmp_path},
            }
        }
    )
    with pytest.raises(ValueError, match="DriveFitPlotStep did not complete"):
        DriveFit(runtime)(sequence)
    assert not list(tmp_path.rglob("result.zarr"))
    assert all(
        run.status == "failed"
        for run in DataProdStore(tmp_path).read_namespace().runs.values()
    )


def test_publication_failure_prevents_tlaloc_application(
    published, sequence, tmp_path, monkeypatch
):
    from tolteca_kids.drive_fit_commit.artifacts import DriveFitCommitTableArtifact

    def fail(*args, **kwargs):
        msg = "commit table export failed"
        raise OSError(msg)

    monkeypatch.setattr(DriveFitCommitTableArtifact, "write", fail)
    runtime = RuntimeContext(
        {
            "drive_fit_commit": {
                "output": {"enabled": True, "path": tmp_path / "products"},
                "tlaloc_output": {"enabled": True, "path": tmp_path / "etc"},
            }
        }
    )
    with pytest.raises(OSError, match="commit table export failed"):
        DriveFitCommit(runtime)([sequence[-1]], reference=published[0])
    assert not (tmp_path / "etc").exists()
