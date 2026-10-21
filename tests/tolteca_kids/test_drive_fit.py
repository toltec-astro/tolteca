"""Drive selection, configured sequence execution and portable results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError
from tollan.config import RuntimeContext
from typer.testing import CliRunner

from tolteca_kids.drive_fit import (
    DriveFit,
    DriveFitGridViz,
    DriveFitPlotStep,
    DriveFitResult,
    DriveFitStep,
    DriveFitStepConfig,
    DriveFitSweepArtifacts,
    DriveFitViz,
    find_best_drive,
    open_drive_fit_product,
    run,
    write_drive_fit_product,
)
from tolteca_kids.drive_fit.cli import app
from tolteca_kids.drive_fit.fit import fit_channel


@pytest.fixture(scope="module")
def sequence(drive_fit_calibration_sequence):
    return drive_fit_calibration_sequence


@pytest.fixture(scope="module")
def working(sequence):
    runtime = RuntimeContext().with_config(
        overrides={
            "drive_fit": {"fit": {"channels": [0, 100, 300, 684]}},
        }
    )
    return DriveFit(runtime)(reversed(sequence))


@pytest.fixture(scope="module")
def fitted(working):
    return DriveFitStep.view(working["interfaces/toltec0"])


def test_root_configuration_is_independent_of_kids():
    runtime = RuntimeContext().with_config(
        overrides={
            "kids": {"a_ref": 0.8},
            "drive_fit": {"fit": {"a_ref": 0.15, "channels": [4, 2]}},
        }
    )
    handler = DriveFit(runtime)
    assert handler.config.fit.a_ref == 0.15
    assert handler.config.fit.channels == [4, 2]
    assert handler.subconfig_key == "drive_fit"
    assert DriveFit.prepare_runtime_config_data({"fit": {"a_ref": 0.1}}) == {
        "drive_fit": {"fit": {"a_ref": 0.1}},
    }
    assert (
        DriveFit(
            RuntimeContext().with_config(
                overrides={"kids": {"drive_fit": {"a_ref": 0.8}}},
            )
        ).config.fit.a_ref
        == 0.2
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"channels": []},
        {"channels": [2, 2]},
        {"channels": [-1]},
        {"n_adrvs_min": 1},
        {"weight_window_Qr": 0},
        {"a_ref": np.nan},
        {"unexpected": 1},
    ],
)
def test_invalid_config(settings):
    with pytest.raises(ValidationError):
        DriveFitStepConfig(**settings)


@pytest.mark.parametrize(
    ("a", "flags", "expected"),
    [
        ([0.0, 0.4, 0.8], [0, 0, 0], (-25.0, 0.2, 1500.0, 0)),
        ([0.4, 0.5, 0.6], [0, 0, 0], (-30.0, 0.4, 1000.0, 1)),
        ([0.0, 0.05, 0.1], [0, 0, 0], (-10.0, 0.1, 3000.0, 1)),
        ([0.1, 0.2, 0.3], [1, 0, 1], (-20.0, 0.2, 2000.0, 1)),
        ([0.0, 2.0, 0.4], [0, 0, 0], (-20.0, 0.2, 2000.0, 0)),
        ([0.195, 0.4, 0.8], [0, 0, 0], (-30.0, 0.195, 1000.0, 0)),
        # Exactly at a_ref at the first point retains the v2 endpoint flag.
        ([0.2, 0.4, 0.8], [0, 0, 0], (-30.0, 0.2, 1000.0, 1)),
    ],
)
def test_production_crossing_and_endpoint_rules(a, flags, expected):
    actual = find_best_drive([-30, -20, -10], a, [1000, 2000, 3000], flags)
    np.testing.assert_array_equal(actual, expected)


def test_no_valid_fit_and_misaligned_arrays():
    selected = find_best_drive([-30, -20], [np.nan, 0.5], [1000, 2000], [1, 1])
    assert np.isnan(selected[:3]).all()
    assert selected[3] == 1
    with pytest.raises(ValueError, match="aligned"):
        find_best_drive([-30, -20], [0.1], [1000, 2000], [0, 0])


def test_failed_reference_keeps_every_power_position(monkeypatch):
    def fail(*args, **kwargs):
        msg = "No resonance"
        raise ValueError(msg)

    monkeypatch.setattr(
        "tolteca_kids.drive_fit.fit.ResonanceFitterSingleTone",
        fail,
    )
    values, errors = fit_channel(
        np.ones((3, 4)),
        np.ones((3, 4), dtype=complex),
        tone_frequency=7e8,
        tau=-36.2,
        window_Qr=12000,
    )
    assert values == [None, None, None]
    assert errors == [
        "ValueError: No resonance",
        "The lowest-power reference fit failed",
        "The lowest-power reference fit failed",
    ]


def test_reference_sequence_matches_production_values(fitted):
    # Independently checked against the original recipe and kid_phase_fit.
    # Full 685-channel equality is recorded by design/probes/drive_fit_parity.py.
    np.testing.assert_array_equal(fitted.data.chan, [0, 100, 300, 684])
    np.testing.assert_array_equal(fitted.data.power, np.arange(-25, -9.5, 0.5))
    np.testing.assert_allclose(
        fitted.data.p_best,
        [-14.173487718020583, -10.0, -13.863999649078753, np.nan],
        rtol=1e-10,
        atol=0,
    )
    np.testing.assert_allclose(
        fitted.data.Qr_best,
        [8190.037972798341, 5849.440336275603, 12654.681962118115, np.nan],
        rtol=1e-10,
        atol=0,
    )
    np.testing.assert_array_equal(fitted.data.p_flag, [0, 1, 0, 1])
    assert fitted.n_success == 2
    assert (fitted.data.sel(chan=684).fit_flag.values == 1).all()
    assert (fitted.data.sel(chan=684).fit_error.values != "").all()
    np.testing.assert_array_equal(
        fitted.table["idx_chan"],
        fitted.data.chan,
    )
    assert len(fitted.data.attrs["inputs"]) == 31


def test_export_reopens_without_sources_or_fitter(
    working, fitted, tmp_path, monkeypatch
):
    path = write_drive_fit_product(working, tmp_path)

    def fail(*args, **kwargs):
        pytest.fail("Saved results must not reopen or fit source data")

    monkeypatch.setattr("tolteca_kids.drive_fit.computation._read_sweep", fail)
    monkeypatch.setattr("tolteca_kids.drive_fit.computation.fit_channel", fail)
    with open_drive_fit_product(path) as tree:
        reopened = DriveFitStep.view(tree)
        xr.testing.assert_identical(reopened.data, fitted.data)
        np.testing.assert_array_equal(
            reopened.table["adrv_best"], fitted.table["adrv_best"]
        )
        DriveFitPlotStep(artifacts={"summary": {"channels": [0, 684]}})(tree)
        figure = DriveFitPlotStep.get_context(tree).artifacts[DriveFitViz].data
        assert figure.layout.xaxis.title.text == "Drive attenuation (dB)"
        grid = DriveFitPlotStep.get_context(tree).artifacts[DriveFitGridViz].data
        assert grid.layout.meta["channels"] == [0, 100, 300, 684]
        sweeps = DriveFitPlotStep.get_context(tree).artifacts[DriveFitSweepArtifacts]
        assert list(sweeps.data) == [0, 100, 300, 684]


def test_discovery_and_cli_use_same_settings(sequence, fitted, tmp_path):
    config = {
        "input_discovery": {
            "inputs": [
                {
                    "sources": [
                        {"glob": str(sequence[0].parent / "*017068*targsweep.nc")},
                    ]
                }
            ]
        },
        "drive_fit": {
            "fit": {"channels": [0, 100, 300, 684]},
            "select": "atten_drive >= 10",
        },
    }
    results = list(run(RuntimeContext().with_config(overrides=config)))
    assert len(results) == 1
    xr.testing.assert_equal(
        DriveFitStep.view(results[0]["interfaces/toltec0"]).data, fitted.data
    )
    config_path = tmp_path / "workflow.yaml"
    # JSON is valid YAML and preserves paths without shell quoting.
    config_path.write_text(json.dumps(config))
    invocation = CliRunner().invoke(
        app,
        [
            "--config_paths",
            str(config_path),
            "--drive_fit.output.enabled",
            "true",
            "--drive_fit.output.path",
            str(tmp_path / "cli"),
            "--drive_fit.plot.enabled",
            "true",
        ],
    )
    assert invocation.exit_code == 0, invocation.output
    [exported] = list((tmp_path / "cli").glob("scratch/*/*/*/index.yaml"))
    with open_drive_fit_product(exported) as saved:
        xr.testing.assert_identical(DriveFitStep.view(saved).data, fitted.data)
    assert list(exported.parent.rglob("drive_fit_summary.html"))
    assert list(exported.parent.rglob("drive_fit_grid.html"))
    assert list(exported.parent.rglob("drive_fit_sweeps.html"))
    from tolteca_datamodels.data_prod import DataProdIndex

    index = DataProdIndex.read(exported)
    names = {Path(item.source_uri).name for item in index.data_items}
    assert {"drive_fit_sweeps.html", "plotly.min.js", "channel_100.html"} <= names


def test_cli_root_override_and_unknown_options():
    runner = CliRunner()
    result = runner.invoke(app, ["--show_config", "--drive_fit.fit.a_ref", "0.15"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["fit"]["a_ref"] == 0.15
    assert runner.invoke(app, ["--show_config", "--unknown", "1"]).exit_code != 0


def test_empty_discovery_reports_no_sweeps(tmp_path):
    runtime = RuntimeContext().with_config(
        overrides={
            "input_discovery": {
                "inputs": [{"sources": [{"glob": str(tmp_path / "*.nc")}]}],
            },
        }
    )
    with pytest.raises(ValueError, match="No target/tune sweeps"):
        list(run(runtime))


def test_input_errors(sequence, data_lmt_path):
    fit = DriveFit(
        RuntimeContext().with_config(
            overrides={"drive_fit": {"fit": {"channels": [0]}}},
        )
    )
    with pytest.raises(ValueError, match="No sweeps"):
        fit([])
    with pytest.raises(ValueError, match="Duplicate"):
        fit([sequence[0], sequence[0]])
    with pytest.raises(ValueError, match="Too few"):
        fit(sequence[:4])
    missing = DriveFit(
        RuntimeContext().with_config(
            overrides={"drive_fit": {"fit": {"channels": [685]}}},
        )
    )
    with pytest.raises(ValueError, match="Channels absent"):
        missing([sequence[0]])
    two_blocks = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*tune.nc"))
    with pytest.raises(ValueError, match="one sweep block"):
        fit([two_blocks])


def test_summary_counts_and_channel_pages_preserve_saved_results(fitted):
    before = fitted.data.copy(deep=True)
    summary = DriveFitViz()(fitted)
    # Both histograms include all finite selections: two accepted and one flagged.
    for trace_ids in ((0, 4), (1, 5)):
        assert [int(np.sum(summary.data[i].y)) for i in trace_ids] == [2, 1]
    assert "1 without a selection" in summary.layout.title.text
    assert all(t.mode == "markers" for t in summary.data if t.type == "scatter")
    assert set(summary.data[2].customdata) == {0, 300}
    grid = DriveFitGridViz(channels=[684, 300, 0, 100], page_size=3, columns=3)(fitted)
    assert len(grid.frames) == 2
    assert "Ch 684" in grid.layout.annotations[0].text
    assert "31 failed" in grid.layout.annotations[0].text
    assert "no selection" in grid.layout.annotations[0].text
    assert np.isnan(grid.data[0].y).all()
    assert len(grid.data[1].x) == 0
    # Channel 300 has its own axis; its recorded curve and selected point survive.
    trace = grid.data[3]
    assert trace.xaxis == "x2"
    row = fitted.data.sel(chan=300)
    np.testing.assert_array_equal(trace.y, np.where(row.fit_flag == 0, row.a, np.nan))
    np.testing.assert_array_equal(grid.data[4].x, [row.p_best.item()])
    # Final page replaces every slot, so prior channels cannot remain visible.
    last = grid.frames[-1]
    assert "Ch 100" in last.layout.annotations[0].text
    assert all(len(t.x) == 0 for t in last.data[3:])
    assert all(t.text == "" for t in last.layout.annotations[1:])
    xr.testing.assert_identical(fitted.data, before)


def test_collated_sweeps_show_saved_circles_models_and_failures(fitted, tmp_path):
    before = fitted.data.copy(deep=True)
    spec = DriveFitSweepArtifacts(channels=[100, 684, 0])
    figures = spec(fitted)
    assert list(figures) == [100, 684, 0]
    assert figures[100].layout.meta["status"] == "flagged"
    assert figures[0].layout.meta["status"] == "accepted"
    for channel, figure in figures.items():
        assert not figure.frames
        assert figure.layout.yaxis2.scaleanchor == "x2"
        assert figure.layout.yaxis2.scaleratio == 1
        assert figure.layout.legend.groupclick == "togglegroup"
        buttons = figure.layout.updatemenus[0].buttons
        assert all(buttons[0].args[0]["visible"])
        assert len(buttons) == 32
        for index, power in enumerate(fitted.data.power.values):
            row = fitted.data.sel(chan=channel, power=power)
            traces = {
                t.meta["role"]: t for t in figure.data if t.meta["power"] == power
            }
            visible = buttons[index + 1].args[0]["visible"]
            assert all(
                show == (t.meta["power"] == power)
                for t, show in zip(figure.data, visible, strict=True)
            )
            assert {t.legendgroup for t in traces.values()} == {str(float(power))}
            assert sum(t.showlegend for t in traces.values()) == 1
            measured = traces["I/Q data"]
            assert len(measured.x) == len(measured.y) == row.sizes["sample"]
            np.testing.assert_allclose(
                traces["Magnitude data"].y,
                20 * np.log10(np.hypot(measured.x, measured.y)),
                rtol=1e-14,
            )
            valid = np.isfinite(row.frequency_fit.values) & (row.fit_flag.item() == 0)
            offset = (row.frequency_fit.values[valid] - row.f_chan.item()) / 1e3
            for role, name in (
                ("Phase data", "phase_data"),
                ("Phase model", "phase_model"),
                ("Phase residual", "phase_residual"),
            ):
                np.testing.assert_array_equal(traces[role].x, offset)
                np.testing.assert_array_equal(traces[role].y, row[name].values[valid])
            np.testing.assert_array_equal(
                traces["I/Q model"].x, row.I_model.values[valid]
            )
            np.testing.assert_array_equal(
                traces["I/Q model"].y, row.Q_model.values[valid]
            )
            np.testing.assert_array_equal(traces["Magnitude model"].x, offset)
            np.testing.assert_array_equal(
                traces["Magnitude model"].y,
                20
                * np.log10(
                    np.abs(row.I_model.values[valid] + 1j * row.Q_model.values[valid])
                ),
            )
            if row.fit_flag.item() == 0:
                np.testing.assert_array_equal(
                    traces["Full phase data"].y, row.phase_full
                )
                circle = traces["Fitted circle"]
                np.testing.assert_allclose(
                    np.hypot(
                        np.asarray(circle.x) - row.I_circle_center.item(),
                        np.asarray(circle.y) - row.Q_circle_center.item(),
                    ),
                    row.circle_radius.item(),
                    rtol=1e-12,
                )
            else:
                assert all(
                    len(t.x) == len(t.y) == 0
                    for role, t in traces.items()
                    if role not in ("Magnitude data", "I/Q data")
                )
    failed = figures[684]
    assert failed.layout.meta["status"] == "no selection"
    assert len(failed.layout.meta["errors"]) == 31
    assert "31 failed" in failed.layout.title.text
    receipts = spec.write(figures, path=tmp_path)
    assert {f.path.name for f in receipts} == {
        "drive_fit_sweeps.html",
        "plotly.min.js",
        "channel_0.html",
        "channel_100.html",
        "channel_684.html",
    }
    browser = (tmp_path / "drive_fit_sweeps.html").read_text()
    assert browser.index("Channel 100") < browser.index("Channel 0")
    page = (tmp_path / "drive_fit_sweeps/channel_684.html").read_text()
    assert 'src="plotly.min.js"' in page
    from html import escape

    assert all(escape(error) in page for _, error in failed.layout.meta["errors"])
    assert "../drive_fit_sweeps.html" in page
    xr.testing.assert_identical(fitted.data, before)


def test_sweep_plot_requires_saved_circle_diagnostics(fitted):
    with pytest.raises(ValueError, match="Rerun drive fitting"):
        DriveFitSweepArtifacts()(DriveFitResult(fitted.data.drop_vars("circle_radius")))
