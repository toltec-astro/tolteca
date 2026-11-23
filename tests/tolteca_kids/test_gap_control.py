"""Gap controls must keep the intended population, calibration and weights fixed."""

import copy
import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def control(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[2] / "examples/tune_model_comparison")
    )
    return importlib.import_module("gap_control")


def test_acceptance_controls_preserve_apt_models_and_weighting(control, tmp_path):
    source: dict[str, Any] = {
        "runtime": {"output_dir": "old"},
        "kids": {"solver": {"parallel_policy": "seq"}},
        "inputs": [
            {
                "data_items": [{"filepath": "raw.nc", "stubs": ["model.nc"]}],
                "cal_items": [{"type": "array_prop_table", "filepath": "old.ecsv"}],
            }
        ],
        "timestream": {
            "output": {"stats": {"eigenvalues": False}},
            **{
                stage: {
                    "output": {"enabled": False},
                    "flagging": {
                        "lower_tod_inv_var_factor": 0.1,
                        "upper_tod_inv_var_factor": 10.0,
                    },
                    "weighting": {"type": "approximate"},
                    "despike": {"enabled": False},
                }
                for stage in ("raw_time_chunk", "processed_time_chunk")
            },
        },
    }
    original = copy.deepcopy(source)
    prepared = tmp_path / "prepared"
    for model in ("legacy-common", "python-full"):
        (prepared / model).mkdir(parents=True)
        (prepared / model / "config.yaml").write_text(yaml.safe_dump(source))
    apt = prepared / "common-apt.ecsv"
    apt.write_text("unchanged intended population")
    control.prepare(tmp_path)
    for name, (_, policy) in control.RUNS.items():
        saved = yaml.safe_load(
            (tmp_path / "gap-control" / name / "source.yaml").read_text()
        )
        assert saved["kids"]["solver"]["sample_acceptance"] == policy
        assert saved["inputs"][0]["data_items"] == original["inputs"][0]["data_items"]
        assert saved["inputs"][0]["cal_items"][0]["filepath"] == str(apt)
        assert saved["timestream"]["processed_time_chunk"]["weighting"] == {
            "type": "approximate"
        }
    assert apt.read_text() == "unchanged intended population"
    assert source == original


@pytest.fixture
def report(control):
    return importlib.import_module("gap_control_report")


@pytest.fixture
def corrected_result():
    order = ["v3-physical", "legacy", "v3-numerical"]
    arrays = ["a1100", "a1400", "a2000"]
    return {
        "order": order,
        "maps": [
            {"array": a, "rms_mJy_beam": [4, 6, 5], "common_annulus_pixels": 20}
            for a in arrays
        ],
        "runs": {
            name: {
                "arrays": [
                    {
                        "array": a,
                        "accepted_channel_samples": 90
                        if name == "v3-physical"
                        else 100,
                        "intended_channels": 5,
                        "contributing_channels": 4,
                        "partially_flagged_channels": 1,
                        "rejected_channel_samples": 10,
                    }
                    for a in reversed(arrays)
                ]
            }
            for name in order
        },
    }


def test_policy_metrics_use_run_identity_and_accepted_sample_denominator(
    report, corrected_result
):
    result = report.policy_summary(corrected_result)
    for row in result["arrays"]:
        assert row["rms_mJy_beam"]["v3-numerical"] == 5
        assert row["physical_to_numerical_rms"] == pytest.approx(0.8)
        assert row["physical_retained_fraction"] == pytest.approx(0.9)
        assert row["additional_rejected_channel_samples"] == 10
    assert result["mapping_speed"]["status"] == "not_validated"


def test_report_keeps_old_maps_out_of_current_findings_and_is_repeatable(
    report, corrected_result
):
    html = (
        '<html><h1>Test</h1><nav>old</nav><h2 id="overview">Old overview</h2>OLD_COUNTS'
    )
    for name in (
        "map-quality",
        "coordinate-control",
        "pca-trace",
        "model-quality",
        "timestream-examples",
        "a1100",
        "a1400",
        "a2000",
    ):
        html += (
            f'<section id="{name}"><h2>{name}</h2><h3>detail</h3>OLD_{name}'
            "<figure><figcaption>Figure 3. Original figure</figcaption>"
            "</figure></section>"
        )
    html += '<h2 id="provenance">Old sources</h2>OLD_SOURCES</html>'
    updated = report.reorganize_html(html, corrected_result)
    main, history = updated.split('<section id="history">')
    assert "OLD_a1100" not in main
    assert "OLD_COUNTS" not in main
    assert "OLD_pca-trace" in history
    assert "OLD_a1100" in history
    assert "OLD_model-quality" in main
    assert "OLD_map-quality" not in updated
    assert "Figure 3. Corrected maps" in main
    assert report.reorganize_html(updated, corrected_result) == updated


def test_general_plot_builder_prefers_corrected_evidence(report, tmp_path, monkeypatch):
    plots = importlib.import_module("array_example_plots")
    (tmp_path / "gap-control.json").write_text("{}")
    sentinel = {"source": "corrected-audit"}
    monkeypatch.setattr(plots, "policy_figures", lambda folder: sentinel)
    assert plots.map_quality(tmp_path) is sentinel
