"""Population audits distinguish beammap, fitted groups and channel contributions."""

import importlib
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.table import Table

from tolteca_kidsproc.model_flags import ModelFitFlag


@pytest.fixture
def population_module(monkeypatch):
    folder = Path(__file__).parents[2] / "examples/tune_model_comparison"
    monkeypatch.syspath_prepend(str(folder))
    return importlib.import_module("array_populations")


@pytest.fixture
def models():
    report = Table(
        {
            "chan": [0, 1, 2, 3],
            "fr": [100.0, np.nan, 200, np.nan],
            "a_nl": [0.1, np.nan, 1, np.nan],
            "flag": [0, int(ModelFitFlag.fit_joint_required), 0, 8192],
        }
    )
    report.meta["joint_models"] = {
        "groups": [
            {
                "group_id": "pair",
                "channels": [0, 1],
                "resonator_ids": ["first", "second"],
                "fr": [100.0, 101],
                "a_nl": [0.1, 0.9],
                "coupling_re": [[1, 0], [1, 1]],
                "coupling_im": [[0, 0], [0, 0]],
            },
            {
                "group_id": "single",
                "channels": [2, 4],
                "resonator_ids": ["third"],
                "fr": [200.0],
                "a_nl": [1.0],
                "coupling_re": [[1], [1]],
                "coupling_im": [[0], [0]],
            },
        ]
    }
    quality = {
        "pair": {"residual_fraction": 0.01, "reduced_chi2": 5.0},
        "single": {"residual_fraction": 0.1, "reduced_chi2": 50.0},
    }
    return report, quality


def test_group_multiplicity_is_not_probe_blending(population_module, models):
    rows = population_module.describe_channels(*models)
    assert rows[0]["group_kind"] == rows[1]["group_kind"] == "multiplet"
    assert rows[0]["channel_kind"] == "single_component"
    assert rows[0]["resonator_id"] == "first"
    assert rows[0]["a_nl_bin"] == "[0.1,0.5)"
    assert rows[1]["channel_kind"] == rows[1]["a_nl_bin"] == "blended"
    assert rows[2]["group_kind"] == "singlet_multiple_probes"
    assert rows[2]["a_nl_bin"] == "[1,infinity)"
    assert rows[3]["channel_kind"] == "unresolved"


def test_report_cannot_substitute_first_multiplet_component(population_module, models):
    report, quality = models
    report["a_nl"][0] = 0.9
    with pytest.raises(AssertionError):
        population_module.describe_channels(report, quality)


def test_counts_preserve_warning_fits_and_count_group_metrics_once(
    population_module, models
):
    descriptions = population_module.describe_channels(*models)
    rows = Table(
        rows=[
            dict(
                **descriptions[i],
                nw=0,
                apt_flag=0,
                model_flag=int(flag),
            )
            for i, flag in (
                (0, 0),
                (1, ModelFitFlag.fit_joint_required),
                (2, ModelFitFlag.fit_at_bound),
                (3, ModelFitFlag.fit_unusable | ModelFitFlag.fit_not_converged),
            )
        ]
    )
    summary = population_module.summarize_population(rows)
    assert summary["channels"] == summary["apt_good"] == 4
    assert summary["quality_state"] == {
        "no_flags": 1,
        "warnings_only": 1,
        "joint_required": 1,
        "unusable": 1,
    }
    assert summary["unique_selected_groups"] == 2
    assert summary["group_metrics"]["group_reduced_chi2"]["median_p90"] == [27.5, 45.5]
    assert summary["flag_bits"]["fit_at_bound"] == 1


def test_common_apt_only_changes_flag_and_preserves_beam_properties(
    population_module, tmp_path
):
    apt = Table(
        {
            "nw": [0, 0],
            "kids_tone": [7, 3],
            "uid": [70, 30],
            "flag": [0, 0],
            "x_t": [1.0, 2.0],
        }
    )
    apt["x_t"].unit = "arcsec"
    original = tmp_path / "beammap.ecsv"
    apt.write(original)
    common = apt.copy()
    common["flag"][1] = 1
    restricted = tmp_path / "common.ecsv"
    common.write(restricted)
    folders = {}
    for name in ("legacy", "common", "python"):
        folder = tmp_path / name / "redu00/152434/raw"
        folder.mkdir(parents=True)
        path = restricted if name == "common" else original
        config = {
            "inputs": [
                {"cal_items": [{"type": "array_prop_table", "filepath": str(path)}]}
            ]
        }
        (folder.parents[1] / "config.yaml").write_text(yaml.safe_dump(config))
        folders[name] = folder
    # Identity join must not rely on row position.
    ledger = Table(
        {
            "nw": [0, 0],
            "channel": [3, 7],
            "apt_uid": [30, 70],
            "apt_flag": [0, 0],
            "eligible": [False, True],
        }
    )
    summary = population_module.apt_audit(folders, ledger)
    assert summary["common_changed_columns"] == ["flag"]
    assert summary["sources"]["common"]["flag_zero"] == 1
    common["x_t"][0] = 5
    common.write(restricted, overwrite=True)
    with pytest.raises(ValueError, match="changed more than its eligibility"):
        population_module.apt_audit(folders, ledger)


def test_exact_channel_and_chunk_overlap_is_retained(population_module, models):
    ledger = Table(
        {
            "array": [0.0] * 4,
            "nw": [0] * 4,
            "channel": [0, 1, 2, 3],
            "apt_flag": [0] * 4,
            "model_flag": [0] * 4,
            "status": ["eligible"] * 4,
        }
    )
    details = {
        (0, k): v for k, v in population_module.describe_channels(*models).items()
    }
    runs = {
        "legacy": [
            {
                "nw": 0,
                "contributing_channel_ids": [0, 1, 2],
                "contributing_chunk_ids": [[0, 1], [1], [0]],
            }
        ],
        "common": [
            {
                "nw": 0,
                "contributing_channel_ids": [0, 1],
                "contributing_chunk_ids": [[0], [1]],
            }
        ],
        "python": [
            {
                "nw": 0,
                "contributing_channel_ids": [0, 2],
                "contributing_chunk_ids": [[1], [0]],
            }
        ],
    }
    rows, summary = population_module.population_audit(ledger, details, runs)
    assert summary["0"]["common_python_overlap"] == {
        "both": 1,
        "common_only": 1,
        "python_only": 1,
    }
    assert rows["python_chunks"].tolist() == [1, 0, 1, 0]
    assert rows["python_chunk_ids"].tolist() == ["[1]", "[]", "[0]", "[]"]
