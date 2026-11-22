"""Post-solving reports keep evidence gaps separate from measured performance."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.tolteca_kidsproc.test_joint_continuation import case

HERE = Path(__file__).parents[2] / "examples/tune_model_comparison"


@pytest.fixture
def recipe():
    return runpy.run_path(str(HERE / "network_postsolve.py"))


@pytest.mark.parametrize(
    "population", ["active_suitable", "unresolved", "dark_candidate"]
)
def test_coordinate_diagnostics_do_not_promote_missing_response_evidence(
    recipe, tmp_path, population
):
    solver, _, fr, r = case(b=np.zeros(2))
    time = np.arange(4096) / 128
    states_fr = fr[0] + np.column_stack([5 * np.sin(2 * np.pi * 8 * time)] * 2)
    states_r = r[0] + np.column_stack([1e-8 * np.cos(2 * np.pi * 8 * time)] * 2)
    evaluation = solver.evaluate(states_fr, states_r), solver.evaluate(fr[:1], r[:1])
    for sub in ("arrays", "cases"):
        (tmp_path / sub).mkdir()
    row = {
        "family": "test",
        "group_id": "g",
        "population": population,
        "loss_detuning_slope_at_probe": 0.03,
    }
    result = recipe["component_diagnostics"](evaluation, "r7", 19, 128, row, tmp_path)
    assert result["coordinate_asd_ratio"] == pytest.approx(1)
    assert ("signal_referred_asd_ratio" in result) == (population == "active_suitable")
    if population == "active_suitable":
        assert result["signal_referred_asd_ratio"] == pytest.approx(1)
    with np.load(tmp_path / "arrays/test-g-r7.npz") as data:
        assert data["x_opt"].shape == (4096,)
        assert data["cross_real"].shape[1:] == (4, 4)


def test_performance_accounting_keeps_all_components_and_no_synthetic_sensitivity():
    report = runpy.run_path(str(HERE / "network_postsolve_report.py"))
    table = pd.DataFrame(
        {
            "family": ["Affine"] * 3,
            "kind": ["Multiplet"] * 3,
            "population": ["active_suitable", "unresolved", "unresolved"],
            "diagnostic_status": ["complete", "complete", "no_complete_valid_estimate"],
            "readout_suitable": [True, True, False],
            "a_fit": [0.1, 0.1, 0.1],
            "coordinate_asd_ratio": [1.4, 1.2, np.nan],
            "signal_referred_asd_ratio": [1.05, np.nan, np.nan],
        }
    )
    summary = report["summaries"](table)
    assert summary.All.sum() == 3
    assert summary.Complete.sum() == 2
    unknown = summary[summary["Evidence cohort"] == "unresolved"].iloc[0]
    assert unknown["Comparison pass"] == 1
    assert unknown["Response comparison"] == 0
    assert np.isnan(unknown["Median signal-referred ASD ratio"])
    matrix = report["matrix"](table)
    assert matrix.All.sum() == 3
    assert matrix["Response comparison"].sum() == 1


def test_catalog_retains_fixed_drive_coefficient_precision(tmp_path):
    import json

    report = runpy.run_path(str(HERE / "network_postsolve_report.py"))
    value = 1.238941923e-15
    path = tmp_path / "catalog.json"
    report["write_catalog"](pd.DataFrame({"b": [value, np.nan]}), path)
    assert json.loads(path.read_text()) == [{"b": value}, {"b": None}]
