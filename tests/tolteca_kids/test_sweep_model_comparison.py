"""Paired sweep evidence must not conflate missing fits and map restrictions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from astropy.table import Table

from tolteca_kidsproc.model_flags import ModelFitFlag


@pytest.fixture
def audit():
    path = (
        Path(__file__).parents[2]
        / "examples/tune_model_comparison/sweep_model_comparison.py"
    )
    spec = importlib.util.spec_from_file_location("sweep_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_residual_uses_measured_excursion_not_large_iq_baseline(audit):
    measured = np.array([99 + 20j, 101 + 20j])
    result = audit.residual_metrics(measured, measured + 2j)
    assert result == {"rms": 2, "fraction": 2}


def test_pair_keeps_unfitted_valid_samples_and_exposes_bad_predictions(audit):
    samples = xr.Dataset(
        {
            "I": ("sample", [1.0, 3, 999, np.nan]),
            "Q": ("sample", [0.0, 0, 0, 0]),
            "I_model": ("sample", [1.0, 5, np.nan, np.nan]),
            "Q_model": ("sample", [0.0, 0, 0, 0]),
            "data_valid": ("sample", [True, True, False, True]),
            "fit_selected": ("sample", [True, False, False, False]),
        }
    )
    result = audit.compare_response(samples, np.array([1, 4, np.nan, np.nan]))
    assert result["sample_count"] == 2
    assert result["fit_sample_count"] == 1
    assert result["legacy_fraction"] == pytest.approx(np.sqrt(0.5))
    assert result["v3_fraction"] == pytest.approx(np.sqrt(2))
    samples["I_model"][1] = np.nan
    result = audit.compare_response(samples, np.array([1, 4, 0, 0]))
    assert result["sample_count"] == 2
    assert result["v3_invalid_predictions"] == 1
    assert np.isnan(result["v3_fraction"])


@pytest.mark.parametrize(
    ("extra_flag", "fit", "probe", "recovery", "expected"),
    [
        (
            ModelFitFlag.fit_no_model,
            "not_selected",
            "bound",
            "invalid",
            "no_model_at_probe",
        ),
        (0, "not_selected", "bound", "invalid", "probe_binding_unresolved"),
        (0, "optimizer_failed", "bound", "invalid", "optimizer_not_converged"),
        (0, "optimizer_converged", "bound", "unresolved", "recovery_unresolved"),
    ],
)
def test_unusable_flag_has_distinct_underlying_causes(  # noqa: PLR0913, PLR0917 -- parametrized states
    audit, extra_flag, fit, probe, recovery, expected
):
    flag = int(ModelFitFlag.fit_unusable) | int(extra_flag)
    assert audit.exclusion_reason(flag, fit, probe, recovery) == expected


def test_joint_requirement_is_not_optimizer_failure(audit):
    assert (
        audit.exclusion_reason(
            int(ModelFitFlag.fit_joint_required),
            "optimizer_converged",
            "bound",
            "joint",
        )
        == "joint_output_not_mapped"
    )


def test_missing_response_stays_in_apt_good_denominator(audit):
    channels = Table(
        {
            "array": [0, 0, 0],
            "apt_flag": [0, 0, 1],
            "candidate_responses": [1, 0, 1],
            "converged_responses": [1, 0, 1],
            "selected_response_exists": [True, False, True],
            "fit_status": [
                "optimizer_converged",
                "not_selected",
                "optimizer_converged",
            ],
            "eligible": [False, False, True],
            "exclusion_reason": [
                "joint_output_not_mapped",
                "no_model_at_probe",
                "map_eligible",
            ],
        }
    )
    responses = Table(
        {
            "array": [0, 0, 0],
            "apt_flag": [0, 0, 1],
            "selected_for_probe": [True, False, True],
            "legacy_fraction": [1.0, 1, 1],
            "v3_fraction": [0.5, 0.01, 0.1],
            "status": ["joint_apt_binding_missing"] * 3,
        }
    )
    result = audit.summarize(channels, responses)["0"]
    assert result["apt_good"] == 2
    assert result["has_fitted_response"] == 1
    assert result["map_eligible"] == 0
    assert result["selected_responses"]["responses"] == 1
    assert result["selected_responses"]["median_paired_ratio"] == 0.5
    assert result["all_responses"]["responses"] == 2
