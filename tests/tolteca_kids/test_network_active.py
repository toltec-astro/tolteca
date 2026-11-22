"""Physical populations require evidence independent of the measured noise."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def recipe():
    """Load the maintained local comparison without executing its driver."""
    return runpy.run_path(
        str(
            Path(__file__).parents[2]
            / "examples/tune_model_comparison/network_active.py"
        )
    )


def edges():
    """Two acquired blocks, each linked to five independent Skydip acquisitions."""
    return pd.DataFrame(
        [
            {
                "block": block,
                "finding_id": str(70 + block),
                "scan": scan,
                "uids": "12",
                "unique": True,
                "residual_hz": 15000.0,
            }
            for block in (0, 1)
            for scan in range(2, 7)
        ]
    )


def test_consensus_does_not_vote_away_an_ambiguous_edge(recipe):
    """Five unique matches cannot overrule a sixth competing identity."""
    data = edges()
    result = recipe["consensus"](data)
    assert (result.association == "supported").all()
    ambiguous = data.iloc[[-1]].assign(scan=7, uids="13", unique=False)
    result = recipe["consensus"](pd.concat([data, ambiguous], ignore_index=True))
    assert not (result.association == "supported").any()
    assert result.loc[result.block == 1, "alternatives"].item() == "12;13"


def test_consensus_requires_distinct_acquisitions_and_unique_target(recipe):
    """Repeated rows are not independent evidence; a detector cannot be cloned."""
    repeated = pd.concat([edges().assign(scan=2)] * 5, ignore_index=True)
    assert not (recipe["consensus"](repeated).association == "supported").any()
    duplicate = edges().query("block == 1").assign(finding_id="99")
    result = recipe["consensus"](pd.concat([edges(), duplicate], ignore_index=True))
    assert not (result.association == "supported").any()
    assert (result.association == "conflicting_target_findings").sum() == 2


def test_exact_receipt_includes_product_realization(recipe):
    """An equal row number in a reprocessed finding table is a different key."""
    binding = {
        "product": {"uid": "proc/run1", "namespace_uid": "namespace"},
        "uid_raw_obs": "tcs-142440-0-2",
        "interface": "toltec0",
        "block": 0,
    }
    other = binding | {"product": binding["product"] | {"uid": "proc/run2"}}
    assert recipe["finding_key"](binding, 12) != recipe["finding_key"](other, 12)


def test_readout_suitability_does_not_reject_nonlinearity_or_select_quiet_noise(recipe):
    """Fit quality and complete solving govern selection, not a or output PSD."""
    row = {
        "group_flag": 0,
        "residual_to_sweep_variation": 0.01,
        "psd_status": "measured",
        "bad_readout_channel": False,
        "parity_status": "pass",
        "native_route": "joint",
        "native_sample_flag_values": "0",
        "a_nl": 4.0,
        "frequency_asd": 1000.0,
    }
    assert recipe["suitability"](row) == []
    row.update(group_flag=2, frequency_asd=0.0001)
    assert recipe["suitability"](row) == ["at_bound"]
    row.update(
        group_flag=0,
        residual_to_sweep_variation=np.nan,
        psd_status="no_complete_valid_estimate",
    )
    assert recipe["suitability"](row) == ["sweep_residual", "incomplete_valid_samples"]


def test_model_plan_cannot_borrow_another_findings_identity(recipe):
    """A supplied same-integer plan with a different seed is not exact evidence."""
    detected = {
        "idx": np.array([3]),
        "f": np.array([5e8]),
        "Qr": np.array([10000.0]),
        "group": np.array([2]),
    }
    plan = {
        "resonator_id": np.array(["3"]),
        "fr_initial": np.array([5e8]),
        "r_initial": np.array([0.00005]),
        "finding_group_id": np.array(["2"]),
    }

    class SavedTable(dict):
        """Minimal table source; array ownership is exercised by the real replay."""

        def array_keys(self):
            """Return saved columns."""
            return self.keys()

    block = {
        "kids_find/detected": SavedTable(detected),
        "model_fit/plan": SavedTable(plan),
    }
    assert len(recipe["checked_finding_plan"](block)) == 1
    plan["fr_initial"] = np.array([5e8 + 1])
    block["model_fit/plan"] = SavedTable(plan)
    with pytest.raises(ValueError, match="not bound"):
        recipe["checked_finding_plan"](block)


def test_skydip_direction_uses_the_timestream_probe_coordinate(recipe):
    """Changing detuning's reference frequency must not change physical loss."""
    beta, error = recipe["response_at_probe"](0.03, 0.002, 500e6, 600e6)
    fr = np.array([501e6, 501e6 + 100])
    dx_reference = np.diff(500e6 / fr - 1).item()
    dx_probe = np.diff(600e6 / fr - 1).item()
    np.testing.assert_allclose(beta * dx_probe, 0.03 * dx_reference, rtol=1e-8)
    assert error == pytest.approx(0.002 * 5 / 6)


def test_conflicting_response_evidence_is_neither_active_nor_confirmed_dark(recipe):
    """A frequency association cannot erase a contradictory response indicator."""
    assert (
        recipe["response_population"](
            dark_evidence=True,
            motion="loading_responsive",
            has_loss_law=True,
            readout_suitable=True,
        )
        == "response_conflict"
    )
    assert (
        recipe["response_population"](
            dark_evidence=False,
            motion="unresolved",
            has_loss_law=False,
            readout_suitable=True,
        )
        == "unresolved"
    )


def test_diagnostics_keep_policy_limits_separate_from_numerical_failures(recipe):
    """An anchor restriction and a tiny parity miss are not optimizer failures."""
    row = {
        "group_flag": 0,
        "residual_to_sweep_variation": 0.02,
        "native_route": "joint",
        "native_sample_flag_values": "256",
        "psd_status": "no_complete_valid_estimate",
        "bad_readout_channel": False,
        "parity_status": "pass",
    }
    assess = recipe["DIAGNOSTICS"]["assess"]
    result = assess(row)
    assert result["fit_adequacy"] == "within_limit"
    assert result["numerical_validity"] == "branch_or_domain_limited"
    assert result["numerical_flags"] == ""
    assert result["selection_policy_limits"] == "domain_or_continuation_limit"
    result = assess(
        row
        | {
            "native_sample_flag_values": "0",
            "psd_status": "measured",
            "parity_status": "tolerance_exceeded",
        }
    )
    assert result["numerical_validity"] == "complete"
    assert result["selection_policy_limits"] == "python_native_tolerance"
    assert result["comparison_exclusions"]
    result = assess(row | {"group_flag": 1, "native_sample_flag_values": "8"})
    assert result["numerical_validity"] == "flagged"
    assert "fit_not_converged" in result["numerical_flags"]
    result = assess(
        row
        | {
            "native_route": "unselected",
            "native_sample_flag_values": "",
            "parity_status": "not_assessed",
        }
    )
    assert result["numerical_validity"] == "not_assessed"
    assert result["selection_policy_limits"] == ""


def test_diagnostic_separation_preserves_the_existing_comparison_cut(recipe):
    """Across fit bits, route states and parity, no relabelling relaxes selection."""
    assess = recipe["DIAGNOSTICS"]["assess"]
    for flag in (0, 1, 2, 4, 8, 16, 32, 64, 2048):
        for residual in (0.049, 0.05, 0.051, np.nan):
            for complete in (True, False):
                for parity in (True, False):
                    row = {
                        "group_flag": flag,
                        "residual_to_sweep_variation": residual,
                        "psd_status": "measured"
                        if complete
                        else "no_complete_valid_estimate",
                        "native_route": "joint",
                        "native_sample_flag_values": "0" if complete else "256",
                        "parity_status": "pass" if parity else "tolerance_exceeded",
                        "bad_readout_channel": False,
                    }
                    expected = (
                        not (flag & (1 | 2 | 4 | 32 | 64 | 2048))
                        and np.isfinite(residual)
                        and residual <= 0.05
                        and complete
                        and parity
                    )
                    assert (not assess(row)["comparison_exclusions"]) == expected
