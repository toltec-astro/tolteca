"""Group topology is evidence, not a nearest-match identity allocator."""

import itertools
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tolteca_kidsproc.response import fit_loss_response
from tolteca_kidsproc.segmentation import ordered_match_support


def test_ordered_support_matches_exhaustive_assignments():
    """Every feasible alternative and every forced edge survives enumeration."""
    for shape in ((2, 2), (3, 2), (2, 3), (4, 2)):
        n, m = shape
        for values in itertools.product((False, True), repeat=n * m):
            allowed = np.array(values).reshape(shape)
            assignments = []
            for qi in itertools.combinations(range(n), min(n, m)):
                for ri in itertools.combinations(range(m), min(n, m)):
                    if all(allowed[i, j] for i, j in zip(qi, ri, strict=True)):
                        a = np.zeros(shape, bool)
                        a[qi, ri] = True
                        assignments.append(a)
            feasible, forced = ordered_match_support(allowed)
            np.testing.assert_array_equal(
                feasible,
                np.any(assignments, axis=0) if assignments else np.zeros(shape, bool),
            )
            np.testing.assert_array_equal(
                forced,
                np.all(assignments, axis=0) if assignments else np.zeros(shape, bool),
            )


def test_failed_coverage_and_empty_groups_do_not_create_edges():
    """An incomplete group is not solved by selecting its nearest surviving pair."""
    feasible, forced = ordered_match_support([[1, 0], [0, 0]])
    assert not feasible.any() and not forced.any()
    assert ordered_match_support(np.empty((0, 4), bool))[0].shape == (0, 4)
    with pytest.raises(ValueError, match="two-dimensional"):
        ordered_match_support([True])


@pytest.fixture
def recipe():
    return runpy.run_path(
        str(
            Path(__file__).parents[2]
            / "examples/tune_model_comparison/network_groups.py"
        )
    )


def test_groups_reconcile_split_fits_without_discarding_unknowns(recipe):
    """All ordered assignments include a companion even without an APT receipt."""
    q = pd.DataFrame(
        {"key": ["a", "b"], "frequency": [10.0, 20.0], "group": ["blend", "blend"]}
    )
    r = pd.DataFrame(
        {"key": [1, 2], "frequency": [11.0, 19.0], "group": ["source1", "source2"]}
    )
    result, groups = recipe["grouped_edges"](q, r, iter([("a", 1), ("a", 2), ("b", 2)]))
    assert set(map(tuple, result[result.forced][["finding_id", "ref_row"]].values)) == {
        ("a", 1),
        ("b", 2),
    }
    assert len(groups) == 1
    # Missing target feature: no forced resolution of two possible source IDs.
    result, _ = recipe["grouped_edges"](q.iloc[:1], r, [("a", 1), ("a", 2)])
    assert not result.forced.any()
    assert result.feasible.all()


def test_equal_frequencies_or_unexplained_group_members_remain_unresolved(recipe):
    q = pd.DataFrame(
        {"key": ["a", "b"], "frequency": [10.0, 10.0], "group": ["g", "g"]}
    )
    r = pd.DataFrame({"key": [1, 2], "frequency": [10.0, 20.0], "group": ["g", "g"]})
    result, groups = recipe["grouped_edges"](q, r, [("a", 1), ("b", 2)])
    assert not result.forced.any() and not groups[0]["ordered"]
    q["frequency"] = [10.0, 20.0]
    result, groups = recipe["grouped_edges"](q, r, [("a", 1)])
    assert not groups[0]["complete"] and not result.feasible.any()


def test_loss_response_preserves_covariance_and_physical_units():
    x = np.linspace(-3e-4, 3e-4, 7)
    r = 7e-5 + 0.03 * x
    cov = np.broadcast_to(np.array([[4e-12, 1e-13], [1e-13, 9e-14]]), (7, 2, 2)).copy()
    fit = fit_loss_response(x, r, cov)
    assert fit["slope"] == pytest.approx(0.03)
    assert fit["intercept"] == pytest.approx(7e-5)
    assert fit["chi2"] < 1e-18
    scaled = fit_loss_response(x * 5, r, cov * np.array([[25, 5], [5, 1]]))
    assert scaled["slope"] == pytest.approx(0.03 / 5)
    assert scaled["slope_error"] == pytest.approx(fit["slope_error"] / 5, rel=1e-5)
    # Correlated parameter errors must influence the reported uncertainty.
    diagonal = cov.copy()
    diagonal[:, 0, 1] = diagonal[:, 1, 0] = 0
    assert fit_loss_response(x, r, diagonal)["slope_error"] != pytest.approx(
        fit["slope_error"], rel=1e-3
    )
    with pytest.raises(ValueError, match="positive covariance"):
        fit_loss_response(x, r, np.zeros_like(cov))


def test_reconciliation_keeps_unknown_receipts_as_vetoes(recipe):
    """A geometrically feasible unknown cannot be removed to manufacture a UID."""
    rows = pd.DataFrame(
        [
            {
                "block": b,
                "finding_id": f"b{b}",
                "scan": s,
                "retained": True,
                "support": True,
                "uids": "12",
            }
            for b in (0, 1)
            for s in range(2, 7)
        ]
    )
    assert (recipe["reconcile_consensus"](rows).association == "supported").all()
    unknown = rows.iloc[[-1]].assign(scan=7, uids="", support=False)
    result = recipe["reconcile_consensus"](pd.concat([rows, unknown]))
    assert not (result.association == "supported").any()
    assert (
        result.query("block == 1").association.item() == "unknown_reference_candidate"
    )
    # An edge ruled out by every complete assignment is no longer a veto.
    unknown["retained"] = False
    result = recipe["reconcile_consensus"](pd.concat([rows, unknown]))
    assert (result.association == "supported").all()


def test_group_consensus_does_not_count_duplicate_scans_or_clone_uid(recipe):
    rows = pd.DataFrame(
        [
            {
                "block": b,
                "finding_id": f"b{b}",
                "scan": s,
                "retained": True,
                "support": True,
                "uids": "12",
            }
            for b in (0, 1)
            for s in range(2, 7)
        ]
    )
    repeated = pd.concat([rows.assign(scan=2)] * 5)
    assert not (
        recipe["reconcile_consensus"](repeated).association == "supported"
    ).any()
    other = rows.query("block == 1").assign(finding_id="extra")
    result = recipe["reconcile_consensus"](pd.concat([rows, other], ignore_index=True))
    assert (result.association == "conflicting_target_findings").sum() == 2
    assert not (result.association == "supported").any()
