"""Joint maps require recorded component identities, not frequency guesses."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from tolteca_kidsproc.model_flags import ModelFitFlag as F

SOURCE = (
    Path(__file__).parents[2]
    / "examples/tune_model_comparison/prepare_joint_map_control.py"
)
SPEC = importlib.util.spec_from_file_location("joint_map_control", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def evidence():
    """An exact plan and two accepted finding/channel pairs."""
    detected = Table(
        {
            "idx": [10, 11],
            "group": [5, 5],
            "f": [500e6, 501e6],
            "Qr": [20000.0, 25000.0],
            "idx_chan": [0, 1],
        }
    )
    plan = Table(
        {
            "group_id": ["g", "g"],
            "resonator_id": ["10", "11"],
            "finding_group_id": ["5", "5"],
            "fr_initial": detected["f"],
            "r_initial": 1 / (2 * detected["Qr"]),
        }
    )
    pairs = Table(
        {
            "idx_det": [10, 11],
            "idx_chan": [0, 1],
            "n_matched_query": [1, 1],
            "n_matched_ref": [1, 1],
            "f_det": detected["f"],
            "f_chan": [499.9e6, 500.9e6],
        }
    )
    return detected, plan, pairs


def test_saved_correspondence_is_required():
    detected, plan, pairs = evidence()
    links = MODULE.component_correspondences(detected, plan, pairs)
    assert links[("g", "10")]["channel"] == 0
    pairs["n_matched_ref"][0] = 2
    links = MODULE.component_correspondences(detected, plan, pairs)
    assert links[("g", "10")]["status"] == "ambiguous_saved_match"
    assert "channel" not in links[("g", "10")]


def test_changed_plan_cannot_reuse_finding_identity():
    detected, plan, pairs = evidence()
    plan["fr_initial"][0] += 1
    with pytest.raises(ValueError, match="not bound"):
        MODULE.component_correspondences(detected, plan, pairs)


def test_mixed_binding_keeps_scalar_rows_and_original_beam_quality(monkeypatch):
    detected, plan, pairs = evidence()
    apt = Table({"nw": [0, 0], "kids_tone": [0, 1], "uid": [100, 101], "flag": [0, 0]})
    common = apt.copy()
    common["flag"] = [0, 1]
    report = Table(
        {"chan": [0, 1], "flag": [0, int(F.fit_joint_required)]},
        meta={
            "joint_models": {
                "groups": [
                    {
                        "group_id": "g",
                        "channels": [0, 1],
                        "resonator_ids": ["10", "11"],
                        "fit_flag": 0,
                    }
                ]
            }
        },
    )
    monkeypatch.setattr(MODULE, "native_group", lambda group: group)
    payload, rows = MODULE.bind_network(apt, common, report, detected, plan, pairs)
    assert payload["channel_rows"] == [{"apt_row": 0, "uid": 100, "channel": 0}]
    assert payload["rows"] == [
        {"uid": 101, "group_id": "g", "resonator_id": "11", "channel": 1}
    ]
    assert {r["status"] for r in rows} == {"joint_added", "retained_scalar_control"}
    np.testing.assert_array_equal(apt["flag"], [0, 0])
    np.testing.assert_array_equal(common["flag"], [0, 1])
    apt["flag"][1] = 1
    payload, rows = MODULE.bind_network(apt, common, report, detected, plan, pairs)
    assert not payload["rows"]
    assert rows[1]["status"] == "original_apt_excluded"


def test_insufficient_probes_are_not_accepted(monkeypatch):
    detected, plan, pairs = evidence()
    apt = Table({"nw": [0, 0], "kids_tone": [0, 1], "uid": [100, 101], "flag": [0, 0]})
    common = apt.copy()
    common["flag"] = [1, 1]
    report = Table(
        {"chan": [0, 1], "flag": [int(F.fit_joint_required)] * 2},
        meta={
            "joint_models": {
                "groups": [
                    {
                        "group_id": "g",
                        "channels": [0],
                        "resonator_ids": ["10", "11"],
                        "fit_flag": 0,
                    }
                ]
            }
        },
    )
    monkeypatch.setattr(MODULE, "native_group", lambda group: group)
    payload, rows = MODULE.bind_network(apt, common, report, detected, plan, pairs)
    assert not payload["rows"]
    assert {r["status"] for r in rows} == {"insufficient_simultaneous_probes"}


def test_report_requires_one_successful_realization(tmp_path, monkeypatch):
    """Failed attempts remain recorded; two successes require explicit review."""
    import json
    import sys

    monkeypatch.syspath_prepend(str(SOURCE.parent))
    if "joint_map_report" in sys.modules:
        report = sys.modules["joint_map_report"]
    else:
        import joint_map_report as report
    for name, code in [("redu00", -6), ("redu01", 0)]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "execution.json").write_text(json.dumps({"exit_code": code}))
    assert report.completed_run(tmp_path).name == "redu01"
    folder = tmp_path / "redu02"
    folder.mkdir()
    (folder / "execution.json").write_text('{"exit_code": 0}')
    with pytest.raises(ValueError, match="found 2"):
        report.completed_run(tmp_path)


def test_broader_report_is_idempotent_and_does_not_claim_scalar_counts_are_current(
    monkeypatch,
):
    monkeypatch.syspath_prepend(str(SOURCE.parent))
    import joint_map_report as report

    old = (
        '<html><nav>Current policy comparison</nav><section id="old">'
        "Current result: gap correction and explicit acceptance</section></html>"
    )
    broader = (
        '<section id="joint-maps"><h2>Broader maps</h2><figure><figcaption>'
        "Current populations.</figcaption></figure></section>"
    )
    first = report.integrate_html(old, broader)
    assert report.integrate_html(first, broader) == first
    assert first.count('id="joint-maps"') == 1
    assert first.count('href="#joint-maps"') == 1
    assert "Current result: gap correction" not in first
