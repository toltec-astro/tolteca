"""Full-network accounting must not confuse fit warnings and identity gaps."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from astropy.table import Table

from tolteca_kidsproc.model_flags import ModelFitFlag as F

SOURCE = (
    Path(__file__).parents[2]
    / "examples/tune_model_comparison/prepare_array_map_control.py"
)
SPEC = importlib.util.spec_from_file_location("array_control", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_warning_only_fits_remain_eligible_and_all_channels_are_accounted_for():
    apt = Table(
        {
            "array": [0] * 6,
            "nw": [0] * 6,
            "kids_tone": np.arange(6),
            "uid": np.arange(100, 106),
            "flag": [0, 0, 0, 0, 1, 0],
        }
    )
    report = Table(
        {
            "chan": np.arange(6),
            "flag": [
                0,
                int(F.fit_residual_high),
                int(F.fit_joint_required),
                int(F.fit_unusable | F.fit_joint_required),
                int(F.fit_unusable),
                int(F.fit_unusable),
            ],
        }
    )
    ledger, counts = MODULE.channel_ledger(apt, report)
    assert ledger["status"].tolist() == [
        "eligible",
        "eligible",
        "joint_apt_binding_missing",
        "fit_unusable",
        "original_apt_excluded",
        "fit_unusable",
    ]
    assert counts["eligible"] == 2
    assert counts["eligible_with_warnings"] == 1
    assert counts["fit_unusable"] == 3
    assert counts["joint_required"] == 2
    assert sum(counts["status"].values()) == len(apt)
    np.testing.assert_array_equal(apt["flag"], [0, 0, 0, 0, 1, 0])


def test_misordered_model_channels_do_not_inherit_apt_identities():
    apt = Table({"kids_tone": [0, 1]})
    report = Table({"chan": [1, 0]})
    with pytest.raises(AssertionError):
        MODULE.channel_ledger(apt, report)


def test_fit_ledger_retains_unrouted_groups_and_counts_shared_components_once(
    monkeypatch,
):
    fit = xr.Dataset(
        {
            "group_id": ("row", ["0", "0", "1"]),
            "resonator_id": ("row", ["a", "a", "b"]),
            "chan": ("row", [0, 1, 2]),
            "fit_status": (
                "row",
                ["optimizer_converged", "optimizer_converged", "optimizer_failed"],
            ),
            "fr": ("row", [500e6, 500e6, 501e6]),
            "r": ("row", [1e-5, 1e-5, 2e-5]),
            "a_nl": ("row", [0.1, 0.1, 0.7]),
        }
    )
    tree = xr.DataTree()
    tree["model_fit/quality"] = xr.Dataset(
        {
            "group_id": ("group", ["0", "1"]),
            "group_flag": ("group", [0, 1]),
        }
    )
    block = SimpleNamespace(tree=tree, meta=SimpleNamespace(block=1))
    monkeypatch.setattr(MODULE.ModelFit, "view", lambda _: SimpleNamespace(fit=fit))
    report = Table(
        meta={"joint_models": {"groups": [{"group_id": "0", "channels": [0]}]}}
    )
    groups, components = MODULE.fit_records(block, report, "toltec0", 0)
    assert len(groups) == len(components) == 2
    assert [r["n_channels"] for r in groups] == [2, 1]
    assert [r["routed"] for r in groups] == [True, False]
    assert groups[0]["n_readout_channels"] == 1
    assert groups[1]["fit_status"] == "optimizer_failed"
