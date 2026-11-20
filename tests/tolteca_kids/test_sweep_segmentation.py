"""Measured support, rather than channel boundaries, defines local fit groups."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from astropy import units as u
from astropy.table import QTable

from tolteca_kids.model_fit import ModelFit
from tolteca_kids.sweep import ReducedSweepView
from tolteca_kids.sweep_groups import _finding_rows
from tolteca_kidsproc.analysis import KidsModelAnalysis
from tolteca_kidsproc.segmentation import (
    CrossSweepPlan,
    group_intervals,
    group_sweep_plans,
    validate_segment_ownership,
)


def sweep_plan(centers, *, channels=(500e6, 500e6 + 50000), bounds=None, mask=None):
    offsets = np.linspace(-100000, 100000, 201)
    sweep = ReducedSweepView(
        xr.Dataset(
            {
                "I": (("chan", "sweep"), np.ones((len(channels), len(offsets)))),
                "Q": (("chan", "sweep"), np.zeros((len(channels), len(offsets)))),
                "f_chan": ("chan", np.asarray(channels)),
            },
            coords={"chan": np.arange(len(channels)) + 10, "sweep": offsets},
        )
    )
    finding = SimpleNamespace(
        detected=QTable(
            {
                "idx": np.arange(len(centers)) + 7,
                "group": np.arange(len(centers)),
                "f": np.asarray(centers) * u.Hz,
                "Qr": np.asarray(centers) / 10000,
            }
        )
    )
    return ModelFit.make_plan(sweep, finding, mask=mask, frequency_bounds=bounds)


def test_interval_closure_retains_broad_and_transitive_support():
    bounds = [[0, 100], [10, 11], [99, 120], [119, 140], [150, 160]]
    labels = np.array(["a", "b", "c", "d", "e"])
    assert group_intervals(bounds, labels).tolist() == ["a", "a", "a", "a", "e"]
    order = [4, 2, 1, 0, 3]
    actual = group_intervals(np.asarray(bounds)[order], labels[order])
    assert actual[np.argsort(order)].tolist() == ["a", "a", "a", "a", "e"]


def test_windows_merge_before_channel_samples_are_selected():
    # Old center-distance grouping left these candidates separate, although
    # their two-FWHM sample windows overlap in both acquired channels.
    plan = sweep_plan([500e6, 500e6 + 30000, 500e6 + 140000])
    assert set(plan.group_id.values[plan.resonator_id.values != "9"]) == {"0"}
    assert set(plan.group_id.values[plan.resonator_id.values == "9"]) == {"2"}
    assert set(plan.finding_group_id.values) == {"0", "1", "2"}
    validate_segment_ownership(plan)
    # Sharing one channel does not join independent frequency windows.
    assert set(plan.group_id.values[plan.chan.values == 11]) == {"0", "2"}


def test_asymmetric_support_keeps_partial_neighbors_and_masked_gaps():
    mask = np.ones((2, 201), bool)
    mask[0, 100:105] = False
    plan = sweep_plan(
        [500e6, 500e6 + 90000],
        mask=mask,
        bounds=np.array(
            [[500e6 - 10000, 500e6 + 95000], [500e6 + 80000, 500e6 + 120000]]
        ),
    )
    assert set(plan.group_id.values) == {"0"}
    assert {
        (int(c), str(r))
        for c, r in zip(plan.chan.values, plan.resonator_id.values, strict=True)
    } == {
        (10, "7"),
        (11, "7"),
        (10, "8"),
        (11, "8"),
    }
    for row in plan.to_dataframe().itertuples():
        assert np.all(mask[row.chan - 10, row.sample_start : row.sample_stop])
    assert (
        len(plan.where((plan.chan == 10) & (plan.resonator_id == "7"), drop=True).row)
        == 2
    )


def test_supplied_overlapping_independent_groups_are_rejected():
    plan = sweep_plan([500e6, 500e6 + 30000])
    plan["group_id"] = plan.finding_group_id.copy()
    with pytest.raises(ValueError, match="samples overlap"):
        validate_segment_ownership(plan)


def test_cross_sweep_merge_and_split_preserve_members_and_locality(tmp_path):
    # Two separate responses approach and blend, then separate again while
    # migrating from channel 10 to 11. Component 9 is unrelated throughout.
    plans = {
        "s1": sweep_plan([500e6 - 40000, 500e6 + 40000, 500e6 + 140000]),
        "s2": sweep_plan([500e6 + 10000, 500e6 + 30000, 500e6 + 140000]),
        "s3": sweep_plan([500e6 + 50000, 500e6 - 30000, 500e6 + 140000]),
    }
    links = xr.Dataset(
        {
            "dataset": ("row", ["s2", "s2", "s3", "s3"]),
            "resonator_id": ("row", ["7", "8", "7", "8"]),
            "dataset_ref": ("row", ["s1"] * 4),
            "resonator_id_ref": ("row", ["7", "8", "7", "8"]),
            "unique": ("row", [False, False, True, True]),
        }
    )
    result = group_sweep_plans(plans, links)
    group = result.groups.joint_group_id.values[
        (result.groups.dataset.values == "s2") & (result.groups.group_id.values == "0")
    ].item()
    selected = result.select(group)
    assert set(selected) == {"s1", "s2", "s3"}
    for plan in selected.values():
        assert set(plan.resonator_id.values) == {"7", "8"}
        assert 11 in plan.chan.values
    assert len(np.unique(result.groups.joint_group_id)) == 4
    # Candidate ambiguity survives; connecting the computation is not acceptance.
    xr.testing.assert_identical(links, result.links)
    path = tmp_path / "groups.zarr"
    result.to_tree().to_zarr(path, zarr_format=2, consolidated=False)
    with xr.open_datatree(path, engine="zarr", consolidated=False) as tree:
        restored = CrossSweepPlan.from_tree(tree)
    for key, plan in restored.select(group).items():
        xr.testing.assert_identical(plan, selected[key])
    with pytest.raises(KeyError):
        result.select("absent")


def test_cross_sweep_unmapped_endpoints_do_not_invent_components():
    plans = {"a": sweep_plan([500e6]), "b": sweep_plan([500e6])}
    links = xr.Dataset(
        {
            "dataset": ("row", ["b"]),
            "resonator_id": ("row", [""]),
            "dataset_ref": ("row", ["a"]),
            "resonator_id_ref": ("row", ["7"]),
        }
    )
    result = group_sweep_plans(plans, links)
    assert len(np.unique(result.groups.joint_group_id)) == 2
    invalid = links.copy(deep=True)
    invalid["resonator_id"] = ("row", ["missing"])
    with pytest.raises(ValueError, match="absent component"):
        group_sweep_plans(plans, invalid)


def test_missing_sweep_support_remains_an_explicit_gap(tmp_path):
    plan = sweep_plan([500e6], mask=np.zeros((2, 201), dtype=bool))
    assert plan.sizes["row"] == 0
    links = xr.Dataset(
        {
            name: ("row", np.array([], dtype=str))
            for name in ("dataset", "resonator_id", "dataset_ref", "resonator_id_ref")
        }
    )
    result = group_sweep_plans({"gap": plan}, links)
    assert result.groups.sizes["row"] == 0
    path = tmp_path / "gap.zarr"
    result.to_tree().to_zarr(path, zarr_format=2)
    with xr.open_datatree(path, engine="zarr") as tree:
        restored = CrossSweepPlan.from_tree(tree)
    assert restored.plans["gap"].attrs["status"] == "no_usable_samples"
    assert restored.links.sizes["row"] == 0


def test_matching_rows_use_exact_finding_ids_not_nearest_frequencies():
    finding = SimpleNamespace(
        detected=QTable(
            {
                "idx": [7, 8],
                "f": [500e6, 500e6 + 30000] * u.Hz,
            }
        )
    )
    plan = sweep_plan([500e6, 500e6 + 30000])
    table = QTable(
        {
            "idx_det": [8, 7, -1],
            "f_det": [500e6 + 30000, 500e6, np.nan] * u.Hz,
        }
    )
    assert _finding_rows(table, finding, plan) == ["8", "7", ""]
    changed = table.copy()
    changed["idx_det"][0] = 7
    with pytest.raises(ValueError, match="exact saved finding disagree"):
        _finding_rows(changed, finding, plan)


def test_fit_follows_a_boundary_crossing_with_distinct_readout_gains():
    channels = np.array([500e6, 500e6 + 120000, 500e6 + 240000])
    offsets = np.linspace(-100000, 100000, 201)
    frequency = channels[:, None] + offsets
    gains = np.array([1 + 0.2j, 1.6 - 0.3j, 0.8 + 0.5j])[:, None]
    analyses = []
    for center in (500e6 + 40000, 500e6 + 140000, 500e6 + 180000):
        signal = (
            gains
            * (1 + (0.02 + 0.01j) * offsets / 1e5)
            * (1 - (6e-6 + 1e-6j) / (1 / 40000 + 1j * (frequency / center - 1)))
        )
        data = xr.Dataset(
            {
                "I": (("chan", "sweep"), signal.real),
                "Q": (("chan", "sweep"), signal.imag),
                "f_chan": ("chan", channels),
            },
            coords={"chan": [10, 11, 12], "sweep": offsets},
        )
        sweep = ReducedSweepView(data)
        finding = SimpleNamespace(
            detected=QTable(
                {
                    "idx": [7],
                    "group": [0],
                    "f": [center] * u.Hz,
                    "Qr": [20000.0],
                }
            )
        )
        plan = ModelFit.make_plan(sweep, finding)
        result = (
            KidsModelAnalysis()
            .make_fit(
                data.assign_coords(frequency=sweep.frequency).kids.multi_sweep, plan
            )["fit"]
            .to_dataset(inherit=False)
        )
        np.testing.assert_allclose(result.fr.values, center, rtol=0, atol=0.1)
        np.testing.assert_allclose(result.r.values, 1 / 40000, rtol=1e-4)
        # Relabel/reorder readout channels without changing their measurements.
        reordered = data.isel(chan=[2, 0, 1]).assign_coords(chan=[22, 20, 21])
        changed = ReducedSweepView(reordered)
        changed_plan = ModelFit.make_plan(changed, finding)
        other = (
            KidsModelAnalysis()
            .make_fit(
                reordered.assign_coords(frequency=changed.frequency).kids.multi_sweep,
                changed_plan,
            )["fit"]
            .to_dataset(inherit=False)
        )
        np.testing.assert_allclose(other.fr.values, center, rtol=0, atol=0.1)
        np.testing.assert_allclose(other.r.values, 1 / 40000, rtol=1e-4)
        analyses.append(set(plan.chan.values))
    assert analyses == [{10, 11}, {10, 11, 12}, {11, 12}]
