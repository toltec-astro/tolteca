"""Coordinate controls must preserve inputs and compare actual shared sky pixels."""

import copy
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def control(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[2] / "examples/tune_model_comparison")
    )
    return importlib.import_module("array_coordinate_control")


def test_coordinate_config_only_changes_output_and_requested_variance_cuts(
    control, tmp_path
):
    source: dict[str, Any] = {
        "runtime": {"output_dir": "original", "n_threads": 2},
        "inputs": [{"data_items": [{"filepath": "raw.nc", "stubs": ["model.nc"]}]}],
        "timestream": {
            stage: {
                "flagging": {
                    "lower_tod_inv_var_factor": 0.25,
                    "upper_tod_inv_var_factor": 10.0,
                    "other_threshold": 7,
                },
                "cleaning": {"enabled": True},
            }
            for stage in ("raw_time_chunk", "processed_time_chunk")
        },
    }
    original = copy.deepcopy(source)
    normal = control.make_config(source, tmp_path)
    uncut = control.make_config(source, tmp_path, variance_cuts=False)
    assert source == original
    expected = copy.deepcopy(original)
    expected["runtime"]["output_dir"] = str(tmp_path.resolve())
    assert normal == expected
    for stage in expected["timestream"].values():
        for key in ("lower_tod_inv_var_factor", "upper_tod_inv_var_factor"):
            stage["flagging"][key] = 0.0
    assert uncut == expected


def map_data(coords, scale=1.0):
    """Construct a sky field with known differences and arbitrary extent."""
    field = coords[:, None] + 2 * coords[None, :]
    return {
        "x": coords,
        "y": coords,
        "signal": scale * field,
        "weight": np.ones(field.shape),
        "coverage": np.ones(field.shape),
        "header": {"UNIT": "mJy/beam"},
    }


def test_three_map_intersection_crops_integer_grids_without_rescaling(control):
    first = map_data(np.arange(-60.0, 61.0, 10.0))
    second = map_data(np.arange(-50.0, 61.0, 10.0), 2.0)
    third = map_data(np.arange(-50.0, 51.0, 10.0), 3.0)
    second["weight"][3, 3] = 0
    third["signal"][7, 7] = np.nan
    items, _, annulus = control.common_maps([first, second, third])
    np.testing.assert_array_equal(items[0]["x"], third["x"])
    radius = np.hypot(third["x"][None, :], third["y"][:, None])
    expected = (radius >= 20) & (radius <= 40)
    expected[3, 3] = False
    expected[7, 7] = False
    np.testing.assert_array_equal(annulus, expected)
    for multiplier, item in enumerate(items, start=1):
        np.testing.assert_array_equal(
            item["signal"][annulus], multiplier * items[0]["signal"][annulus]
        )


def test_coordinate_comparison_rejects_fractional_pixel_offsets(control):
    coords = np.arange(-50.0, 51.0, 10.0)
    with pytest.raises(ValueError, match="fractional offset"):
        control.common_maps([map_data(coords), map_data(coords + 0.25)])


def test_coverage_threshold_uses_full_map_not_cropped_peak(control):
    wide = map_data(np.arange(-70.0, 71.0, 10.0))
    wide["coverage"][0, 0] = 100
    narrow = map_data(np.arange(-50.0, 51.0, 10.0))
    with pytest.raises(ValueError, match="Insufficient common annulus"):
        control.common_maps([wide, narrow])


def test_statistics_separates_eligibility_weights_and_actual_contributions(control):
    stats = {
        "apt_array": np.array([0, 0, 1, 2]),
        "apt_nw": np.array([0, 0, 7, 11]),
        "apt_kids_tone": np.array([1, 2, 1, 1]),
        "weights": np.ones((2, 4)),
        "flagged_frac": np.array([[0, 1, 0, 0], [0, 0, 0, 0]]),
    }
    changed = copy.deepcopy(stats)
    changed["flagged_frac"][0, 1] = 0.0
    changed["weights"][0, 2] = 2.0
    rows = control.compare_statistics(stats, changed)
    assert rows[0]["contributors"] == [2, 2]
    assert rows[0]["contributing_chunk_channels"] == [3, 4]
    assert rows[0]["changed_contribution_entries"] == 1
    assert rows[0]["all_weights_equal"]
    assert rows[1]["changed_contribution_entries"] == 0
    assert not rows[1]["all_weights_equal"]
    assert rows[2]["all_weights_equal"]
    changed["apt_kids_tone"][0] = 100
    with pytest.raises(AssertionError):
        control.compare_statistics(stats, changed)
