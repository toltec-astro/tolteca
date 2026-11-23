"""Scientific report counts must follow saved weights and actual sky pixels."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table
from netCDF4 import Dataset


@pytest.fixture
def report_module(monkeypatch):
    folder = Path(__file__).parents[2] / "examples/tune_model_comparison"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "array_report", folder / "array_map_report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_saved_stats_exclude_flagged_nonpositive_and_missing_weights(
    report_module, tmp_path
):
    with Dataset(tmp_path / "example_stats.nc", "w") as data:
        data.createDimension("n_chunks", 2)
        data.createDimension("n_dets", 6)
        weights = data.createVariable("weights", "f8", ("n_chunks", "n_dets"))
        flags = data.createVariable("flagged_frac", "f8", ("n_chunks", "n_dets"))
        weights[:] = [[1, 2, 0, -1, np.nan, 3], [1, 2, 1, -1, np.nan, 3]]
        flags[:] = [[0, 1, 0, 0, 0, 0], [0.5, 1, 0, 0, 0, 0]]
        weights[:, 5] = np.ma.masked
        data.createVariable("apt_nw", "i4", ("n_dets",))[:] = [0, 0, 1, 1, 7, 7]
        data.createVariable("apt_array", "i4", ("n_dets",))[:] = [0, 0, 0, 0, 1, 1]
        data.createVariable("apt_kids_tone", "i4", ("n_dets",))[:] = [0, 1, 0, 1, 0, 1]
    rows = report_module.statistics(tmp_path)
    assert [r["contributing_channels"] for r in rows] == [1, 1, 0]
    assert [r["contributing_in_all_chunks"] for r in rows] == [1, 0, 0]
    assert [r["contributing_channel_chunks"] for r in rows] == [2, 1, 0]
    assert sum(r["channels"] for r in rows) == 6
    assert [r["contributing_channel_ids"] for r in rows] == [[0], [0], []]


def test_report_rejects_map_use_outside_selected_model_eligibility(report_module):
    ledger = Table(
        {"nw": [0, 0], "channel": [3, 7], "apt_flag": [0, 0], "eligible": [True, False]}
    )
    rows = [{"nw": 0, "contributing_channel_ids": [3, 7]}]
    report_module.validate_contributions(ledger, {"legacy": rows})
    with pytest.raises(
        ValueError, match="python network 0 used ineligible channels: \\[7\\]"
    ):
        report_module.validate_contributions(ledger, {"python": rows})


def test_historical_stats_use_uid_not_ledger_row_order(report_module, tmp_path):
    with Dataset(tmp_path / "old_stats.nc", "w") as data:
        data.createDimension("chunk", 1)
        data.createDimension("det", 2)
        data.createVariable("weights", "f8", ("chunk", "det"))[:] = [[1, 1]]
        data.createVariable("flagged_frac", "f8", ("chunk", "det"))[:] = [[0, 1]]
        data.createVariable("apt_nw", "i4", ("det",))[:] = [0, 0]
        data.createVariable("apt_array", "i4", ("det",))[:] = [0, 0]
        data.createVariable("apt_uid", "f8", ("det",))[:] = [18, 15]
    ledger = Table(
        {"nw": [0, 0], "apt_uid": [15, 18], "channel": [3, 7], "apt_flag": [0, 0]}
    )
    with pytest.raises(ValueError, match="source APT channel ledger"):
        report_module.statistics(tmp_path)
    [row] = report_module.statistics(tmp_path, channel_ledger=ledger)
    assert row["contributing_channel_ids"] == [7]


def test_background_uses_common_covered_annulus_without_signal_rescaling(report_module):
    coords = np.arange(-40.0, 41.0, 10.0)
    radius = np.hypot(coords[None, :], coords[:, None])
    left = np.arange(radius.size, dtype=float).reshape(radius.shape)
    right = 2 * left
    right[2, 2] = np.nan
    result = report_module.background(coords, coords, left, right)
    expected = (radius >= 20) & (radius <= 40) & np.isfinite(right)
    assert result["pixels"] == expected.sum()
    assert result["legacy_rms"] == pytest.approx(np.std(left[expected], ddof=1))
    assert result["python_rms"] == pytest.approx(2 * result["legacy_rms"])


def test_signal_scale_retains_negative_structure_and_ignores_uncovered_pixels(
    report_module,
):
    base = {
        "x": np.array([-1.0, 1.0]),
        "y": np.array([-1.0, 1.0]),
        "signal": np.array([[-20.0, 10], [5, 30]]),
        "weight": np.ones((2, 2)),
        "coverage": np.ones((2, 2)),
    }
    weight = np.array([[0.0, 1], [1, 1]])
    other = dict(base, signal=np.array([[-1000.0, 5], [40, 0]]), weight=weight)
    assert report_module.signal_limits(base, other) == (-20.0, 40.0)
