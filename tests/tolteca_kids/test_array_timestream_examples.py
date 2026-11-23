"""Report examples follow sweep evidence and preserve common sampling/pixels."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table


@pytest.fixture
def examples(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[2] / "examples/tune_model_comparison")
    )
    return importlib.import_module("array_timestream_examples")


def test_selection_uses_median_sweep_quality_and_complete_good_groups(examples):
    groups, rows = [], []
    for size in (1, 2):
        for i, error in enumerate((0.01, 0.02, 0.04, 0.015, 0.019, 0.03)):
            gid = f"{size}-{i}"
            channels = list(range(10 * i, 10 * i + size))
            groups.append(
                {
                    "group_id": gid,
                    "channels": channels,
                    "resonator_ids": [str(c) for c in channels],
                    "a_nl": [i * 0.5] * size,
                    "fit_flag": int(i == 5),
                }
            )
            rows.extend(
                (
                    0,
                    True,
                    gid,
                    c,
                    int(i == 3),
                    "failed" if i == 4 else "optimizer_converged",
                    error,
                )
                for c in channels
            )
    responses = Table(
        rows=rows,
        names=(
            "nw",
            "selected_for_probe",
            "group_id",
            "channel",
            "apt_flag",
            "fit_status",
            "v3_fraction",
        ),
    )
    chosen = examples.select_cases(responses, groups)
    assert [c["group"]["group_id"] for c in chosen] == ["1-1", "2-1"]
    assert [c["selection"]["pool"] for c in chosen] == [3, 3]
    # Removing one channel cannot turn a partial group into a representative fit.
    responses = responses[
        ~((responses["group_id"] == "2-1") & (responses["channel"] == 11))
    ]
    assert examples.select_cases(responses, groups)[1]["group"]["group_id"] != "2-1"


@pytest.mark.parametrize(
    ("valid", "interval"),
    [
        ([False, False], (0, 0)),
        ([True, True, False, True, True, True, False], (3, 6)),
        ([True, False, True], (0, 1)),
    ],
)
def test_spectral_interval_does_not_close_invalid_gaps(examples, valid, interval):
    assert examples.longest_valid_interval(valid) == interval


def test_map_uncertainty_and_rms_use_same_pixels(examples):
    plots = importlib.import_module("array_example_plots")
    coords = np.arange(-40.0, 41.0, 10.0)
    radius = np.hypot(coords[:, None], coords[None, :])
    signal = np.arange(radius.size, dtype=float).reshape(radius.shape)
    left = {
        "x": coords,
        "y": coords,
        "signal": signal,
        "weight": np.full(radius.shape, 4.0),
        "coverage": np.ones(radius.shape),
        "header": {"UNIT": "mJy/beam"},
    }
    weight = np.full(radius.shape, 16.0)
    weight[2, 2] = 0
    right = dict(left, signal=2 * signal, weight=weight)
    result = plots.annulus_measurement(left, right)
    keep = (radius >= 20) & (radius <= 40) & (weight > 0)
    assert result["pixels"] == int(keep.sum())
    assert result["formal_weight_error_median"] == [0.5, 0.25]
    assert result["annulus_rms"] == pytest.approx(
        [np.std(signal[keep], ddof=1), 2 * np.std(signal[keep], ddof=1)]
    )
