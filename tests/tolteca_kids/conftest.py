"""Recorded inputs for file-based pipeline integration tests."""

import os
from pathlib import Path

import pytest

N_CALIBRATION_SWEEPS = 31


@pytest.fixture
def target_sweep_path(data_lmt_path):
    """Return the recorded target-sweep input without opening it."""
    return (
        data_lmt_path / "toltec/tcs/toltec0/"
        "toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc"
    )


@pytest.fixture
def timestream_path(data_lmt_path):
    """Return the recorded timestream input without opening it."""
    return (
        data_lmt_path / "toltec/tcs/toltec0/"
        "toltec0_145609_000_0002_2025_10_31_09_27_57.nc"
    )


@pytest.fixture(scope="module")
def drive_fit_calibration_sequence():
    """Return the unchanged 31-sweep calibration baseline for v2 parity assertions."""
    root = Path(
        os.environ.get(
            "TOLTECA_DRIVE_FIT_DATA",
            Path(__file__).resolve().parents[3]
            / "tolteca_cal_data/toltec/drive_fit/ics-17068/toltec0",
        )
    )
    paths = sorted(root.rglob("*017068*targsweep.nc"))
    if len(paths) != N_CALIBRATION_SWEEPS:
        pytest.skip(
            f"31-sweep calibration unavailable at {root}; set TOLTECA_DRIVE_FIT_DATA"
        )
    return paths
