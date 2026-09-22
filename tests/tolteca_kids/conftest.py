"""Recorded inputs for file-based pipeline integration tests."""

import pytest


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
