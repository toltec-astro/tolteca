"""Shared pytest fixtures for TolTEC data model tests.

Provides access to TolTEC files through the deployed ``data_lmt`` fixture. All
fixtures that require real data call ``pytest.skip`` when the files are absent,
so the test suite degrades gracefully outside a development deployment.

Directory layout::

    data_lmt/
    └── toltec/
        └── tcs/
            └── toltec0/
                ├── *_vnasweep.nc
                ├── *_targsweep.nc
                └── *_tune.nc
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

# ── Path fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def toltec0_data_path(data_lmt_path: Path) -> Path:
    """Path to the ``toltec0`` reference data directory.

    Skips the test if the directory does not exist.
    """
    path = data_lmt_path / "toltec" / "tcs" / "toltec0"
    if not path.is_dir():
        pytest.skip(f"toltec0 reference data not found at {path}")
    return path


# ── File fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def real_vnasweep_file(toltec0_data_path: Path) -> Path:
    """Path to the first available ``*_vnasweep.nc`` reference file."""
    files = sorted(toltec0_data_path.glob("*_vnasweep.nc"))
    if not files:
        pytest.skip(f"No *_vnasweep.nc files found in {toltec0_data_path}")
    return files[0]


@pytest.fixture(scope="session")
def real_targsweep_file(toltec0_data_path: Path) -> Path:
    """Path to the first available ``*_targsweep.nc`` reference file."""
    files = sorted(toltec0_data_path.glob("*_targsweep.nc"))
    if not files:
        pytest.skip(f"No *_targsweep.nc files found in {toltec0_data_path}")
    return files[0]


@pytest.fixture(scope="session")
def real_tune_file(toltec0_data_path: Path) -> Path:
    """Path to the first available ``*_tune.nc`` reference file."""
    files = sorted(toltec0_data_path.glob("*_tune.nc"))
    if not files:
        pytest.skip(f"No *_tune.nc files found in {toltec0_data_path}")
    return files[0]
