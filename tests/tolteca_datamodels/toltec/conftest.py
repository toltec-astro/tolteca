"""Shared pytest fixtures for TolTEC data model tests.

Provides access to ``tolteca_ref_data/`` reference files.  All fixtures that
require real data call ``pytest.skip`` when the files are absent, so the test
suite degrades gracefully on machines that do not have the reference data
checked out.

Directory layout (relative to package root ``tolteca/``)::

    tolteca_ref_data/
    └── data_lmt/
        └── toltec/
            └── tcs/
                └── toltec0/
                    ├── *_vnasweep.nc
                    ├── *_targsweep.nc
                    └── *_tune.nc
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Resolved once at import time so the skipif conditions are cheap.
_PACKAGE_ROOT = Path(__file__).parents[3]
_REF_DATA_ROOT = _PACKAGE_ROOT / "tolteca_ref_data"
_TOLTEC0_DIR = _REF_DATA_ROOT / "data_lmt" / "toltec" / "tcs" / "toltec0"


# ── Path fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def tolteca_ref_data_path() -> Path:
    """Root path of the tolteca reference data tree.

    Skips the test if ``tolteca_ref_data/`` does not exist next to the package.
    """
    if not _REF_DATA_ROOT.exists():
        pytest.skip(f"tolteca_ref_data not found at {_REF_DATA_ROOT}")
    return _REF_DATA_ROOT


@pytest.fixture(scope="session")
def toltec0_data_path(tolteca_ref_data_path: Path) -> Path:  # noqa: ARG001
    """Path to the ``toltec0`` reference data directory.

    Skips the test if the directory does not exist.
    """
    if not _TOLTEC0_DIR.exists():
        pytest.skip(f"toltec0 reference data not found at {_TOLTEC0_DIR}")
    return _TOLTEC0_DIR


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
