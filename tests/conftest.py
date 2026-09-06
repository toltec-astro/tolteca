"""Shared fixtures for the TolTECA test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).parents[1]
_LOCAL_DEV_LOCATION = _PACKAGE_ROOT.parent / "toltec_astro_dev"


@pytest.fixture(scope="session")
def data_lmt_path() -> Path:
    """Return the deployed, read-only ``data_lmt`` test fixture.

    An activated deployment is authoritative. When tests are run directly
    from this repository, use the Python workspace's development location.
    """
    deploy_root = os.environ.get("TOLTECA_DEPLOY_ROOT")
    location_root = (
        Path(deploy_root).expanduser()
        if deploy_root is not None
        else _LOCAL_DEV_LOCATION
    )
    path = location_root / "run" / "data_lmt"
    if not path.is_dir():
        pytest.skip(f"deployed data_lmt test fixture not found at {path}")
    return path
