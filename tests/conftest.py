"""Shared fixtures for the TolTECA test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import xarray as xr

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


@pytest.fixture
def dummy_tel_path(tmp_path, data_lmt_path):
    """Synthetic telescope companion with a dummy receiver and minimal headers."""
    path = tmp_path / "tel_dummy_instru_2025-10-31_145609_00_0002.nc"
    reference = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    with xr.open_dataset(reference) as data:
        telescope = data[
            [
                "Header.File.Name",
                "Header.Dcs.ObsNum",
                "Header.Dcs.SubObsNum",
                "Header.Dcs.ScanNum",
                "Header.Dcs.ObsPgm",
                "Header.Dcs.ObsGoal",
                "Header.Dcs.IntegrationTime",
                "Header.Source.Ra",
                "Header.Source.Dec",
                "Header.TelescopeBackend.Master",
                "Data.TelescopeBackend.TelTime",
                "Data.TelescopeBackend.TelAzAct",
            ]
        ]
        telescope.drop_encoding().assign(
            {
                "Header.File.Name": str(path),
                "Header.Dcs.Receiver": "Dummy_Instru",
            }
        ).to_netcdf(path)
    return path
