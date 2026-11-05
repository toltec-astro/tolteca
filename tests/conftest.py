"""Shared fixtures for the TolTECA test suite."""

from __future__ import annotations

import pytest
import xarray as xr


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
