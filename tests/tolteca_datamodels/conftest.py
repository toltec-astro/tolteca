"""Read-only acquisition records for filename and metadata tests."""

import csv
import sqlite3
from contextlib import closing

import pytest
import xarray as xr


@pytest.fixture(scope="session")
def lmtmc_records(data_lmt_path):
    """Return the recorded telescope observations from the LMTMC slice."""
    path = data_lmt_path.resolve().parent / "test_fixtures/lmtmc_slice.csv"
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="session")
def toltecdb_records(data_lmt_path):
    """Return readout records with their database master and observation labels."""
    path = data_lmt_path.resolve().parent / "test_fixtures/toltecdb_slice.sqlite"
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        return list(
            db.execute(
                "SELECT toltec.*, master.label AS master_label, "
                "obstype.label AS kind_label FROM toltec "
                "JOIN master ON toltec.Master = master.id "
                "JOIN obstype ON toltec.ObsType = obstype.id ORDER BY toltec.id"
            )
        )


@pytest.fixture(scope="session")
def kids_readout_settings(data_lmt_path):
    """Read recorded settings for toy datasets, not acquisition-format proof."""
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145606*_targsweep.nc")
    with xr.open_dataset(path) as data:
        return {
            f"Header.Toltec.{key}": data[f"Header.Toltec.{key}"].item()
            for key in (
                "DriveAtten",
                "SenseAtten",
                "TargSweepObsNum",
                "TargSweepSubObsNum",
                "TargSweepScanNum",
                "MaxNumTones",
                "Filename",
                "RepeatLevel",
            )
        }


@pytest.fixture(scope="session")
def kids_filepath_orig(kids_readout_settings):
    """Return the recorded acquisition path for metadata-model tests."""
    return kids_readout_settings["Header.Toltec.Filename"].decode().rstrip("\x00 ")


@pytest.fixture(scope="session")
def kids_design_data(data_lmt_path):
    """Read recorded design/ADC values for toy metadata datasets."""
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145606*_targsweep.nc")
    with xr.open_dataset(path) as data:
        return data[["Header.Toltec.Xloc", "Header.Toltec.AdcSnapData"]].load()
