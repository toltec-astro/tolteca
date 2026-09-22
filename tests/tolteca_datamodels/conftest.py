"""Read-only acquisition records for filename and metadata tests."""

import csv
import sqlite3
from contextlib import closing

import pytest


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
