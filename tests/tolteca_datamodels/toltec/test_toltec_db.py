"""Unit tests for TolTEC database constants and mappings."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
import xarray as xr

from tolteca_datamodels.lmt import open_lmt_tel
from tolteca_datamodels.lmt.types import LmtInfo
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.toltec_db import (
    ToltecDBRawObsMasterCode,
    ToltecDBRawObsTypeCode,
)
from tolteca_datamodels.toltec.types import ToltecDataKind, ToltecInfo


@pytest.fixture
def toltecdb_slice(data_lmt_path):
    path = data_lmt_path.resolve().parent / "test_fixtures/toltecdb_slice.sqlite"
    if not path.is_file():
        pytest.skip(f"reference toltecdb slice not found at {path}")
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as db:
        yield db


def test_constants_against_reference_database(toltecdb_slice):
    masters = dict(toltecdb_slice.execute("SELECT id, label FROM master"))
    for master in ToltecDBRawObsMasterCode:
        assert masters[master.value] == master.name
        assert masters[master.value].lower() == master.get_master(master)
        label = ToltecInfo.master_code_master[master]
        assert label == masters[master.value].lower()
        assert type(label) is str
        assert ToltecInfo.master_master_code[label] == master.value
    assert set(ToltecInfo.master_code_master) == set(ToltecDBRawObsMasterCode)
    assert set(ToltecInfo.master_master_code) == set(ToltecInfo.masters)
    for master, label in LmtInfo.master_code_master.items():
        assert label == ToltecInfo.master_code_master[master]
        assert LmtInfo.master_master_code[label] == ToltecInfo.master_master_code[label]
    # This database entry is not a supported acquisition master. The readout
    # header documents code 3 as TCS_SIM, so do not infer a file master from it.
    assert masters.keys() - set(ToltecDBRawObsMasterCode) == {3}
    assert masters[3] == "REPEAT"
    with pytest.raises(ValueError):
        ToltecDBRawObsMasterCode(3)
    with pytest.raises(KeyError):
        ToltecInfo.master_code_master[3]

    obs_types = dict(toltecdb_slice.execute("SELECT id, label FROM obstype"))
    assert obs_types == {member.value: member.name for member in ToltecDBRawObsTypeCode}
    assert ToltecInfo.obs_type_code_obs_type == {
        code: label.lower() for code, label in obs_types.items()
    }


def test_reference_telescope_masters_match_database(toltecdb_slice, data_lmt_path):
    masters = dict(toltecdb_slice.execute("SELECT id, label FROM master"))
    seen = set()
    for path in sorted((data_lmt_path / "tel").glob("*.nc")):
        with open_lmt_tel(path) as ds:
            meta = ds.lmt_tel.meta
            assert meta.master_code == ds["Header.TelescopeBackend.Master"].item()
            assert meta.master == masters[meta.master_code].lower()
            assert meta.master == LmtInfo.master_code_master[meta.master_code]
            assert LmtInfo.master_master_code[meta.master] == meta.master_code
            seen.add(meta.master_code)
    assert seen == {0}


def test_reference_file_headers_match_database(toltecdb_slice, data_lmt_path):
    """Compare the readout headers to independent database rows and labels."""
    rows = toltecdb_slice.execute(
        "SELECT DISTINCT FileName, Master, ObsType, RoachIndex, ObsNum, "
        "SubObsNum, ScanNum FROM toltec WHERE Valid = 1"
    )
    masters = dict(toltecdb_slice.execute("SELECT id, label FROM master"))
    kinds = {
        "Nominal": ToltecDataKind.raw_timestream,
        "Timestream": ToltecDataKind.raw_timestream,
        "VNA": ToltecDataKind.vnasweep,
        "TARG": ToltecDataKind.targsweep,
        "TUNE": ToltecDataKind.tune,
    }
    obs_types = dict(toltecdb_slice.execute("SELECT id, label FROM obstype"))
    seen = set()
    for filename, master, obs_type_code, roach, obsnum, subobsnum, scannum in rows:
        path = data_lmt_path / Path(filename).relative_to("/data_lmt")
        if not path.is_file():
            continue
        with xr.open_dataset(path) as ds:
            for key, expected in (
                ("Master", master),
                ("ObsType", obs_type_code),
                ("RoachIndex", roach),
                ("ObsNum", obsnum),
                ("SubObsNum", subobsnum),
                ("ScanNum", scannum),
            ):
                assert ds[f"Header.Toltec.{key}"].item() == expected, path.name
        with open_toltec_kids(path) as ds:
            assert ds.toltec_kids.data_kind == kinds[obs_types[obs_type_code]]
            meta = ds.toltec_kids.meta
            assert meta.master_code == master
            assert meta.master == masters[master].lower()
            assert meta.obs_type_code == obs_type_code
            assert (meta.roach, meta.obsnum, meta.subobsnum, meta.scannum) == (
                roach,
                obsnum,
                subobsnum,
                scannum,
            )
        seen.add(obs_types[obs_type_code])
    assert seen == {"Nominal", "VNA", "TARG", "TUNE"}


class TestToltecDBRawObsMasterCode:
    """Test ToltecDBRawObsMasterCode enum."""

    def test_enum_values(self) -> None:
        """Test that enum values match database constants."""
        assert ToltecDBRawObsMasterCode.TCS == 0
        assert ToltecDBRawObsMasterCode.ICS == 1
        assert ToltecDBRawObsMasterCode.CLIP == 2

    @pytest.mark.parametrize(
        ("code", "label"),
        [
            (ToltecDBRawObsMasterCode.TCS, "tcs"),
            (ToltecDBRawObsMasterCode.ICS, "ics"),
            (ToltecDBRawObsMasterCode.CLIP, "clip"),
        ],
    )
    def test_get_master(self, code, label):
        master = ToltecDBRawObsMasterCode.get_master(code)
        assert master == label
        assert type(master) is str


class TestToltecDBRawObsTypeCode:
    """Test ToltecDBRawObsTypeCode enum."""

    @pytest.mark.parametrize(
        ("code", "label"),
        [
            (ToltecDBRawObsTypeCode.Nominal, "nominal"),
            (ToltecDBRawObsTypeCode.Timestream, "timestream"),
            (ToltecDBRawObsTypeCode.VNA, "vna"),
            (ToltecDBRawObsTypeCode.TARG, "targ"),
            (ToltecDBRawObsTypeCode.TUNE, "tune"),
        ],
    )
    def test_get_obs_type(self, code, label):
        obs_type = ToltecDBRawObsTypeCode.get_obs_type(code)
        assert obs_type == label
        assert type(obs_type) is str

    def test_enum_values(self) -> None:
        """Test that enum values match database constants."""
        assert ToltecDBRawObsTypeCode.Nominal == 0
        assert ToltecDBRawObsTypeCode.Timestream == 1
        assert ToltecDBRawObsTypeCode.VNA == 2
        assert ToltecDBRawObsTypeCode.TARG == 3
        assert ToltecDBRawObsTypeCode.TUNE == 4
