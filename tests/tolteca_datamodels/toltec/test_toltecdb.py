"""Unit tests for TolTEC database constants and mappings."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
import xarray as xr

from tolteca_datamodels.toltec import open_toltec
from tolteca_datamodels.toltec.toltecdb import (
    ToltecDBRawObsMaster,
    ToltecDBRawObsType,
)
from tolteca_datamodels.toltec.types import ToltecDataKind, ToltecMasterType


@pytest.fixture
def toltecdb_slice(data_lmt_path):
    path = data_lmt_path.resolve().parent / "test_fixtures/toltecdb_slice.sqlite"
    if not path.is_file():
        pytest.skip(f"reference toltecdb slice not found at {path}")
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as db:
        yield db


def test_constants_against_reference_database(toltecdb_slice):
    masters = dict(toltecdb_slice.execute("SELECT id, label FROM master"))
    for master in ToltecDBRawObsMaster:
        assert masters[master.value] == master.name
        assert masters[master.value].lower() == master.get_master_type(master)
    # This database entry is not a supported ToltecMasterType. The readout
    # header documents code 3 as TCS_SIM, so do not infer a file master from it.
    assert masters.keys() - set(ToltecDBRawObsMaster) == {3}
    assert masters[3] == "REPEAT"
    with pytest.raises(ValueError):
        ToltecDBRawObsMaster(3)

    obs_types = dict(toltecdb_slice.execute("SELECT id, label FROM obstype"))
    assert obs_types == {member.value: member.name for member in ToltecDBRawObsType}


def test_reference_file_headers_match_database(toltecdb_slice, data_lmt_path):
    """Compare the readout headers to independent database rows and labels."""
    rows = toltecdb_slice.execute(
        "SELECT DISTINCT FileName, Master, ObsType, RoachIndex, ObsNum, "
        "SubObsNum, ScanNum FROM toltec WHERE Valid = 1"
    )
    masters = dict(toltecdb_slice.execute("SELECT id, label FROM master"))
    kinds = {
        "Nominal": ToltecDataKind.RawTimeStream,
        "Timestream": ToltecDataKind.RawTimeStream,
        "VNA": ToltecDataKind.VnaSweep,
        "TARG": ToltecDataKind.TargetSweep,
        "TUNE": ToltecDataKind.Tune,
    }
    obs_types = dict(toltecdb_slice.execute("SELECT id, label FROM obstype"))
    seen = set()
    for filename, master, obs_type, roach, obsnum, subobsnum, scannum in rows:
        path = data_lmt_path / Path(filename).relative_to("/data_lmt")
        if not path.is_file():
            continue
        with xr.open_dataset(path) as ds:
            for key, expected in (
                ("Master", master),
                ("ObsType", obs_type),
                ("RoachIndex", roach),
                ("ObsNum", obsnum),
                ("SubObsNum", subobsnum),
                ("ScanNum", scannum),
            ):
                assert ds[f"Header.Toltec.{key}"].item() == expected, path.name
        with open_toltec(path) as ds:
            assert ds.toltec_kids.data_kind == kinds[obs_types[obs_type]]
            meta = ds.toltec_kids.meta
            assert meta.master == masters[master].lower()
            assert (meta.roach, meta.obsnum, meta.subobsnum, meta.scannum) == (
                roach,
                obsnum,
                subobsnum,
                scannum,
            )
        seen.add(obs_types[obs_type])
    assert seen == {"Nominal", "VNA", "TARG", "TUNE"}


class TestToltecDBRawObsMaster:
    """Test ToltecDBRawObsMaster enum."""

    def test_enum_values(self) -> None:
        """Test that enum values match database constants."""
        assert ToltecDBRawObsMaster.TCS == 0
        assert ToltecDBRawObsMaster.ICS == 1
        assert ToltecDBRawObsMaster.CLIP == 2

    def test_get_master_type_tcs(self) -> None:
        """Test getting master type for TCS."""
        master_type = ToltecDBRawObsMaster.get_master_type(ToltecDBRawObsMaster.TCS)
        assert master_type == ToltecMasterType.tcs

    def test_get_master_type_ics(self) -> None:
        """Test getting master type for ICS."""
        master_type = ToltecDBRawObsMaster.get_master_type(ToltecDBRawObsMaster.ICS)
        assert master_type == ToltecMasterType.ics

    def test_get_master_type_clip(self) -> None:
        """Test getting master type for CLIP."""
        master_type = ToltecDBRawObsMaster.get_master_type(ToltecDBRawObsMaster.CLIP)
        assert master_type == ToltecMasterType.clip

    def test_all_masters_mapped(self) -> None:
        """Test that all master enum values have master type mappings."""
        for master in ToltecDBRawObsMaster:
            master_type = ToltecDBRawObsMaster.get_master_type(master)
            assert master_type is not None
            assert isinstance(master_type, ToltecMasterType)


class TestToltecDBRawObsType:
    """Test ToltecDBRawObsType enum."""

    def test_enum_values(self) -> None:
        """Test that enum values match database constants."""
        assert ToltecDBRawObsType.Nominal == 0
        assert ToltecDBRawObsType.Timestream == 1
        assert ToltecDBRawObsType.VNA == 2
        assert ToltecDBRawObsType.TARG == 3
        assert ToltecDBRawObsType.TUNE == 4

    def test_get_data_kind_nominal(self) -> None:
        """Test getting data kind for nominal observation."""
        data_kind = ToltecDBRawObsType.get_data_kind(ToltecDBRawObsType.Nominal)
        assert data_kind == ToltecDataKind.RawTimeStream

    def test_get_data_kind_timestream(self) -> None:
        """Test getting data kind for timestream."""
        data_kind = ToltecDBRawObsType.get_data_kind(ToltecDBRawObsType.Timestream)
        assert data_kind == ToltecDataKind.RawTimeStream

    def test_get_data_kind_vna(self) -> None:
        """Test getting data kind for VNA sweep."""
        data_kind = ToltecDBRawObsType.get_data_kind(ToltecDBRawObsType.VNA)
        assert data_kind == ToltecDataKind.VnaSweep

    def test_get_data_kind_targ(self) -> None:
        """Test getting data kind for target sweep."""
        data_kind = ToltecDBRawObsType.get_data_kind(ToltecDBRawObsType.TARG)
        assert data_kind == ToltecDataKind.TargetSweep

    def test_get_data_kind_tune(self) -> None:
        """Test getting data kind for tune."""
        data_kind = ToltecDBRawObsType.get_data_kind(ToltecDBRawObsType.TUNE)
        assert data_kind == ToltecDataKind.Tune

    def test_all_obs_types_mapped(self) -> None:
        """Test that all obs type enum values have data kind mappings."""
        for obs_type in ToltecDBRawObsType:
            data_kind = ToltecDBRawObsType.get_data_kind(obs_type)
            assert data_kind is not None
            assert isinstance(data_kind, ToltecDataKind)

    def test_obs_type_to_data_kind_consistency(self) -> None:
        """Test consistency between observation type and data kind."""
        # VNA and TARG should map to sweep types
        vna_kind = ToltecDBRawObsType.get_data_kind(ToltecDBRawObsType.VNA)
        targ_kind = ToltecDBRawObsType.get_data_kind(ToltecDBRawObsType.TARG)
        assert vna_kind in (ToltecDataKind.VnaSweep, ToltecDataKind.RawSweep)
        assert targ_kind in (ToltecDataKind.TargetSweep, ToltecDataKind.RawSweep)

        # Nominal and Timestream should map to timestream types
        nominal_kind = ToltecDBRawObsType.get_data_kind(ToltecDBRawObsType.Nominal)
        ts_kind = ToltecDBRawObsType.get_data_kind(ToltecDBRawObsType.Timestream)
        assert nominal_kind in (ToltecDataKind.RawTimeStream,)
        assert ts_kind in (ToltecDataKind.RawTimeStream,)
