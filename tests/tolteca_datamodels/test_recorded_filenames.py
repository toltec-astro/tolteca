"""Filename discovery agrees with independently recorded acquisition metadata."""

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tollan.utils.fileloc import fileloc

from tolteca_datamodels.file import FilepathInfo
from tolteca_datamodels.lmt import LmtFilepathInfo
from tolteca_datamodels.toltec.file import (
    ToltecFilepathInfo,
    get_info_from_source,
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.types import ToltecDataKind


def test_lmtmc_filenames(lmtmc_records):
    assert len(lmtmc_records) == 12
    for row in lmtmc_records:
        info = FilepathInfo.from_source(row["FileName"])
        assert type(info) is LmtFilepathInfo
        assert info.prefix == "tel_toltec"
        assert info.interface == row["Instrument"] == "tel"
        assert Path(row["FileName"]).parent.name == info.interface
        assert (info.instru, info.instru_component) == ("lmt", "tcs")
        assert (info.obsnum, info.subobsnum, info.scannum) == tuple(
            map(int, row["ObsNum"].split("."))
        ), row["FileName"]
        assert info.suffix is None and info.ext == "nc"
        assert info.master == "tcs"
        source = get_info_from_source(row["FileName"])
        assert source is not None
        assert source.file_prefix == info.prefix
        assert source.file_ut == info.ut
        assert source.obsnum == info.obsnum
        assert source.data_kind == ToltecDataKind.lmt_tel.value


def test_toltecdb_filenames(toltecdb_records):
    kinds = {
        "Nominal": (None, ToltecDataKind.raw_timestream),
        "VNA": ("vnasweep", ToltecDataKind.vnasweep),
        "TARG": ("targsweep", ToltecDataKind.targsweep),
        "TUNE": ("tune", ToltecDataKind.tune),
    }
    assert len(toltecdb_records) == 120
    for row in toltecdb_records:
        info = FilepathInfo.from_source(row["FileName"])
        assert isinstance(info, ToltecFilepathInfo)
        assert info.prefix == info.interface == f"toltec{row['RoachIndex']}"
        assert Path(row["FileName"]).parent.name == info.interface
        assert type(info.roach) is int
        assert info.roach == row["RoachIndex"]
        assert (info.instru, info.instru_component) == ("toltec", "roach")
        assert info.master == row["master_label"].lower()
        assert (info.obsnum, info.subobsnum, info.scannum) == (
            row["ObsNum"],
            row["SubObsNum"],
            row["ScanNum"],
        ), row["FileName"]
        suffix, kind = kinds[row["kind_label"]]
        assert (info.suffix, info.ext) == (suffix, "nc")
        source = get_info_from_source(row["FileName"])
        assert source is not None
        assert source.file_prefix == info.prefix
        assert source.file_ut == info.ut
        assert source.obsnum == info.obsnum
        assert source.data_kind == kind.value
        assert info.ut is not None
        assert (
            type(info).model_validate_json(info.model_dump_json(round_trip=True))
            == info
        )


def test_filename_ut_is_not_database_time(toltecdb_records):
    row = next(
        row
        for row in toltecdb_records
        if row["ObsNum"] == 145611 and row["ScanNum"] == 1 and row["RoachIndex"] == 3
    )
    info = FilepathInfo.from_source(row["FileName"])
    assert isinstance(info, ToltecFilepathInfo)
    assert info.ut == datetime(2025, 10, 31, 9, 36, 8, tzinfo=UTC)
    assert row["Time"] == "09:36:09"


def test_recorded_filename_fields_match_v2(lmtmc_records, toltecdb_records):
    interpreter = os.environ.get("TOLTECA_V2_PYTHON")
    if not interpreter:
        pytest.skip("Set TOLTECA_V2_PYTHON to compare the installed v2 parser")
    paths = [row["FileName"] for row in [*lmtmc_records, *toltecdb_records]]
    result = subprocess.run(  # noqa: S603 -- explicit local comparison interpreter
        [
            interpreter,
            "-c",
            """
import json
import sys
from tolteca_datamodels.toltec.file import SourceInfoModel

fields = ("interface", "roach", "obsnum", "subobsnum", "scannum",
          "file_timestamp", "file_suffix", "file_ext")
records = [SourceInfoModel.model_validate(path) for path in json.load(sys.stdin)]
print(json.dumps([{key: getattr(record, key) for key in fields}
                  for record in records], default=str))
""",
        ],
        input=json.dumps(paths),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    for path, old in zip(paths, json.loads(result.stdout), strict=True):
        info = FilepathInfo.from_source(path)
        assert isinstance(info, (LmtFilepathInfo, ToltecFilepathInfo))
        assert info.prefix == old["interface"], path
        for field in ("roach", "obsnum", "subobsnum", "scannum"):
            assert getattr(info, field, None) == old[field], (path, field)
        # Filename time is independent of header/catalog observation timestamps.
        assert info.ut == datetime.fromisoformat(old["file_timestamp"]), path
        assert info.suffix == old["file_suffix"], path
        assert info.ext == old["file_ext"].removeprefix("."), path


def test_recorded_catalog_selection(lmtmc_records, toltecdb_records, monkeypatch):
    paths = [row["FileName"] for row in [*lmtmc_records, *toltecdb_records]]

    def unexpected(*args, **kwargs):
        pytest.fail("Filename-only catalog opened a source")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", unexpected)
        catalog = guess_info_from_sources(paths)
    assert len(catalog) == 132
    telescopes = catalog.toltec_file.query('interface == "tel"')
    assert [
        str(fileloc(r.source).path) for r in list(telescopes.itertuples(index=False))
    ] == [row["FileName"] for row in lmtmc_records]
    assert catalog.toltec_file.query("data_kind == lmt_tel.value").index.equals(
        telescopes.index
    )
    sweeps = catalog.toltec_file.query('file_suffix == "vnasweep"')
    assert [
        str(fileloc(r.source).path) for r in list(sweeps.itertuples(index=False))
    ] == [row["FileName"] for row in toltecdb_records if row["kind_label"] == "VNA"]
    assert catalog.toltec_file.query("data_kind == vnasweep.value").index.equals(
        sweeps.index
    )
