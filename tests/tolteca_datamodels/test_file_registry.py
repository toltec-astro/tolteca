"""Normalized filename keys and eager parser records."""

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pytest

import tolteca_datamodels.file as file_module
from tolteca_datamodels.file import (
    FileIdentifier,
    FilepathInfo,
    FilepathInfoMatch,
)
from tolteca_datamodels.toltec.file import _parse_master_str
from tolteca_datamodels.toltec.types import ToltecDataKind


def test_toltec_master_parser_uses_interface_without_filename_prefix(toltecdb_records):
    row = toltecdb_records[0]
    path = Path(row["FileName"])
    fields = {"interface": f"toltec{row['RoachIndex']}"}
    assert _parse_master_str(fields, path) == {
        "master_str": row["master_label"].lower()
    }


def test_ignored_absent_and_empty_values_are_distinct():
    keys = [
        FilepathInfoMatch(ext="nc"),
        FilepathInfoMatch(ext="nc", suffix=None),
        FilepathInfoMatch(ext="nc", suffix=""),
    ]
    values = dict(zip(keys, ["ignored", "absent", "empty"], strict=True))
    assert len(values) == 3
    assert [values[key] for key in keys] == ["ignored", "absent", "empty"]
    assert keys[0].suffix is ...
    assert keys[0].prefix is ...


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (
            "tel_toltec_2025-10-31_145609_00_0002.nc",
            datetime(2025, 10, 31, tzinfo=UTC),
        ),
        (
            "toltec8_145609_000_0002_2025_10_31_09_27_57.nc",
            datetime(2025, 10, 31, 9, 27, 57, tzinfo=UTC),
        ),
    ],
)
def test_pattern_parsers_run_once(monkeypatch, filename, expected):
    calls = []
    parser = next(
        parser
        for pattern, parser in FileIdentifier.filename_pattern_parser.items()
        if pattern.match(filename)
    )

    def trace(index, parse):
        def wrapped(fields, path):
            calls.append(index)
            before = fields.copy()
            result = parse(fields, path)
            assert fields == before
            return result

        return wrapped

    monkeypatch.setattr(
        FileIdentifier,
        "filename_pattern_parser",
        {
            pattern: parser._replace(
                parsers=tuple(trace(i, parse) for i, parse in enumerate(parser.parsers))
            )
            for pattern, parser in FileIdentifier.filename_pattern_parser.items()
        },
    )
    info = FilepathInfo.from_path(filename)
    assert info is not None
    assert calls == list(range(len(parser.parsers)))
    assert info.ut == expected
    assert (info.obsnum, info.subobsnum, info.scannum) == (145609, 0, 2)
    calls.clear()
    _ = info.interface, info.instru, info.instru_component, info.ut
    info.model_dump()
    info.model_dump_json()
    assert type(info).model_validate_json(info.model_dump_json(round_trip=True)) == info
    assert calls == []


def test_ordered_parsers_share_captures_and_see_previous_updates(monkeypatch):
    name = "tel_toltec_2025-10-31_145609_00_0002.nc"
    expected = FilepathInfo.from_path(name)
    pattern, parser = next(
        (pattern, parser)
        for pattern, parser in FileIdentifier.filename_pattern_parser.items()
        if pattern.match(name)
    )
    captures = {}
    extract = file_module.dict_from_regex_match

    def trace_extract(*args, **kwargs):
        nonlocal captures
        captures = extract(*args, **kwargs)
        return captures

    def increment_obsnum(fields, path):
        assert fields is captures
        assert fields["prefix"] == "tel_toltec"
        assert path == Path.cwd() / name
        return {"obsnum": int(fields["obsnum"]) + 1}

    def restore_obsnum(fields, path):
        assert fields is captures
        assert fields["obsnum"] == 145610
        assert fields["suffix"] is None
        return {"obsnum": fields["obsnum"] - 1}

    monkeypatch.setattr(file_module, "dict_from_regex_match", trace_extract)
    monkeypatch.setattr(
        FileIdentifier,
        "filename_pattern_parser",
        {
            pattern: parser._replace(
                parsers=(increment_obsnum, restore_obsnum, *parser.parsers)
            )
        },
    )
    assert FilepathInfo.from_path(name) == expected
    assert captures is not None
    assert captures["interface"] == "tel"
    assert captures["obsnum"] == 145609


@pytest.mark.parametrize(
    ("name", "ut_invalid"),
    [
        ("tel_toltec_2025-10-31_145609_00_0002.nc", "2025_10_31_09_27_57"),
        ("toltec8_145609_000_0002_2025_10_31_09_27_57.nc", "2025-10-31"),
    ],
)
def test_ut_parser_is_specific_to_the_matched_pattern(monkeypatch, name, ut_invalid):
    extract = file_module.dict_from_regex_match

    def invalid_capture(*args, **kwargs):
        fields = extract(*args, **kwargs)
        if fields is not None:
            fields["ut"] = ut_invalid
        return fields

    monkeypatch.setattr(file_module, "dict_from_regex_match", invalid_capture)
    with pytest.raises(ValueError):
        FilepathInfo.from_path(name)


@pytest.mark.parametrize(
    "first",
    [
        "tolteca_datamodels.file",
        "tolteca_datamodels.lmt.file",
        "tolteca_datamodels.toltec.file",
        "tolteca_datamodels.lmt.metadata",
        "tolteca_datamodels.toltec.metadata",
        "tolteca_datamodels.lmt_base.file",
        "tolteca_datamodels.lmt_base.metadata",
    ],
)
def test_import_order_and_one_time_assembly(first):
    script = """
import importlib
import sys

importlib.import_module(sys.argv[1])
from tolteca_datamodels.file import FilepathInfo, FileIdentifier

assert len(FileIdentifier.filename_pattern_parser) == 4
patterns_before = dict(FileIdentifier.filename_pattern_parser)
kinds_before = dict(FileIdentifier.filepath_info_data_kind)
for module in ("tolteca_datamodels.file", "tolteca_datamodels.lmt.file",
               "tolteca_datamodels.toltec.file"):
    importlib.import_module(module)
assert FileIdentifier.filename_pattern_parser == patterns_before
assert FileIdentifier.filepath_info_data_kind == kinds_before

names = (
    "tel_toltec_2025-10-31_145609_00_0002.nc",
    "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
)
for name in names:
    assert FilepathInfo.from_path(name) is not None
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script, first],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_each_recorded_name_has_one_parser(lmtmc_records, toltecdb_records):
    for row in [*lmtmc_records, *toltecdb_records]:
        name = row["FileName"].rsplit("/", 1)[-1]
        assert (
            sum(
                pattern.fullmatch(name) is not None
                for pattern in FileIdentifier.filename_pattern_parser
            )
            == 1
        )


def test_shared_record_defers_registry_assembly_until_parsing():
    script = """
import sys
from tolteca_datamodels.lmt_base import FilepathInfo
from tolteca_datamodels.lmt_base.metadata import LmtRawObsMetadata

assert "tolteca_datamodels.file" not in sys.modules
info = FilepathInfo.from_path("tel_toltec_2025-10-31_145609_00_0002.nc")
assert info.interface == "tel"
from tolteca_datamodels.file import FilepathInfo as PublicFilepathInfo
from tolteca_datamodels.toltec.file import RoachFilepathInfo
assert FilepathInfo is PublicFilepathInfo
assert issubclass(RoachFilepathInfo, FilepathInfo)
assert isinstance(FilepathInfo.from_path(
    "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
), RoachFilepathInfo)
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_toltec_owns_all_data_kind_declarations():
    script = """
import importlib
from tolteca_datamodels.file import FileIdentifier

kinds = dict(FileIdentifier.filepath_info_data_kind)
FileIdentifier.filepath_info_data_kind.clear()
importlib.reload(importlib.import_module("tolteca_datamodels.lmt.file"))
assert not FileIdentifier.filepath_info_data_kind
importlib.reload(importlib.import_module("tolteca_datamodels.toltec.file"))
assert FileIdentifier.filepath_info_data_kind == kinds
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("suffix", "kind"),
    [
        ("vnasweep", ToltecDataKind.VnaSweep),
        ("targsweep", ToltecDataKind.TargetSweep),
        ("targetsweep", ToltecDataKind.TargetSweep),
        ("tune", ToltecDataKind.Tune),
        ("vnasweep_processed", ToltecDataKind.ReducedVnaSweep),
        ("targsweep_processed", ToltecDataKind.ReducedTargetSweep),
        ("tune_processed", ToltecDataKind.ReducedTargetSweep),
        ("sweep", ToltecDataKind.ReducedSweep),
        ("timestream_processed", ToltecDataKind.SolvedTimeStream),
        ("timestream", ToltecDataKind.RawTimeStream),
        (None, ToltecDataKind.RawTimeStream),
        ("", ToltecDataKind.Unknown),
    ],
)
def test_netcdf_kind_uses_the_full_suffix(suffix, kind):
    # Classification vocabulary, not evidence of additional acquisition formats.
    assert (
        FileIdentifier.guess_data_kinds(
            pa.table(
                {
                    "instru": ["toltec"],
                    "instru_component": ["roach"],
                    "suffix": [suffix],
                    "ext": ["nc"],
                }
            )
        )[0]
        is kind
    )


@pytest.mark.parametrize(
    ("proc_suffix", "ext", "kind"),
    [
        ("targamps", "dat", ToltecDataKind.TargAmpsDat),
        ("kids_find", "ecsv", ToltecDataKind.KidsPropTable),
        ("kids_fit", "ecsv", ToltecDataKind.KidsPropTable),
        ("chan_prop", "ecsv", ToltecDataKind.KidsPropTable),
        ("adrv", "ecsv", ToltecDataKind.KidsPropTable),
    ],
)
def test_product_kind_has_one_key_per_processing_tag(proc_suffix, ext, kind):
    assert (
        FileIdentifier.guess_data_kinds(
            pa.table(
                {
                    "instru": ["toltec"],
                    "instru_component": ["roach"],
                    "proc_suffix": [proc_suffix],
                    "ext": [ext],
                }
            )
        )[0]
        is kind
    )
    keys = [
        key
        for key in FileIdentifier.filepath_info_data_kind
        if key.ext == ext and key.proc_suffix == proc_suffix
    ]
    assert keys == [
        FilepathInfoMatch(
            instru="toltec", instru_component="roach", proc_suffix=proc_suffix, ext=ext
        )
    ]
    assert keys[0].suffix is ...


def test_classification_accepts_parsed_records_without_reparsing(monkeypatch):
    roach = FilepathInfo.from_path(
        "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
    )
    tel = FilepathInfo.from_path("tel_toltec_2025-10-31_145609_00_0002.nc")
    assert roach is not None and tel is not None

    def unexpected(*args, **kwargs):
        pytest.fail("Metadata classification attempted filename parsing")

    monkeypatch.setattr(FilepathInfo, "from_path", unexpected)
    assert FileIdentifier.guess_data_kind(roach) == ToltecDataKind.VnaSweep
    assert FileIdentifier.guess_data_kind(tel) == ToltecDataKind.LmtTel
    assert FileIdentifier.guess_data_kinds(pa.table({"ext": [None]})) == [
        ToltecDataKind.Unknown
    ]
