"""Normalized filename keys and eager parser records."""

import subprocess
import sys
from datetime import UTC, datetime

import pytest

import tolteca_datamodels.file as file_module
from tolteca_datamodels.file import DataKindKey, FilepathInfo, guess_data_kind
from tolteca_datamodels.toltec.types import ToltecDataKind


def test_ignored_absent_and_empty_values_are_distinct():
    keys = [
        DataKindKey(ext="nc"),
        DataKindKey(ext="nc", suffix=None),
        DataKindKey(ext="nc", suffix=""),
    ]
    values = dict(zip(keys, ["ignored", "absent", "empty"], strict=True))
    assert len(values) == 3
    assert [values[key] for key in keys] == ["ignored", "absent", "empty"]
    assert keys[0].suffix is ...
    assert keys[0].prefix is ...


def test_lookup_requires_the_same_normalized_key():
    key = DataKindKey(instru="device", suffix="sweep", ext="nc")
    kinds = {key: "sweep"}
    assert kinds[DataKindKey(instru="device", suffix="sweep", ext="nc")] == "sweep"
    # Ellipsis is an ordinary key value, not automatic wildcard matching.
    assert kinds.get(key._replace(prefix="device0")) is None


def test_capture_dispatch_converts_time_and_numbers_once(monkeypatch):
    calls = []

    def trace(name, convert):
        def wrapped(value):
            calls.append(name)
            return convert(value)

        return wrapped

    monkeypatch.setattr(
        file_module,
        "filename_field_types",
        {
            name: trace(name, convert)
            for name, convert in file_module.filename_field_types.items()
        },
    )
    info = FilepathInfo.from_path("tel_toltec_2025-10-31_145609_00_0002.nc")
    assert info is not None
    assert sorted(calls) == ["obsnum", "scannum", "subobsnum", "ut"]
    assert info.ut == datetime(2025, 10, 31, tzinfo=UTC)
    assert (info.obsnum, info.subobsnum, info.scannum) == (145609, 0, 2)
    calls.clear()
    _ = info.interface, info.instru, info.instru_component, info.ut
    info.model_dump()
    info.model_dump_json()
    assert calls == []


@pytest.mark.parametrize(
    "first",
    [
        "tolteca_datamodels.file",
        "tolteca_datamodels.lmt.file",
        "tolteca_datamodels.toltec.file",
        "tolteca_datamodels.lmt.metadata",
    ],
)
def test_import_order_and_one_time_assembly(first):
    script = """
import importlib
import sys

importlib.import_module(sys.argv[1])
from tolteca_datamodels.file import FilepathInfo, filename_patterns, data_kind_map

assert len(filename_patterns) == 4
patterns_before = dict(filename_patterns)
kinds_before = dict(data_kind_map)
for module in ("tolteca_datamodels.file", "tolteca_datamodels.lmt.file",
               "tolteca_datamodels.toltec.file"):
    importlib.import_module(module)
assert filename_patterns == patterns_before
assert data_kind_map == kinds_before

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
                for pattern in file_module.filename_patterns
            )
            == 1
        )


def test_classification_accepts_metadata_without_a_filepath(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Metadata classification attempted filename parsing")

    monkeypatch.setattr(FilepathInfo, "from_path", unexpected)
    assert (
        guess_data_kind(
            instru="toltec", instru_component="roach", suffix="vnasweep", ext="nc"
        )
        == ToltecDataKind.VnaSweep
    )
    assert guess_data_kind(prefix="tel_toltec", ext="nc") == ToltecDataKind.LmtTel
    assert guess_data_kind() == ToltecDataKind.Unknown
