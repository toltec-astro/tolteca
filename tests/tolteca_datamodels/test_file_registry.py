"""Normalized filename keys and eager parser records."""

import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime

import pytest

import tolteca_datamodels.file as file_module
from tolteca_datamodels.file import FileKindKey, FilepathInfo


def test_ignored_absent_and_empty_values_are_distinct():
    keys = [
        FileKindKey(ext="nc"),
        FileKindKey(ext="nc", suffix=None),
        FileKindKey(ext="nc", suffix=""),
    ]
    values = dict(zip(keys, ["ignored", "absent", "empty"], strict=True))
    assert len(values) == 3
    assert [values[key] for key in keys] == ["ignored", "absent", "empty"]
    assert keys[0].suffix is ...
    assert keys[0].prefix is ...


def test_lookup_requires_the_same_normalized_key():
    key = FileKindKey(instru="device", suffix="sweep", ext="nc")
    kinds = {key: "sweep"}
    assert kinds[FileKindKey(instru="device", suffix="sweep", ext="nc")] == "sweep"
    # Ellipsis is an ordinary key value, not automatic wildcard matching.
    assert kinds.get(key._replace(prefix="device0")) is None


def test_capture_dispatch_converts_time_and_numbers_once(monkeypatch):
    calls = []

    def trace(name, convert):
        def wrapped(value):
            calls.append(name)
            return convert(value)

        return wrapped

    specs = tuple(
        replace(
            spec,
            type_dispatcher={
                name: trace(name, convert)
                for name, convert in spec.type_dispatcher.items()
            },
        )
        for spec in file_module._get_file_specs()
    )
    monkeypatch.setattr(file_module, "_get_file_specs", lambda: specs)
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
from tolteca_datamodels.file import FilepathInfo, _get_file_specs

names = (
    "tel_toltec_2025-10-31_145609_00_0002.nc",
    "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
)
for name in names:
    assert FilepathInfo.from_path(name) is not None
assert _get_file_specs.cache_info().misses == 1
assert _get_file_specs() is _get_file_specs()
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script, first],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_each_recorded_name_has_one_parser(lmtmc_records, toltecdb_records):
    specs = file_module._get_file_specs()
    for row in [*lmtmc_records, *toltecdb_records]:
        name = row["FileName"].rsplit("/", 1)[-1]
        assert sum(spec.pattern.fullmatch(name) is not None for spec in specs) == 1
