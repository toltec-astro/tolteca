"""Normalized filename keys and eager parser records."""

import re
import subprocess
import sys
from datetime import UTC, datetime
from itertools import permutations
from pathlib import Path
from typing import assert_type

import pytest
from tollan.accessor.pandas.model_table import ModelColumn
from tollan.config import RuntimeContext
from tollan.utils.fileloc import fileloc

import tolteca_datamodels.core.filepath as file_module
from tolteca_datamodels.core.filepath import register_filepath_parser
from tolteca_datamodels.core.source import SourceOpener
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.input_discovery import InputDiscovery
from tolteca_datamodels.toltec import file as toltec_file
from tolteca_datamodels.toltec.filepath import (
    ToltecFilepathInfo,
    ToltecLmtFilepathInfo,
)
from tolteca_datamodels.toltec.types import ToltecDataKind


@pytest.fixture
def registry(monkeypatch):
    """Isolate parser registrations."""
    monkeypatch.setattr(file_module, "_filepath_parsers", [])
    return file_module


def test_registration_is_explicit_and_selects_result_model(registry):

    class Filename(FilepathInfo):
        number: int

    class SpecificFilename(Filename):
        pass

    assert FilepathInfo.from_source("example-1", raise_on_error=False) is None
    register_filepath_parser(
        pattern=r"^example-(?P<number>\d+)$", parsers=(), filepath_info_cls=Filename
    )
    assert type(FilepathInfo.from_source("example-1")) is Filename
    register_filepath_parser(
        pattern=r"^example-(?P<number>\d+)$",
        parsers=(),
        filepath_info_cls=SpecificFilename,
    )
    assert type(FilepathInfo.from_source("example-1")) is SpecificFilename


@pytest.mark.parametrize("order", list(permutations(range(4))))
def test_factory_selects_most_specialized_match(registry, order):

    class A(FilepathInfo):
        number: int

    class B(A):
        pass

    class C(B):
        pass

    class D(C):
        pass

    models = (A, B, C, D)
    for level in order:
        register_filepath_parser(
            pattern=rf"^example-(?P<number>[{level}-3])$",
            parsers=(),
            filepath_info_cls=models[level],
        )

    # Higher numbers match progressively deeper classes, independent of order.
    for number, expected in enumerate(models):
        source = f"example-{number}"
        result = FilepathInfo.from_source(source)
        assert type(result) is expected
        assert isinstance(result, A)
        assert result.number == number


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("raise_on_error", [False, True])
def test_unrelated_matches_raise_before_conversion(registry, reverse, raise_on_error):

    class First(FilepathInfo):
        pass

    class Second(FilepathInfo):
        pass

    def unexpected(data, path):
        pytest.fail("Ambiguous filename ran a field converter")

    models = (Second, First) if reverse else (First, Second)
    for model in models:
        register_filepath_parser(
            pattern=r"^ambiguous\.nc$", parsers=(unexpected,), filepath_info_cls=model
        )
    with pytest.raises(RuntimeError, match="Ambiguous acquisition filename"):
        FilepathInfo.from_source("ambiguous.nc", raise_on_error=raise_on_error)


@pytest.mark.parametrize("on_error", ["record", "raise"])
@pytest.mark.parametrize("entry_point", ["accessor", "discovery"])
def test_filename_ambiguity_propagates_through_batch_workflows(
    registry, on_error, entry_point
):
    class First(FilepathInfo):
        pass

    class Second(FilepathInfo):
        pass

    for model in (First, Second):
        register_filepath_parser(
            pattern=r"^ambiguous\.nc$", parsers=(), filepath_info_cls=model
        )

    if entry_point == "accessor":
        with pytest.raises(RuntimeError, match="Ambiguous acquisition filename"):
            FileAccessor.from_sources(["ambiguous.nc"], on_error=on_error)
    else:
        discovery = InputDiscovery(
            RuntimeContext(
                {"input_discovery": {"inputs": [{"sources": ["ambiguous.nc"]}]}}
            )
        )
        with pytest.raises(RuntimeError, match="Ambiguous acquisition filename"):
            discovery.discover(on_error=on_error)


def test_identical_registration_is_idempotent(registry):

    class Filename(FilepathInfo):
        pass

    for pattern in (r"^example\.nc$", re.compile(r"^example\.nc$")):
        register_filepath_parser(
            pattern=pattern, parsers=(), filepath_info_cls=Filename
        )
    assert type(FilepathInfo.from_source("example.nc")) is Filename


@pytest.mark.parametrize("invalid", [object(), str, None])
def test_registration_rejects_non_filepath_models(registry, invalid):
    with pytest.raises(TypeError, match="FilepathInfo subclass"):
        register_filepath_parser(
            pattern="example", parsers=(), filepath_info_cls=invalid
        )
    assert registry._filepath_parsers == []


def test_filename_models_reuse_their_own_prepared_columns(registry, monkeypatch):
    class First(FilepathInfo):
        number: int

    class Second(First):
        label: str

    register_filepath_parser(
        pattern=r"^first-(?P<number>\d+)$", parsers=(), filepath_info_cls=First
    )
    number_column = First._columns["file_number"]
    register_filepath_parser(
        pattern=r"^second-(?P<number>\d+)-(?P<label>[a-z]+)$",
        parsers=(),
        filepath_info_cls=Second,
    )
    assert Second._columns["file_number"] == number_column
    assert set(First._columns) == {"file_number"}
    assert set(Second._columns) == {"file_number", "file_label"}

    def unexpected(*args, **kwargs):
        pytest.fail("Table construction rebuilt registered filename columns")

    monkeypatch.setattr(registry, "_filepath_column_mapper", unexpected)
    register_filepath_parser(
        pattern=r"^first-(?P<number>\d+)$", parsers=(), filepath_info_cls=First
    )
    for _ in range(2):
        table = FileAccessor.from_sources(["first-1", "second-2-a"], accessor_types=())
        assert table.file_number.tolist() == [1, 2]
        assert table.file_label.isna().tolist() == [True, False]
        assert table.file_label.iloc[1] == "a"


def test_incompatible_destination_values_fail_at_population_not_registration(registry):
    class First(FilepathInfo):
        number: int

    class Second(FilepathInfo):
        label: str
        number: str

    register_filepath_parser(
        pattern=r"^first-(?P<number>\d+)$", parsers=(), filepath_info_cls=First
    )
    register_filepath_parser(
        pattern=r"^second-(?P<label>[a-z]+)-(?P<number>[a-z]+)$",
        parsers=(),
        filepath_info_cls=Second,
    )
    first = FilepathInfo.from_source("first-1")
    second = FilepathInfo.from_source("second-a-text")
    assert isinstance(first, First)
    assert isinstance(second, Second)
    assert first.number == 1
    assert second.number == "text"
    with pytest.raises((TypeError, ValueError)):
        FileAccessor.from_sources(["first-1", "second-a-text"], accessor_types=())


def test_filename_projections_can_supply_one_column_from_different_attributes(registry):
    class First(FilepathInfo):
        number: int

    class Second(FilepathInfo):
        label: str

    First._columns = {"file_number": ModelColumn("number", dtype="Int64")}
    Second._columns = {
        "file_number": ModelColumn("label", dtype="Int64", converter=int)
    }
    register_filepath_parser(
        pattern=r"^first-(?P<number>\d+)$", parsers=(), filepath_info_cls=First
    )
    register_filepath_parser(
        pattern=r"^second-(?P<label>\d+)$", parsers=(), filepath_info_cls=Second
    )
    table = FileAccessor.from_sources(
        ["second-2", "first-1", "unrecognized", "second-3"], accessor_types=()
    )
    assert table.file_number.take([0, 1, 3]).tolist() == [2, 1, 3]
    assert table.file_number.isna().tolist() == [False, False, True, False]
    assert str(table.file_number.dtype) == "Int64"


def test_opener_rules_use_projected_names_not_reader_attributes(registry, monkeypatch):

    class Filename(FilepathInfo):
        format_name: str

    Filename._columns = {"file_ext": ModelColumn("format_name", dtype="string")}
    register_filepath_parser(
        pattern=r"^cached-(?P<format_name>ecsv)$",
        parsers=(),
        filepath_info_cls=Filename,
    )

    def opener(path):
        pytest.fail("Matching an opener must not open a file")

    monkeypatch.setattr(
        FileAccessor,
        "_source_opener",
        SourceOpener(rules=(({"ext": "ecsv"}, opener),)),
    )
    table = FileAccessor.from_sources(["cached-ecsv"], accessor_types=())
    assert FileAccessor(table, accessor_types=())._get_opener_choices() == [
        {FileAccessor: opener}
    ]


def test_standalone_lmt_parsing_and_registered_specializations():
    script = """
import sys
from tolteca_datamodels.lmt.filepath import LmtFilepathInfo
from tolteca_datamodels.core.filepath import FilepathInfo

assert 'tolteca_datamodels.toltec' not in sys.modules
assert 'tolteca_dummy_instru' not in sys.modules
names = [f'{p}_2025-10-31_145609_00_0002.nc'
         for p in ('tel_toltec', 'tel_dummy_instru', 'tel_unregistered')]
before = [FilepathInfo.from_source(name) for name in names]
assert all(type(info) is LmtFilepathInfo for info in before)
assert all(info.data_models == ['lmt'] for info in before)
assert all(type(FilepathInfo.from_source(name)) is LmtFilepathInfo for name in names)

from tolteca_datamodels.toltec.filepath import ToltecLmtFilepathInfo
from tolteca_dummy_instru.filepath import DummyInstruLmtFilepathInfo
after = [FilepathInfo.from_source(name) for name in names]
assert [type(info) for info in after] == [
    ToltecLmtFilepathInfo, DummyInstruLmtFilepathInfo, LmtFilepathInfo]
assert [info.data_models for info in after] == [
    ['lmt', 'toltec'], ['lmt', 'dummy_instru'], ['lmt']]
assert all(info.instru == 'lmt' for info in after)
assert [FilepathInfo.from_source(name) for name in names] == after
assert [info.model_dump(exclude={'data_models'}) for info in after] == [
    info.model_dump(exclude={'data_models'}) for info in before]
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("as_fileloc", [False, True])
@pytest.mark.parametrize(
    "prefix", ["", "host:/data_lmt/tel/", "https://example.org/tel/"]
)
def test_from_source_selects_registered_subclass(prefix, as_fileloc):
    source = f"{prefix}tel_toltec_2025-10-31_145609_00_0002.nc"
    if as_fileloc:
        source = fileloc(source)
    info = FilepathInfo.from_source(source)
    assert_type(info, FilepathInfo | None)
    assert type(info) is ToltecLmtFilepathInfo


@pytest.mark.parametrize("raise_on_error", [False, True])
@pytest.mark.parametrize(
    ("source", "error"),
    [
        ("tel_toltec_2025-02-30_145609_00_0002.nc", ValueError),
        (
            "toltec13_145609_000_0002_2025_10_31_09_27_57.nc",
            KeyError,
        ),
    ],
)
def test_recognized_filename_errors_propagate(source, error, raise_on_error):
    # Impossible dates and out-of-range boards are invalid acquisition names.
    with pytest.raises(error):
        FilepathInfo.from_source(source, raise_on_error=raise_on_error)


def test_toltec_master_parser_uses_interface_without_filename_prefix(toltecdb_records):
    row = toltecdb_records[0]
    path = Path(row["FileName"])
    filepath_data = {"interface": f"toltec{row['RoachIndex']}"}
    assert ToltecFilepathInfo._parse_master(filepath_data, path) == {
        "master": row["master_label"].lower()
    }


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
    validated = []
    parser = next(
        parser
        for parser in file_module._filepath_parsers
        if parser.pattern.match(filename)
        and parser.filepath_info_cls is type(FilepathInfo.from_source(filename))
    )

    def trace(index, parse):
        def wrapped(filepath_data, path):
            calls.append(index)
            before = filepath_data.copy()
            result = parse(filepath_data, path)
            assert filepath_data == before
            return result

        return wrapped

    validate = parser.filepath_info_cls.model_validate

    def validate_once(captures, **kwargs):
        validated.append(captures)
        return validate(captures, **kwargs)

    monkeypatch.setattr(parser.filepath_info_cls, "model_validate", validate_once)
    monkeypatch.setattr(
        file_module,
        "_filepath_parsers",
        [
            parser._replace(
                parsers=tuple(trace(i, parse) for i, parse in enumerate(parser.parsers))
            )
            for parser in file_module._filepath_parsers
        ],
    )
    info = FilepathInfo.from_source(filename)
    assert isinstance(info, (ToltecLmtFilepathInfo, ToltecFilepathInfo))
    assert len(validated) == 1
    assert calls == list(range(len(parser.parsers)))
    assert info.ut == expected
    assert (info.obsnum, info.subobsnum, info.scannum) == (145609, 0, 2)
    calls.clear()
    _ = info.interface, info.instru, info.interface_kind, info.ut
    info.model_dump()
    info.model_dump_json()
    assert type(info).model_validate_json(info.model_dump_json(round_trip=True)) == info
    assert calls == []


def test_ordered_parsers_share_captures_and_see_previous_updates(monkeypatch):
    name = "tel_toltec_2025-10-31_145609_00_0002.nc"
    expected = FilepathInfo.from_source(name)
    parser = next(
        parser
        for parser in file_module._filepath_parsers
        if parser.filepath_info_cls is type(expected) and parser.pattern.match(name)
    )
    captures = None

    def increment_obsnum(filepath_data, path):
        nonlocal captures
        captures = filepath_data
        assert filepath_data["prefix"] == "tel_toltec"
        assert path == Path.cwd() / name
        return {"obsnum": int(filepath_data["obsnum"]) + 1}

    def restore_obsnum(filepath_data, path):
        assert filepath_data is captures
        assert filepath_data["obsnum"] == 145610
        assert filepath_data["suffix"] is None
        return {"obsnum": filepath_data["obsnum"] - 1}

    monkeypatch.setattr(
        file_module,
        "_filepath_parsers",
        [parser._replace(parsers=(increment_obsnum, restore_obsnum, *parser.parsers))],
    )
    assert FilepathInfo.from_source(name) == expected
    assert captures is not None
    assert captures["interface"] == "tel"
    assert captures["obsnum"] == 145609


@pytest.mark.parametrize(
    "first",
    [
        "tolteca_datamodels.file",
        "tolteca_datamodels.core.filepath",
        "tolteca_datamodels.lmt.filepath",
        "tolteca_datamodels.toltec.filepath",
        "tolteca_datamodels.lmt",
        "tolteca_datamodels.toltec",
        "tolteca_datamodels.toltec.file",
        "tolteca_datamodels.lmt.metadata",
        "tolteca_datamodels.toltec.metadata",
        "tolteca_datamodels.lmt_base",
        "tolteca_datamodels.lmt_base.filepath",
        "tolteca_datamodels.lmt_base.metadata",
    ],
)
def test_import_order_and_one_time_assembly(first):
    script = """
import importlib
import sys

importlib.import_module(sys.argv[1])
import tolteca_datamodels.core.filepath as file_module
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.core.types import DataModelInfo
from tolteca_datamodels.toltec.file import ToltecFileAccessor, guess_info_from_sources

name = "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
info = FilepathInfo.from_source(name)
assert info.roach == 8

class TestDataModel(DataModelInfo):
    name = "test"

class TestFiles(FileAccessor[TestDataModel]):
    pass

table = TestFiles.from_sources([])
assert {
    "source", "source_loc", "source_loc_is_local",
    "source_loc_host", "source_loc_scheme", "source_trace", "file_roach"
} <= set(table)
assert "data_kind_toltec" not in table
table = guess_info_from_sources([])
assert "file_roach" in table
columns = ToltecFileAccessor._metadata_columns
assert columns["obsnum"].attr == "obsnum"
assert columns is ToltecFileAccessor._metadata_columns

from tolteca_datamodels.toltec import file as toltec_file

patterns_before = list(file_module._filepath_parsers)
kinds_before = (
    toltec_file.ToltecFileAccessor.from_sources([name]).data_kind_toltec.tolist()
)
for module in ("tolteca_datamodels.file", "tolteca_datamodels.lmt",
               "tolteca_datamodels.toltec.file"):
    importlib.import_module(module)
assert file_module._filepath_parsers == patterns_before
assert (
    toltec_file.ToltecFileAccessor.from_sources([name]).data_kind_toltec.tolist()
    == kinds_before
)

names = (
    "tel_toltec_2025-10-31_145609_00_0002.nc",
    "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
)
for name in names:
    assert FilepathInfo.from_source(name) is not None
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script, first],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_recorded_names_resolve_to_instrument_result(lmtmc_records, toltecdb_records):
    for row in [*lmtmc_records, *toltecdb_records]:
        name = row["FileName"].rsplit("/", 1)[-1]
        info = FilepathInfo.from_source(name)
        expected = (
            ToltecLmtFilepathInfo if name.startswith("tel_") else ToltecFilepathInfo
        )
        assert type(info) is expected


def test_classification_is_owned_by_explicit_instrument_handlers():
    script = """
import sys
from tolteca_datamodels.core.file import FileAccessor
from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.lmt.file import LmtFileAccessor

name = "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
parsed = FilepathInfo.from_source(name)
assert parsed.data_models == ["toltec"]
assert "tolteca_datamodels.toltec.file" not in sys.modules
assert not hasattr(FilepathInfo, "filepath_info_data_kind")
for cls in (FileAccessor, LmtFileAccessor):
    table = cls.from_sources([name])
    assert table.file_prefix.tolist() == ["toltec8"]
    assert "data_kind_toltec" not in table
    assert "data_kind_dummy_instru" not in table

from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.types import ToltecDataKind

assert ToltecFileAccessor.from_sources([name]).data_kind_toltec.tolist() == [
    ToltecDataKind.vnasweep.value
]
assert FilepathInfo.from_source(name) == parsed

from tolteca_dummy_instru.file import DummyInstruFileAccessor
from tolteca_dummy_instru.types import DummyInstruDataKind
from tolteca_dummy_instru import filepath

table = FileAccessor.from_sources(
    [name, "chan01-tcs-1-0-2-20251031T120000.nc", "unrecognized.nc"],
    accessor_types=(ToltecFileAccessor, DummyInstruFileAccessor),
)
assert table.data_kind_toltec.tolist() == [
    ToltecDataKind.vnasweep.value,
    ToltecDataKind.unknown.value,
    ToltecDataKind.unknown.value,
]
assert table.data_kind_dummy_instru.tolist() == [
    DummyInstruDataKind.unknown.value,
    DummyInstruDataKind.counts.value,
    DummyInstruDataKind.unknown.value,
]
assert "data_kind" not in table
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
        ("vnasweep", ToltecDataKind.vnasweep),
        ("targsweep", ToltecDataKind.targsweep),
        ("targetsweep", ToltecDataKind.targsweep),
        ("tune", ToltecDataKind.tune),
        ("vnasweep_processed", ToltecDataKind.reduced_vnasweep),
        ("targsweep_processed", ToltecDataKind.reduced_targsweep),
        ("tune_processed", ToltecDataKind.reduced_targsweep),
        ("sweep", ToltecDataKind.reduced_sweep),
        ("timestream_processed", ToltecDataKind.solved_timestream),
        ("timestream", ToltecDataKind.raw_timestream),
        (None, ToltecDataKind.raw_timestream),
        ("", ToltecDataKind.unknown),
    ],
)
def test_netcdf_kind_uses_the_full_suffix(suffix, kind):
    # Classification vocabulary, not evidence of additional acquisition formats.
    table = toltec_file.guess_info_from_sources(
        ["toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"]
    )
    table["file_suffix"] = suffix
    assert table.toltec_file._guess_data_kinds() is None
    assert table["data_kind_toltec"].tolist() == [kind.value]


def test_classification_accepts_parsed_records_without_reparsing(monkeypatch):
    paths = [
        "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
        "tel_toltec_2025-10-31_145609_00_0002.nc",
    ]
    roach, tel = map(FilepathInfo.from_source, paths)
    assert roach is not None and tel is not None
    table = toltec_file.ToltecFileAccessor.from_sources(paths)

    def unexpected(*args, **kwargs):
        pytest.fail("Metadata classification attempted filename parsing")

    monkeypatch.setattr(FilepathInfo, "from_source", unexpected)
    table = table.toltec_file.update_from_models(
        [roach, tel],
        columns=roach._columns | tel._columns,
    )
    assert table.toltec_file._guess_data_kinds() is None
    assert table["data_kind_toltec"].tolist() == [
        ToltecDataKind.vnasweep.value,
        ToltecDataKind.lmt_tel.value,
    ]
