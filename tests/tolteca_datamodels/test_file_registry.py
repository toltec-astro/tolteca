"""Normalized filename keys and eager parser records."""

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, assert_type

import pytest
from tollan.utils.fileloc import fileloc

import tolteca_datamodels._filepath as file_module
from tolteca_datamodels.file import FileAccessor, FilepathInfo
from tolteca_datamodels.lmt import LmtFilepathInfo
from tolteca_datamodels.toltec import file as toltec_file
from tolteca_datamodels.toltec.file import ToltecFilepathInfo
from tolteca_datamodels.toltec.types import ToltecDataKind


@pytest.mark.parametrize("model", [LmtFilepathInfo, ToltecFilepathInfo])
def test_models_register_their_own_patterns_and_static_parsers(model):
    for pattern, parsers in model._filename_patterns.items():
        registered = file_module._filename_pattern_parser[pattern]
        assert registered.filepath_info_cls is model
        assert registered.parsers is parsers
        assert all(
            isinstance(model.__dict__[parse.__name__], staticmethod)
            for parse in parsers
        )
    assert "_filename_patterns" not in model.model_fields
    assert not hasattr(model, "filename_pattern_parser")


def test_inherited_patterns_do_not_change_registry_ownership(monkeypatch):
    registry = file_module._filename_pattern_parser.copy()
    monkeypatch.setattr(file_module, "_filename_pattern_parser", registry.copy())

    class Inherited(LmtFilepathInfo):
        pass

    assert Inherited._filename_patterns is LmtFilepathInfo._filename_patterns
    assert file_module._filename_pattern_parser == registry
    name = "tel_toltec_2025-10-31_145609_00_0002.nc"
    assert type(FilepathInfo.from_source(name)) is LmtFilepathInfo
    pattern, parsers = next(iter(LmtFilepathInfo._filename_patterns.items()))

    class Declared(LmtFilepathInfo):
        _filename_patterns: ClassVar = {pattern: parsers}

    assert file_module._filename_pattern_parser[pattern].filepath_info_cls is Declared
    assert type(FilepathInfo.from_source(name)) is Declared
    for accepted_types in ((LmtFilepathInfo,), (Declared,)):
        assert (
            type(FilepathInfo.from_source(name, accepted_types=accepted_types))
            is Declared
        )


def test_subclass_cannot_shadow_shared_parser_registry(monkeypatch):
    monkeypatch.setattr(
        file_module,
        "_filename_pattern_parser",
        file_module._filename_pattern_parser.copy(),
    )
    pattern, parsers = next(iter(LmtFilepathInfo._filename_patterns.items()))

    class Declared(LmtFilepathInfo):
        _filename_pattern_parser: ClassVar = {}
        _filename_patterns: ClassVar = {pattern: parsers}

    assert not Declared._filename_pattern_parser
    assert file_module._filename_pattern_parser[pattern].filepath_info_cls is Declared
    assert (
        type(FilepathInfo.from_source("tel_toltec_2025-10-31_145609_00_0002.nc"))
        is Declared
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
    assert type(info) is LmtFilepathInfo
    assert "filename_pattern_parser" not in info.model_dump()
    telescope = LmtFilepathInfo.from_source(source)
    assert_type(telescope, LmtFilepathInfo | None)
    assert telescope == info


def test_subclass_factory_rejects_other_instrument():
    source = "toltec8_145609_000_0002_2025_10_31_09_27_57.nc"
    assert LmtFilepathInfo.from_source(source, raise_on_error=False) is None
    with pytest.raises(
        ValueError, match="Unrecognized acquisition filename for LmtFilepathInfo"
    ):
        LmtFilepathInfo.from_source(source)


@pytest.mark.parametrize(
    "accepted_types",
    [
        None,
        (),
        (LmtFilepathInfo,),
        (ToltecFilepathInfo,),
        (LmtFilepathInfo, ToltecFilepathInfo),
        (FilepathInfo,),
    ],
)
@pytest.mark.parametrize(
    ("source", "model"),
    [
        ("tel_toltec_2025-10-31_145609_00_0002.nc", LmtFilepathInfo),
        ("toltec8_145609_000_0002_2025_10_31_09_27_57.nc", ToltecFilepathInfo),
    ],
)
def test_factory_limits_results_to_accepted_types(source, model, accepted_types):
    source = fileloc(f"archive:/data/{source}")
    result = FilepathInfo.from_source(
        source, accepted_types=accepted_types, raise_on_error=False
    )
    if accepted_types is None or issubclass(model, accepted_types):
        assert type(result) is model
        assert result == model.from_source(source)
    else:
        assert result is None
        with pytest.raises(ValueError, match="Unrecognized acquisition filename"):
            FilepathInfo.from_source(source, accepted_types=accepted_types)


def test_accepted_types_only_narrow_the_calling_class():
    tel = "tel_toltec_2025-10-31_145609_00_0002.nc"
    kids = "toltec8_145609_000_0002_2025_10_31_09_27_57.nc"
    assert isinstance(
        LmtFilepathInfo.from_source(tel, accepted_types=(FilepathInfo,)),
        LmtFilepathInfo,
    )
    for source in (tel, kids):
        assert (
            LmtFilepathInfo.from_source(
                source, accepted_types=(ToltecFilepathInfo,), raise_on_error=False
            )
            is None
        )


@pytest.mark.parametrize("factory", [FilepathInfo, LmtFilepathInfo])
def test_excluded_types_are_not_matched_parsed_or_validated(monkeypatch, factory):
    extract = file_module.dict_from_regex_match

    def match_accepted_only(pattern, filename):
        parser = file_module._filename_pattern_parser[pattern]
        assert issubclass(parser.filepath_info_cls, LmtFilepathInfo)
        return extract(pattern, filename)

    def unexpected(*args, **kwargs):
        pytest.fail("Excluded filepath type was parsed or validated")

    monkeypatch.setattr(file_module, "dict_from_regex_match", match_accepted_only)
    monkeypatch.setattr(ToltecFilepathInfo, "model_validate", unexpected)
    monkeypatch.setattr(
        file_module,
        "_filename_pattern_parser",
        {
            pattern: (
                parser
                if issubclass(parser.filepath_info_cls, LmtFilepathInfo)
                else parser._replace(parsers=(unexpected,))
            )
            for pattern, parser in file_module._filename_pattern_parser.items()
        },
    )
    # Invalid date deliberately tests rejection without parsing foreign data.
    source = "toltec8_145609_000_0002_2025_02_30_09_27_57.nc"
    kwargs = {"accepted_types": (LmtFilepathInfo,)} if factory is FilepathInfo else {}
    assert factory.from_source(source, raise_on_error=False, **kwargs) is None


@pytest.mark.parametrize("raise_on_error", [False, True])
@pytest.mark.parametrize(
    ("source", "model", "error"),
    [
        ("tel_toltec_2025-02-30_145609_00_0002.nc", LmtFilepathInfo, ValueError),
        (
            "toltec13_145609_000_0002_2025_10_31_09_27_57.nc",
            ToltecFilepathInfo,
            KeyError,
        ),
    ],
)
def test_accepted_filename_errors_propagate(source, model, error, raise_on_error):
    # Impossible dates and out-of-range boards are invalid acquisition names.
    with pytest.raises(error):
        FilepathInfo.from_source(
            source, accepted_types=(model,), raise_on_error=raise_on_error
        )


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
@pytest.mark.parametrize(
    "accepted_types", [None, (LmtFilepathInfo, ToltecFilepathInfo)]
)
def test_pattern_parsers_run_once(monkeypatch, filename, expected, accepted_types):
    calls = []
    validated = []
    parser = next(
        parser
        for pattern, parser in file_module._filename_pattern_parser.items()
        if pattern.match(filename)
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

    def validate_once(captures):
        validated.append(captures)
        return validate(captures)

    monkeypatch.setattr(parser.filepath_info_cls, "model_validate", validate_once)
    monkeypatch.setattr(
        file_module,
        "_filename_pattern_parser",
        {
            pattern: parser._replace(
                parsers=tuple(trace(i, parse) for i, parse in enumerate(parser.parsers))
            )
            for pattern, parser in file_module._filename_pattern_parser.items()
        },
    )
    info = FilepathInfo.from_source(filename, accepted_types=accepted_types)
    assert isinstance(info, (LmtFilepathInfo, ToltecFilepathInfo))
    assert len(validated) == 1
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
    expected = FilepathInfo.from_source(name)
    pattern, parser = next(
        (pattern, parser)
        for pattern, parser in file_module._filename_pattern_parser.items()
        if pattern.match(name)
    )
    captures = {}
    extract = file_module.dict_from_regex_match

    def trace_extract(*args, **kwargs):
        nonlocal captures
        captures = extract(*args, **kwargs)
        return captures

    def increment_obsnum(filepath_data, path):
        assert filepath_data is captures
        assert filepath_data["prefix"] == "tel_toltec"
        assert path == Path.cwd() / name
        return {"obsnum": int(filepath_data["obsnum"]) + 1}

    def restore_obsnum(filepath_data, path):
        assert filepath_data is captures
        assert filepath_data["obsnum"] == 145610
        assert filepath_data["suffix"] is None
        return {"obsnum": filepath_data["obsnum"] - 1}

    monkeypatch.setattr(file_module, "dict_from_regex_match", trace_extract)
    monkeypatch.setattr(
        file_module,
        "_filename_pattern_parser",
        {
            pattern: parser._replace(
                parsers=(increment_obsnum, restore_obsnum, *parser.parsers)
            )
        },
    )
    assert FilepathInfo.from_source(name) == expected
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
        captures = extract(*args, **kwargs)
        if captures is not None:
            captures["ut"] = ut_invalid
        return captures

    monkeypatch.setattr(file_module, "dict_from_regex_match", invalid_capture)
    with pytest.raises(ValueError):
        FilepathInfo.from_source(name)


@pytest.mark.parametrize(
    "first",
    [
        "tolteca_datamodels.file",
        "tolteca_datamodels._filepath",
        "tolteca_datamodels.lmt._filepath",
        "tolteca_datamodels.toltec._filepath",
        "tolteca_datamodels.lmt",
        "tolteca_datamodels.toltec",
        "tolteca_datamodels.toltec.file",
        "tolteca_datamodels.lmt.metadata",
        "tolteca_datamodels.toltec.metadata",
        "tolteca_datamodels.lmt_base",
        "tolteca_datamodels.lmt_base._filepath",
        "tolteca_datamodels.lmt_base.metadata",
    ],
)
@pytest.mark.parametrize("operation", ["parse", "columns", "metadata_columns"])
def test_import_order_and_one_time_assembly(first, operation):
    script = """
import importlib
import sys

importlib.import_module(sys.argv[1])
import tolteca_datamodels._filepath as file_module
from tolteca_datamodels.file import (
    FileAccessor, FilepathInfo,
)

name = "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
if sys.argv[2] == "parse":
    info = FilepathInfo.from_source(name)
    assert info.roach == 8
elif sys.argv[2] == "columns":
    class Catalog(FileAccessor):
        pass
    catalog = Catalog.from_sources([])
    assert list(catalog) == [
        "source", "source_loc", "source_loc_is_local",
        "source_loc_host", "source_loc_scheme"
    ]
    from tolteca_datamodels.toltec.file import guess_info_from_sources
    catalog = guess_info_from_sources([])
    assert "file_roach" in catalog
else:
    from tolteca_datamodels.toltec.file import ToltecFileAccessor
    columns = ToltecFileAccessor._metadata_columns
    assert columns["obsnum"].attr == "obsnum"
    assert columns is ToltecFileAccessor._metadata_columns

from tolteca_datamodels.toltec import file as toltec_file

assert len(file_module._filename_pattern_parser) == 4
patterns_before = dict(file_module._filename_pattern_parser)
kinds_before = dict(toltec_file._filepath_info_data_kind)
for module in ("tolteca_datamodels.file", "tolteca_datamodels.lmt",
               "tolteca_datamodels.toltec.file"):
    importlib.import_module(module)
assert file_module._filename_pattern_parser == patterns_before
assert toltec_file._filepath_info_data_kind == kinds_before

names = (
    "tel_toltec_2025-10-31_145609_00_0002.nc",
    "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
)
for name in names:
    assert FilepathInfo.from_source(name) is not None
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script, first, operation],
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
                for pattern in file_module._filename_pattern_parser
            )
            == 1
        )


def test_builtin_formats_are_registered_on_import():
    script = """
import sys
from pydantic import TypeAdapter
from tolteca_datamodels.file import FilepathInfo
from tolteca_datamodels.lmt_base._filepath import FilepathOrigMetadataMixin
from tolteca_datamodels.lmt_base.metadata import LmtRawObsMetadata

from tolteca_datamodels._filepath import FilepathInfo as FilepathInfoModel
assert FilepathInfo is FilepathInfoModel
schema = TypeAdapter(LmtRawObsMetadata).json_schema(mode="serialization")
assert "filepath_info_orig" not in schema["properties"]
schema = FilepathOrigMetadataMixin[FilepathInfo].model_json_schema(mode="serialization")
assert "filepath_info_orig" in schema["properties"]
assert "tolteca_datamodels.lmt._filepath" in sys.modules
assert "tolteca_datamodels.toltec._filepath" in sys.modules
assert "tolteca_datamodels.toltec.file" not in sys.modules
info = FilepathInfo.from_source("tel_toltec_2025-10-31_145609_00_0002.nc")
assert info.interface == "tel"

from tolteca_datamodels.toltec.file import ToltecFilepathInfo
assert issubclass(ToltecFilepathInfo, FilepathInfo)
assert isinstance(FilepathInfo.from_source(
    "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
), ToltecFilepathInfo)
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
from tolteca_datamodels.file import FilepathInfo
from tolteca_datamodels.toltec import file as toltec_file
ToltecFileAccessor = toltec_file.ToltecFileAccessor
kinds = dict(toltec_file._filepath_info_data_kind)
toltec_file._filepath_info_data_kind.clear()
importlib.reload(importlib.import_module("tolteca_datamodels.lmt._filepath"))
assert not toltec_file._filepath_info_data_kind
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    importlib.reload(toltec_file)
assert toltec_file._filepath_info_data_kind == kinds
assert not hasattr(FilepathInfo, "filepath_info_data_kind")
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
    assert table["data_kind"].tolist() == [kind.value]


def test_classification_accepts_parsed_records_without_reparsing(monkeypatch):
    paths = [
        "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
        "tel_toltec_2025-10-31_145609_00_0002.nc",
    ]
    roach, tel = map(FilepathInfo.from_source, paths)
    assert roach is not None and tel is not None

    def unexpected(*args, **kwargs):
        pytest.fail("Metadata classification attempted filename parsing")

    monkeypatch.setattr(FilepathInfo, "from_source", unexpected)
    table = FileAccessor.from_sources(paths)
    table = table.toltec_file.update_from_models(
        [roach, tel],
        columns=toltec_file.ToltecFileAccessor._filepath_info_columns,
    )
    assert table.toltec_file._guess_data_kinds() is None
    assert table["data_kind"].tolist() == [
        ToltecDataKind.vnasweep.value,
        ToltecDataKind.lmt_tel.value,
    ]
