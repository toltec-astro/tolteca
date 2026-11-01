"""Source validation and universal filename projection, without file I/O."""

import subprocess
import sys
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from tollan.utils.fileloc import FileLoc, fileloc

from tolteca_datamodels.core.file import (
    FileAccessorBase,
    SourceTraceKind,
)
from tolteca_datamodels.core.filepath import FilepathInfo, register_filepath_parser

_SOURCE_DTYPES = {
    "source": "object",
    "source_loc": "object",
    "source_loc_is_local": "boolean",
    "source_loc_host": "string",
    "source_loc_scheme": "string",
    "filepath_info": "object",
    "source_trace": "object",
}


def test_universal_preparation_does_not_evaluate_accessor_rules(monkeypatch):
    from tollan.accessor.pandas.model_table import RecordRuleMapper

    def unexpected(*args, **kwargs):
        pytest.fail("Universal source preparation evaluated a capability policy")

    monkeypatch.setattr(RecordRuleMapper, "map_table", unexpected)
    file_accessor = FileAccessorBase.from_sources(["first.nc", None])
    assert file_accessor.table.file_ext.iloc[0] == "nc"
    assert file_accessor.table.data_models.iloc[0] == frozenset()
    assert list(file_accessor.source_trace.iloc[0].iter_entries()) == []
    assert file_accessor.source_trace.iloc[1].has_error


@pytest.mark.parametrize("container", [list, tuple, iter, set, np.array, pd.Series])
def test_source_collections_preserve_iteration_order(container):
    sources = container(["first.nc", "second.nc"])
    expected = ["first.nc", "second.nc"] if container is iter else list(sources)

    table = FileAccessorBase.from_sources(sources).table

    assert table.source.tolist() == expected


@pytest.mark.parametrize(
    "sources",
    ["single.nc", b"single.nc", {"source": "single.nc"}, None, 1],
)
def test_invalid_outer_collections_raise_in_record_mode(sources):
    with pytest.raises(ValidationError):
        FileAccessorBase.from_sources(sources)


def test_source_iteration_failure_is_not_recorded_as_a_bad_entry():
    def sources():
        yield "first.nc"
        msg = "Cannot enumerate remaining sources"
        raise ValueError(msg)

    with pytest.raises(ValidationError, match="Cannot enumerate remaining sources"):
        FileAccessorBase.from_sources(sources())


def test_sources_are_retained_and_locations_validate_without_mutating_input(tmp_path):
    location = fileloc("archive:/unrecognized.nc")
    path = tmp_path / "cached.nc"
    pair = {"source": location, "source_loc": None}
    before = pair.copy()
    sources = ["relative.nc", path, pair]
    table = FileAccessorBase.from_sources(iter(sources)).table

    assert pair == before
    assert table.dtypes.astype(str).to_dict() == _SOURCE_DTYPES | {
        "file_ext": "string",
        "file_data_models": "object",
        "data_models": "object",
    }
    for row, source in zip(
        table.itertuples(index=False), [sources[0], path, location], strict=True
    ):
        assert row.source is source
        assert isinstance(row.source_loc, FileLoc)
        assert row.source_loc == fileloc(source)
        assert type(row.filepath_info) is FilepathInfo
        assert row.file_ext == row.filepath_info.ext == "nc"
        assert row.data_models == row.file_data_models == frozenset()
    assert table.source_loc.iloc[2] is location
    assert table.source_loc_is_local.tolist() == [True, True, False]
    assert table.source_loc_host.tolist() == ["", "", "archive"]
    assert table.source_loc_scheme.tolist() == ["file", "file", "file"]


def test_explicit_location_is_authoritative_for_an_opaque_resource():
    from tolteca_datamodels.toltec.filepath import ToltecLmtFilepathInfo

    location = fileloc("https://archive/tel_toltec_2025-10-31_145609_00_0002.nc")
    with BytesIO(b"unread") as resource:
        table = FileAccessorBase.from_sources(
            [{"source": resource, "source_loc": location}]
        ).table
        row = table.iloc[0]
        assert row.source is resource
        assert row.source_loc is location
        assert type(row.filepath_info) is ToltecLmtFilepathInfo
        assert (row.file_prefix, row.file_obsnum) == ("tel_toltec", 145609)
        assert (row.source_loc_host, row.source_loc_scheme) == ("archive", "https")
        assert not row.source_loc_is_local
        assert not resource.closed and resource.tell() == 0


@pytest.mark.parametrize(
    "entry",
    [
        object(),
        {"source": "valid.nc", "source_loc": object()},
        {"source_loc": "missing-source.nc"},
        {"source": None, "source_loc": "missing-source.nc"},
        {"source": "valid.nc", "unknown_key": "other.nc"},
    ],
)
def test_invalid_pairs_keep_their_row_and_do_not_interrupt_other_sources(entry):
    table = FileAccessorBase.from_sources(["before.nc", entry, "after.nc"]).table
    assert len(table) == 3
    assert table.iloc[[0, 2]].source.tolist() == ["before.nc", "after.nc"]
    assert table.drop(columns="source_trace").iloc[1].isna().all()
    assert [
        [entry.kind for entry in cell.iter_entries()] for cell in table.source_trace
    ] == [
        [],
        [SourceTraceKind.source_validation_failed],
        [],
    ]
    failure = next(table.source_trace.iloc[1].iter_entries())
    assert failure.context == {"exception_type": "ValidationError"}
    assert failure.message
    with pytest.raises(ValidationError):
        FileAccessorBase.from_sources([entry], on_error="raise")


def test_empty_input_has_typed_source_columns_and_no_invented_filename_columns():
    table = FileAccessorBase.from_sources([]).table
    assert table.empty
    assert table.dtypes.astype(str).to_dict() == _SOURCE_DTYPES


def test_all_invalid_sources_still_have_typed_columns_and_distinct_traces():
    table = FileAccessorBase.from_sources([None, object()]).table
    assert len(table) == 2
    assert table.dtypes.astype(str).to_dict() == _SOURCE_DTYPES
    assert table.drop(columns="source_trace").isna().all().all()
    assert table.source_trace.iloc[0] is not table.source_trace.iloc[1]
    assert [
        [entry.kind for entry in cell.iter_entries()] for cell in table.source_trace
    ] == [[SourceTraceKind.source_validation_failed]] * 2


def test_mixed_filename_types_preserve_rows_and_project_their_own_fields():
    from tolteca_datamodels.lmt.filepath import LmtFilepathInfo
    from tolteca_datamodels.toltec.filepath import (
        ToltecFilepathInfo,
        ToltecLmtFilepathInfo,
    )
    from tolteca_dummy_instru.filepath import (
        DummyInstruFilepathInfo,
        DummyInstruLmtFilepathInfo,
    )

    sources = [
        "tel_other_2025-10-31_145609_00_0002.nc",
        "toltec8_145605_000_0001_2025_10_31_09_20_06_tune.nc",
        "tel_toltec_2025-10-31_145609_00_0002.nc",
        "chan01-tcs-1-0-2-20251031T120000.nc",
        "tel_dummy_instru_2025-10-31_000001_00_0002.nc",
        "unrecognized.nc",
    ]
    sources.append(sources[1])
    file_accessor = FileAccessorBase.from_sources(sources)
    table = file_accessor.table
    assert table.source.tolist() == sources
    assert [type(info) for info in table.filepath_info] == [
        LmtFilepathInfo,
        ToltecFilepathInfo,
        ToltecLmtFilepathInfo,
        DummyInstruFilepathInfo,
        DummyInstruLmtFilepathInfo,
        FilepathInfo,
        ToltecFilepathInfo,
    ]
    assert table.data_models.dropna().tolist() == [
        {"lmt"},
        {"toltec"},
        {"lmt", "toltec"},
        {"dummy_instru"},
        {"lmt", "dummy_instru"},
        frozenset(),
        {"toltec"},
    ]
    assert table.data_models.iloc[5] == frozenset()
    assert table.file_obsnum.dropna().tolist() == [145609, 145605, 145609, 1, 1, 145605]
    assert table.file_roach.dropna().tolist() == [8, 8]
    assert table.file_channel.dropna().tolist() == [1]
    for name in ("obsnum", "roach", "channel", "interface"):
        pd.testing.assert_series_equal(table[name], table[f"file_{name}"].rename(name))
    assert table.file_prefix.iloc[2] == "tel_toltec"
    assert "prefix" not in table
    assert "ext" not in table
    assert table.file_ext.tolist() == ["nc"] * len(sources)
    assert "data_kind_toltec" not in table
    assert "uid_raw_obs" not in table
    assert "io_obj" not in table

    selected = file_accessor.query('contains(data_models, "dummy_instru")')
    assert selected.index.tolist() == [3, 4]
    for position, row in zip([3, 4], selected.itertuples(index=False), strict=True):
        assert row.filepath_info is table.filepath_info.iloc[position]
        assert row.source is sources[position]


def test_mixed_projection_leaves_other_models_fields_missing(monkeypatch):
    from tolteca_datamodels.core import filepath

    monkeypatch.setattr(filepath, "_filepath_parsers", [])

    class First(FilepathInfo):
        value: int

    class Second(FilepathInfo):
        other: int

    register_filepath_parser(
        pattern=r"^first-(?P<value>\d+)$",
        parsers=(lambda *_: {"ext": "", "data_models": []},),
        filepath_info_cls=First,
    )
    register_filepath_parser(
        pattern=r"^second-(?P<other>\d+)$",
        parsers=(lambda *_: {"ext": "", "data_models": []},),
        filepath_info_cls=Second,
    )
    table = FileAccessorBase.from_sources(
        [None, "first-1", "unrecognized", "second-2", None]
    ).table
    assert table.filepath_info.isna().tolist() == [True, False, True, False, True]
    assert table.file_value.dropna().tolist() == [1]
    assert table.file_other.dropna().tolist() == [2]
    assert table.file_value.isna().tolist() == [True, False, True, True, True]
    assert table.file_other.isna().tolist() == [True, True, True, False, True]
    assert table.file_value.dtype == table.file_other.dtype == pd.Int64Dtype()


def test_mixed_projection_rejects_conflicting_destination_dtypes(monkeypatch):
    class IntegerInfo(FilepathInfo):
        value: int

    class FloatingInfo(FilepathInfo):
        value: float

    parsed = iter(
        [
            IntegerInfo(value=1, ext="nc", data_models=frozenset()),
            FloatingInfo(value=2.0, ext="nc", data_models=frozenset()),
        ]
    )
    monkeypatch.setattr(
        FilepathInfo, "from_source", lambda *args, **kwargs: next(parsed)
    )

    with pytest.raises(TypeError, match="Conflicting dtypes for column 'file_value'"):
        FileAccessorBase.from_sources(["integer.nc", "floating.nc"])


def test_recognized_invalid_filename_is_not_treated_as_unrecognized():
    from tolteca_datamodels.lmt import filepath  # noqa: F401 -- register parser

    with pytest.raises(ValueError, match="does not match format"):
        FileAccessorBase.from_sources(
            ["tel_toltec_2025-99-31_145609_00_0002.nc"], on_error="raise"
        )


def test_binding_and_slicing_retain_source_objects_without_reparsing(monkeypatch):
    table = FileAccessorBase.from_sources(["first.nc", "second.nc", "first.nc"]).table
    table.index = [8, 8, 2]

    def unexpected(*args, **kwargs):
        pytest.fail("Binding reparsed a filename")

    monkeypatch.setattr(FilepathInfo, "from_source", unexpected)
    selected = table.iloc[[2, 0]]
    accessor = FileAccessorBase(selected)
    assert accessor.table is selected
    for name in (
        "source",
        "source_loc",
        "filepath_info",
        "data_models",
        "source_trace",
    ):
        pd.testing.assert_series_equal(getattr(accessor, name), selected[name])
    assert accessor.source_loc.iloc[0] is table.source_loc.iloc[2]
    assert accessor.filepath_info.iloc[0] is table.filepath_info.iloc[2]
    assert accessor.data_models.iloc[0] is table.data_models.iloc[2]
    assert accessor.source_trace.iloc[0] is table.source_trace.iloc[2]
    assert accessor.source.index.tolist() == [2, 8]
    accessor.table["note"] = ["third", "first"]
    assert selected.note.tolist() == ["third", "first"]
    assert "note" not in table


def test_neutral_import_uses_later_registrations_without_loading_a_workflow():
    script = r"""
import sys
from tolteca_datamodels.core import file as core_file
from tolteca_datamodels.core.file import FileAccessorBase
from tolteca_datamodels.core.filepath import FilepathInfo, register_filepath_parser

assert core_file.FileAccessorBase is FileAccessorBase
assert not hasattr(core_file, "FileAccessor")
assert not hasattr(core_file.accessor, "FileAccessor")
info = FileAccessorBase.from_sources(["entry-1.nc"]).table.filepath_info.iloc[0]
assert type(info) is FilepathInfo and info.ext == "nc"
assert info.data_models == frozenset()

class Entry(FilepathInfo):
    number: int

register_filepath_parser(
    pattern=r"^entry-(?P<number>\d+)\.(?P<ext>nc)$",
    parsers=(lambda *_: {"data_models": []},), filepath_info_cls=Entry,
)
table = FileAccessorBase.from_sources(["entry-1.nc"]).table
assert type(table.filepath_info.iloc[0]) is Entry
assert table.file_number.iloc[0] == 1
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules
    for prefix in (
        "tolteca_datamodels.core.data_model",
        "tolteca_datamodels.data_model",
        "tolteca_datamodels.lmt_base", "tolteca_datamodels.lmt",
        "tolteca_datamodels.toltec", "tolteca_dummy_instru",
    )
)
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
