"""Filename issues, failure policy, and row-local reporting."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest
from pydantic import ValidationError
from tollan.accessor.pandas.model_table import ModelColumn
from tollan.utils.log import logger

from tolteca_datamodels.core.file import (
    FileAccessorBase,
    SourceTrace,
    SourceTraceEntry,
    SourceTraceKind,
)
from tolteca_datamodels.core.filepath import FilepathInfo
from tolteca_datamodels.lmt import filepath  # noqa: F401 -- register telescope parser


def test_trace_records_issues_and_leaves_successful_rows_empty():
    sources = [
        "tel_toltec_2025-99-31_145609_00_0002.nc",
        "generic.nc",
        "/",
        "tel_toltec_2025-10-31_145609_00_0002.nc",
    ]
    table = FileAccessorBase.from_sources(sources).table
    assert table.source.tolist() == sources
    assert table.filepath_info.isna().tolist() == [True, False, True, False]
    assert table.filepath_info.iloc[1].data_models == frozenset()
    assert table.filepath_info.iloc[1].ext == "nc"
    assert all(isinstance(cell, SourceTrace) for cell in table.source_trace)
    assert [
        [entry.kind for entry in cell.iter_entries()] for cell in table.source_trace
    ] == [
        [SourceTraceKind.filepath_parse_failed],
        [],
        [SourceTraceKind.filepath_rule_unmatched],
        [],
    ]
    assert [cell.has_error for cell in table.source_trace] == [
        True,
        False,
        False,
        False,
    ]
    failed = next(table.source_trace.iloc[0].iter_entries())
    unmatched = next(table.source_trace.iloc[2].iter_entries())
    assert failed.is_error
    assert failed.message is not None
    assert "does not match format" in failed.message
    assert failed.context == {"exception_type": "ValueError"}
    assert not unmatched.is_error
    assert unmatched.message is None and unmatched.context is None
    assert len({id(entries) for entries in table.source_trace}) == 4
    assert table.source_loc.iloc[0] is not None
    assert table.source_loc.iloc[2] is not None

    # Traces stay with rows, independently of index labels or source order.
    table.index = [8, 8, 2, 3]
    selected = table.iloc[[3, 0]]
    assert selected.source_trace.iloc[0] is table.source_trace.iloc[3]
    assert list(selected.source_trace.iloc[0].iter_entries()) == []
    assert next(selected.source_trace.iloc[1].iter_entries()) is failed
    assert selected.filepath_info.notna().sum() == 1
    summary = FileAccessorBase(selected).source_trace_summary()
    assert summary.to_dict("records") == [
        {"kind": "filepath_parse_failed", "count": 1, "total": 2},
    ]
    selected.source_trace.iloc[0].record(failed)
    selected.source_trace.iloc[0].record(failed)
    assert table.source_trace.iloc[3].has_error
    assert FileAccessorBase(selected).source_trace_summary().to_dict("records") == [
        {"kind": "filepath_parse_failed", "count": 2, "total": 2},
    ]


def test_generic_filename_is_successful_base_parse_in_raise_mode():
    file_accessor = FileAccessorBase.from_sources(["unrecognized.nc"], on_error="raise")
    table = file_accessor.table
    assert table.filepath_info.iloc[0].data_models == frozenset()
    assert table.file_ext.iloc[0] == "nc"
    assert list(table.source_trace.iloc[0].iter_entries()) == []
    assert not table.source_trace.iloc[0].has_error
    summary = file_accessor.source_trace_summary()
    assert summary.empty
    assert summary.columns.tolist() == ["kind", "count", "total"]


def test_traces_are_independent_across_rows_and_batches():
    sources = ["tel_toltec_2025-99-31_145609_00_0002.nc", None, "generic.nc"]
    first = FileAccessorBase.from_sources(sources)
    second = FileAccessorBase.from_sources(sources)
    expected = [
        [SourceTraceKind.filepath_parse_failed],
        [SourceTraceKind.source_validation_failed],
        [],
    ]
    cells = [*first.source_trace, *second.source_trace]
    assert [
        [entry.kind for entry in cell.iter_entries()] for cell in cells
    ] == expected * 2
    assert len({id(cell) for cell in cells}) == 6
    first.source_trace.iloc[2].record(next(first.source_trace.iloc[0].iter_entries()))
    assert list(second.source_trace.iloc[2].iter_entries()) == []
    assert [len(list(cell.iter_entries())) for cell in cells] == [1, 1, 1, 1, 1, 0]
    assert [cell.has_error for cell in cells] == [True, True, True, True, True, False]


def test_record_updates_error_state_from_only_the_new_entry(monkeypatch):
    issue = SourceTraceEntry(kind=SourceTraceKind.filepath_rule_unmatched)
    error = SourceTraceEntry(kind=SourceTraceKind.filepath_parse_failed)
    later_error = SourceTraceEntry(kind=SourceTraceKind.source_validation_failed)
    trace = SourceTrace()
    assert list(trace.iter_entries()) == list(trace.iter_error_entries()) == []
    trace.record(issue)
    assert not trace.has_error

    checked = []

    def is_error(entry):
        checked.append(entry)
        return entry.kind.value.endswith("_failed")

    monkeypatch.setattr(SourceTraceEntry, "is_error", property(is_error))
    trace.record(error)
    trace.record(issue)
    trace.record(later_error)
    assert checked == [error, issue, later_error]
    assert list(trace.iter_entries()) == [issue, error, later_error]
    assert list(trace.iter_error_entries()) == [error, later_error]
    assert trace.has_error


def test_summary_uses_recorded_kinds_without_scanning_entries(monkeypatch):
    table = FileAccessorBase.from_sources([None, "/", "generic.nc"]).table
    table.source_trace.iloc[0].record(next(table.source_trace.iloc[0].iter_entries()))

    def unexpected(*args):
        pytest.fail("Summary scanned a row's entries")

    monkeypatch.setattr(SourceTrace, "iter_entries", unexpected)

    selected = table.iloc[[1, 0, 0]]
    assert FileAccessorBase(selected).source_trace_summary().to_dict("records") == [
        {"kind": "filepath_rule_unmatched", "count": 1, "total": 3},
        {"kind": "source_validation_failed", "count": 2, "total": 3},
    ]


@pytest.mark.parametrize(
    "table",
    [pd.DataFrame(), pd.DataFrame(index=[8, 8, 2]), pd.DataFrame({"note": ["kept"]})],
)
def test_binding_and_empty_summary_do_not_require_or_create_columns(table):
    before = table.copy()
    file_accessor = FileAccessorBase(table)
    assert file_accessor.table is table
    assert file_accessor.source_trace_summary().to_dict("list") == {
        "kind": [],
        "count": [],
        "total": [],
    }
    for name in ("source", "source_loc", "source_trace"):
        with pytest.raises(KeyError, match=name):
            getattr(file_accessor, name)
    pd.testing.assert_frame_equal(table, before)


def test_accessor_collector_records_in_table_traces_by_position_not_index_label():
    table = pd.DataFrame(index=[8, 8, 2])
    accessor = FileAccessorBase(table)
    collector = accessor._make_source_trace_collector(on_error="record")
    pd.testing.assert_series_equal(accessor.source_trace, table["source_trace"])
    assert table.columns.tolist() == ["source_trace"]
    assert accessor.source_trace.dtype == object
    assert collector.on_error == "record"
    collector.record(1, SourceTraceKind.filepath_rule_unmatched)
    collector.record(0, SourceTraceKind.source_validation_failed)
    collector.record(1, SourceTraceKind.filepath_parse_failed)
    assert [entry.kind for entry in table.source_trace.iloc[1].iter_entries()] == [
        SourceTraceKind.filepath_rule_unmatched,
        SourceTraceKind.filepath_parse_failed,
    ]
    assert [cell.has_error for cell in table.source_trace] == [True, True, False]
    assert len({id(cell) for cell in table.source_trace}) == 3
    assert not hasattr(table.source_trace.iloc[0], "on_error")
    later = accessor._make_source_trace_collector(on_error="raise")
    assert later.on_error == "raise"
    assert all(
        original is current
        for original, current in zip(collector.traces, later.traces, strict=True)
    )
    assert accessor.source_trace_summary().to_dict("records") == [
        {"kind": "filepath_parse_failed", "count": 1, "total": 3},
        {"kind": "filepath_rule_unmatched", "count": 1, "total": 3},
        {"kind": "source_validation_failed", "count": 1, "total": 3},
    ]


def test_factory_populates_the_table_that_already_owns_its_traces(monkeypatch):
    tables = []
    traces = []

    class ObservedFileAccessor(FileAccessorBase):
        def __init__(self, data):
            super().__init__(data)
            assert data.shape == (3, 0)
            tables.append(data)

    parse = FilepathInfo.from_source

    def parse_after_sources(source, **kwargs):
        table = tables[0]
        if not traces:
            traces.extend(table.source_trace)
        assert table.source.iloc[1] == "generic.nc"
        assert table.source_loc.notna().tolist() == [False, True, True]
        assert next(traces[0].iter_entries()).kind == (
            SourceTraceKind.source_validation_failed
        )
        return parse(source, **kwargs)

    monkeypatch.setattr(FilepathInfo, "from_source", parse_after_sources)
    result = ObservedFileAccessor.from_sources(
        [None, "generic.nc", "tel_toltec_2025-99-31_145609_00_0002.nc"]
    )
    assert type(result) is ObservedFileAccessor
    assert len(tables) == 1
    assert result.table is tables[0]
    assert all(
        original is current
        for original, current in zip(traces, result.source_trace, strict=True)
    )
    assert next(traces[2].iter_entries()).kind == SourceTraceKind.filepath_parse_failed


def test_error_iteration_and_logging_do_not_rescan_or_reclassify_entries(monkeypatch):
    table = FileAccessorBase.from_sources(
        ["/", "generic.nc", "tel_toltec_2025-99-31_145609_00_0002.nc"]
    ).table
    trace = table.source_trace.iloc[2]
    error = next(trace.iter_error_entries())
    trace.record(SourceTraceEntry(kind=SourceTraceKind.filepath_rule_unmatched))
    trace.record(error)

    def is_error(entry):
        pytest.fail("Previously recorded entry was reclassified")

    def unexpected(*args):
        pytest.fail("Error iteration scanned all issue entries")

    monkeypatch.setattr(SourceTrace, "iter_entries", unexpected)
    monkeypatch.setattr(SourceTraceEntry, "is_error", property(is_error))
    errors = list(trace.iter_error_entries())
    assert errors == [error]
    assert [cell.has_error for cell in table.source_trace] == [False, False, True]
    FileAccessorBase(table)._log_traces()


@pytest.mark.parametrize("on_error", [None, "skip"])
def test_invalid_error_policy_is_validated_before_consuming_sources(on_error):
    def sources():
        pytest.fail("Invalid policy consumed the source iterator")
        yield

    with pytest.raises(ValidationError, match="literal_error"):
        FileAccessorBase.from_sources(sources(), on_error=on_error)


def test_recorded_entry_cannot_change_its_kind():
    entry = SourceTraceEntry(
        kind=SourceTraceKind.filepath_parse_failed,
        message="Invalid date",
        context={"exception_type": "ValueError"},
    )
    trace = SourceTrace()
    trace.record(entry)
    with pytest.raises(FrozenInstanceError):
        entry.kind = SourceTraceKind.filepath_rule_unmatched  # ty: ignore[invalid-assignment]
    assert trace.has_error


def test_empty_trace_summary_retains_its_columns():
    file_accessor = FileAccessorBase.from_sources([])
    assert file_accessor.source_trace.dtype == object
    summary = file_accessor.source_trace_summary()
    assert summary.empty
    assert summary.columns.tolist() == ["kind", "count", "total"]


@pytest.mark.parametrize("error", [RuntimeError("ambiguous parser"), TypeError("bug")])
def test_registration_and_programming_errors_are_not_recorded(error, monkeypatch):
    def parse(*args, **kwargs):
        raise error

    monkeypatch.setattr(FilepathInfo, "from_source", parse)
    with pytest.raises(type(error), match=str(error)):
        FileAccessorBase.from_sources(["example.nc"])


def test_projection_failure_is_not_suppressed_as_a_filename_failure(monkeypatch):
    class BrokenProjection(FilepathInfo):
        value: str

    monkeypatch.setattr(
        FileAccessorBase,
        "_filepath_columns",
        classmethod(lambda *_: {"number": ModelColumn("value", dtype="Int64")}),
    )
    info = BrokenProjection(value="not-a-number", ext="nc", data_models=frozenset())
    monkeypatch.setattr(FilepathInfo, "from_source", lambda *args, **kwargs: info)
    with pytest.raises(ValueError, match="invalid literal"):
        FileAccessorBase.from_sources(["example.nc"])


def test_parse_failure_and_summary_are_available_in_debug_logging():
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        FileAccessorBase.from_sources(
            ["tel_toltec_2025-99-31_145609_00_0002.nc", "generic.nc"]
        )
    finally:
        logger.remove(sink)
    output = "\n".join(messages)
    assert "filepath_parse_failed" in output
    assert "ValueError" in output
    assert "Source processing trace" in output
    assert "1/2 filenames parsed" in output
    assert "2025-99-31" in output
