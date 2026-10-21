"""Source processing traces, retries, and resource ownership."""

import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from tolteca_datamodels.core.file import SourceTraceEvent, source_trace_issues
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.toltec.file import (
    ToltecFileAccessor,
    get_info_from_source,
    guess_info_from_sources,
    guess_or_read_info_from_sources,
    read_info_from_sources,
)
from tolteca_dummy_instru.file import DummyInstruFileAccessor


@pytest.fixture
def tel_path(data_lmt_path):
    return data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"


@pytest.fixture(params=[LmtFileAccessor, ToltecFileAccessor, DummyInstruFileAccessor])
def accessor_cls(request):
    return request.param


@pytest.fixture
def accessor_tel_path(accessor_cls, tel_path, dummy_tel_path):
    return dummy_tel_path if accessor_cls is DummyInstruFileAccessor else tel_path


@pytest.mark.parametrize("stage", ["filepath", "open", "metadata"])
@pytest.mark.parametrize("exception_type", [None, "ValueError"])
def test_trace_event_validates_and_dumps_named_fields(stage, exception_type):
    data = {"stage": stage, "outcome": "error", "message": "Identification failed"}
    if exception_type is not None:
        data["exception_type"] = exception_type
    event = SourceTraceEvent.model_validate(data)
    assert event.model_dump(mode="json") == data | {"exception_type": exception_type}
    assert SourceTraceEvent.model_validate_json(event.model_dump_json()) == event
    with pytest.raises(ValidationError, match="frozen_instance"):
        event.message = "Changed"  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("update", "error"),
    [
        ({"stage": "reduce"}, "literal_error"),
        ({"message": None}, "string_type"),
        ({"message": b"bytes"}, "string_type"),
        ({"exception_type": ValueError("failure")}, "string_type"),
        ({"unexpected": "value"}, "extra_forbidden"),
    ],
)
def test_trace_event_rejects_invalid_data(update, error):
    with pytest.raises(ValidationError, match=error):
        SourceTraceEvent.model_validate(
            {
                "stage": "filepath",
                "outcome": "unrecognized",
                "message": "No filename match",
            }
            | update
        )


def test_filename_failure_and_nonmatch_are_distinct(tel_path):
    invalid = tel_path.with_name(tel_path.name.replace("2025-10-31", "2025-99-31"))
    with pytest.raises(ValueError, match="does not match"):
        FilepathInfo.from_source(invalid, raise_on_error=False)
    with pytest.raises(ValueError, match="does not match"):
        LmtFileAccessor.from_sources([invalid], on_error="raise")
    table = LmtFileAccessor.from_sources(
        [invalid, "unknown.nc", tel_path], on_error="record"
    )
    (error,) = table.source_trace.map(source_trace_issues).iloc[0]
    assert error.stage == "filepath" and error.exception_type == "ValueError"
    assert table.source_trace.map(source_trace_issues).iloc[1] == [
        SourceTraceEvent(
            stage="filepath",
            outcome="unrecognized",
            message="No accepted filename pattern matched",
        )
    ]
    assert table.source_trace.map(source_trace_issues).iloc[2] == []
    assert pd.isna(table.obsnum.iloc[0])
    assert table.obsnum.iloc[2] == 145609
    assert table.filepath_info.iloc[:2].tolist() == [None, None]
    assert table.filepath_info.iloc[2].obsnum == 145609
    assert table.source_trace.iloc[1] is not table.source_trace.iloc[2]


@pytest.mark.parametrize("on_error", ["raise", "record"])
def test_guess_mode_records_nonmatches_without_opening(
    accessor_cls, monkeypatch, on_error
):
    def unexpected(*args, **kwargs):
        pytest.fail("Filename identification attempted I/O")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    table = accessor_cls.from_sources(["renamed.nc"], mode="guess", on_error=on_error)
    assert table.source_trace.map(source_trace_issues).iloc[0] == [
        SourceTraceEvent(
            stage="filepath",
            outcome="unrecognized",
            message="No accepted filename pattern matched",
        )
    ]
    assert table.obsnum.isna().all()
    assert "errors" not in table


def test_summary_counts_scope_and_outcomes_per_row(tel_path, dummy_tel_path):
    table = ToltecFileAccessor.from_sources([tel_path, dummy_tel_path, "renamed.nc"])
    summary = table.toltec_file.source_trace_summary()
    counts = summary.set_index(["stage", "outcome"])["count"].to_dict()
    assert counts[("filepath", "success")] == 2
    assert counts[("filepath", "unrecognized")] == 1
    assert counts[("scope", "in_scope")] == 1
    assert counts[("scope", "out_of_scope")] == 1
    assert counts[("scope", "unrecognized")] == 1
    assert counts[("metadata", "skipped")] == 3
    assert summary.total.eq(3).all()
    selected = table.iloc[:1]
    assert selected.toltec_file.source_trace_summary().total.eq(1).all()
    assert not table.metadata_read.any()


def test_identification_issue_is_resolved_by_successful_retry(
    accessor_cls, accessor_tel_path
):
    tel_path = accessor_tel_path
    table = accessor_cls.from_sources(
        [{"source": xr.Dataset(), "source_loc": "renamed.nc"}], mode="read"
    )
    assert [
        item.stage for item in table.source_trace.map(source_trace_issues).iloc[0]
    ] == [
        "filepath",
        "identify",
    ]
    with (
        xr.open_dataset(tel_path) as source,
        accessor_cls(table).open(io_objs=[source], on_error="raise") as live,
    ):
        updated = accessor_cls(live).update_from_source_metadata(on_error="raise")
    assert [
        item.stage for item in updated.source_trace.map(source_trace_issues).iloc[0]
    ] == ["filepath"]
    assert updated.obsnum.iloc[0] == 145609
    assert updated.filepath_info.iloc[0] is None
    # Header recovery establishes scope, not a successful filename parse.
    refreshed = accessor_cls(updated).update_from_source_metadata(on_error="raise")
    assert refreshed.filepath_info.iloc[0] is None
    scope = [
        event for event in refreshed.source_trace.iloc[0] if event.stage == "scope"
    ]
    assert scope[-1].outcome == "in_scope"


def test_filename_results_follow_filtered_sorted_rows(tel_path, dummy_tel_path):
    table = ToltecFileAccessor.from_sources([tel_path, "renamed.nc", dummy_tel_path])
    infos = table.filepath_info.tolist()
    table.index = [4, 4, 9]
    table["order"] = [2, 0, 1]
    selected = table.toltec_file.query("order < 2").sort_values(
        "order", ascending=False
    )
    assert selected.index.tolist() == [9, 4]
    assert selected.filepath_info.iloc[0] is infos[2]
    assert selected.filepath_info.iloc[1] is None
    choices = selected.toltec_file._get_reader_choices()
    assert choices == [{}, {ToltecFileAccessor: None}]
    assert table.toltec_file.filepath_info.iloc[0] is infos[0]
    assert selected.toltec_file.source_trace_summary().total.eq(2).all()


@pytest.mark.parametrize(
    "accessor_cls",
    [FileAccessor, LmtFileAccessor, ToltecFileAccessor, DummyInstruFileAccessor],
)
@pytest.mark.parametrize(
    "operation",
    ["read", "guess_or_read", "read_metadata", "read_metadata_inplace", "open"],
)
@pytest.mark.parametrize("on_error", ["record", "raise"])
def test_incompatible_payload_honors_error_policy_and_preserves_values(
    accessor_cls, tel_path, dummy_tel_path, operation, on_error
):
    location = dummy_tel_path if accessor_cls is DummyInstruFileAccessor else tel_path
    if operation == "guess_or_read":
        location = "renamed.nc"
    data = xr.Dataset()
    closed = []
    data.set_close(lambda: closed.append(True))
    sources = [{"source": data, "source_loc": location}]
    table = accessor_cls.from_sources(sources)
    before = table.copy()

    message = (
        "No supported source recognized"
        if operation == "guess_or_read"
        else "Not an LMT telescope dataset"
    )

    def inspect():
        if operation in {"read", "guess_or_read"}:
            return accessor_cls.from_sources(sources, mode=operation, on_error=on_error)
        if operation == "read_metadata":
            return accessor_cls(table).update_from_source_metadata(on_error=on_error)
        if operation == "read_metadata_inplace":
            accessor_cls(table).update_from_source_metadata(
                inplace=True, on_error=on_error
            )
            return table
        with accessor_cls(table).open(on_error=on_error) as result:
            assert result.io_obj.iloc[0] is data
            assert result.accessor_obj.iloc[0] is None
        return result

    if on_error == "raise":
        with pytest.raises(ValueError, match=message):
            inspect()
    else:
        result = inspect()
        pd.testing.assert_frame_equal(
            result.drop(columns="source_trace"), before.drop(columns="source_trace")
        )
        error = result.source_trace.map(source_trace_issues).iloc[0][-1]
        assert error.exception_type == "ValueError"
        assert message in error.message
    pd.testing.assert_frame_equal(
        table.drop(columns="source_trace"), before.drop(columns="source_trace")
    )
    if on_error == "raise" or operation != "read_metadata_inplace":
        pd.testing.assert_frame_equal(table, before)
    assert not closed


@pytest.mark.parametrize("operation", ["read", "guess_or_read", "read_metadata"])
@pytest.mark.parametrize("on_error", ["record", "raise"])
def test_metadata_identification_failure_closes_owned_sources_and_obeys_stop_policy(
    tel_path, tmp_path, monkeypatch, operation, on_error
):
    path = tmp_path / "unrecognized.nc"
    xr.Dataset().to_netcdf(path)
    open_dataset = xr.open_dataset
    opened, closed = [], []

    def tracked_open(path):
        data = open_dataset(path)
        opened.append(path)
        close = data._close
        assert close is not None

        def tracked_close():
            closed.append(path)
            close()

        data.set_close(tracked_close)
        return data

    monkeypatch.setattr(xr, "open_dataset", tracked_open)
    sources = [
        path,
        {"source": tel_path, "source_loc": "renamed.nc"},
    ]
    table = LmtFileAccessor.from_sources(sources)
    before = table.copy()

    def inspect():
        if operation == "read_metadata":
            table.lmt_file.update_from_source_metadata(inplace=True, on_error=on_error)
            return table
        return LmtFileAccessor.from_sources(sources, mode=operation, on_error=on_error)

    if on_error == "raise":
        with pytest.raises(ValueError, match="No supported source recognized"):
            inspect()
        assert opened == [path]
        pd.testing.assert_frame_equal(table, before)
    else:
        result = inspect()
        assert opened == [path, tel_path]
        assert (
            result.source_trace.map(source_trace_issues).iloc[0][-1].exception_type
            == "ValueError"
        )
        assert result.obsnum.iloc[1] == 145609
        assert [
            item.stage for item in result.source_trace.map(source_trace_issues).iloc[1]
        ] == ["filepath"]
    assert closed == opened


@pytest.mark.parametrize(
    "factory",
    [guess_info_from_sources, guess_or_read_info_from_sources, read_info_from_sources],
)
def test_helpers_forward_error_policy(factory):
    # Invalid writer date intentionally exercises validation, not acquisition naming.
    path = "tel_toltec_2025-99-31_145609_00_0002.nc"
    table = factory([path], on_error="record")
    assert table.source_trace.map(source_trace_issues).iloc[0][0].stage == "filepath"
    row = get_info_from_source(path, on_error="record")
    assert (
        source_trace_issues(row.source_trace)
        == table.source_trace.map(source_trace_issues).iloc[0][:1]
    )


def test_record_mode_does_not_hide_invalid_arguments():
    with pytest.raises(ValidationError):
        FileAccessor.from_sources(
            [{"source_loc": "missing-source.nc"}], on_error="record"
        )
    with pytest.raises(ValidationError):
        FileAccessor.from_sources([], on_error="ignore")  # ty: ignore[invalid-argument-type]


def test_record_mode_empty_table():
    table = LmtFileAccessor.from_sources([], on_error="record")
    assert table.source_trace.map(source_trace_issues).dtype == object and table.empty
    with table.lmt_file.open(on_error="record") as opened:
        assert (
            opened.empty
            and opened.source_trace.map(source_trace_issues).dtype == object
        )


@pytest.mark.parametrize("mode", ["read", "guess_or_read"])
def test_open_failure_does_not_try_metadata(
    accessor_cls, accessor_tel_path, tmp_path, mode
):
    tel_path = accessor_tel_path
    table = accessor_cls.from_sources(
        [tmp_path / "missing.nc", tel_path], mode=mode, on_error="record"
    )
    unidentified, error = table.source_trace.map(source_trace_issues).iloc[0]
    assert unidentified.stage == "filepath" and unidentified.exception_type is None
    assert error.stage == "open" and error.exception_type == "FileNotFoundError"
    assert table.source_trace.map(source_trace_issues).iloc[1] == []


def test_multiple_stages_are_preserved_when_header_read_recovers_identifiers(
    accessor_cls, accessor_tel_path
):
    tel_path = accessor_tel_path
    invalid_location = tel_path.name.replace("2025-10-31", "2025-99-31")
    with xr.open_dataset(tel_path) as source:
        corrupt = source.drop_vars("Header.Source.Ra")
        pairs = [
            {"source": source, "source_loc": invalid_location},
            {"source": corrupt, "source_loc": invalid_location},
        ]
        table = accessor_cls.from_sources(
            pairs, mode="guess_or_read", on_error="record"
        )
        assert [
            error.stage for error in table.source_trace.map(source_trace_issues).iloc[0]
        ] == ["filepath"]
        assert table.obsnum.iloc[0] == 145609
        assert [
            error.stage for error in table.source_trace.map(source_trace_issues).iloc[1]
        ] == [
            "filepath",
            "metadata",
        ]
        assert (
            "Header.Source.Ra"
            in table.source_trace.map(source_trace_issues).iloc[1][-1].message
        )
        assert pd.isna(table.obsnum.iloc[1])


@pytest.mark.parametrize("inplace", [False, True])
def test_retries_replace_stage_errors_without_mutating_copies(
    accessor_cls, accessor_tel_path, inplace
):
    tel_path = accessor_tel_path
    with xr.open_dataset(tel_path) as source:
        corrupt = source.drop_vars("Header.Source.Ra")
        table = accessor_cls.from_sources(
            [
                {"source": corrupt, "source_loc": tel_path},
                {"source": source, "source_loc": tel_path},
            ],
            mode="read",
            on_error="record",
        )
        table.index = [8, 8]
        errors_before = table.source_trace.map(source_trace_issues).iloc[0]
        again = accessor_cls(table).update_from_source_metadata(on_error="record")
        assert len(again.source_trace.map(source_trace_issues).iloc[0]) == 1
        assert again.source_trace.map(source_trace_issues).iloc[0] == errors_before
        assert again.source_trace.map(source_trace_issues).iloc[0] is not errors_before
        with accessor_cls(again).open(io_objs=[source, source]) as live:
            before = live.copy()
            result = accessor_cls(live).update_from_source_metadata(inplace=inplace)
            if inplace:
                assert result is None
                result = live
            else:
                pd.testing.assert_frame_equal(live, before)
            assert result.source_trace.map(source_trace_issues).tolist() == [[], []]
            assert result.obsnum.tolist() == [145609, 145609]
            assert result.index.tolist() == [8, 8]
        assert table.source_trace.map(source_trace_issues).iloc[0] == errors_before


def test_skipped_read_does_not_clear_previous_failure(accessor_cls):
    table = accessor_cls.from_sources(["archive:/unknown.nc"], on_error="record")
    table["data_models"] = [[]]
    event = SourceTraceEvent(
        stage="metadata",
        outcome="error",
        message="Previous failure",
        exception_type="ValueError",
    )
    table["source_trace"] = pd.Series([[event]], dtype=object).array
    updated = accessor_cls(table).update_from_source_metadata(on_error="record")
    assert updated.source_trace.map(source_trace_issues).iloc[0] == [event]


def test_failed_open_rows_are_not_retried_by_metadata_read(
    accessor_cls, accessor_tel_path, tmp_path, monkeypatch
):
    tel_path = accessor_tel_path
    source_open = xr.open_dataset
    calls, closed = [], []

    def open_dataset(path):
        calls.append(path)
        ds = source_open(path)
        close = ds._close
        assert close is not None

        def close_dataset():
            closed.append(path)
            close()

        ds.set_close(close_dataset)
        return ds

    monkeypatch.setattr(xr, "open_dataset", open_dataset)
    missing = tmp_path / "missing.nc"
    table = accessor_cls.from_sources([missing, tel_path, tel_path])
    table.index = pd.Index([8, 3, 8], name="input_row")
    before = table.copy()
    with accessor_cls(table).open() as opened:
        pd.testing.assert_index_equal(opened.index, table.index)
        assert opened.io_obj.iloc[0] is None
        assert opened.io_obj.iloc[1] is opened.io_obj.iloc[2]
        assert [
            row.source
            for row in opened.itertuples(index=False)
            if row.io_obj is not None
        ] == [tel_path, tel_path]
        assert calls == [missing, tel_path]
        assert [
            error.stage
            for error in opened.source_trace.map(source_trace_issues).iloc[0]
        ] == [
            "filepath",
            "open",
        ]
        assert opened.source_trace.map(source_trace_issues).iloc[1:].tolist() == [
            [],
            [],
        ]
        issues_before = opened.source_trace.map(source_trace_issues).tolist()
        assert not opened.metadata_read.any()
        accessor_cls(opened).update_from_source_metadata(inplace=True)
        assert calls == [missing, tel_path]
        assert not closed
        assert opened.metadata_read.tolist() == [False, True, True]
        assert pd.isna(opened.obsnum.iloc[0])
        assert opened.obsnum.iloc[1:].tolist() == [145609, 145609]
        assert opened.t0_mjd.iloc[1:].notna().all()
        assert opened.source_trace.map(source_trace_issues).tolist() == issues_before
    assert closed == [tel_path] and "io_obj" not in opened
    pd.testing.assert_frame_equal(table, before)


def test_borrowed_objects_survive_recorded_metadata_failure(
    accessor_cls, accessor_tel_path
):
    tel_path = accessor_tel_path
    with xr.open_dataset(tel_path) as source:
        closed = []
        corrupt = source.drop_vars("Header.Source.Ra")
        corrupt.set_close(lambda: closed.append(True))
        table = accessor_cls.from_sources([{"source": corrupt, "source_loc": tel_path}])
        with accessor_cls(table).open() as opened:
            assert opened.io_obj.iloc[0] is corrupt
            accessor_cls(opened).update_from_source_metadata(inplace=True)
            assert (
                opened.source_trace.map(source_trace_issues).iloc[0][0].stage
                == "metadata"
            )
        assert not closed


@pytest.mark.parametrize("supply", ["source", "io_objs"])
@pytest.mark.parametrize("on_error", ["raise", "record"])
def test_unrecognized_borrowed_payload_is_rejected_without_closing(
    accessor_cls, accessor_tel_path, supply, on_error
):
    tel_path = accessor_tel_path
    # A valid container without any instrument data tests identification failure.
    data = xr.Dataset()
    closed = []
    data.set_close(lambda: closed.append(True))
    sources = [tel_path, "archive:/unknown.nc"]
    options = {}
    if supply == "source":
        sources[1] = {"source": data, "source_loc": sources[1]}
    elif supply == "io_objs":
        options["io_objs"] = [None, data]
    table = accessor_cls.from_sources(sources)
    before = table.copy()
    if on_error == "raise":
        with (
            pytest.raises(ValueError, match="No supported source recognized"),
            accessor_cls(table).open(on_error="raise", **options),
        ):
            pytest.fail("Unrecognized payload was accepted")
    else:
        with accessor_cls(table).open(**options) as opened:
            assert opened.io_obj.iloc[0] is not None
            assert opened.io_obj.iloc[1] is data
            assert opened.accessor_obj.iloc[1] is None
            assert opened.obsnum.iloc[0] == 145609
            errors = opened.source_trace.map(source_trace_issues).iloc[1]
            assert [error.stage for error in errors] == ["filepath", "identify"]
            assert errors[-1].exception_type == "ValueError"
            assert accessor_cls.__name__ in errors[-1].message
    assert not closed
    pd.testing.assert_frame_equal(table, before)


@pytest.mark.parametrize("on_error", ["raise", "record"])
def test_unrecognized_owned_payload_closes_and_is_not_enriched(
    accessor_cls, accessor_tel_path, tmp_path, monkeypatch, on_error
):
    tel_path = accessor_tel_path
    path = tmp_path / "unrecognized.nc"
    xr.Dataset().to_netcdf(path)
    open_dataset = xr.open_dataset
    opened_paths, closed_paths = [], []

    def tracked_open(path):
        data = open_dataset(path)
        opened_paths.append(path)
        close = data._close
        assert close is not None

        def tracked_close():
            closed_paths.append(path)
            close()

        data.set_close(tracked_close)
        return data

    monkeypatch.setattr(xr, "open_dataset", tracked_open)
    table = accessor_cls.from_sources([tel_path, path, path])
    before = table.copy()
    if on_error == "raise":
        with (
            pytest.raises(ValueError, match="No supported source recognized"),
            accessor_cls(table).open(on_error="raise"),
        ):
            pytest.fail("Unrecognized payload was accepted")
    else:
        with accessor_cls(table).open() as live:
            assert live.io_obj.iloc[0] is not None
            assert live.io_obj.iloc[1] is live.io_obj.iloc[2]
            assert live.accessor_obj.iloc[1:].tolist() == [None, None]
            for errors in live.source_trace.map(source_trace_issues).iloc[1:]:
                assert [error.stage for error in errors] == ["filepath", "identify"]
            assert not closed_paths
    assert opened_paths == [tel_path, path]
    assert closed_paths == [path, tel_path]
    pd.testing.assert_frame_equal(table, before)


@pytest.mark.parametrize("accessor_cls", [LmtFileAccessor, ToltecFileAccessor])
def test_unrecognized_borrowed_tree_is_rejected(accessor_cls):
    tree = xr.DataTree()
    closed = []
    tree.set_close(lambda: closed.append(True))
    table = accessor_cls.from_sources(
        [{"source": tree, "source_loc": "archive:/unknown.nc"}]
    )
    with (
        pytest.raises(ValueError, match="No supported source recognized"),
        accessor_cls(table).open(on_error="raise"),
    ):
        pytest.fail("Unrecognized tree was accepted")
    assert not closed


@pytest.mark.parametrize("error_type", [RuntimeError, AssertionError, TypeError])
def test_unexpected_exceptions_are_not_recorded(tel_path, monkeypatch, error_type):
    from tolteca_datamodels.lmt.tel import LmtTelIOMapper

    def broken_reader(*args, **kwargs):
        msg = "Reader implementation failure"
        raise error_type(msg)

    monkeypatch.setattr(LmtTelIOMapper, "get_metadata", broken_reader)
    with pytest.raises(error_type, match="implementation failure"):
        LmtFileAccessor.from_sources([tel_path], mode="read", on_error="record")


def test_combined_tables_accept_rows_without_trace_column(tel_path):
    guessed = LmtFileAccessor.from_sources([tel_path], on_error="raise").drop(
        columns="source_trace"
    )
    recorded = LmtFileAccessor.from_sources([tel_path], on_error="record")
    table = pd.concat([guessed, recorded])
    result = table.lmt_file.update_from_source_metadata(on_error="record")
    assert result.source_trace.map(source_trace_issues).tolist() == [[], []]
    assert pd.isna(table.source_trace.iloc[0])


def test_raising_failure_preserves_previous_trace_and_metadata(tel_path):
    with xr.open_dataset(tel_path) as source:
        corrupt = source.drop_vars("Header.Source.Ra")
        table = LmtFileAccessor.from_sources(
            [
                {"source": source, "source_loc": tel_path},
                {"source": corrupt, "source_loc": tel_path},
            ],
            on_error="record",
        )
        prior = SourceTraceEvent(
            stage="metadata",
            outcome="error",
            message="Earlier failure",
            exception_type="ValueError",
        )
        table["source_trace"] = pd.Series([[prior], []], dtype=object).array
        before = table.copy()
        with pytest.raises(KeyError):
            table.lmt_file.update_from_source_metadata(inplace=True, on_error="raise")
        pd.testing.assert_frame_equal(table, before)


def test_successful_open_retry_clears_open_error(tel_path, tmp_path):
    source = tmp_path / "later.nc"
    table = LmtFileAccessor.from_sources([source], mode="read", on_error="record")
    assert [
        item.stage for item in table.source_trace.map(source_trace_issues).iloc[0]
    ] == ["filepath", "open"]
    source.symlink_to(tel_path)
    with table.lmt_file.open() as opened:
        opened.lmt_file.update_from_source_metadata(inplace=True)
        assert [
            item.stage for item in opened.source_trace.map(source_trace_issues).iloc[0]
        ] == ["filepath"]
        assert opened.obsnum.iloc[0] == 145609
    assert [
        item.stage for item in table.source_trace.map(source_trace_issues).iloc[0]
    ] == ["filepath", "open"]
