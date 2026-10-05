"""Source identification diagnostics, retries, and resource ownership."""

import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from tolteca_datamodels.file import FileAccessor, FilepathInfo, SourceInfoDiagnostic
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


@pytest.mark.parametrize("stage", ["filepath", "open", "metadata"])
@pytest.mark.parametrize("exception_type", [None, "ValueError"])
def test_diagnostic_validates_and_dumps_named_fields(stage, exception_type):
    data = {"stage": stage, "message": "Identification failed"}
    if exception_type is not None:
        data["exception_type"] = exception_type
    diagnostic = SourceInfoDiagnostic.model_validate(data)
    assert diagnostic.model_dump(mode="json") == data | {
        "exception_type": exception_type
    }
    assert (
        SourceInfoDiagnostic.model_validate_json(diagnostic.model_dump_json())
        == diagnostic
    )
    with pytest.raises(ValidationError, match="frozen_instance"):
        diagnostic.message = "Changed"  # ty: ignore[invalid-assignment]


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
def test_diagnostic_rejects_invalid_data(update, error):
    with pytest.raises(ValidationError, match=error):
        SourceInfoDiagnostic.model_validate(
            {"stage": "filepath", "message": "No filename match"} | update
        )


def test_filename_failure_and_nonmatch_are_distinct(tel_path):
    invalid = tel_path.with_name(tel_path.name.replace("2025-10-31", "2025-99-31"))
    with pytest.raises(ValueError, match="does not match"):
        FilepathInfo.from_source(invalid, raise_on_error=False)
    with pytest.raises(ValueError, match="does not match"):
        LmtFileAccessor.from_sources([invalid])
    table = LmtFileAccessor.from_sources(
        [invalid, "unknown.nc", tel_path], on_error="record"
    )
    (error,) = table.diagnostics.iloc[0]
    assert error.stage == "filepath" and error.exception_type == "ValueError"
    assert table.diagnostics.iloc[1] == [
        SourceInfoDiagnostic(
            stage="filepath", message="No accepted filename pattern matched"
        )
    ]
    assert table.diagnostics.iloc[2] == []
    assert pd.isna(table.obsnum.iloc[0])
    assert table.obsnum.iloc[2] == 145609
    assert table.diagnostics.iloc[1] is not table.diagnostics.iloc[2]


@pytest.mark.parametrize("on_error", ["raise", "record"])
def test_guess_mode_records_nonmatches_without_opening(
    accessor_cls, monkeypatch, on_error
):
    def unexpected(*args, **kwargs):
        pytest.fail("Filename identification attempted I/O")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    monkeypatch.setattr(accessor_cls, "_source_opener", xr.open_dataset)
    table = accessor_cls.from_sources(["renamed.nc"], mode="guess", on_error=on_error)
    assert table.diagnostics.iloc[0] == [
        SourceInfoDiagnostic(
            stage="filepath", message="No accepted filename pattern matched"
        )
    ]
    assert table.obsnum.isna().all()
    assert "errors" not in table


def test_filename_retry_replaces_or_clears_diagnostic(tel_path):
    from tollan.utils.fileloc import fileloc

    from tolteca_datamodels.file import _SourceInfoDiagnostics

    table = LmtFileAccessor.from_sources(["renamed.nc"])
    previous = table.diagnostics.iloc[0]
    diagnostics = _SourceInfoDiagnostics("record")
    table.lmt_file._guess_columns(diagnostics=diagnostics)
    diagnostics.update_table(table)
    assert table.diagnostics.iloc[0] == previous
    assert table.diagnostics.iloc[0] is not previous
    table["source_loc"] = pd.Series([fileloc(tel_path)], dtype=object)
    diagnostics = _SourceInfoDiagnostics("record")
    table.lmt_file._guess_columns(diagnostics=diagnostics)
    diagnostics.update_table(table)
    assert table.diagnostics.iloc[0] == []
    assert table.obsnum.iloc[0] == 145609


def test_unknown_metadata_diagnostic_is_cleared_by_successful_retry(
    accessor_cls, tel_path
):
    table = accessor_cls.from_sources(
        [{"source": xr.Dataset(), "source_loc": "renamed.nc"}], mode="read"
    )
    assert [item.stage for item in table.diagnostics.iloc[0]] == [
        "filepath",
        "metadata",
    ]
    with xr.open_dataset(tel_path) as source:
        table["source"] = pd.Series([source], dtype=object)
        updated = accessor_cls(table).read_metadata()
    assert [item.stage for item in updated.diagnostics.iloc[0]] == ["filepath"]
    assert updated.obsnum.iloc[0] == 145609


@pytest.mark.parametrize(
    "factory",
    [guess_info_from_sources, guess_or_read_info_from_sources, read_info_from_sources],
)
def test_helpers_forward_error_policy(factory):
    # Invalid writer date intentionally exercises validation, not acquisition naming.
    path = "tel_toltec_2025-99-31_145609_00_0002.nc"
    table = factory([path], on_error="record")
    assert table.diagnostics.iloc[0][0].stage == "filepath"
    row = get_info_from_source(path, on_error="record")
    assert row.diagnostics == table.diagnostics.iloc[0][:1]


def test_record_mode_does_not_hide_invalid_arguments():
    with pytest.raises(ValidationError):
        FileAccessor.from_sources(
            [{"source_loc": "missing-source.nc"}], on_error="record"
        )
    with pytest.raises(ValidationError):
        FileAccessor.from_sources([], on_error="ignore")  # ty: ignore[invalid-argument-type]


def test_record_mode_empty_table():
    table = LmtFileAccessor.from_sources([], on_error="record")
    assert table.diagnostics.dtype == object and table.empty
    with table.lmt_file.open(on_error="record") as opened:
        assert opened.empty and opened.diagnostics.dtype == object


@pytest.mark.parametrize("mode", ["read", "guess_or_read"])
def test_open_failure_does_not_try_metadata(accessor_cls, tel_path, tmp_path, mode):
    table = accessor_cls.from_sources(
        [tmp_path / "missing.nc", tel_path], mode=mode, on_error="record"
    )
    unidentified, error = table.diagnostics.iloc[0]
    assert unidentified.stage == "filepath" and unidentified.exception_type is None
    assert error.stage == "open" and error.exception_type == "FileNotFoundError"
    assert table.diagnostics.iloc[1] == []


def test_multiple_stages_are_preserved_when_header_read_recovers_identifiers(
    accessor_cls, tel_path
):
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
        assert [error.stage for error in table.diagnostics.iloc[0]] == ["filepath"]
        assert table.obsnum.iloc[0] == 145609
        assert [error.stage for error in table.diagnostics.iloc[1]] == [
            "filepath",
            "metadata",
        ]
        assert "Header.Source.Ra" in table.diagnostics.iloc[1][-1].message
        assert pd.isna(table.obsnum.iloc[1])


@pytest.mark.parametrize("inplace", [False, True])
def test_retries_replace_stage_errors_without_mutating_copies(
    accessor_cls, tel_path, inplace
):
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
        errors_before = table.diagnostics.iloc[0]
        again = accessor_cls(table).read_metadata(on_error="record")
        assert len(again.diagnostics.iloc[0]) == 1
        assert again.diagnostics.iloc[0] == errors_before
        assert again.diagnostics.iloc[0] is not errors_before
        again["source"] = pd.Series([source, source], dtype=object).array
        before = again.copy()
        result = accessor_cls(again).read_metadata(inplace=inplace)
        if inplace:
            assert result is None
            result = again
        else:
            pd.testing.assert_frame_equal(again, before)
        assert result.diagnostics.tolist() == [[], []]
        assert result.obsnum.tolist() == [145609, 145609]
        assert result.index.tolist() == [8, 8]
        assert table.diagnostics.iloc[0] == errors_before


def test_skipped_read_does_not_clear_previous_failure(accessor_cls):
    table = accessor_cls.from_sources(["archive:/unknown.nc"], on_error="record")
    diagnostic = SourceInfoDiagnostic(
        stage="metadata", message="Previous failure", exception_type="ValueError"
    )
    table["diagnostics"] = pd.Series([[diagnostic]], dtype=object).array
    updated = accessor_cls(table).read_metadata(on_error="record")
    assert updated.diagnostics.iloc[0] == [diagnostic]


def test_failed_open_rows_are_not_retried_by_metadata_read(
    accessor_cls, tel_path, tmp_path, monkeypatch
):
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
    monkeypatch.setattr(accessor_cls, "_source_opener", xr.open_dataset)
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
        assert [error.stage for error in opened.diagnostics.iloc[0]] == [
            "filepath",
            "open",
        ]
        assert opened.diagnostics.iloc[1:].tolist() == [[], []]
    assert closed == [tel_path] and "io_obj" not in opened
    pd.testing.assert_frame_equal(table, before)


def test_borrowed_objects_survive_recorded_metadata_failure(accessor_cls, tel_path):
    with xr.open_dataset(tel_path) as source:
        closed = []
        corrupt = source.drop_vars("Header.Source.Ra")
        corrupt.set_close(lambda: closed.append(True))
        table = accessor_cls.from_sources([{"source": corrupt, "source_loc": tel_path}])
        with accessor_cls(table).open() as opened:
            assert opened.io_obj.iloc[0] is corrupt
            assert opened.diagnostics.iloc[0][0].stage == "metadata"
        assert not closed


@pytest.mark.parametrize("supply", ["source", "io_objs", "column"])
@pytest.mark.parametrize("on_error", ["raise", "record"])
def test_unrecognized_borrowed_payload_is_rejected_without_closing(
    accessor_cls, tel_path, supply, on_error
):
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
    if supply == "column":
        table["io_obj"] = pd.Series([None, data], dtype=object).array
    before = table.copy()
    if on_error == "raise":
        with (
            pytest.raises(ValueError, match="No supported dataset recognized"),
            accessor_cls(table).open(on_error="raise", **options),
        ):
            pytest.fail("Unrecognized payload was accepted")
    else:
        with accessor_cls(table).open(**options) as opened:
            assert opened.io_obj.iloc[0] is not None
            assert opened.io_obj.iloc[1] is None
            assert opened.obsnum.iloc[0] == 145609
            errors = opened.diagnostics.iloc[1]
            assert [error.stage for error in errors] == ["filepath", "open"]
            assert errors[-1].exception_type == "ValueError"
            assert accessor_cls.__name__ in errors[-1].message
    assert not closed
    pd.testing.assert_frame_equal(table, before)


@pytest.mark.parametrize("on_error", ["raise", "record"])
def test_unrecognized_owned_payload_closes_and_is_not_enriched(
    accessor_cls, tel_path, tmp_path, monkeypatch, on_error
):
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
    monkeypatch.setattr(accessor_cls, "_source_opener", xr.open_dataset)
    table = accessor_cls.from_sources([tel_path, path, path])
    before = table.copy()
    if on_error == "raise":
        with (
            pytest.raises(ValueError, match="No supported dataset recognized"),
            accessor_cls(table).open(on_error="raise"),
        ):
            pytest.fail("Unrecognized payload was accepted")
    else:
        with accessor_cls(table).open() as live:
            assert live.io_obj.iloc[0] is not None
            assert live.io_obj.iloc[1:].tolist() == [None, None]
            for errors in live.diagnostics.iloc[1:]:
                assert [error.stage for error in errors] == ["filepath", "open"]
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
        pytest.raises(ValueError, match="No supported dataset recognized"),
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


def test_combined_tables_accept_rows_without_diagnostics_column(tel_path):
    guessed = LmtFileAccessor.from_sources([tel_path])
    recorded = LmtFileAccessor.from_sources([tel_path], on_error="record")
    table = pd.concat([guessed, recorded])
    result = table.lmt_file.read_metadata(on_error="record")
    assert result.diagnostics.tolist() == [[], []]
    assert pd.isna(table.diagnostics.iloc[0])


def test_default_failure_preserves_previous_diagnostics_and_metadata(tel_path):
    with xr.open_dataset(tel_path) as source:
        corrupt = source.drop_vars("Header.Source.Ra")
        table = LmtFileAccessor.from_sources(
            [
                {"source": source, "source_loc": tel_path},
                {"source": corrupt, "source_loc": tel_path},
            ],
            on_error="record",
        )
        prior = SourceInfoDiagnostic(
            stage="metadata", message="Earlier failure", exception_type="ValueError"
        )
        table["diagnostics"] = pd.Series([[prior], []], dtype=object).array
        before = table.copy()
        with pytest.raises(KeyError):
            table.lmt_file.read_metadata(inplace=True)
        pd.testing.assert_frame_equal(table, before)


def test_successful_open_retry_clears_open_error(tel_path, tmp_path):
    source = tmp_path / "later.nc"
    table = LmtFileAccessor.from_sources([source], mode="read", on_error="record")
    assert [item.stage for item in table.diagnostics.iloc[0]] == ["filepath", "open"]
    source.symlink_to(tel_path)
    with table.lmt_file.open() as opened:
        assert [item.stage for item in opened.diagnostics.iloc[0]] == ["filepath"]
        assert opened.obsnum.iloc[0] == 145609
    assert [item.stage for item in table.diagnostics.iloc[0]] == ["filepath", "open"]
