"""Per-row publication of complete metadata and classification updates."""

from typing import ClassVar

import pandas as pd
import pytest
import xarray as xr
from tollan.accessor.pandas.model_table import ModelColumn

from tolteca_datamodels.core.file import source_trace_issues
from tolteca_datamodels.core.source import SourceReaderRules
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.lmt.tel import LmtTelAccessor
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.types import ToltecDataKind


@pytest.fixture
def projection_case(data_lmt_path):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    with xr.open_dataset(path) as original:
        # Deliberate header/filename differences test authoritative replacement.
        sources = [
            original.assign({"Header.Dcs.ObsNum": number}) for number in (10, 20, 30)
        ]
        closed = []
        for source in sources:
            source.set_close(lambda: closed.append(True))
        yield path, sources, closed


def make_handlers(reads, conversion, dtype="Int64"):
    class Reader(LmtTelAccessor):
        @property
        def meta(self):
            meta = super().meta
            reads.append(meta.obsnum)
            return meta

    rules = SourceReaderRules(rules=(({}, Reader),))

    class ObservationColumns(LmtFileAccessor):
        _reader_rules = rules
        _metadata_columns: ClassVar = {"obsnum": ModelColumn("obsnum", dtype="Int64")}

    class AdditionalColumns(ToltecFileAccessor):
        _reader_rules = rules
        _metadata_columns: ClassVar = {
            "header_obsnum": ModelColumn("obsnum", dtype=dtype, converter=conversion)
        }

        @staticmethod
        def _classify_metadata(reader, metadata):
            return None

    return ObservationColumns, AdditionalColumns


def invalid_projection(number):
    if number == 20:
        msg = "Cannot project this observation"
        raise ValueError(msg)
    return number


def invalid_integer(number):
    return "invalid integer" if number == 20 else number


@pytest.mark.parametrize("failure", ["converter", "dtype"])
@pytest.mark.parametrize("on_error", ["record", "raise"])
@pytest.mark.parametrize(
    "operation", ["read_metadata", "inplace", "open", "read", "guess_or_read"]
)
def test_projection_failure_preserves_whole_row(
    projection_case, failure, on_error, operation
):
    path, sources, closed = projection_case
    reads = []

    handlers = make_handlers(
        reads, invalid_projection if failure == "converter" else invalid_integer
    )
    location = "renamed.nc" if operation == "guess_or_read" else path
    pairs = [{"source": source, "source_loc": location} for source in sources]
    table = FileAccessor.from_sources(pairs, accessor_types=handlers)
    if operation not in {"read", "guess_or_read"}:
        table.index = pd.Index([8, 3, 8], name="row")
    before = table.copy()

    def inspect():
        accessor = FileAccessor(table, accessor_types=handlers)
        if operation in {"read", "guess_or_read"}:
            return FileAccessor.from_sources(
                pairs, accessor_types=handlers, mode=operation, on_error=on_error
            )
        if operation == "inplace":
            accessor.update_from_source_metadata(inplace=True, on_error=on_error)
            return table
        if operation == "read_metadata":
            return accessor.update_from_source_metadata(on_error=on_error)
        with accessor.open(on_error=on_error) as opened:
            assert all(
                obj is source
                for obj, source in zip(opened.io_obj, sources, strict=True)
            )
            FileAccessor(opened, accessor_types=handlers).update_from_source_metadata(
                inplace=True, on_error=on_error
            )
        return opened

    if on_error == "raise":
        with pytest.raises(ValueError):
            inspect()
        pd.testing.assert_frame_equal(table, before)
    else:
        result = inspect()
        pd.testing.assert_frame_equal(
            result.iloc[[1]][before.columns].drop(columns="source_trace"),
            before.iloc[[1]].drop(columns="source_trace"),
        )
        assert pd.isna(result.header_obsnum.iloc[1])
        assert result.metadata_read.tolist() == [True, False, True]
        assert result.obsnum.iloc[[0, 2]].tolist() == [10, 30]
        assert result.header_obsnum.iloc[[0, 2]].tolist() == [10, 30]
        assert result.data_kind_toltec.iloc[[0, 2]].tolist() == [
            ToltecDataKind.unknown.value,
            ToltecDataKind.unknown.value,
        ]
        error = result.source_trace.map(source_trace_issues).iloc[1][-1]
        assert error.stage == "metadata" and error.exception_type == "ValueError"
        assert result.index.equals(before.index)
        if operation != "inplace":
            pd.testing.assert_frame_equal(table, before)
    assert reads == [10, 20, 30]
    assert not closed


@pytest.mark.parametrize("column", ["header_obsnum", "data_kind_toltec"])
@pytest.mark.parametrize("on_error", ["record", "raise"])
def test_assignment_error_cannot_publish_earlier_columns(
    projection_case, column, on_error
):
    path, sources, closed = projection_case
    handlers = make_handlers([], None)
    table = FileAccessor.from_sources(
        [{"source": source, "source_loc": path} for source in sources],
        accessor_types=handlers,
    )
    table[column] = pd.Categorical([999] * len(table), categories=[999])
    before = table.copy()
    # An incompatible destination dtype is a programming error, not a bad file.
    with pytest.raises(TypeError):
        FileAccessor(table, accessor_types=handlers).update_from_source_metadata(
            inplace=True, on_error=on_error
        )
    pd.testing.assert_frame_equal(table, before)
    assert not closed


@pytest.mark.parametrize("projected_dtype", ["Int64", "Float64"])
@pytest.mark.parametrize("on_error", ["record", "raise"])
def test_fractional_values_cannot_silently_replace_integers(
    projection_case, projected_dtype, on_error
):
    path, sources, closed = projection_case
    handlers = make_handlers([], lambda number: number + 0.5, projected_dtype)
    table = FileAccessor.from_sources(
        [{"source": source, "source_loc": path} for source in sources],
        accessor_types=handlers,
    )
    table["header_obsnum"] = pd.array([3] * len(table), dtype="Int64")
    before = table.copy()
    # Both projection conversion and assignment into existing integer columns
    # must fail without publishing another handler's values or classifications.
    with pytest.raises(TypeError):
        FileAccessor(table, accessor_types=handlers).update_from_source_metadata(
            inplace=True, on_error=on_error
        )
    pd.testing.assert_frame_equal(table, before)
    assert not closed


def test_prepared_columns_preserve_failed_rows_and_table_structure(projection_case):
    path, sources, closed = projection_case
    reads = []
    handlers = make_handlers(reads, None)
    table = FileAccessor.from_sources(
        [
            {"source": sources[0], "source_loc": path},
            {"source": "https://example.org/unstaged.nc", "source_loc": path},
            {"source": sources[2], "source_loc": path},
        ],
        accessor_types=handlers,
    )
    table.index = pd.MultiIndex.from_tuples(
        [(8, "a"), (3, "b"), (8, "a")], names=["batch", "row"]
    )
    table.attrs["description"] = "Selected observations"
    table["notes"] = ["first", "remote", "last"]
    table["header_obsnum"] = pd.array([1, 2, 3], dtype="Int64")
    before = table.copy()

    result = FileAccessor(table, accessor_types=handlers).update_from_source_metadata(
        inplace=True
    )

    assert result is None
    pd.testing.assert_frame_equal(
        table.iloc[[1]][before.columns].drop(columns="source_trace"),
        before.iloc[[1]].drop(columns="source_trace"),
    )
    assert table.source_trace.iloc[1][-1].stage == "open"
    assert table.source_trace.iloc[1][-1].outcome == "error"
    assert table.metadata_read.tolist() == [True, False, True]
    pd.testing.assert_frame_equal(
        table[before.columns].drop(
            columns=[
                "obsnum",
                "header_obsnum",
                "data_kind_toltec",
                "metadata_read",
                "source_trace",
            ]
        ),
        before.drop(
            columns=[
                "obsnum",
                "header_obsnum",
                "data_kind_toltec",
                "metadata_read",
                "source_trace",
            ]
        ),
    )
    assert table.obsnum.iloc[::2].tolist() == [10, 30]
    assert table.header_obsnum.tolist() == [10, 2, 30]
    assert table.data_kind_toltec.iloc[::2].tolist() == [
        ToltecDataKind.unknown.value,
        ToltecDataKind.unknown.value,
    ]
    assert table[before.columns].dtypes.equals(before.dtypes)
    assert table.attrs == before.attrs
    assert reads == [10, 30]
    assert not closed


def test_failed_projection_does_not_add_columns_and_retry_clears_diagnostic(
    projection_case,
):
    path, sources, _ = projection_case

    def invalid(number):
        msg = "Projection failed"
        raise ValueError(msg)

    bad_handlers = make_handlers([], invalid)
    table = FileAccessor.from_sources(
        [{"source": sources[0], "source_loc": path}], accessor_types=bad_handlers
    )
    before = table.copy()
    FileAccessor(table, accessor_types=bad_handlers).update_from_source_metadata(
        inplace=True
    )
    pd.testing.assert_frame_equal(
        table.drop(columns="source_trace"), before.drop(columns="source_trace")
    )
    assert len(table.source_trace.map(source_trace_issues).iloc[0]) == 1
    result = FileAccessor(
        table, accessor_types=make_handlers([], None)
    ).update_from_source_metadata()
    assert result.source_trace.map(source_trace_issues).iloc[0] == []
    assert result.obsnum.iloc[0] == 10
    assert len(table.source_trace.map(source_trace_issues).iloc[0]) == 1
