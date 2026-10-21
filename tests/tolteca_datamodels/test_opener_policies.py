"""Per-data-model opening, shared resolution and resource ownership."""

from types import SimpleNamespace
from typing import ClassVar

import netCDF4
import pandas as pd
import pytest
import xarray as xr
from tollan.accessor.pandas.model_table import ModelColumn
from tollan.utils.fileloc import fileloc

from tolteca_datamodels.core.file import FileAccessor, source_trace_issues
from tolteca_datamodels.core.file._resolution import _SourceOpenerResolver
from tolteca_datamodels.core.source import (
    SourceOpener,
    SourceReaderRules,
    source_opener,
)
from tolteca_datamodels.core.types import DataModelInfo


class _First(DataModelInfo):
    name = "first"


class _Second(DataModelInfo):
    name = "second"


@pytest.fixture
def backends():
    opened = []
    closed = []

    class Resource:
        @property
        def meta(self):
            return SimpleNamespace(value=42)

        def close(self):
            closed.append(self)

    def first(path):
        obj = Resource()
        opened.append(("first", obj))
        return obj

    def second(path):
        obj = Resource()
        opened.append(("second", obj))
        return obj

    class Reader:
        def __init__(self, obj):
            self.source = obj

        @property
        def meta(self):
            return self.source.meta

    class FirstFiles(FileAccessor[_First]):
        _source_opener = SourceOpener(rules=(({"ext": "nc"}, first),))
        _reader_rules = SourceReaderRules(rules=(({}, Reader),))

    class SecondFiles(FileAccessor[_Second]):
        _source_opener = SourceOpener(rules=(({"ext": "nc"}, second),))
        _reader_rules = SourceReaderRules(rules=(({}, Reader),))

    return FirstFiles, SecondFiles, Resource, opened, closed


@pytest.mark.parametrize("mode", ["open", "read"])
@pytest.mark.parametrize("on_error", ["record", "raise"])
def test_conflicting_opener_choices_raise_before_opening(backends, mode, on_error):
    first, second, _, opened, _ = backends
    table = FileAccessor.from_sources(["renamed.nc"], accessor_types=(first, second))
    accessor = FileAccessor(table, accessor_types=(first, second))

    def run():
        if mode == "open":
            with accessor.open(on_error=on_error):
                pytest.fail("Ambiguous opening entered a resource scope")
        else:
            accessor.update_from_source_metadata(on_error=on_error)

    with pytest.raises(RuntimeError, match="Ambiguous source openers"):
        run()
    assert opened == []


@pytest.mark.parametrize("supplied", [False, True])
def test_borrowed_payload_bypasses_conflicting_openers(backends, supplied):
    first, second, resource, opened, closed = backends
    borrowed = resource()
    source = "renamed.nc" if supplied else borrowed
    table = FileAccessor.from_sources(
        [{"source": source, "source_loc": "renamed.nc"}], accessor_types=(first, second)
    )
    accessor = FileAccessor(table, accessor_types=(first, second))
    with accessor.open(io_objs=[borrowed] if supplied else None) as live:
        assert live.io_obj.iloc[0] is borrowed
        assert live.accessor_obj.iloc[0].source is borrowed
    if not supplied:
        accessor.update_from_source_metadata()
    assert not opened and not closed


def test_current_memberships_select_backend_and_reuse_by_backend(backends):
    first, second, _, opened, closed = backends
    table = FileAccessor.from_sources(["same.nc"] * 3, accessor_types=(first, second))
    table["data_models"] = [["first"], ["second"], ["first"]]
    accessor = FileAccessor(table, accessor_types=(first, second))
    with accessor.open(on_error="raise") as live:
        assert [name for name, _ in opened] == ["first", "second"]
        assert live.io_obj.iloc[0] is live.io_obj.iloc[2]
        assert live.io_obj.iloc[0] is not live.io_obj.iloc[1]
        assert not closed
    assert closed == [opened[1][1], opened[0][1]]
    assert len(first.from_sources(["renamed.nc"], mode="read")) == 1
    assert opened[-1][0] == "first"


def test_shared_selected_callable_opens_once_across_distinct_policies(backends):
    first, second, _, opened, closed = backends

    class SharedFiles(second):
        _source_opener = SourceOpener(rules=first._source_opener.rules)

    assert first._source_opener is not SharedFiles._source_opener
    table = FileAccessor.from_sources(
        ["same.nc"] * 2, accessor_types=(first, SharedFiles)
    )
    with FileAccessor(table, accessor_types=(first, SharedFiles)).open() as live:
        assert live.io_obj.iloc[0] is live.io_obj.iloc[1]
        assert live.accessor_obj.iloc[0].source is live.io_obj.iloc[0]
        assert len(opened) == 1
    assert closed == [opened[0][1]]


def test_shared_policy_matches_batch_once_and_retains_supporting_handlers(monkeypatch):
    calls = []
    match = SourceOpener.map_table

    def tracked(policy, table, **kwargs):
        calls.append(policy)
        return match(policy, table, **kwargs)

    monkeypatch.setattr(SourceOpener, "map_table", tracked)
    resolver = _SourceOpenerResolver(
        handlers={"first": source_opener, "second": source_opener}
    )
    choices = resolver.match_table(pd.DataFrame({"ext": ["nc", "other"]}))
    assert len(calls) == 1
    resolved = resolver({key: values[0] for key, values in choices.items()})
    assert resolved is not None
    opener, handlers = resolved
    assert opener is source_opener({"ext": "nc"})
    assert handlers == ("first", "second")
    assert resolver({key: values[1] for key, values in choices.items()}) is None


def test_callable_identity_does_not_require_hashability_or_callable_equality():
    class Opener:
        def __call__(self, path):
            pytest.fail("Selection must not open a source")

        def __eq__(self, other):
            return True

        __hash__ = None

    first, second = Opener(), Opener()
    resolver = _SourceOpenerResolver(handlers={})
    assert resolver({"first": first, "second": first}) == (first, ("first", "second"))
    with pytest.raises(RuntimeError, match="Ambiguous source openers"):
        resolver({"first": first, "second": second})


def test_data_model_opens_netcdf4_and_reads_its_native_object(tmp_path):
    path = tmp_path / "unnamed.nc"
    xr.Dataset({"signal": ("sample", [1.0, 2.0])}).to_netcdf(path)
    opened = []

    def open_netcdf4(path):
        obj = netCDF4.Dataset(path)
        opened.append(obj)
        return obj

    class Reader:
        def __init__(self, obj):
            self.obj = obj

        @property
        def meta(self):
            return SimpleNamespace(sample_count=len(self.obj.dimensions["sample"]))

    class Files(FileAccessor[_First]):
        _source_opener = SourceOpener(
            rules=(({"ext": "nc"}, open_netcdf4), *source_opener.rules)
        )
        _reader_rules = SourceReaderRules(
            rules=(({}, Reader),),
            prepare_record=lambda obj: {} if isinstance(obj, netCDF4.Dataset) else None,
        )
        _metadata_columns: ClassVar = {
            "sample_count": ModelColumn("sample_count", dtype="Int64")
        }

    table = Files.from_sources([path], mode="read", on_error="raise")
    assert table.sample_count.tolist() == [2]
    assert not opened[0].isopen()
    with Files(table).open(on_error="raise") as live:
        assert isinstance(live.io_obj.iloc[0], netCDF4.Dataset)
        assert live.accessor_obj.iloc[0].obj is live.io_obj.iloc[0]
        assert live.io_obj.iloc[0].isopen()
    assert not opened[1].isopen()
    with source_opener.open({"ext": "nc"}, source=path) as default:
        assert isinstance(default, xr.Dataset)


@pytest.mark.parametrize("operation", ["read", "refresh", "open"])
def test_table_opener_receives_original_sources(backends, tmp_path, operation):
    first, _, resource, _, _ = backends
    received = []
    sources = [
        str(tmp_path / "string.nc"),
        tmp_path / "path.nc",
        fileloc(tmp_path / "location.nc"),
        "https://example.invalid/remote.nc",
        fileloc("archive:/remote-location.nc"),
    ]

    def acquire(source):
        received.append(source)
        return resource()

    class Files(first):
        _source_opener = SourceOpener(rules=(({"ext": "nc"}, acquire),))
        _metadata_columns: ClassVar = {"value": ModelColumn("value", dtype="Int64")}

    entries = [
        {"source": source, "source_loc": f"identification-{index}.nc"}
        for index, source in enumerate(sources)
    ]
    if operation == "read":
        result = Files.from_sources(entries, mode="read", on_error="raise")
    else:
        table = Files.from_sources(entries)
        if operation == "refresh":
            result = Files(table).update_from_source_metadata(on_error="raise")
        else:
            with Files(table).open(
                update_from_metadata=True, on_error="raise"
            ) as result:
                assert result.accessor_obj.notna().all()
    assert len(received) == len(sources)
    assert all(
        actual is expected for actual, expected in zip(received, sources, strict=True)
    )
    assert result.value.tolist() == [42] * len(sources)
    assert result.metadata_read.all()


@pytest.mark.parametrize("operation", ["read", "refresh", "open"])
@pytest.mark.parametrize("on_error", ["record", "raise"])
def test_opening_failure_has_one_policy_and_closes_owned_resources(
    backends, operation, on_error
):
    first, _, resource, _, closed = backends
    opened = []

    def acquire(source):
        if source == "broken.nc":
            message = "Missing format header"
            raise KeyError(message)
        if operation != "open" and opened:
            assert closed == opened
        obj = resource()
        opened.append(obj)
        return obj

    class Files(first):
        _source_opener = SourceOpener(rules=(({"ext": "nc"}, acquire),))
        _metadata_columns: ClassVar = {"value": ModelColumn("value", dtype="Int64")}

    table = Files.from_sources(["first.nc", "broken.nc", "last.nc"])
    before = table.copy()

    def run():
        if operation == "read":
            return Files.from_sources(table.source, mode="read", on_error=on_error)
        if operation == "refresh":
            Files(table).update_from_source_metadata(inplace=True, on_error=on_error)
            return table
        with Files(table).open(update_from_metadata=True, on_error=on_error) as result:
            assert not closed
        return result

    if on_error == "raise":
        with pytest.raises(KeyError, match="Missing format header"):
            run()
        assert len(opened) == 1
        pd.testing.assert_frame_equal(table, before)
    else:
        result = run()
        assert result.metadata_read.tolist() == [True, False, True]
        assert result.value.iloc[[0, 2]].tolist() == [42, 42]
        assert pd.isna(result.value.iloc[1])
        [error] = [
            event
            for event in source_trace_issues(result.source_trace.iloc[1])
            if event.stage == "open"
        ]
        assert error.exception_type == "KeyError"
        assert len(opened) == 2
    assert len(closed) == len(opened)
    assert all(obj in closed for obj in opened)
