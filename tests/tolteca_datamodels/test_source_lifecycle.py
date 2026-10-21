"""Opening, explicit metadata reads and resource-independent reader reuse."""

from types import SimpleNamespace
from typing import ClassVar

import pandas as pd
import pytest
from tollan.accessor.pandas.model_table import ModelColumn

from tolteca_datamodels.core.file import source_trace_issues
from tolteca_datamodels.core.source import (
    SourceOpener,
    SourceReaderRules,
)
from tolteca_datamodels.core.types import DataModelInfo
from tolteca_datamodels.file import FileAccessor


class _LifecycleDataModel(DataModelInfo):
    name = "lifecycle"


@pytest.fixture
def lifecycle(monkeypatch):
    calls = []
    resources = []

    class Resource:
        closed = False
        fail = False
        no_metadata = False

        def close(self):
            self.closed = True

    def acquire(path):
        calls.append("open")
        resource = Resource()
        resources.append(resource)
        return resource

    def identify(resource):
        calls.append("identify")
        return {}

    class Reader:
        def __init__(self, resource):
            calls.append("bind")
            self.source = resource

        @property
        def meta(self):
            assert not self.source.closed
            calls.append("read")
            if self.source.fail:
                message = "Broken metadata"
                raise ValueError(message)
            if self.source.no_metadata:
                return None
            return SimpleNamespace(value=42)

    class Files(FileAccessor[_LifecycleDataModel]):
        _source_opener = SourceOpener(rules=(({"ext": "nc"}, acquire),))
        _reader_rules = SourceReaderRules(
            rules=(({"ready": True}, Reader),),
            prepare_record=lambda obj: identify(obj) | {"ready": True},
        )
        _metadata_columns: ClassVar = {"value": ModelColumn("value", dtype="Int64")}

    return Files, calls, resources


def test_open_stops_after_identification_and_live_update_reuses_accessor(lifecycle):
    files, calls, resources = lifecycle
    table = files.from_sources(["renamed.nc"])
    assert table.metadata_read.tolist() == [False]
    with files(table).open(on_error="raise") as live:
        assert calls == ["open", "identify", "bind"]
        assert "value" not in live
        assert live.accessor_obj.iloc[0].source is live.io_obj.iloc[0]
        reader = live.accessor_obj.iloc[0]
        with files(live).open(on_error="raise") as nested:
            assert nested.accessor_obj.iloc[0] is reader
            assert calls == ["open", "identify", "bind"]
        files(live).update_from_source_metadata(inplace=True, on_error="raise")
        assert calls == ["open", "identify", "bind", "read"]
        assert live.value.tolist() == [42]
        assert live.metadata_read.tolist() == [True]
    assert resources[0].closed
    assert "io_obj" not in live and "accessor_obj" not in live
    assert "value" not in table
    assert table.metadata_read.tolist() == [False]


@pytest.mark.parametrize("mode", ["read", "guess_or_read"])
def test_constructor_and_manual_read_share_live_reader_path(lifecycle, mode):
    files, calls, resources = lifecycle
    table = files.from_sources(["renamed.nc"], mode=mode, on_error="raise")
    assert calls == ["open", "identify", "bind", "read"]
    assert table.metadata_read.tolist() == [True]
    assert resources[0].closed
    assert "io_obj" not in table and "accessor_obj" not in table
    with files(table).open(on_error="raise") as live:
        assert calls == ["open", "identify", "bind", "read", "open", "bind"]
        assert live.accessor_obj.iloc[0].source is resources[1]
        files(live).update_from_source_metadata(inplace=True, on_error="raise")
        assert calls.count("read") == 1
        files(live).update_from_source_metadata(
            inplace=True, force=True, on_error="raise"
        )
        assert calls[-2:] == ["bind", "read"]
        assert calls.count("identify") == 1
    assert all(resource.closed for resource in resources)


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_combined_open_updates_only_missing_metadata(lifecycle, mode):
    files, calls, _ = lifecycle
    table = files.from_sources(["renamed.nc"], mode=mode)
    with files(table).open(update_from_metadata=True) as live:
        assert live.metadata_read.tolist() == [True]
        assert live.value.tolist() == [42]
    assert calls.count("read") == 1
    with files(live).open(update_from_metadata=True, force=True) as reread:
        assert reread.value.tolist() == [42]
    assert calls.count("read") == 2


def test_cached_metadata_update_does_not_open_source(lifecycle):
    files, calls, _ = lifecycle
    table = files.from_sources(["renamed.nc"], mode="read")
    calls.clear()
    refreshed = files(table).update_from_source_metadata()
    assert calls == []
    assert refreshed.metadata_read.tolist() == [True]
    files(table).update_from_source_metadata(force=True)
    assert calls == ["open", "bind", "read"]


def test_failed_read_and_forced_retry_preserve_status_values_and_history(lifecycle):
    files, _, resources = lifecycle
    with files(files.from_sources(["renamed.nc"])).open() as live:
        resources[0].fail = True
        files(live).update_from_source_metadata(inplace=True)
        assert live.metadata_read.tolist() == [False]
        assert "value" not in live
        resources[0].fail = False
        files(live).update_from_source_metadata(inplace=True)
        assert live.metadata_read.tolist() == [True]
        assert not any(
            event.stage == "metadata"
            for event in source_trace_issues(live.source_trace.iloc[0])
        )
        resources[0].fail = True
        files(live).update_from_source_metadata(inplace=True, force=True)
        assert live.metadata_read.tolist() == [True]
        assert live.value.tolist() == [42]
        events = [
            event.outcome
            for event in live.source_trace.iloc[0]
            if event.stage == "metadata"
        ]
        assert events[-3:] == ["error", "success", "error"]
        summary = files(live).source_trace_summary()
        outcome = summary.query('stage == "metadata"').iloc[0]
        assert outcome.outcome == "error" and outcome["count"] == 1
        assert outcome.total == 1


def test_changing_selected_handlers_identifies_with_the_new_reader(lifecycle):
    files, calls, _ = lifecycle
    table = files.from_sources(["renamed.nc"], mode="read")
    other_calls = []

    class OtherReader:
        selected = "other"

        def __init__(self, resource):
            other_calls.append(resource)
            self.source = resource

    class OtherFiles(FileAccessor[_LifecycleDataModel]):
        _source_opener = files._source_opener
        _reader_rules = SourceReaderRules(
            rules=(({}, OtherReader),),
        )

    calls.clear()
    with FileAccessor(table, accessor_types=(OtherFiles,)).open(
        on_error="raise"
    ) as live:
        assert calls == ["open"]
        assert other_calls == [live.io_obj.iloc[0]]
        assert live.accessor_obj.iloc[0].selected == "other"
        assert live.value.tolist() == [42]
        assert live.metadata_read.tolist() == [True]


def test_live_update_can_be_copied_without_changing_source_table(lifecycle):
    files, calls, _ = lifecycle
    with files(files.from_sources(["renamed.nc"])).open() as live:
        before = live.copy()
        result = files(live).update_from_source_metadata()
        assert result.value.tolist() == [42]
        pd.testing.assert_frame_equal(live, before)
        assert result.accessor_obj.iloc[0] is live.accessor_obj.iloc[0]
        assert calls == ["open", "identify", "bind", "read"]


def test_missing_metadata_does_not_claim_completion_or_erase_previous_values(lifecycle):
    files, calls, resources = lifecycle
    with files(files.from_sources(["renamed.nc"])).open() as live:
        resources[0].no_metadata = True
        files(live).update_from_source_metadata(inplace=True)
        assert live.metadata_read.tolist() == [False]
        assert "value" not in live
        assert live.source_trace.iloc[0][-1].outcome == "skipped"

        resources[0].no_metadata = False
        files(live).update_from_source_metadata(inplace=True)
        assert live.metadata_read.tolist() == [True]
        assert live.value.tolist() == [42]

        resources[0].no_metadata = True
        files(live).update_from_source_metadata(inplace=True, force=True)
        assert live.metadata_read.tolist() == [True]
        assert live.value.tolist() == [42]
        assert live.source_trace.iloc[0][-1].outcome == "skipped"
        assert calls.count("read") == 3
        assert calls.count("identify") == 1
