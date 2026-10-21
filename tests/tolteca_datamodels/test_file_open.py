"""File-table resource scopes independent of the file format."""

from dataclasses import dataclass
from io import StringIO, TextIOBase
from types import SimpleNamespace
from typing import ClassVar

import pandas as pd
import pytest
from tollan.accessor.pandas.model_table import ModelColumn
from tollan.utils.fileloc import fileloc

from tests.source_helpers import patch_metadata_read
from tolteca_datamodels.core.file import source_trace_issues
from tolteca_datamodels.core.obs_spec import FieldConstraint, ObsSpecBase, Selector
from tolteca_datamodels.core.source import (
    SourceOpener,
    SourceReaderRules,
)
from tolteca_datamodels.core.source.opener import _local_path
from tolteca_datamodels.core.types import DataModelInfo
from tolteca_datamodels.file import FileAccessor


class _TextDataModel(DataModelInfo):
    name = "text"


class _UnconfiguredDataModel(DataModelInfo):
    name = "unconfigured"


class _ChannelDataModel(DataModelInfo):
    name = "channel"


@pytest.fixture
def text_files(tmp_path, monkeypatch):
    path = tmp_path / "observation.txt"
    path.write_text("observation\n")
    owned = []

    def open_text(source):
        resource = _local_path(source).open()
        owned.append(resource)
        return resource

    class TextReader:
        def __init__(self, source):
            self.source = source

        @property
        def meta(self):
            return SimpleNamespace(readable=self.source.readable())

    def prepare_record(data):
        return {} if isinstance(data, TextIOBase) else None

    class TextFiles(FileAccessor[_TextDataModel]):
        _source_opener = SourceOpener(rules=(({"ext": "txt"}, open_text),))
        _reader_rules = SourceReaderRules(
            rules=(({}, TextReader),), prepare_record=prepare_record
        )
        _metadata_columns: ClassVar = {
            "readable": ModelColumn("readable", dtype="boolean")
        }

    return TextFiles, path, owned


def test_non_xarray_scope_borrows_deduplicates_and_uses_subclass_metadata(text_files):
    accessor_cls, path, owned = text_files
    with StringIO("borrowed") as borrowed:
        table = accessor_cls.from_sources(
            [path, {"source": borrowed, "source_loc": "archive:/borrowed.txt"}, path]
        )
        table.index = pd.Index([8, 3, 8], name="input_row")
        table.attrs["label"] = "observations"
        before = table.copy()
        with accessor_cls(table).open() as live:
            assert len(owned) == 1
            assert live.io_obj.iloc[0] is live.io_obj.iloc[2] is owned[0]
            assert live.io_obj.iloc[1] is borrowed
            assert "readable" not in live
            assert live.accessor_obj.iloc[0].source is owned[0]
            accessor_cls(live).update_from_source_metadata(inplace=True)
            assert live.readable.tolist() == [True] * 3
            pd.testing.assert_index_equal(live.index, table.index)
            assert live.attrs == table.attrs
            with accessor_cls(live.iloc[[2, 1]]).open() as nested:
                assert nested.io_obj.iloc[0] is owned[0]
                assert nested.io_obj.iloc[1] is borrowed
            assert "io_obj" not in nested
            assert "accessor_obj" not in nested
            assert not owned[0].closed and not borrowed.closed
        assert owned[0].closed and not borrowed.closed
        assert "io_obj" not in live
        assert "accessor_obj" not in live
        assert live.readable.tolist() == [True] * 3
        pd.testing.assert_frame_equal(table, before)


def test_metadata_failure_cleans_up_temporary_column_and_owned_handles(
    text_files, monkeypatch
):
    accessor_cls, path, owned = text_files

    def fail(reader):
        msg = "Metadata failed"
        raise RuntimeError(msg)

    patch_metadata_read(monkeypatch, accessor_cls._reader_rules, fail)
    FailedFiles = accessor_cls

    table = FailedFiles.from_sources([path])
    before = table.copy()

    def read():
        with FailedFiles(table).open() as live:
            assert live.accessor_obj.iloc[0] is not None
            tables.append(live)
            FailedFiles(live).update_from_source_metadata(inplace=True)

    tables = []
    with pytest.raises(RuntimeError, match="Metadata failed"):
        read()
    assert len(owned) == 1
    assert owned[0].closed
    assert "io_obj" not in tables[0] and "accessor_obj" not in tables[0]
    pd.testing.assert_frame_equal(table, before)


def test_unsupported_format_is_reported_by_generic_opening(tmp_path):
    class Unconfigured(FileAccessor[_UnconfiguredDataModel]):
        pass

    table = Unconfigured.from_sources([tmp_path / "observation.txt"])
    with (
        pytest.raises(ValueError, match="No source opener matches"),
        Unconfigured(table).open(on_error="raise"),
    ):
        pytest.fail("The unconfigured accessor selected a file reader")


def test_local_source_aliases_share_handle_without_reusing_it_for_remote(text_files):
    accessor_cls, path, owned = text_files
    table = accessor_cls.from_sources(
        [path, path.as_uri(), fileloc(path), f"archive:{path}"]
    )

    with accessor_cls(table).open() as live:
        assert len(owned) == 1
        assert all(obj is owned[0] for obj in live.io_obj.iloc[:3])
        assert live.io_obj.iloc[3] is None
        accessor_cls(live).update_from_source_metadata(inplace=True)
        assert live.readable.iloc[:3].all()
        assert pd.isna(live.readable.iloc[3])
        assert live.source_trace.map(source_trace_issues).iloc[3][-1].stage == "open"
        assert (
            "staged locally"
            in live.source_trace.map(source_trace_issues).iloc[3][-1].message
        )
        assert not owned[0].closed
    assert owned[0].closed


def test_subclass_declares_selector_layout_without_lmt_fields():
    @dataclass(frozen=True)
    class ChannelSpec(ObsSpecBase):
        channel: Selector = None

        _field_constraints: ClassVar = {"channel": FieldConstraint(kind="integer")}
        _primary_field: ClassVar = "channel"

    class ChannelFiles(FileAccessor[_ChannelDataModel]):
        _obs_spec_cls = ChannelSpec

    table = ChannelFiles.from_sources(["one", "two", "three"])
    table["channel"] = [2, 0, 1]
    table.index = [4, 8, 4]
    expected = table.iloc[[0, 2]]
    pd.testing.assert_frame_equal(ChannelFiles(table).query_obs_spec("(1,3)"), expected)
    pd.testing.assert_frame_equal(
        ChannelFiles(table).query_obs_spec(ChannelSpec.parse("(1,3)")), expected
    )

    class Unconfigured(FileAccessor[_UnconfiguredDataModel]):
        pass

    with pytest.raises(NotImplementedError, match="No observation-selector class"):
        Unconfigured(table).query_obs_spec("1")
