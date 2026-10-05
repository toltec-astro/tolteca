"""File-table resource scopes independent of the file format."""

from dataclasses import dataclass
from io import StringIO, TextIOBase
from typing import ClassVar

import pandas as pd
import pytest
from tollan.utils.fileloc import fileloc

from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.obs_spec import FieldConstraint, ObsSpecBase, Selector


@pytest.fixture
def text_files(tmp_path):
    path = tmp_path / "observation.txt"
    path.write_text("observation\n")
    owned = []

    def open_text(path):
        resource = path.open()
        owned.append(resource)
        return resource

    class TextFiles(FileAccessor):
        _io_obj_types = (TextIOBase,)
        _source_opener = open_text

        def _read_metadata(self, rows, *, diagnostics):
            self._obj["readable"] = [row.io_obj.readable() for _, row in rows]

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
            assert live.readable.tolist() == [True] * 3
            pd.testing.assert_index_equal(live.index, table.index)
            assert live.attrs == table.attrs
            with accessor_cls(live.iloc[[2, 1]]).open() as nested:
                assert nested.io_obj.iloc[0] is owned[0]
                assert nested.io_obj.iloc[1] is borrowed
            assert "io_obj" not in nested
            assert not owned[0].closed and not borrowed.closed
        assert owned[0].closed and not borrowed.closed
        assert "io_obj" not in live
        assert live.readable.tolist() == [True] * 3
        pd.testing.assert_frame_equal(table, before)


def test_metadata_failure_cleans_up_temporary_column_and_owned_handles(text_files):
    accessor_cls, path, owned = text_files
    temporary_tables = []

    class FailedFiles(accessor_cls):
        def _read_metadata(self, rows, *, diagnostics):
            temporary_tables.append(self._obj)
            msg = "Metadata failed"
            raise RuntimeError(msg)

    table = FailedFiles.from_sources([path])
    before = table.copy()
    with (
        pytest.raises(RuntimeError, match="Metadata failed"),
        FailedFiles(table).open(),
    ):
        pytest.fail("The scope yielded despite a metadata failure")
    assert len(owned) == len(temporary_tables) == 1
    assert owned[0].closed
    assert "io_obj" not in temporary_tables[0]
    pd.testing.assert_frame_equal(table, before)


def test_base_accessor_does_not_choose_a_file_reader(tmp_path):
    table = FileAccessor.from_sources([tmp_path / "observation.txt"])
    with (
        pytest.raises(NotImplementedError, match="No source opener"),
        FileAccessor(table).open(),
    ):
        pytest.fail("The base accessor selected a file reader")


def test_local_source_aliases_share_handle_without_reusing_it_for_remote(text_files):
    accessor_cls, path, owned = text_files
    table = accessor_cls.from_sources(
        [path, path.as_uri(), fileloc(path), f"archive:{path}"]
    )

    # Read the three usable rows while keeping the remote row and its diagnostic.
    class Files(accessor_cls):
        def _read_metadata(self, rows, *, diagnostics):
            assert [position for position, _ in rows] == [0, 1, 2]

    with Files(table).open() as live:
        assert len(owned) == 1
        assert all(obj is owned[0] for obj in live.io_obj.iloc[:3])
        assert live.io_obj.iloc[3] is None
        assert live.diagnostics.iloc[3][-1].stage == "open"
        assert "staged locally" in live.diagnostics.iloc[3][-1].message
        assert not owned[0].closed
    assert owned[0].closed


def test_subclass_declares_selector_layout_without_lmt_fields():
    @dataclass(frozen=True)
    class ChannelSpec(ObsSpecBase):
        channel: Selector = None

        _field_constraints: ClassVar = {"channel": FieldConstraint(kind="integer")}
        _primary_field: ClassVar = "channel"

    class ChannelFiles(FileAccessor):
        _obs_spec_cls = ChannelSpec

    table = ChannelFiles.from_sources(["one", "two", "three"])
    table["channel"] = [2, 0, 1]
    table.index = [4, 8, 4]
    expected = table.iloc[[0, 2]]
    pd.testing.assert_frame_equal(ChannelFiles(table).query_obs_spec("(1,3)"), expected)
    pd.testing.assert_frame_equal(
        ChannelFiles(table).query_obs_spec(ChannelSpec.parse("(1,3)")), expected
    )
    with pytest.raises(NotImplementedError, match="No observation-selector class"):
        FileAccessor(table).query_obs_spec("1")
