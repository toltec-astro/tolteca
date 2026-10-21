"""Stable handler selections for table creation and subsequent access."""

import pandas as pd
import pytest
from pydantic import ValidationError

from tolteca_datamodels.core.data_model import DataModelSpec
from tolteca_datamodels.core.filepath import FilepathInfo
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_dummy_instru.file import DummyInstruFileAccessor


def test_partial_is_lazy_frozen_and_keeps_accessor_order(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Creating a view must not parse or open sources")

    monkeypatch.setattr(FilepathInfo, "from_source", unexpected)
    monkeypatch.setattr(FileAccessor, "_open_source", unexpected)
    files = FileAccessor.partial(
        accessor_types=(ToltecFileAccessor, LmtFileAccessor, ToltecFileAccessor)
    )
    assert files.accessor_types == (ToltecFileAccessor, LmtFileAccessor)
    with pytest.raises(ValidationError, match="frozen"):
        files.accessor_types = ()


@pytest.mark.parametrize("entry_point", ["view", "constructor", "factory"])
@pytest.mark.parametrize("handler", [str, FileAccessor])
def test_all_entry_points_require_concrete_accessor_types(handler, entry_point):
    def create():
        if entry_point == "view":
            FileAccessor.partial(accessor_types=(handler,))
        elif entry_point == "constructor":
            FileAccessor(
                pd.DataFrame(columns=["source", "source_loc"]),
                accessor_types=(handler,),
            )
        else:
            FileAccessor.from_sources([], accessor_types=(handler,))

    with pytest.raises(ValidationError):
        create()


def test_factory_and_view_apply_each_accessor_once_in_declared_order():
    calls = []

    class First(LmtFileAccessor):
        @staticmethod
        def _guess_derived_columns(table):
            calls.append("first")
            return pd.DataFrame({"value": [1] * len(table)}, index=table.index)

    class Second(LmtFileAccessor):
        @staticmethod
        def _guess_derived_columns(table):
            calls.append("second")
            return pd.DataFrame({"value": [2] * len(table)}, index=table.index)

    classes = (First, Second, First)
    sources = ["tel_toltec_2025-10-31_145609_00_0002.nc"]
    direct = FileAccessor.from_sources(sources, accessor_types=classes)
    assert calls == ["first", "second"]
    assert direct.value.tolist() == [2]
    calls.clear()
    via_view = FileAccessor.partial(accessor_types=classes).from_sources(sources)
    assert calls == ["first", "second"]
    pd.testing.assert_frame_equal(direct, via_view)


def test_narrowing_cannot_add_accessor_types_or_change_parent():
    files = FileAccessor.partial(accessor_types=(ToltecFileAccessor, LmtFileAccessor))
    narrowed = files.partial(
        accessor_types=(DummyInstruFileAccessor, ToltecFileAccessor)
    )
    assert narrowed.accessor_types == (ToltecFileAccessor,)
    assert files.accessor_types == (ToltecFileAccessor, LmtFileAccessor)
    assert narrowed.partial(accessor_types=(LmtFileAccessor,)).accessor_types == ()
    assert files.partial(accessor_types=()).accessor_types == ()
    assert ToltecFileAccessor.partial(
        accessor_types=(ToltecFileAccessor, DummyInstruFileAccessor)
    ).accessor_types == (ToltecFileAccessor,)


def test_same_view_builds_queries_and_rebinds_without_filtering_foreign_rows():
    files = FileAccessor.partial(accessor_types=(ToltecFileAccessor,))
    sources = [
        "toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc",
        "chan01-tcs-1-0-2-20251031T120000.nc",
        "unidentified.nc",
    ]
    table = files.from_sources(sources)
    assert len(table) == len(sources)
    assert table.filepath_info.iloc[1] is not None
    assert "dummy_instru" in table.data_models.iloc[1]
    assert "data_kind_toltec" in table
    assert "data_kind_dummy_instru" not in table
    original = table.copy(deep=True)
    query = "bits_any_set(data_kind_toltec, ToltecDataKind.tune)"
    selected = files(table).query(query)
    assert selected.interface.tolist() == ["toltec0"]
    assert files(table.iloc[::-1]).query(query).index.tolist() == [0]
    pd.testing.assert_frame_equal(table, original)


def test_empty_accessor_types_keep_filename_information_without_opening(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("No handlers means no source reads")

    monkeypatch.setattr(FileAccessor, "_open_source", unexpected)
    files = FileAccessor.partial(accessor_types=())
    table = files.from_sources(
        ["tel_toltec_2025-10-31_145609_00_0002.nc", "unidentified.nc"], mode="read"
    )
    assert len(table) == 2
    assert table.obsnum.iloc[0] == 145609
    assert not table.metadata_read.any()
    assert not any(name.startswith("data_kind_") for name in table)


def test_view_uses_explicit_unregistered_accessor_types_and_ignores_later_registration(
    monkeypatch,
):
    from tolteca_datamodels.core import data_model

    monkeypatch.setattr(data_model._data_models, "_entries", {})

    class LocalFiles(ToltecFileAccessor):
        pass

    files = FileAccessor.partial(accessor_types=(LocalFiles,))
    DataModelSpec(file_accessor_cls=DummyInstruFileAccessor).register()
    assert files.accessor_types == (LocalFiles,)
    table = files.from_sources(["unidentified.nc"])
    assert "data_kind_toltec" in table
    assert "data_kind_dummy_instru" not in table
