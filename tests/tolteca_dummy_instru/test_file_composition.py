"""Nested dummy capabilities use one concrete table workflow per row."""

import pandas as pd
import pytest

from tolteca_datamodels import data_model as assembly
from tolteca_datamodels.core.file import (
    CompositeFileAccessor,
    CompositeFileWorkflow,
    FileAccessorBase,
    FileAccessorView,
    SourceTraceKind,
)
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.lmt.data_model import data_model_spec as lmt_spec
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.lmt_base.file import LmtBaseFileWorkflow
from tolteca_datamodels.toltec.data_model import data_model_spec as toltec_spec
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_dummy_instru.data_model import data_model_spec as dummy_spec
from tolteca_dummy_instru.file import (
    DummyInstruCounterFileAccessor,
    DummyInstruFileAccessor,
    DummyInstruTelescopeFileAccessor,
)

_sources = [
    "chan01-tcs-1-0-2-20251031T120000.nc",
    "tel_dummy_instru_2025-10-31_000001_00_0002.nc",
    "tel_toltec_2025-10-31_145609_00_0002.nc",
    "chan03-ics-2-0-1-20251031T120000.nc",
    "unknown.nc",
    None,
]


@pytest.fixture(autouse=True)
def registered_models(monkeypatch):
    monkeypatch.setattr(
        assembly._data_models,
        "_entries",
        {"lmt": lmt_spec, "toltec": toltec_spec, "dummy_instru": dummy_spec},
    )


def test_nested_dummy_composition_derives_each_row_once(monkeypatch):
    calls = []
    derive = LmtBaseFileWorkflow._prepare_derived_columns

    def counted(self, positions):
        calls.append((type(self.file_accessor), positions))
        return derive(self, positions)

    monkeypatch.setattr(LmtBaseFileWorkflow, "_prepare_derived_columns", counted)
    file_accessor = FileAccessor.from_sources(_sources)
    assert calls == [
        (DummyInstruCounterFileAccessor, [0, 3]),
        (DummyInstruTelescopeFileAccessor, [1]),
        (ToltecFileAccessor, [2]),
    ]
    assert len(file_accessor.table) == len(_sources)
    assert file_accessor.table.channel.dropna().tolist() == [1, 3]
    assert file_accessor.table.uid_raw_obs_file.iloc[0] == "tcs-1-0-2-chan01"
    assert file_accessor.table.uid_raw_obs_file.iloc[1] == "tcs-1-0-2-tel"
    assert list(file_accessor.source_trace.iloc[1].iter_entries()) == []
    assert [
        entry.kind for entry in file_accessor.source_trace.iloc[4].iter_entries()
    ] == [SourceTraceKind.file_accessor_rule_unmatched]
    assert [
        entry.kind for entry in file_accessor.source_trace.iloc[5].iter_entries()
    ] == [SourceTraceKind.source_validation_failed]


@pytest.mark.parametrize(
    "selected_cls",
    [
        DummyInstruFileAccessor,
        DummyInstruCounterFileAccessor,
        DummyInstruTelescopeFileAccessor,
    ],
)
def test_selecting_a_group_or_child_matches_its_direct_workflow(selected_cls):
    direct = selected_cls.from_sources(_sources)
    selected = FileAccessorView.from_sources(_sources, accessor_types=(selected_cls,))
    pd.testing.assert_frame_equal(selected.table, direct.table)
    pd.testing.assert_frame_equal(
        selected.query("bits_any_set(data_kind_dummy_instru, unknown)"),
        direct.query("bits_any_set(data_kind_dummy_instru, unknown)"),
    )


def test_partial_cannot_restore_a_sibling_and_does_not_modify_rows():
    file_accessor = DummyInstruFileAccessor.from_sources(_sources)
    before = file_accessor.table.copy(deep=True)
    counter = file_accessor.partial(accessor_types=(DummyInstruCounterFileAccessor,))
    selected = counter.partial(accessor_types=(DummyInstruFileAccessor,))
    assert selected.accessor_types == (DummyInstruCounterFileAccessor,)
    assert (
        selected.partial(
            accessor_types=(DummyInstruTelescopeFileAccessor,)
        ).accessor_types
        == ()
    )
    assert (
        file_accessor.partial(accessor_types=(ToltecFileAccessor,)).accessor_types == ()
    )
    pd.testing.assert_frame_equal(file_accessor.table, before)
    file_accessor.table.index = [4, 4, 2, 4, 1, 1]
    ((positions, child),) = selected.iter_accessors()
    assert positions == [0, 3]
    assert child.table is file_accessor.table
    child.table["visited"] = False
    column = child.table.columns.get_loc("visited")
    assert isinstance(column, int)
    child.table.iloc[positions, column] = True
    assert file_accessor.table.visited.tolist() == [
        True,
        False,
        False,
        True,
        False,
        False,
    ]


def test_repeated_composition_members_are_dispatched_once_and_not_auto_discovered():
    class UndeclaredCounter(DummyInstruCounterFileAccessor):
        pass

    class RepeatedWorkflow(CompositeFileWorkflow):
        _component_accessor_types = (
            DummyInstruFileAccessor,
            DummyInstruCounterFileAccessor,
            DummyInstruFileAccessor,
        )

    class Repeated(CompositeFileAccessor[RepeatedWorkflow]):
        pass

    file_accessor = Repeated.from_sources(_sources)
    assert file_accessor.view.accessor_types == (
        DummyInstruCounterFileAccessor,
        DummyInstruTelescopeFileAccessor,
    )
    assert (
        file_accessor.partial(accessor_types=(UndeclaredCounter,)).accessor_types == ()
    )
    assert [
        (positions, type(child)) for positions, child in file_accessor.iter_accessors()
    ] == [
        ([0, 3], DummyInstruCounterFileAccessor),
        ([1], DummyInstruTelescopeFileAccessor),
    ]


def test_leaf_matching_uses_current_values_without_snapshot_fallback():
    file_accessor = DummyInstruFileAccessor(
        FileAccessorBase.from_sources([_sources[1]] * 3).table
    )
    assert file_accessor.table.interface.tolist() == ["tel"] * 3
    assert [
        (positions, type(child)) for positions, child in file_accessor.iter_accessors()
    ] == [([0, 1, 2], DummyInstruTelescopeFileAccessor)]
    file_accessor.table["interface"] = ["tel", "unrecognized", None]
    assert [
        (positions, type(child)) for positions, child in file_accessor.iter_accessors()
    ] == [([0], DummyInstruTelescopeFileAccessor)]
    assert file_accessor.table.file_interface.tolist() == ["tel"] * 3
    del file_accessor.table["interface"]
    assert (
        list(file_accessor.iter_accessors()) == []
    )  # Matching does not prepare metadata.
    file_accessor.table["interface"] = ["tel"] * 3
    file_accessor.table["data_models"] = pd.array([["foreign"], [], None], dtype=object)
    assert list(file_accessor.iter_accessors()) == []


def test_restricting_the_composite_removes_its_precedence_over_generic_lmt():
    file_accessor = FileAccessorBase.from_sources([_sources[1]])
    full = FileAccessorView(file_accessor, (LmtFileAccessor, DummyInstruFileAccessor))
    assert [type(child) for _, child in full.iter_accessors()] == [
        DummyInstruTelescopeFileAccessor
    ]
    restricted = full.partial(
        accessor_types=(LmtFileAccessor, DummyInstruCounterFileAccessor)
    )
    assert [type(child) for _, child in restricted.iter_accessors()] == [
        LmtFileAccessor
    ]


def test_cyclic_composition_fails_without_changing_the_table():
    class FirstWorkflow(CompositeFileWorkflow):
        pass

    class First(CompositeFileAccessor[FirstWorkflow]):
        pass

    class SecondWorkflow(CompositeFileWorkflow):
        _component_accessor_types = (First,)

    class Second(CompositeFileAccessor[SecondWorkflow]):
        pass

    FirstWorkflow._component_accessor_types = (Second,)
    file_accessor = First(pd.DataFrame({"note": ["unchanged"]}))
    with pytest.raises(ValueError, match="Cyclic file-accessor composition"):
        file_accessor.partial(accessor_types=(First,))
    assert file_accessor.table.to_dict("list") == {"note": ["unchanged"]}
