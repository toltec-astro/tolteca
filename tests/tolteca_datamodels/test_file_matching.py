"""Filename evidence and selected capabilities remain independent."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from tolteca_datamodels import data_model as assembly
from tolteca_datamodels.core.data_model import DataModelSpec
from tolteca_datamodels.core.file import (
    CompositeFileWorkflow,
    FileAccessorBase,
    FileAccessorView,
    FileAccessorWorkflow,
    SourceTraceKind,
)
from tolteca_datamodels.core.source import SourceOpener
from tolteca_datamodels.core.types import DataModelInfo
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.lmt.data_model import data_model_spec as lmt_spec
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.toltec.data_model import data_model_spec as toltec_spec
from tolteca_datamodels.toltec.file import ToltecFileAccessor, ToltecFileWorkflow
from tolteca_dummy_instru.data_model import data_model_spec as dummy_spec
from tolteca_dummy_instru.file import (
    DummyInstruCounterFileAccessor,
    DummyInstruFileAccessor,
    DummyInstruTelescopeFileAccessor,
)

_sources = [
    "tel_toltec_2025-10-31_145609_00_0002.nc",
    "toltec8_145605_000_0001_2025_10_31_09_20_06_tune.nc",
    "hwpr_145605_000_0001_2025_10_31_09_20_06.nc",
    "tel_dummy_instru_2025-10-31_145609_00_0002.nc",
    "chan01-tcs-1-0-2-20251031T120000.nc",
]


@pytest.fixture(autouse=True)
def builtins_only(monkeypatch):
    monkeypatch.setattr(
        assembly._data_models, "_entries", {"lmt": lmt_spec, "toltec": toltec_spec}
    )


@pytest.mark.parametrize(
    ("accessor_cls", "unmatched"),
    [
        (FileAccessor, [False, False, False, False, True]),
        (LmtFileAccessor, [False, True, True, False, True]),
        (ToltecFileAccessor, [False, False, False, True, True]),
        (DummyInstruFileAccessor, [True, True, True, False, False]),
    ],
)
def test_guess_matches_current_memberships_without_discarding_or_opening(
    accessor_cls, unmatched, monkeypatch
):
    def unexpected(*args, **kwargs):
        pytest.fail("Filename compatibility opened a resource")

    monkeypatch.setattr(SourceOpener, "open", unexpected)
    file_accessor = accessor_cls.from_sources(_sources, on_error="raise")
    assert type(file_accessor) is accessor_cls
    assert file_accessor.source.tolist() == _sources
    assert file_accessor.table.data_models.tolist() == [
        {"lmt", "toltec"},
        {"toltec"},
        {"toltec"},
        {"lmt", "dummy_instru"},
        {"dummy_instru"},
    ]
    for trace, has_issue, memberships in zip(
        file_accessor.source_trace,
        unmatched,
        file_accessor.table.data_models,
        strict=True,
    ):
        entries = list(trace.iter_entries())
        assert len(entries) == int(has_issue)
        assert not trace.has_error
        if entries:
            entry = entries[0]
            assert entry.kind is SourceTraceKind.file_accessor_rule_unmatched
            assert entry.message is None
            assert entry.context == {
                "accessor_types": [
                    f"{cls.__module__}.{cls.__qualname__}"
                    for cls in FileAccessorView(
                        file_accessor, (accessor_cls,)
                    ).accessor_types
                ],
                "data_models": sorted(memberships),
            }
    assert file_accessor.table.file_data_models.equals(file_accessor.table.data_models)


def test_unidentified_differs_from_invalid_source_and_both_retain_rows():
    file_accessor = ToltecFileAccessor.from_sources(["renamed.nc", None, "/"])
    assert len(file_accessor.table) == 3
    assert file_accessor.table.filepath_info.iloc[0].data_models == frozenset()
    assert [
        [entry.kind for entry in trace.iter_entries()]
        for trace in file_accessor.source_trace
    ] == [
        [SourceTraceKind.file_accessor_rule_unmatched],
        [SourceTraceKind.source_validation_failed],
        [
            SourceTraceKind.filepath_rule_unmatched,
            SourceTraceKind.file_accessor_rule_unmatched,
        ],
    ]
    assert [trace.has_error for trace in file_accessor.source_trace] == [
        False,
        True,
        False,
    ]
    assert file_accessor.source_trace_summary().set_index("kind")[
        "count"
    ].to_dict() == {
        "file_accessor_rule_unmatched": 2,
        "source_validation_failed": 1,
        "filepath_rule_unmatched": 1,
    }
    core = FileAccessorBase.from_sources([*_sources, "renamed.nc"])
    assert core.source_trace_summary().empty


def test_specs_select_capabilities_but_do_not_change_their_matching_rules():
    class AdditionalModel(DataModelInfo):
        name = "additional"

    file_accessor = FileAccessor.from_sources([_sources[0], "renamed.nc"])
    first = file_accessor.view
    spec = DataModelSpec(info_cls=AdditionalModel, file_accessor_cls=ToltecFileAccessor)
    assembly.data_models.register(spec)
    assert first.accessor_types == (LmtFileAccessor, ToltecFileAccessor)
    entries_before = [
        tuple(trace.iter_entries()) for trace in file_accessor.source_trace
    ]
    selected = file_accessor.partial(accessor_types=(ToltecFileAccessor,))
    assert selected.table is file_accessor.table
    assert selected.accessor_types == (ToltecFileAccessor,)
    assert file_accessor.view.accessor_types == first.accessor_types
    assert [
        tuple(trace.iter_entries()) for trace in file_accessor.source_trace
    ] == entries_before
    assert not selected.partial(accessor_types=(LmtFileAccessor,)).accessor_types
    assert not selected.partial(accessor_types=()).accessor_types
    assert (
        selected.partial(accessor_types=(FileAccessorBase,)).accessor_types
        == selected.accessor_types
    )

    # A spec is a high-level association, not a new local matching rule.
    file_accessor.table["data_models"] = [["additional"], []]
    assert list(selected.iter_accessors()) == []
    (entry,) = file_accessor.source_trace.iloc[0].iter_entries()
    assert entry.kind is SourceTraceKind.file_accessor_rule_unmatched

    models = assembly.DataModelSet.from_select(["additional"])
    view = FileAccessorView(file_accessor, accessor_types=models.file_accessor_types)
    assert view.accessor_types == (ToltecFileAccessor,)


def test_later_matching_uses_current_values_not_original_evidence_or_history():
    file_accessor = ToltecFileAccessor.from_sources([_sources[4]])
    view = FileAccessorView(file_accessor, (ToltecFileAccessor,))
    (earlier,) = file_accessor.source_trace.iloc[0].iter_entries()
    file_accessor.table["data_models"] = [["toltec"]]
    assert list(view.iter_accessors()) == [([0], file_accessor)]
    assert list(file_accessor.source_trace.iloc[0].iter_entries()) == [earlier]
    assert file_accessor.table.file_data_models.iloc[0] == {"dummy_instru"}
    assert earlier.context["data_models"] == ["dummy_instru"]

    current = ["foreign"]
    file_accessor.table["data_models"] = [current]
    assert list(view.iter_accessors()) == []
    current.append("toltec")
    entries = list(file_accessor.source_trace.iloc[0].iter_entries())
    assert len(entries) == 1
    assert entries[0].context["data_models"] == ["foreign"]
    assert not file_accessor.source_trace.iloc[0].has_error


def test_explicit_extension_registration_changes_root_capabilities_not_parsing():
    before = FileAccessor.from_sources([_sources[4]])
    previous_view = before.view
    assembly.data_models.register(dummy_spec)
    after = FileAccessor.from_sources([_sources[4]])
    assert before.table.filepath_info.iloc[0] == after.table.filepath_info.iloc[0]
    assert before.source_trace_summary().kind.tolist() == [
        "file_accessor_rule_unmatched"
    ]
    assert after.source_trace_summary().empty
    assert DummyInstruCounterFileAccessor not in previous_view.accessor_types
    assert DummyInstruCounterFileAccessor in after.view.accessor_types
    assert DummyInstruTelescopeFileAccessor in after.view.accessor_types
    assert before.table.channel.tolist() == [1]
    assert "data_kind_dummy_instru" not in before.table
    assert after.table.channel.tolist() == [1]
    assert DummyInstruFileAccessor.from_sources(
        [_sources[4]]
    ).table.channel.tolist() == [1]


def test_partial_matching_a_bare_table_tolerates_missing_memberships():
    table = pd.DataFrame(
        {
            "data_models": [
                ["toltec"],
                np.array(["toltec"]),
                [],
                None,
                pd.NA,
                ["foreign"],
            ]
        }
    )
    file_accessor = ToltecFileAccessor(table)
    view = FileAccessorView(file_accessor, (ToltecFileAccessor,))
    assert "source_trace" not in table
    assert list(view.iter_accessors()) == [([0, 1], file_accessor)]
    assert file_accessor.source_trace_summary().to_dict("records") == [
        {"kind": "file_accessor_rule_unmatched", "count": 4, "total": 6},
    ]
    assert [
        next(trace.iter_entries()).context["data_models"]
        for trace in file_accessor.source_trace.iloc[2:]
    ] == [[], [], [], ["foreign"]]
    no_memberships = LmtFileAccessor(pd.DataFrame(index=[3]))
    assert (
        list(FileAccessorView(no_memberships, (LmtFileAccessor,)).iter_accessors())
        == []
    )
    assert no_memberships.source_trace_summary().kind.tolist() == [
        "file_accessor_rule_unmatched"
    ]


@pytest.mark.parametrize(
    ("module", "class_name", "source", "membership"),
    [
        ("tolteca_datamodels.lmt.file", "LmtFileAccessor", _sources[0], "lmt"),
        ("tolteca_datamodels.toltec.file", "ToltecFileAccessor", _sources[1], "toltec"),
        (
            "tolteca_dummy_instru.file",
            "DummyInstruFileAccessor",
            _sources[4],
            "dummy_instru",
        ),
    ],
)
def test_concrete_workflow_does_not_construct_or_import_data_model_specs(
    module, class_name, source, membership
):
    script = f"""
import sys
from {module} import {class_name}
file_accessor = {class_name}.from_sources([{source!r}])
assert {membership!r} in file_accessor.table.data_models.iloc[0]
assert file_accessor.table.obsnum.notna().all()
assert not any(name.endswith('.data_model') for name in sys.modules)
"""
    subprocess.run(  # noqa: S603 -- fixed import-boundary probe
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("accessor_cls", "membership"),
    [
        (LmtFileAccessor, "lmt"),
        (ToltecFileAccessor, "toltec"),
        (DummyInstruFileAccessor, "dummy_instru"),
    ],
)
def test_selected_factory_matches_direct_usage(
    accessor_cls,
    membership,
):
    assembly.data_models.register(dummy_spec)
    selected = FileAccessorView.from_sources(
        _sources, accessor_types=(accessor_cls,), on_error="raise"
    )
    table = selected.table
    direct = accessor_cls.from_sources(_sources).table
    pd.testing.assert_frame_equal(table, direct)
    assert table.obsnum.notna().all()
    assert table.uid_obs.notna().tolist() == [
        membership in values for values in table.data_models
    ]
    assert table.file_obsnum.notna().all()


def test_concrete_workflow_invokes_its_hooks_without_composite_orchestration(
    monkeypatch,
):
    calls = []

    class ExtensionWorkflow(FileAccessorWorkflow):
        _file_match_rules = ToltecFileWorkflow._file_match_rules

        def _prepare_derived_columns(self, positions):
            calls.append(positions)
            assert self.table.filepath_info.notna().all()
            return {"source_format": self.table.file_ext.take(positions).str.upper()}

    class Extension(FileAccessorBase[ExtensionWorkflow]):
        pass

    def unexpected(*args, **kwargs):
        pytest.fail("Direct guessing depends on composite orchestration")

    with monkeypatch.context() as direct:
        direct.setattr(FileAccessorView, "__init__", unexpected)
        direct.setattr(CompositeFileWorkflow, "_concrete_workflows", unexpected)
        direct.setattr(CompositeFileWorkflow, "_compile_match_rules", unexpected)
        file_accessor = Extension.from_sources(_sources)
    assert calls == [[0, 1, 2]]
    assert file_accessor.table.source_format.iloc[:3].tolist() == ["NC"] * 3
    assert file_accessor.table.source_format.iloc[3:].isna().all()
    assert file_accessor.table.file_ext.tolist() == ["nc"] * len(_sources)
    assert file_accessor.source_trace_summary().to_dict("records") == [
        {"kind": "file_accessor_rule_unmatched", "count": 2, "total": 5}
    ]
    # Root composition uses the same workflow without needing a view either.
    composed = FileAccessor.from_sources(_sources)
    assert composed.table.obsnum.iloc[:4].notna().all()
    assert composed.table.obsnum.iloc[4] == 1
    assert pd.isna(composed.table.uid_raw_obs.iloc[4])
    assert composed.source_trace_summary().to_dict("records") == [
        {"kind": "file_accessor_rule_unmatched", "count": 1, "total": 5}
    ]


def test_empty_selection_prepares_metadata_but_adds_no_derived_columns():
    view = FileAccessorView.from_sources(_sources, accessor_types=())
    assert view.table.filepath_info.notna().all()
    assert view.table.obsnum.notna().all()
    assert "uid_obs" not in view.table
    assert view.file_accessor.source_trace_summary().empty


def test_guess_retains_matching_diagnostics_when_derivation_fails():
    table = FileAccessorBase.from_sources([_sources[1], _sources[4]]).table
    table["uid_raw_obs_file"] = pd.array([0, 0], dtype="Int64")
    file_accessor = ToltecFileAccessor(table)
    with pytest.raises((TypeError, ValueError)):
        file_accessor._workflow.prepare(mode="guess", on_error="raise")
    assert table.filepath_info.notna().all()
    assert table.uid_raw_obs_file.tolist() == [0, 0]
    assert file_accessor.source_trace_summary().to_dict("records") == [
        {"kind": "file_accessor_rule_unmatched", "count": 1, "total": 2}
    ]


def test_dispatch_assigns_overlap_once_without_duplicating_rows_or_traces():
    file_accessor = FileAccessorBase.from_sources([*_sources, "renamed.nc"])
    view = FileAccessorView(
        file_accessor, accessor_types=(LmtFileAccessor, ToltecFileAccessor)
    )
    groups = list(view.iter_accessors())
    assert [(positions, type(accessor)) for positions, accessor in groups] == [
        ([0, 1, 2], ToltecFileAccessor),
        ([3], LmtFileAccessor),
    ]
    assert all(accessor.table is file_accessor.table for _, accessor in groups)
    assert "accessor_cls" not in file_accessor.table
    assert len(file_accessor.table) == 6
    assert "uid_obs" not in file_accessor.table  # Matching does not derive columns.
    assert [
        [entry.kind for entry in trace.iter_entries()]
        for trace in file_accessor.source_trace
    ] == [
        [],
        [],
        [],
        [],
        [SourceTraceKind.file_accessor_rule_unmatched],
        [SourceTraceKind.file_accessor_rule_unmatched],
    ]


@pytest.mark.parametrize(
    ("accessor_types", "expected"),
    [
        (
            (LmtFileAccessor, ToltecFileAccessor),
            {ToltecFileAccessor: [0, 1, 2], LmtFileAccessor: [3]},
        ),
        (
            (ToltecFileAccessor, LmtFileAccessor),
            {ToltecFileAccessor: [0, 1, 2], LmtFileAccessor: [3]},
        ),
        (
            (LmtFileAccessor, ToltecFileAccessor, DummyInstruFileAccessor),
            {
                ToltecFileAccessor: [0, 1, 2],
                DummyInstruTelescopeFileAccessor: [3],
                DummyInstruCounterFileAccessor: [4],
            },
        ),
    ],
)
def test_composite_policy_matches_once_and_derives_each_row_once(
    accessor_types, expected, monkeypatch
):
    from tollan.accessor.pandas.model_table import RecordRuleMapper

    from tolteca_datamodels.lmt_base.file import LmtBaseFileWorkflow

    calls = []
    policies = []
    derive = LmtBaseFileWorkflow._prepare_derived_columns
    map_table = RecordRuleMapper.map_table

    def counted_derive(self, positions):
        calls.append((type(self.file_accessor), positions))
        assert self.table is file_accessor.table
        return derive(self, positions)

    def counted_match(self, table, **kwargs):
        if self is CompositeFileWorkflow._compile_match_rules(selected.accessor_types):
            policies.append(self)
        return map_table(self, table, **kwargs)

    monkeypatch.setattr(LmtBaseFileWorkflow, "_prepare_derived_columns", counted_derive)
    file_accessor = FileAccessorBase.from_sources(_sources)
    file_accessor.table.index = [5, 5, 8, 2, 1]
    selected = FileAccessorView(file_accessor, accessor_types=accessor_types)
    monkeypatch.setattr(RecordRuleMapper, "map_table", counted_match)
    selected._workflow.prepare(mode="guess", on_error="raise")
    assert len(policies) == 1
    assert calls == list(expected.items())
    positions = [position for _, rows in calls for position in rows]
    assert len(positions) == len(set(positions))
    assert len(file_accessor.table) == len(_sources)
    assert file_accessor.table.uid_raw_obs_file.iloc[0] == "tcs-145609-0-2-tel"


def test_iteration_rematches_filtered_and_reordered_rows():
    file_accessor = FileAccessor.from_sources([*_sources, "renamed.nc", None])
    file_accessor.table.index = [9, 9, 2, 4, 4, 1, 5]
    assert [
        (positions, type(accessor))
        for positions, accessor in file_accessor.iter_accessors()
    ] == [
        ([0, 1, 2], ToltecFileAccessor),
        ([3], LmtFileAccessor),
    ]
    reordered = FileAccessor(file_accessor.table.iloc[[3, 0, 3, 4, 6]])
    groups = list(reordered.iter_accessors())
    assert [(positions, type(accessor)) for positions, accessor in groups] == [
        ([0, 2], LmtFileAccessor),
        ([1], ToltecFileAccessor),
    ]
    assert reordered.table.index.tolist() == [4, 9, 4, 4, 5]
    assert all(accessor.table is reordered.table for _, accessor in groups)


def test_iteration_uses_partial_capabilities_and_updates_the_original_table():
    file_accessor = FileAccessor.from_sources(
        [_sources[0], _sources[1], "renamed.nc", None]
    )
    file_accessor.table.index = [5, 5, 5, 5]
    before = file_accessor.table.copy(deep=True)
    selected = file_accessor.partial(accessor_types=(LmtFileAccessor,))
    pd.testing.assert_frame_equal(file_accessor.table, before)
    groups = list(selected.iter_accessors())
    assert len(groups) == 1
    positions, accessor = groups[0]
    assert positions == [0]
    assert type(accessor) is LmtFileAccessor
    assert accessor.table is file_accessor.table
    accessor.table["visited"] = pd.array(
        [False] * len(file_accessor.table), dtype="boolean"
    )
    column = accessor.table.columns.get_loc("visited")
    assert isinstance(column, int)
    accessor.table.iloc[positions, column] = True
    assert file_accessor.table.visited.tolist() == [True, False, False, False]
    assert [
        (positions, type(accessor))
        for positions, accessor in file_accessor.iter_accessors()
    ] == [([0, 1], ToltecFileAccessor)]


def test_empty_selection_does_not_evaluate_rules_or_record_issues():
    file_accessor = FileAccessor.from_sources([_sources[0]])
    assert list(file_accessor.partial(accessor_types=()).iter_accessors()) == []
    assert file_accessor.source_trace_summary().empty
    assert list(FileAccessor.from_sources([]).iter_accessors()) == []


def test_each_operation_matches_current_values_once(monkeypatch):
    from tollan.accessor.pandas.model_table import RecordRuleMapper

    file_accessor = FileAccessor.from_sources(
        [_sources[0], _sources[3], "renamed.nc", None]
    )
    view = file_accessor.view
    calls = []
    map_table = RecordRuleMapper.map_table

    def counted_match(self, table, **kwargs):
        calls.append(table)
        return map_table(self, table, **kwargs)

    monkeypatch.setattr(RecordRuleMapper, "map_table", counted_match)
    # Use current metadata, even when it differs from the filename snapshot.
    file_accessor.table["data_models"] = pd.array(
        [["lmt"], ["lmt", "dummy_instru"], [], None], dtype=object
    )
    assert [
        (positions, type(accessor)) for positions, accessor in view.iter_accessors()
    ] == [([0, 1], LmtFileAccessor)]
    assert len(calls) == 1 and calls[0] is file_accessor.table
    # A later operation must use the then-current metadata, not cached groups.
    file_accessor.table["data_models"] = pd.array(
        [["lmt"], ["toltec"], [], None], dtype=object
    )
    assert [
        (positions, type(accessor)) for positions, accessor in view.iter_accessors()
    ] == [([0], LmtFileAccessor), ([1], ToltecFileAccessor)]
    assert len(calls) == 2


def test_concrete_derivation_excludes_invalid_source_rows():
    class AnyFileAccessorWorkflow(FileAccessorWorkflow):
        _file_match_rules = ({},)

        def _prepare_derived_columns(self, positions):
            return {"handled": pd.Series(data=True, index=positions, dtype="boolean")}

    class AnyFileAccessor(FileAccessorBase[AnyFileAccessorWorkflow]):
        pass

    file_accessor = AnyFileAccessor.from_sources(["renamed.nc", None])
    assert file_accessor.table.handled.iloc[0]
    assert pd.isna(file_accessor.table.handled.iloc[1])
    assert "accessor_cls" not in file_accessor.table
    assert list(file_accessor.source_trace.iloc[0].iter_entries()) == []
    assert [
        entry.kind for entry in file_accessor.source_trace.iloc[1].iter_entries()
    ] == [SourceTraceKind.source_validation_failed]


def test_workflow_without_match_rules_is_inactive_directly_and_in_composition():
    class FilenameOnly(FileAccessorBase[FileAccessorWorkflow]):
        pass

    for file_accessor in (
        FilenameOnly.from_sources(["example.nc"]),
        FileAccessorView.from_sources(["example.nc"], accessor_types=(FilenameOnly,)),
    ):
        with file_accessor.open(on_error="raise") as table:
            assert table.io_obj.isna().all()
            assert table.accessor_obj.isna().all()
        file_accessor.read_metadata(on_error="raise")
        assert FileAccessorBase(file_accessor.table).source_trace_summary().empty


def test_dispatch_policy_is_cached_but_selection_and_current_values_are_not():
    compile_rules = CompositeFileWorkflow._compile_match_rules
    both = compile_rules((LmtFileAccessor, ToltecFileAccessor))
    assert compile_rules((LmtFileAccessor, ToltecFileAccessor)) is both
    lmt = compile_rules((LmtFileAccessor,))
    record = {"data_models": ["lmt", "toltec"]}
    assert both(record) is ToltecFileAccessor
    assert lmt(record) is LmtFileAccessor
    record["data_models"] = ["lmt"]
    assert both(record) is LmtFileAccessor
    assert compile_rules(())(record) is None


def test_cyclic_supersession_is_rejected_when_compiling_policy():
    from graphlib import CycleError

    class FirstWorkflow(FileAccessorWorkflow):
        pass

    class First(FileAccessorBase[FirstWorkflow]):
        pass

    class SecondWorkflow(FileAccessorWorkflow):
        _supersedes_workflow_types = (FirstWorkflow,)

    class Second(FileAccessorBase[SecondWorkflow]):
        pass

    FirstWorkflow._supersedes_workflow_types = (SecondWorkflow,)
    with pytest.raises(CycleError):
        CompositeFileWorkflow._compile_match_rules((First, Second))
