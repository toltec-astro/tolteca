"""File-table metadata updates, independent of individual reader implementations."""

import re
from dataclasses import dataclass
from typing import ClassVar

import pandas as pd
import pytest
from tollan.accessor.pandas.model_table import (
    Contains,
    ModelColumn,
    ModelTableAccessor,
    RecordRuleMapper,
)
from tollan.utils.log import logger

from tolteca_datamodels.core.file import (
    FileAccessorBase,
    FileAccessorView,
    FileAccessorWorkflow,
)
from tolteca_datamodels.core.source import SourceReaderRules
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.types import ToltecDataKind


@dataclass
class _Metadata:
    obsnum: int
    quality: float


@dataclass
class _Resource:
    metadata: _Metadata | Exception | None
    reads: int = 0
    bindings: int = 0


class _Reader:
    def __init__(self, resource):
        self.resource = resource
        resource.bindings += 1

    @property
    def meta(self):
        self.resource.reads += 1
        value = self.resource.metadata
        if isinstance(value, Exception):
            raise value
        return value


class _FileAccessorWorkflow(FileAccessorWorkflow):
    _file_match_rules = ({},)
    _source_reader_rules = SourceReaderRules(rules=(({}, _Reader),))
    _metadata_columns: ClassVar = {
        _Metadata: {
            "obsnum": ModelColumn("obsnum", dtype="Int64"),
            "quality": ModelColumn("quality", dtype="Float64"),
        },
    }


class _FileAccessor(FileAccessorBase[_FileAccessorWorkflow]):
    pass


def _make_file_accessor(*resources):
    file_accessor = _FileAccessor.from_sources(
        [{"source": resource, "source_loc": "example.nc"} for resource in resources]
    )
    file_accessor.table.index = ["repeated"] * len(resources)
    file_accessor.table["obsnum"] = pd.array([0] * len(resources), dtype="Int64")
    file_accessor.table["quality"] = pd.array([0.0] * len(resources), dtype="Float64")
    return file_accessor


def test_selected_factory_logs_metadata_failures_and_completed_summary():
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        file_accessor = FileAccessorView.from_sources(
            [
                {"source": _Resource(_Metadata(42, 0.5)), "source_loc": "good.nc"},
                {
                    "source": _Resource(ValueError("Incomplete metadata")),
                    "source_loc": "bad.nc",
                },
            ],
            accessor_types=(_FileAccessor,),
            mode="read",
        )
    finally:
        logger.remove(sink)

    output = "\n".join(messages)
    assert "metadata_read_failed" in output
    assert "Incomplete metadata" in output
    assert "bad.nc" in output
    summary = FileAccessorBase(file_accessor.table).source_trace_summary()
    assert summary.to_dict("records") == [
        {"kind": "metadata_read_failed", "count": 1, "total": 2}
    ]
    assert re.search(r"metadata_read_failed\s+1\s+2\b", output)


def test_repeated_metadata_failures_retain_latest_details_per_kind():
    resource = _Resource(ValueError("First attempt"))
    file_accessor = _make_file_accessor(resource)
    file_accessor.read_metadata()
    trace = file_accessor.source_trace.iloc[0]
    (first,) = trace.iter_error_entries()

    resource.metadata = ValueError("Latest attempt")
    file_accessor.read_metadata()
    (latest,) = trace.iter_entries()
    assert latest.kind.value == "metadata_read_failed"
    assert latest.message == "Latest attempt"
    assert first.message == "First attempt"
    assert list(trace.iter_error_entries()) == [latest]
    assert file_accessor.source_trace_summary().to_dict("records") == [
        {"kind": "metadata_read_failed", "count": 1, "total": 1}
    ]

    resource.metadata = _Metadata(42, 0.5)
    file_accessor.read_metadata()
    assert file_accessor.table.obsnum.tolist() == [42]
    assert list(trace.iter_error_entries()) == [latest]
    assert trace.has_error
    assert {"io_obj", "accessor_obj"}.isdisjoint(file_accessor.table)


def _error_kinds(file_accessor):
    return [
        [entry.kind.value for entry in trace.iter_error_entries()]
        for trace in file_accessor.source_trace
    ]


def test_each_metadata_read_reuses_live_readers_and_projects_once(monkeypatch):
    resources = [_Resource(_Metadata(10, 0.5)), _Resource(_Metadata(20, 1.5))]
    file_accessor = _make_file_accessor(*resources)
    projections = []
    update = ModelTableAccessor.update_from_models
    selections = {"opener": 0, "reader": 0}
    map_table = RecordRuleMapper.map_table

    def record_selection(self, *args, **kwargs):
        if self is file_accessor._workflow._source_opener:
            selections["opener"] += 1
        elif self is file_accessor._workflow._source_reader_rules:
            selections["reader"] += 1
        return map_table(self, *args, **kwargs)

    def record_update(self, models, **kwargs):
        projections.append(list(models))
        return update(self, models, **kwargs)

    def unexpected_derivation(self, positions):
        pytest.fail("Metadata reading reran filename derivation")

    monkeypatch.setattr(ModelTableAccessor, "update_from_models", record_update)
    monkeypatch.setattr(RecordRuleMapper, "map_table", record_selection)
    monkeypatch.setattr(
        _FileAccessorWorkflow, "_prepare_derived_columns", unexpected_derivation
    )
    with file_accessor.open() as table:
        assert [r.reads for r in resources] == [0, 0]
        assert file_accessor.read_metadata() is file_accessor
        assert table.obsnum.tolist() == [10, 20]
        assert projections == [[0, 1]]
        readers = list(table.accessor_obj)
        assert file_accessor.read_metadata() is file_accessor
        assert projections == [[0, 1], [0, 1]]
        assert list(table.accessor_obj) == readers
        assert [(r.bindings, r.reads) for r in resources] == [(1, 2), (1, 2)]
        assert selections == {"opener": 1, "reader": 1}
    assert {"io_obj", "accessor_obj"}.isdisjoint(file_accessor.table)
    with file_accessor.open():
        assert [r.reads for r in resources] == [2, 2]
        file_accessor.read_metadata()
        assert [r.reads for r in resources] == [3, 3]
    assert projections == [[0, 1], [0, 1], [0, 1]]
    assert selections == {"opener": 2, "reader": 2}


def test_metadata_read_reuses_live_subset_and_opens_remaining_rows():
    class TelescopeWorkflow(_FileAccessorWorkflow):
        _file_match_rules = ({"data_models": Contains(value="lmt")},)

    class TelescopeAccessor(FileAccessorBase[TelescopeWorkflow]):
        pass

    resources = [_Resource(_Metadata(10, 0.5)), _Resource(_Metadata(20, 1.5))]
    locations = [
        "tel_toltec_2025-10-31_145609_00_0002.nc",
        "toltec0_145609_000_0002_2025_10_31_09_27_57.nc",
    ]
    file_accessor = _FileAccessor.from_sources(
        [
            {"source": resource, "source_loc": location}
            for resource, location in zip(resources, locations, strict=True)
        ]
    )
    file_accessor.table.index = ["repeated"] * len(resources)

    with TelescopeAccessor(file_accessor.table).open() as table:
        first_reader = table.accessor_obj.iloc[0]
        assert table.accessor_obj.iloc[1] is None
        file_accessor.read_metadata()
        assert [(r.bindings, r.reads) for r in resources] == [(1, 1), (1, 1)]
        assert table.obsnum.tolist() == [10, 20]
        assert table.accessor_obj.iloc[0] is first_reader
        assert table.accessor_obj.iloc[1] is None

    assert {"io_obj", "accessor_obj"}.isdisjoint(file_accessor.table)


def test_metadata_reads_repeat_after_resources_are_released():
    resources = [_Resource(_Metadata(10, 0.5)), _Resource(_Metadata(20, 1.5))]
    file_accessor = _make_file_accessor(*resources)
    file_accessor.read_metadata()
    assert [(r.bindings, r.reads) for r in resources] == [(1, 1), (1, 1)]
    assert {"io_obj", "accessor_obj"}.isdisjoint(file_accessor.table)
    file_accessor.read_metadata()
    assert file_accessor.table.obsnum.tolist() == [10, 20]
    assert [(r.bindings, r.reads) for r in resources] == [(2, 2), (2, 2)]


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_factory_modes_read_unidentified_sources_through_matching_workflow(mode):
    resource = _Resource(_Metadata(10, 0.5))
    file_accessor = _FileAccessor.from_sources(
        [{"source": resource, "source_loc": "example.nc"}], mode=mode
    )
    assert resource.reads == int(mode != "guess")
    if mode != "guess":
        assert file_accessor.table.obsnum.tolist() == [10]
    assert {"io_obj", "accessor_obj"}.isdisjoint(file_accessor.table)
    file_accessor.read_metadata()
    assert resource.reads == int(mode != "guess") + 1
    assert file_accessor.table.obsnum.tolist() == [10]


@pytest.mark.parametrize(
    ("mode", "expected_reads"),
    [("guess", [0, 0, 0]), ("guess_or_read", [0, 0, 1]), ("read", [1, 0, 1])],
)
def test_factory_modes_match_once_and_leave_foreign_sources_unread(
    mode, expected_reads, monkeypatch
):
    class TelescopeWorkflow(_FileAccessorWorkflow):
        _file_match_rules = ({"data_models": Contains(value="lmt")},)

    class TelescopeFileAccessor(FileAccessorBase[TelescopeWorkflow]):
        pass

    resources = [_Resource(_Metadata(145606, 0.5)) for _ in range(3)]
    locations = [
        "tel_toltec_2025-10-31_145606_00_0001.nc",
        "toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc",
        "renamed.nc",
    ]
    matcher = TelescopeWorkflow._compile_file_match_rules(TelescopeFileAccessor)
    match_calls = 0
    map_table = RecordRuleMapper.map_table

    def count_matching(self, *args, **kwargs):
        nonlocal match_calls
        if self is matcher:
            match_calls += 1
        return map_table(self, *args, **kwargs)

    monkeypatch.setattr(RecordRuleMapper, "map_table", count_matching)
    file_accessor = TelescopeFileAccessor.from_sources(
        [
            {"source": resource, "source_loc": location}
            for resource, location in zip(resources, locations, strict=True)
        ],
        mode=mode,
    )
    assert [resource.reads for resource in resources] == expected_reads
    assert match_calls == 1
    assert {"io_obj", "accessor_obj"}.isdisjoint(file_accessor.table)


@pytest.mark.parametrize("on_error", ["record", "raise"])
def test_read_failure_preserves_values(on_error):
    file_accessor = _make_file_accessor(
        _Resource(_Metadata(10, 0.5)), _Resource(ValueError("bad header"))
    )
    before = file_accessor.table.drop(columns="source_trace").copy()
    if on_error == "raise":
        with pytest.raises(ValueError, match="bad header"):
            file_accessor.read_metadata(on_error=on_error)
        pd.testing.assert_frame_equal(
            file_accessor.table.drop(columns="source_trace"), before
        )
    else:
        file_accessor.read_metadata(on_error=on_error)
        assert file_accessor.table.obsnum.tolist() == [10, 0]
        assert file_accessor.table.quality.tolist() == [0.5, 0.0]
    assert _error_kinds(file_accessor) == [[], ["metadata_read_failed"]]
    assert {"io_obj", "accessor_obj"}.isdisjoint(file_accessor.table)


@pytest.mark.parametrize("failure_source", ["metadata", "reader_columns"])
def test_projection_failure_leaves_all_destination_columns_unchanged(
    failure_source, monkeypatch
):
    file_accessor = _make_file_accessor(_Resource(_Metadata(10, 0.5)))
    if failure_source == "metadata":
        file_accessor.table["quality"] = pd.array([0], dtype="Int64")
    else:
        file_accessor.table["classification"] = pd.array([0], dtype="Int64")

        def reader_columns(self, reads):
            return {
                "new_reader_column": pd.Series(7, index=list(reads), dtype="Int64"),
                "classification": pd.Series(1.5, index=list(reads), dtype="Float64"),
            }

        monkeypatch.setattr(
            _FileAccessorWorkflow, "_prepare_reader_columns", reader_columns
        )
    before = file_accessor.table.drop(columns="source_trace").copy()
    with pytest.raises(TypeError):
        file_accessor.read_metadata()
    pd.testing.assert_frame_equal(
        file_accessor.table.drop(columns="source_trace"), before
    )
    assert _error_kinds(file_accessor) == [[]]


def test_reader_without_metadata_remains_openable_and_records_issue(monkeypatch):
    def reader_columns(self, reads):
        return {
            "classification": pd.Series(
                [7] * len(reads), index=list(reads), dtype="Int64"
            )
        }

    monkeypatch.setattr(
        _FileAccessorWorkflow, "_prepare_reader_columns", reader_columns
    )
    resource = _Resource(None)
    file_accessor = _make_file_accessor(resource)
    with file_accessor.open() as table:
        file_accessor.read_metadata()
        assert table.accessor_obj.iloc[0].resource is resource
        assert table.obsnum.tolist() == [0]
        assert table.classification.tolist() == [7]
    trace = file_accessor.source_trace.iloc[0]
    assert [entry.kind.value for entry in trace.iter_entries()] == [
        "metadata_unavailable"
    ]


def test_toltec_metadata_overrides_guesses_and_retains_snapshots(data_lmt_path):
    paths = [
        data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc",
        data_lmt_path
        / "toltec/tcs/toltec0/toltec0_145609_000_0002_2025_10_31_09_27_57.nc",
        data_lmt_path / "toltec/tcs/hwpr/hwpr_156755_000_0002_2026_05_16_12_33_28.nc",
        "other.nc",
    ]
    file_accessor = ToltecFileAccessor.from_sources(paths)
    snapshots = file_accessor.table.filter(regex="^file_").copy()
    file_accessor.table["obsnum"] = pd.array([0] * 4, dtype="Int64")
    file_accessor.table["uid_raw_obs"] = "guessed"
    assert file_accessor.read_metadata() is file_accessor
    assert file_accessor.table.obsnum.tolist() == [145609, 145609, 156755, 0]
    assert file_accessor.table.uid_raw_obs.tolist() == [
        "tcs-145609-0-2",
        "tcs-145609-0-2",
        "tcs-156755-0-2",
        "guessed",
    ]
    assert file_accessor.table.data_kind_toltec.tolist() == [
        ToltecDataKind.lmt_tel.value,
        ToltecDataKind.raw_timestream.value,
        ToltecDataKind.hwpr.value,
        ToltecDataKind.unknown.value,
    ]
    pd.testing.assert_frame_equal(file_accessor.table[snapshots.columns], snapshots)
    assert _error_kinds(file_accessor) == [[], [], [], ["source_open_failed"]]
    assert {"io_obj", "accessor_obj"}.isdisjoint(file_accessor.table)


def test_authoritative_telescope_membership_replaces_toltec_classification(
    dummy_tel_path,
):
    file_accessor = ToltecFileAccessor.from_sources(
        [
            {
                "source": dummy_tel_path,
                "source_loc": "tel_toltec_2025-10-31_145609_00_0002.nc",
            }
        ]
    )
    assert file_accessor.table.data_kind_toltec.tolist() == [
        ToltecDataKind.lmt_tel.value
    ]
    file_accessor.read_metadata(on_error="raise")
    assert file_accessor.table.data_models.tolist() == [{"lmt", "dummy_instru"}]
    assert file_accessor.table.file_data_models.tolist() == [{"lmt", "toltec"}]
    assert file_accessor.table.data_kind_toltec.tolist() == [
        ToltecDataKind.unknown.value
    ]


@pytest.mark.parametrize("sources", [[], ["other.nc"]])
def test_metadata_update_with_no_matching_rows(sources):
    file_accessor = ToltecFileAccessor.from_sources(sources)
    before = file_accessor.table.drop(columns="source_trace").copy()
    file_accessor.read_metadata()
    pd.testing.assert_frame_equal(
        file_accessor.table.drop(columns="source_trace"), before
    )
