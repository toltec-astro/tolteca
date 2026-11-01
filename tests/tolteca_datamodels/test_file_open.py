"""Concrete accessors open and identify the sources of the rows they handle."""

import pandas as pd
import pytest
import xarray as xr

from tolteca_datamodels.core.file import (
    FileAccessorBase,
    FileAccessorView,
    FileAccessorWorkflow,
)
from tolteca_datamodels.core.source import SourceReaderRules
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.lmt.tel import LmtTelAccessor
from tolteca_datamodels.toltec.file import ToltecFileAccessor, ToltecFileWorkflow
from tolteca_datamodels.toltec.hwpr import ToltecHwprAccessor
from tolteca_datamodels.toltec.kids.core import ToltecKidsAccessor

_TEL = "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
_KIDS = "toltec/tcs/toltec0/toltec0_145609_000_0002_2025_10_31_09_27_57.nc"
_HWPR = "toltec/tcs/hwpr/hwpr_156755_000_0002_2026_05_16_12_33_28.nc"


def _error_kinds(file_accessor):
    return [
        [entry.kind.value for entry in trace.iter_error_entries()]
        for trace in file_accessor.table.source_trace
    ]


@pytest.fixture
def closed_datasets(monkeypatch):
    """Identities of datasets closed while the test runs."""
    closed = []
    close = xr.Dataset.close

    def record_close(self):
        closed.append(id(self))
        close(self)

    monkeypatch.setattr(xr.Dataset, "close", record_close)
    return closed


@pytest.mark.parametrize("accessor_cls", [ToltecFileAccessor, FileAccessor])
def test_open_identifies_matching_rows_and_restores_the_table(
    accessor_cls, data_lmt_path
):
    sources = [*(data_lmt_path / path for path in (_TEL, _KIDS, _HWPR)), "example.nc"]
    file_accessor = accessor_cls.from_sources(sources)
    before = file_accessor.table.drop(columns="source_trace").copy()
    for name in ("io_obj", "accessor_obj"):
        with pytest.raises(KeyError, match=name):
            getattr(file_accessor, name)
    with file_accessor.open() as table:
        assert table is file_accessor.table
        pd.testing.assert_series_equal(file_accessor.io_obj, table["io_obj"])
        pd.testing.assert_series_equal(
            file_accessor.accessor_obj, table["accessor_obj"]
        )
        assert [type(obj) for obj in file_accessor.accessor_obj] == [
            LmtTelAccessor,
            ToltecKidsAccessor,
            ToltecHwprAccessor,
            type(None),
        ]
        assert [type(obj) for obj in file_accessor.io_obj] == [
            *[xr.Dataset] * 3,
            type(None),
        ]
    for name in ("io_obj", "accessor_obj"):
        with pytest.raises(KeyError, match=name):
            getattr(file_accessor, name)
    pd.testing.assert_frame_equal(
        file_accessor.table.drop(columns="source_trace"), before
    )
    assert _error_kinds(file_accessor) == [[], [], [], ["source_open_failed"]]


def test_open_leaves_rows_of_other_accessors_unopened(data_lmt_path):
    file_accessor = LmtFileAccessor.from_sources(
        [data_lmt_path / _TEL, data_lmt_path / _KIDS]
    )
    with file_accessor.open() as table:
        assert [type(obj) for obj in table.accessor_obj] == [LmtTelAccessor, type(None)]
        assert table.io_obj.iloc[1] is None


def test_open_closes_owned_resources_and_keeps_borrowed_ones(
    data_lmt_path, closed_datasets
):
    kids = data_lmt_path / _KIDS
    with (
        xr.open_dataset(kids) as supplied,
        xr.open_dataset(data_lmt_path / _TEL) as source_obj,
    ):
        file_accessor = ToltecFileAccessor.from_sources(
            [
                kids,
                str(kids),
                {"source": supplied, "source_loc": kids},
                {"source": source_obj, "source_loc": data_lmt_path / _TEL},
            ]
        )
        seen = []

        def use_then_fail():
            with file_accessor.open() as table:
                seen.extend(zip(table.io_obj, table.accessor_obj, strict=True))
                msg = "inside the scope"
                raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="inside the scope"):
            use_then_fail()
        io_objs, accessor_objs = zip(*seen, strict=True)
        owned = io_objs[0]
        assert io_objs == (owned, owned, supplied, source_obj)
        assert isinstance(accessor_objs[3], LmtTelAccessor)
        assert closed_datasets == [id(owned)]
        assert {"io_obj", "accessor_obj"}.isdisjoint(file_accessor.table)


def test_open_records_failures_per_row(tmp_path, data_lmt_path):
    product = tmp_path / "toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep"
    housekeeping = tmp_path / "toltec_hk_156755_000_0002_2026_05_16_12_33_28.nc"
    xr.Dataset().to_netcdf(housekeeping)
    telescope = tmp_path / "tel_toltec_2025-10-31_145609_00_0002.nc"
    xr.Dataset().to_netcdf(telescope)
    file_accessor = ToltecFileAccessor.from_sources(
        [
            tmp_path / "toltec0_145609_000_0002_2025_10_31_09_27_57.nc",
            f"{product}_targamps.dat",
            housekeeping,
            telescope,
            data_lmt_path / _KIDS,
        ]
    )
    with file_accessor.open() as table:
        assert [type(obj) for obj in table.io_obj] == [
            type(None),
            type(None),
            xr.Dataset,
            xr.Dataset,
            xr.Dataset,
        ]
        assert table.accessor_obj.iloc[:4].isna().all()
        assert isinstance(table.accessor_obj.iloc[4], ToltecKidsAccessor)
    assert _error_kinds(file_accessor) == [
        ["source_open_failed"],
        [],
        ["reader_creation_failed"],
        ["reader_creation_failed"],
        [],
    ]
    assert [
        entry.kind.value for entry in file_accessor.source_trace.iloc[1].iter_entries()
    ] == ["opener_rule_unmatched"]


def test_opener_rule_miss_remains_informational_with_raise_policy(tmp_path):
    unsupported = tmp_path / "toltec0_145606_000_0001_2025_10_31_09_22_14_targamps.dat"
    file_accessor = ToltecFileAccessor.from_sources([unsupported])
    with file_accessor.open(on_error="raise") as table:
        assert table.io_obj.iloc[0] is None
        assert table.accessor_obj.isna().all()
    assert _error_kinds(file_accessor) == [[]]
    assert [
        [entry.kind.value for entry in trace.iter_entries()]
        for trace in file_accessor.source_trace
    ] == [["opener_rule_unmatched"]]


@pytest.mark.parametrize(
    "source_loc",
    ["toltec_hk_156755_000_0002_2026_05_16_12_33_28.nc", "unrecognized.nc"],
)
def test_exhausted_reader_selection_raises_on_request(source_loc):
    file_accessor = ToltecFileAccessor.from_sources(
        [{"source": xr.Dataset(), "source_loc": source_loc}]
    )
    with (
        pytest.raises(ValueError, match="No .*reader recognizes"),
        file_accessor.open(on_error="raise"),
    ):
        pass
    assert _error_kinds(file_accessor) == [["reader_creation_failed"]]
    assert {"io_obj", "accessor_obj"}.isdisjoint(file_accessor.table)


def test_reader_fallback_records_one_failure_after_all_policies(data_lmt_path):
    class UnsupportedFilesWorkflow(FileAccessorWorkflow):
        _file_match_rules = ({"interface": "unsupported"},)

    class UnsupportedFiles(FileAccessorBase[UnsupportedFilesWorkflow]):
        pass

    with xr.open_dataset(data_lmt_path / _KIDS) as kids:
        file_accessor = FileAccessorView.from_sources(
            [
                {"source": kids, "source_loc": "renamed.nc"},
                {"source": xr.Dataset(), "source_loc": "unrecognized.nc"},
                "unknown.unsupported",
            ],
            accessor_types=(UnsupportedFiles, ToltecFileAccessor),
        )
        with file_accessor.open() as table:
            assert isinstance(table.accessor_obj.iloc[0], ToltecKidsAccessor)
            assert table.accessor_obj.iloc[1:].isna().all()
        assert [
            [
                entry.kind.value
                for entry in trace.iter_entries()
                if entry.kind.value != "file_accessor_rule_unmatched"
            ]
            for trace in file_accessor.table.source_trace
        ] == [
            [],
            ["reader_creation_failed"],
            ["opener_rule_unmatched"],
        ]
        assert [trace.has_error for trace in file_accessor.table.source_trace] == [
            False,
            True,
            False,
        ]


@pytest.mark.parametrize("on_error", ["record", "raise"])
def test_reader_inspection_errors_follow_error_policy(monkeypatch, on_error):
    def inspect(resource):
        msg = "Cannot inspect routing headers"
        raise OSError(msg)

    monkeypatch.setattr(
        ToltecFileWorkflow,
        "_source_reader_rules",
        SourceReaderRules(rules=(), prepare_record=inspect),
    )
    file_accessor = ToltecFileAccessor.from_sources(
        [
            {"source": xr.Dataset(), "source_loc": name}
            for name in (_KIDS, "unrecognized.nc")
        ]
    )
    if on_error == "raise":
        with (
            pytest.raises(OSError, match="routing headers"),
            file_accessor.open(on_error=on_error),
        ):
            pass
        assert _error_kinds(file_accessor) == [["reader_creation_failed"], []]
    else:
        with file_accessor.open(on_error=on_error) as table:
            assert table.accessor_obj.isna().all()
        assert _error_kinds(file_accessor) == [["reader_creation_failed"]] * 2
    for trace in file_accessor.source_trace:
        for entry in trace.iter_error_entries():
            assert entry.context == {"exception_type": "OSError"}
            assert entry.message == "Cannot inspect routing headers"


def test_reader_construction_failure_does_not_try_another_policy():
    inspected = []

    class BrokenReader:
        def __init__(self, resource):
            msg = "Required header is missing"
            raise ValueError(msg)

    def inspect(resource):
        inspected.append(resource)

    class FirstFileAccessorWorkflow(FileAccessorWorkflow):
        _file_match_rules = ({"interface": "first"},)
        _source_reader_rules = SourceReaderRules(rules=(({}, BrokenReader),))

    class FirstFileAccessor(FileAccessorBase[FirstFileAccessorWorkflow]):
        pass

    class SecondFileAccessorWorkflow(FileAccessorWorkflow):
        _file_match_rules = ({"interface": "second"},)
        _source_reader_rules = SourceReaderRules(rules=(), prepare_record=inspect)

    class SecondFileAccessor(FileAccessorBase[SecondFileAccessorWorkflow]):
        pass

    resource = xr.Dataset()
    file_accessor = FileAccessorView.from_sources(
        [{"source": resource, "source_loc": "renamed.nc"}],
        accessor_types=(FirstFileAccessor, SecondFileAccessor),
    )
    with file_accessor.open() as table:
        assert table.io_obj.iloc[0] is resource
        assert table.accessor_obj.iloc[0] is None
    assert inspected == []
    assert _error_kinds(file_accessor) == [["reader_creation_failed"]]
    entry = next(file_accessor.table.source_trace.iloc[0].iter_error_entries())
    assert entry.message == "Required header is missing"
    assert entry.context == {"exception_type": "ValueError"}


def test_open_raises_the_first_failure_on_request(tmp_path, data_lmt_path):
    file_accessor = ToltecFileAccessor.from_sources(
        [
            data_lmt_path / _KIDS,
            tmp_path / "toltec0_145609_000_0002_2025_10_31_09_27_57.nc",
        ]
    )
    with pytest.raises(FileNotFoundError), file_accessor.open(on_error="raise"):
        pass
    assert _error_kinds(file_accessor) == [[], ["source_open_failed"]]
    assert "io_obj" not in file_accessor.table


def test_open_accepts_an_empty_table():
    file_accessor = ToltecFileAccessor.from_sources([])
    with file_accessor.open() as table:
        assert table.io_obj.empty
    assert "io_obj" not in file_accessor.table


def test_unidentified_source_records_one_failure_for_a_shared_opener(
    tmp_path, data_lmt_path
):
    file_accessor = FileAccessor.from_sources(
        [tmp_path / "missing.nc", data_lmt_path / _TEL]
    )
    with file_accessor.open() as table:
        assert table.io_obj.iloc[0] is None
        assert isinstance(table.accessor_obj.iloc[1], LmtTelAccessor)
    assert _error_kinds(file_accessor) == [["source_open_failed"], []]
