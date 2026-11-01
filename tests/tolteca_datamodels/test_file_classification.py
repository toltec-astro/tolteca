"""Independent instrument flag columns in the filename-only workflow."""

from typing import ClassVar

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from tollan.accessor.pandas.model_table import RecordRuleMapper

from tolteca_datamodels import data_model as assembly
from tolteca_datamodels.core.file import (
    FileAccessorBase,
    FileAccessorView,
    FileAccessorWorkflow,
)
from tolteca_datamodels.core.file.workflow import _apply_derived_column_updates
from tolteca_datamodels.core.source import SourceOpener
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.lmt.data_model import data_model_spec as lmt_spec
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.toltec.data_model import data_model_spec as toltec_spec
from tolteca_datamodels.toltec.file import ToltecFileAccessor, ToltecFileWorkflow
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_dummy_instru.data_model import data_model_spec as dummy_spec
from tolteca_dummy_instru.file import (
    DummyInstruCounterFileWorkflow,
    DummyInstruFileAccessor,
)
from tolteca_dummy_instru.types import DummyInstruDataKind

_sources = [
    "tel_toltec_2025-10-31_145609_00_0002.nc",
    "toltec8_145605_000_0001_2025_10_31_09_20_06_tune.nc",
    "hwpr_145605_000_0001_2025_10_31_09_20_06.nc",
    "tel_dummy_instru_2025-10-31_145609_00_0002.nc",
    "chan01-tcs-1-0-2-20251031T120000.nc",
    "renamed.nc",
    None,
]


@pytest.fixture(autouse=True)
def builtin_registry(monkeypatch):
    monkeypatch.setattr(
        assembly._data_models, "_entries", {"lmt": lmt_spec, "toltec": toltec_spec}
    )


def test_mixed_classification_runs_once_per_disjoint_group_without_opening(monkeypatch):
    assembly.data_models.register(dummy_spec)
    mappers = {
        id(ToltecFileWorkflow._data_kind_rule_mapper): "toltec",
        id(DummyInstruCounterFileWorkflow._data_kind_rule_mapper): "dummy",
    }
    calls = []
    map_table = RecordRuleMapper.map_table

    def counted(self, table, **kwargs):
        if id(self) in mappers:
            calls.append((mappers[id(self)], table.index.tolist()))
        return map_table(self, table, **kwargs)

    def unexpected(*args, **kwargs):
        pytest.fail("Filename classification opened a resource")

    monkeypatch.setattr(RecordRuleMapper, "map_table", counted)
    monkeypatch.setattr(SourceOpener, "open", unexpected)
    file_accessor = FileAccessor.from_sources(_sources)
    table = file_accessor.table
    assert calls == [("toltec", [0, 1, 2]), ("dummy", [3]), ("dummy", [4])]
    assert table.data_kind_toltec.tolist() == [
        kind.value
        for kind in [ToltecDataKind.lmt_tel, ToltecDataKind.tune, ToltecDataKind.hwpr]
        + [ToltecDataKind.unknown] * 4
    ]
    assert table.data_kind_dummy_instru.tolist() == [
        kind.value
        for kind in [DummyInstruDataKind.unknown] * 3
        + [DummyInstruDataKind.telescope, DummyInstruDataKind.counts]
        + [DummyInstruDataKind.unknown] * 2
    ]
    assert "data_kind" not in table
    assert str(table.data_kind_toltec.dtype) == "Int64"
    assert str(table.data_kind_dummy_instru.dtype) == "Int64"
    assert len(table) == len(_sources)


@pytest.mark.parametrize(
    ("accessor_cls", "column", "unknown", "foreign"),
    [
        (
            ToltecFileAccessor,
            "data_kind_toltec",
            ToltecDataKind.unknown.value,
            _sources[4],
        ),
        (
            DummyInstruFileAccessor,
            "data_kind_dummy_instru",
            DummyInstruDataKind.unknown.value,
            _sources[1],
        ),
    ],
)
def test_defaults_exist_without_any_matching_group(
    accessor_cls, column, unknown, foreign
):
    for sources in ([], [foreign, "unrecognized.nc", None, "/"]):
        table = accessor_cls.from_sources(sources).table
        assert table[column].tolist() == [unknown] * len(sources)
        assert str(table[column].dtype) == "Int64"
    assert not any(
        isinstance(name, str) and name.startswith("data_kind")
        for name in FileAccessorBase.from_sources(_sources).table
    )
    assert not any(
        isinstance(name, str) and name.startswith("data_kind")
        for name in LmtFileAccessor.from_sources(_sources).table
    )


def test_filename_rule_matrix():
    # Acquisition and product naming conventions; the final two are deliberately
    # unsupported cases, not new conventions inferred from synthetic filenames.
    cases = [
        ("vnasweep.nc", ToltecDataKind.vnasweep),
        ("targsweep.nc", ToltecDataKind.targsweep),
        ("targetsweep.nc", ToltecDataKind.targsweep),
        ("tune.nc", ToltecDataKind.tune),
        ("vnasweep_processed.nc", ToltecDataKind.reduced_vnasweep),
        ("targsweep_processed.nc", ToltecDataKind.reduced_targsweep),
        ("tune_processed.nc", ToltecDataKind.reduced_targsweep),
        ("sweep.nc", ToltecDataKind.reduced_sweep),
        ("timestream_processed.nc", ToltecDataKind.solved_timestream),
        ("timestream.nc", ToltecDataKind.raw_timestream),
        *[
            (f"{suffix}.txt", ToltecDataKind.kids_model_params_table)
            for suffix in ("vnasweep", "targsweep", "tune")
        ],
        ("tune_targamps.dat", ToltecDataKind.targ_amps_dat),
        *[
            (f"tune_{suffix}.ecsv", ToltecDataKind.kids_prop_table)
            for suffix in ("kids_find", "kids_fit", "chan_prop", "adrv")
        ],
        ("not_known.nc", ToltecDataKind.unknown),
        ("vnasweep.nc.gz", ToltecDataKind.unknown),
    ]
    prefix = "toltec8_145605_000_0001_2025_10_31_09_20_06"
    paths = [f"{prefix}_{suffix}" for suffix, _ in cases]
    tel_source, hwpr_source = _sources[0], _sources[2]
    assert isinstance(tel_source, str)
    assert isinstance(hwpr_source, str)
    auxiliary = [
        (f"{prefix}.nc", ToltecDataKind.raw_timestream),
        (tel_source, ToltecDataKind.lmt_tel),
        (tel_source.replace("tel_toltec_", "tel_toltec2_"), ToltecDataKind.lmt_tel2),
        (hwpr_source, ToltecDataKind.hwpr),
        (hwpr_source.replace("hwpr_", "toltec_hk_"), ToltecDataKind.housekeeping),
        ("wyatt_2025-10-31_145605_000_0001.nc", ToltecDataKind.wyatt),
    ]
    paths.extend(path for path, _ in auxiliary)
    file_accessor = ToltecFileAccessor.from_sources(paths, on_error="raise")
    assert file_accessor.table.data_kind_toltec.tolist() == [
        kind.value for _, kind in cases + auxiliary
    ]


@pytest.mark.parametrize("accessor_cls", [ToltecFileAccessor, DummyInstruFileAccessor])
def test_direct_and_selected_factory_classification_are_equivalent(accessor_cls):
    assembly.data_models.register(dummy_spec)
    direct = accessor_cls.from_sources(_sources)
    selected = FileAccessorView.from_sources(_sources, accessor_types=(accessor_cls,))
    assert_frame_equal(selected.table, direct.table)
    column = (
        "data_kind_toltec"
        if accessor_cls is ToltecFileAccessor
        else "data_kind_dummy_instru"
    )
    expression = f"bits_any_set({column}, unknown)"
    assert_frame_equal(selected.query(expression), direct.query(expression))


def test_partial_selection_preserves_sources_and_unselected_classifications():
    assembly.data_models.register(dummy_spec)
    file_accessor = FileAccessor.from_sources(_sources)
    table = file_accessor.table
    before = table.copy(deep=True)
    selected = file_accessor.partial(accessor_types=(ToltecFileAccessor,))
    selected.query("bits_any_set(data_kind_toltec, unknown)")
    assert selected.table is table
    assert_frame_equal(table, before)


def test_classification_uses_current_rule_fields_and_positional_updates():
    file_accessor = ToltecFileAccessor.from_sources([_sources[1]] * 4)
    file_accessor.table.index = [5, 5, 5, 2]
    file_accessor.table["suffix"] = pd.array(
        [None, "", "targsweep", "unused"], dtype="string"
    )
    _apply_derived_column_updates(
        file_accessor.table, [([2, 0, 1], file_accessor._workflow)]
    )
    assert file_accessor.table.data_kind_toltec.tolist() == [
        ToltecDataKind.raw_timestream.value,
        ToltecDataKind.unknown.value,
        ToltecDataKind.targsweep.value,
        ToltecDataKind.tune.value,
    ]
    assert file_accessor.table.file_suffix.tolist() == ["tune"] * 4


def test_query_namespaces_keep_overlapping_flags_separate():
    assembly.data_models.register(dummy_spec)
    file_accessor = FileAccessor.from_sources(_sources)
    before = file_accessor.table.copy(deep=True)
    expressions = [
        ("bits_any_set(data_kind_toltec, ToltecDataKind.raw_sweep)", [1]),
        ("bits_all_set(data_kind_toltec, ToltecDataKind.tune)", [1]),
        ("bits_any_set(data_kind_dummy_instru, DummyInstruDataKind.counts)", [4]),
        (
            "bits_any_set(data_kind_dummy_instru, DummyInstruDataKind.unknown)",
            [0, 1, 2, 5, 6],
        ),
        ("bits_any_set(data_kind_toltec, ToltecDataKind.unknown)", [3, 4, 5, 6]),
        (
            (
                "bits_any_set(data_kind_toltec, "
                "ToltecDataKind.raw_sweep | ToltecDataKind.hwpr) & obsnum > 0"
            ),
            [1, 2],
        ),
    ]
    for expression, positions in expressions:
        assert_frame_equal(
            file_accessor.query(expression), file_accessor.table.iloc[positions]
        )
    with pytest.raises(pd.errors.UndefinedVariableError, match="unknown"):
        file_accessor.query("bits_any_set(data_kind_toltec, unknown)")
    selected = file_accessor.partial(accessor_types=(ToltecFileAccessor,))
    with pytest.raises(pd.errors.UndefinedVariableError, match="DummyInstruDataKind"):
        selected.query(expressions[2][0])
    assert_frame_equal(file_accessor.table, before)
    file_accessor.table["unknown"] = range(len(_sources))
    assert ToltecFileAccessor(file_accessor.table).query(
        "unknown == 2"
    ).index.tolist() == [2]


def test_conflicting_extension_query_symbols_raise():
    class FirstWorkflow(FileAccessorWorkflow):
        _query_symbols: ClassVar = {"shared": object()}

    class First(FileAccessorBase[FirstWorkflow]):
        pass

    class SecondWorkflow(FileAccessorWorkflow):
        _query_symbols: ClassVar = {"shared": object()}

    class Second(FileAccessorBase[SecondWorkflow]):
        pass

    view = FileAccessorView(FileAccessorBase(pd.DataFrame({"a": [1]})), (First, Second))
    with pytest.raises(ValueError, match="Conflicting query symbol: 'shared'"):
        view.query("a == shared")


def test_recorded_toltecdb_filenames_support_sweep_queries(toltecdb_records):
    paths = [row["FileName"] for row in toltecdb_records]
    file_accessor = ToltecFileAccessor.from_sources(paths)
    expected = [
        i
        for i, path in enumerate(paths)
        if path.endswith(("_vnasweep.nc", "_targsweep.nc", "_tune.nc"))
    ]
    assert expected
    assert_frame_equal(
        file_accessor.query("bits_any_set(data_kind_toltec, raw_sweep)"),
        file_accessor.table.iloc[expected],
    )
