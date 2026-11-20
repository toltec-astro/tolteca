"""Filename snapshots, marked metadata, and scoped derived columns."""

import subprocess
import sys
from typing import Annotated

import pandas as pd
import pytest
from tollan.accessor.pandas.model_table import ModelColumnMapper

from tolteca_datamodels.core.file import FileAccessorBase
from tolteca_datamodels.core.file.workflow import _apply_derived_column_updates
from tolteca_datamodels.core.filepath import FilepathInfo, MetadataField
from tolteca_datamodels.file import FileAccessor, file_table_from_sources
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.lmt.filepath import LmtFilepathInfo
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.filepath import (
    ToltecFilepathInfo,
    ToltecLmtFilepathInfo,
)
from tolteca_dummy_instru.file import (
    DummyInstruCounterFileWorkflow,
    DummyInstruFileAccessor,
)
from tolteca_dummy_instru.filepath import (
    DummyInstruFilepathInfo,
    DummyInstruLmtFilepathInfo,
)


@pytest.mark.parametrize(
    ("model_cls", "accessor_cls", "specific_aliases", "source"),
    [
        (
            LmtFilepathInfo,
            LmtFileAccessor,
            (),
            "tel_toltec_2025-10-31_145609_00_0002.nc",
        ),
        (
            ToltecFilepathInfo,
            ToltecFileAccessor,
            ("roach",),
            "toltec8_145605_000_0001_2025_10_31_09_20_06_tune.nc",
        ),
        (
            DummyInstruFilepathInfo,
            DummyInstruFileAccessor,
            ("channel",),
            "chan01-tcs-1-0-2-20251031T120000.nc",
        ),
    ],
)
def test_marked_metadata_is_available_before_concrete_derivation(
    model_cls, accessor_cls, specific_aliases, source
):
    columns = FileAccessor._filepath_columns(model_cls)
    obs_fields = {
        "interface",
        "instru",
        "interface_kind",
        "master",
        "obsnum",
        "subobsnum",
        "scannum",
        *specific_aliases,
    }
    expected = [f"file_{name}" for name in model_cls.model_fields]
    assert list(columns) == expected
    assert all(column.default is None for column in columns.values())
    assert columns["file_ut"].dtype == "datetime64[ns, UTC]"
    assert columns == FileAccessorBase._filepath_columns(model_cls)
    assert columns == accessor_cls._filepath_columns(model_cls)
    aliases = FileAccessorBase._filepath_metadata_aliases(model_cls)
    assert aliases == {name: f"file_{name}" for name in (*obs_fields, "data_models")}
    base_table = FileAccessorBase.from_sources([source]).table
    table = accessor_cls.from_sources([source]).table
    for name in obs_fields:
        assert name not in columns
        assert columns[f"file_{name}"].attr == name
        pd.testing.assert_series_equal(table[name], table[f"file_{name}"].rename(name))
        pd.testing.assert_series_equal(table[name], base_table[name])
    assert columns["file_obsnum"].dtype == "Int64"
    assert columns["file_master"].dtype == "string"
    assert "file_quartet" not in columns and "quartet" not in columns


def test_replacing_current_memberships_preserves_filename_snapshot():
    file_accessor = FileAccessorBase.from_sources(
        [
            "tel_toltec_2025-10-31_145609_00_0002.nc",
            None,
        ]
    )
    original = file_accessor.table.filepath_info.iloc[0].data_models
    snapshot = file_accessor.table.file_data_models.iloc[0]
    current = file_accessor.table.data_models.iloc[0]
    assert current == snapshot == original == {"lmt", "toltec"}
    assert isinstance(current, frozenset)
    file_accessor.table["data_models"] = [current | {"other"}, None]
    assert file_accessor.table.data_models.iloc[0] == {"lmt", "toltec", "other"}
    assert snapshot == original == {"lmt", "toltec"}
    assert file_accessor.table.file_data_models.dtype == object
    assert file_accessor.table.data_models.dtype == object
    assert pd.isna(file_accessor.table.data_models.iloc[1])


def test_projection_inspects_concrete_fields_without_changing_models():
    class Extended(DummyInstruFilepathInfo):
        readout_code: Annotated[int, MetadataField]
        payload: dict[str, int]
        filename_note: str | None = None

    class Inherited(Extended):
        pass

    columns = FileAccessor._filepath_columns(Extended)
    assert columns["file_readout_code"].attr == "readout_code"
    assert columns["file_readout_code"].dtype == "Int64"
    assert "readout_code" not in columns
    assert columns["file_channel"].attr == "channel"
    assert "channel" not in columns
    aliases = FileAccessor._filepath_metadata_aliases(Extended)
    assert aliases["channel"] == "file_channel"
    assert aliases["readout_code"] == "file_readout_code"
    assert "filename_note" not in aliases
    assert "payload" not in aliases
    assert FileAccessor._filepath_metadata_aliases(Inherited) == aliases
    assert "payload" not in columns and "file_payload" not in columns
    assert "filename_note" not in columns
    assert columns["file_filename_note"].attr == "filename_note"
    assert columns["file_filename_note"].dtype == "string"
    assert FileAccessor._filepath_columns(Inherited) == columns
    assert "file_readout_code" not in FileAccessor._filepath_columns(
        DummyInstruFilepathInfo
    )


@pytest.mark.parametrize("model", [ToltecLmtFilepathInfo, DummyInstruLmtFilepathInfo])
def test_thin_telescope_subclasses_keep_telescope_columns(model):
    assert FileAccessor._filepath_columns(model) == FileAccessor._filepath_columns(
        LmtFilepathInfo
    )


def test_repeated_batches_reuse_prepared_filename_projections(monkeypatch):
    class Layout(FilepathInfo):
        channel: int

    parsed = Layout(ext="nc", data_models=frozenset(), channel=7)
    monkeypatch.setattr(FilepathInfo, "from_source", lambda *args, **kwargs: parsed)
    prepare = ModelColumnMapper.__call__
    preparations = []

    def count_preparation(self, model_cls):
        preparations.append(model_cls)
        return prepare(self, model_cls)

    monkeypatch.setattr(ModelColumnMapper, "__call__", count_preparation)
    for accessor_cls in (FileAccessorBase, FileAccessor):
        first = accessor_cls.from_sources(["first.nc", "second.nc"]).table
        assert first.file_channel.tolist() == [7, 7]
        count = len(preparations)
        again = accessor_cls.from_sources(["third.nc"]).table
        assert again.file_channel.tolist() == [7]
        assert len(preparations) == count
    assert preparations.count(Layout) > 0


def test_explicit_composition_only_adds_derived_columns_to_prepared_metadata():
    from tolteca_datamodels.core.file import FileAccessorView

    sources = [
        "chan01-tcs-1-0-2-20251031T120000.nc",
        "tel_toltec_2025-10-31_145609_00_0002.nc",
    ]
    view = FileAccessorView.from_sources(
        sources, accessor_types=(LmtFileAccessor, DummyInstruFileAccessor)
    )
    table = view.table
    assert table.channel.dropna().tolist() == [1]
    assert table.obsnum.tolist() == [1, 145609]
    assert table.file_channel.equals(table.channel.rename("file_channel"))
    assert FileAccessor.from_sources(sources).table.channel.dropna().tolist() == [1]
    assert FileAccessorBase.from_sources(sources).table.obsnum.tolist() == [1, 145609]
    assert "uid_raw_obs" not in FileAccessorBase.from_sources(sources).table
    assert view.file_accessor._filepath_columns(DummyInstruFilepathInfo) == (
        FileAccessor._filepath_columns(DummyInstruFilepathInfo)
    )


def test_derivation_uses_table_columns_without_accessing_models(monkeypatch):
    sources = [
        "toltec8_145605_000_0001_2025_10_31_09_20_06_tune.nc",
        "tel_toltec_2025-10-31_145609_00_0002.nc",
        "chan01-tcs-1-0-2-20251031T120000.nc",
    ]
    table = FileAccessorBase.from_sources(sources).table
    table.index = [4, 4, 1]
    table = table.drop(columns="filepath_info")
    table["data_models"] = [["current"] for _ in range(len(table))]
    table["obsnum"] = pd.array([9, 9, 9], dtype="Int64")
    table.loc[:, "file_obsnum"] = pd.array([101, None, 303], dtype="Int64")
    before = table.copy(deep=True)

    def unexpected(*args, **kwargs):
        pytest.fail("Column derivation reprojected models")

    monkeypatch.setattr(FileAccessorBase, "update_from_models", unexpected)
    file_accessor = ToltecFileAccessor(table)
    _apply_derived_column_updates(
        file_accessor.table, [([1, 0], file_accessor._workflow)]
    )

    assert table.index.tolist() == [4, 4, 1]
    assert table.obsnum.tolist() == [9, 9, 9]  # Current values, not snapshots.
    assert table.roach.iloc[0] == 8
    assert table.roach.iloc[1:].isna().all()
    assert table.uid_raw_obs_file.iloc[:2].tolist() == [
        "9-0-1-toltec8",
        "tcs-9-0-2-tel",
    ]
    assert pd.isna(table.uid_raw_obs_file.iloc[2])
    for name in (
        "master",
        "obsnum",
        "subobsnum",
        "scannum",
        "interface",
        "interface_kind",
        "instru",
        "roach",
    ):
        if f"file_{name}" in table:
            assert table[name].dtype == table[f"file_{name}"].dtype
    pd.testing.assert_frame_equal(
        table[before.columns].drop(columns="obsnum"), before.drop(columns="obsnum")
    )
    table.iloc[0, table.columns.tolist().index("obsnum")] = 777
    assert table.file_obsnum.iloc[0] == 101
    unchanged = table.copy(deep=True)
    _apply_derived_column_updates(file_accessor.table, [([], file_accessor._workflow)])
    pd.testing.assert_frame_equal(table, unchanged)


@pytest.mark.parametrize("incompatible", ["uid_raw_obs_file", "data_kind_toltec"])
def test_derived_assignment_failure_leaves_all_destinations_unchanged(incompatible):
    table = pd.DataFrame(
        {
            "file_obsnum": pd.array([1, 2], dtype="Int64"),
            "obsnum": pd.array([9, 9], dtype="Int64"),
            "file_interface": pd.array(["tel", "toltec8"], dtype="string"),
            "interface": pd.array(["tel", "toltec8"], dtype="string"),
            "master": pd.array(["tcs", "ics"], dtype="string"),
            "subobsnum": pd.array([0, 0], dtype="Int64"),
            "scannum": pd.array([1, 1], dtype="Int64"),
        },
        index=[5, 5],
    )
    table[incompatible] = (
        pd.array(["bad", "bad"], dtype="string")
        if incompatible == "data_kind_toltec"
        else pd.array([0, 0], dtype="Int64")
    )
    before = table.copy(deep=True)
    file_accessor = ToltecFileAccessor(table)
    with pytest.raises((TypeError, ValueError)):
        _apply_derived_column_updates(
            file_accessor.table, [([0, 1], file_accessor._workflow)]
        )
    pd.testing.assert_frame_equal(table, before)


@pytest.mark.parametrize("failure", ["defaults", "preparation", "assignment"])
def test_composite_derivation_failure_preserves_defaults_and_earlier_groups(
    failure, monkeypatch
):
    from tolteca_datamodels.core.file import FileAccessorView, SourceTraceKind

    file_accessor = FileAccessorBase.from_sources(
        [
            "toltec8_145605_000_0001_2025_10_31_09_20_06_tune.nc",
            "chan01-tcs-1-0-2-20251031T120000.nc",
            "renamed.nc",
        ]
    )
    file_accessor.table.index = [4, 4, 1]
    file_accessor.table["obsnum"] = pd.array([9, 9, 9], dtype="Int64")
    file_accessor.table["channel"] = pd.array([7, 7, 7], dtype="Int64")
    file_accessor.table["data_kind_toltec"] = pd.array([123, 123, 123], dtype="Int64")
    before = file_accessor.table.copy(deep=True)
    prepare = DummyInstruCounterFileWorkflow._prepare_derived_columns

    def fail_late(self, positions=None):
        # Neither preparing defaults nor a preceding group changes the table.
        pd.testing.assert_frame_equal(self.table, before)
        if failure != "assignment":
            message = "Failed preparing dummy columns"
            raise ValueError(message)
        assert positions is not None
        derived = prepare(self, positions)
        derived["channel"] = pd.Series(
            ["not a channel"], index=self.table.index.take(positions), dtype="string"
        )
        return derived

    hook = (
        "_prepare_derived_column_defaults"
        if failure == "defaults"
        else "_prepare_derived_columns"
    )
    monkeypatch.setattr(DummyInstruCounterFileWorkflow, hook, fail_late)
    view = FileAccessorView(
        file_accessor, accessor_types=(ToltecFileAccessor, DummyInstruFileAccessor)
    )
    with pytest.raises((TypeError, ValueError)):
        view._workflow.prepare(mode="guess", on_error="raise")

    pd.testing.assert_frame_equal(file_accessor.table, before)
    assert "data_kind_dummy_instru" not in file_accessor.table
    assert "uid_raw_obs" not in file_accessor.table
    assert [
        entry.kind for entry in file_accessor.source_trace.iloc[2].iter_entries()
    ] == [SourceTraceKind.file_accessor_rule_unmatched]


def test_composed_guess_projects_sources_and_filenames_only_once(monkeypatch):
    calls = []
    update = FileAccessorBase.update_from_models

    def counted(self, *args, **kwargs):
        calls.append(kwargs["columns"])
        return update(self, *args, **kwargs)

    monkeypatch.setattr(FileAccessorBase, "update_from_models", counted)
    file_accessor = FileAccessor.from_sources(
        ["tel_toltec_2025-10-31_145609_00_0002.nc"]
    )
    # Both LMT and TolTEC match this file, but neither reprojects its model.
    assert len(calls) == 2
    assert file_accessor.table.obsnum.tolist() == [145609]
    assert file_accessor.table.file_obsnum.equals(
        file_accessor.table.obsnum.rename("file_obsnum")
    )
    assert file_accessor.table.uid_raw_obs_file.tolist() == ["tcs-145609-0-2-tel"]


def test_uid_derivation_uses_current_values_and_preserves_unselected_rows():
    table = pd.DataFrame(
        {
            "master": ["tcs", None, "ics", "tcs"],
            "obsnum": [101, 102, 103, 104],
            "subobsnum": [1, 0, None, None],
            "scannum": [2, 3, None, 1],
            "interface": ["tel", "toltec8", "hwpr", "tel"],
            "uid_raw_obs_file": ["keep", "old", "old", "old"],
        },
        index=[7, 7, 2, 9],
    ).astype(
        {
            "master": "string",
            "obsnum": "Int64",
            "subobsnum": "Int64",
            "scannum": "Int64",
            "interface": "string",
            "uid_raw_obs_file": "string",
        }
    )
    file_accessor = ToltecFileAccessor(table)
    _apply_derived_column_updates(
        file_accessor.table, [([3, 1, 2], file_accessor._workflow)]
    )
    assert table.index.tolist() == [7, 7, 2, 9]
    assert table.uid_raw_obs_file.iloc[0] == "keep"
    assert table.uid_raw_obs_file.iloc[1] == "102-0-3-toltec8"
    assert table.uid_raw_obs_file.iloc[2:].isna().all()
    assert table.quartet.iloc[1:3].tolist() == ["102-0-3", "ics-103"]
    assert pd.isna(table.quartet.iloc[3])
    assert table.uid_obs.iloc[1:].tolist() == ["102", "ics-103", "tcs-104"]
    assert table.uid_raw_obs.iloc[1] == "102-0-3"
    assert table.uid_raw_obs.take([0, 2, 3]).isna().all()
    assert table.interface.tolist() == ["tel", "toltec8", "hwpr", "tel"]


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_table_helper_matches_accessor_preparation(mode, data_lmt_path, tmp_path):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    renamed = tmp_path / "renamed.nc"
    renamed.symlink_to(path)
    sources = [path, renamed]
    expected = FileAccessor.from_sources(sources, mode=mode, on_error="raise").table
    table = file_table_from_sources(sources, mode=mode, on_error="raise")
    pd.testing.assert_frame_equal(
        table.drop(columns="source_trace"), expected.drop(columns="source_trace")
    )
    assert [list(trace.iter_entries()) for trace in table.source_trace] == [
        list(trace.iter_entries()) for trace in expected.source_trace
    ]
    if mode == "guess":
        assert "obs_goal" not in table
        assert table.obsnum.isna().tolist() == [False, True]
    else:
        assert table.obsnum.tolist() == [145609, 145609]
        assert table.obs_goal.notna().tolist() == [mode == "read", True]
    assert {"io_obj", "accessor_obj"}.isdisjoint(table)


@pytest.mark.parametrize("mode", ["guess_or_read", "read"])
def test_table_helper_applies_source_error_policy(mode, tmp_path):
    source = tmp_path / "missing.nc"
    table = file_table_from_sources([source], mode=mode)
    assert len(table) == 1
    errors = list(table.source_trace.iloc[0].iter_error_entries())
    assert [entry.kind.value for entry in errors] == ["source_open_failed"]
    with pytest.raises(FileNotFoundError):
        file_table_from_sources([source], mode=mode, on_error="raise")


def test_assembled_entry_point_loads_parsers_and_builtin_policies():
    script = r"""
import sys
from tolteca_datamodels.core.file import FileAccessorBase
from tolteca_datamodels.file import FileAccessor, file_table_from_sources
from tolteca_dummy_instru.filepath import DummyInstruFilepathInfo

assert FileAccessor is not FileAccessorBase
assert issubclass(FileAccessor, FileAccessorBase)
sources = [
    "tel_toltec_2025-10-31_145609_00_0002.nc",
    "toltec8_145605_000_0001_2025_10_31_09_20_06_tune.nc",
    "chan01-tcs-1-0-2-20251031T120000.nc",
]
table = file_table_from_sources(sources)
assert table.obsnum.iloc[:2].tolist() == [145609, 145605]
assert table.obsnum.tolist() == [145609, 145605, 1]
assert table.roach.dropna().tolist() == [8]
assert table.interface.iloc[:2].tolist() == ["tel", "toltec8"]
assert table.file_interface.tolist() == ["tel", "toltec8", "chan01"]
assert type(table.filepath_info.iloc[2]) is DummyInstruFilepathInfo
assert table.file_channel.dropna().tolist() == [1]
assert table.channel.dropna().tolist() == [1]
base = FileAccessorBase.from_sources(sources)
assert base.table.obsnum.tolist() == [145609, 145605, 1]
from tolteca_datamodels.data_model import data_models
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.toltec.file import ToltecFileAccessor
assert data_models.lmt.file_accessor_cls is LmtFileAccessor
assert data_models.toltec.file_accessor_cls is ToltecFileAccessor
"""
    subprocess.run(  # noqa: S603 -- fixed local import-boundary probe.
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
