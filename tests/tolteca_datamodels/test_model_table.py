"""Duck-typed projection, explicit enrichment, and flat selection."""

import subprocess
import sys
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from tolteca_datamodels.model_table import ModelColumn, ModelTableAccessor


class Measurements(ModelTableAccessor):
    columns: ClassVar = {
        "channel": ModelColumn("channel", dtype="Int64"),
        "amplitude": ModelColumn("amplitude", default=None, dtype="Float64"),
    }


def test_mixed_duck_types_only_read_declared_attributes():
    class Reading(BaseModel):
        channel: int
        amplitude: float

        def model_dump(self, **kwargs):
            pytest.fail("Projection serialized a model")

        @property
        def expensive(self):
            pytest.fail("Projection read an undeclared attribute")

    values = [Reading(channel=1, amplitude=2.5), SimpleNamespace(channel=2)]
    table = Measurements.from_models(iter(values), columns=Measurements.columns)
    assert table["channel"].tolist() == [1, 2]
    assert table["amplitude"].iloc[0] == 2.5
    assert pd.isna(table["amplitude"].iloc[1])
    table.loc[0, "amplitude"] = 8
    assert next(Measurements(table).query("amplitude > 3").itertuples()).amplitude == 8
    assert values[0].amplitude == 2.5
    assert list(Measurements.from_models([], columns=Measurements.columns)) == list(
        table
    )


def test_paths_are_duck_typed_and_defaults_are_explicit():
    table = ModelTableAccessor.from_models(
        [SimpleNamespace(meta=SimpleNamespace(obsnum=3), kind=2), object()],
        columns={
            "obsnum": ModelColumn("meta.obsnum", default=None, dtype="Int64"),
            "data_kind": ModelColumn("kind", default=None, dtype="Int64"),
        },
    )
    assert table.iloc[0].tolist() == [3, 2]
    assert table.iloc[1].isna().all()
    with pytest.raises(AttributeError):
        ModelTableAccessor.from_models([object()], columns={"x": ModelColumn("x")})


def test_projection_query_neither_binds_nor_reconstructs_models():
    table = pd.DataFrame({"channel": [3, 1, 2]}, index=["same", "same", "last"])
    pd.testing.assert_frame_equal(
        ModelTableAccessor(table).query("channel < 3"), table.iloc[1:]
    )
    assert not hasattr(ModelTableAccessor, "to_models")
    assert not hasattr(ModelTableAccessor, "_get_query_resolvers")
    assert not hasattr(ModelTableAccessor, "_record_model_cls")
    assert not hasattr(ModelTableAccessor, "_columns")


def test_projection_requires_explicit_columns():
    with pytest.raises(TypeError, match="columns"):
        ModelTableAccessor.from_models([])  # ty: ignore[missing-argument]
    with pytest.raises(TypeError, match="columns"):
        ModelTableAccessor(pd.DataFrame()).update_from_models([])  # ty: ignore[no-matching-overload]


def test_multiple_passes_overwrite_nulls_and_preserve_unrelated_data():
    table = Measurements.from_models(
        [
            SimpleNamespace(channel=1, amplitude=2),
            SimpleNamespace(channel=2, amplitude=3),
            SimpleNamespace(channel=3, amplitude=4),
        ],
        columns=Measurements.columns,
    )
    table.index = pd.Index(["same", "same", "last"], name="observation")
    table["note"] = ["a", "b", "c"]
    table.attrs["batch"] = "test"
    before = table.copy()
    updated = Measurements(table).update_from_models(
        {1: SimpleNamespace(channel=20, amplitude=None)},
        columns=Measurements.columns,
    )
    assert updated["channel"].tolist() == [1, 20, 3]
    assert pd.isna(updated["amplitude"].iloc[1])
    pd.testing.assert_index_equal(updated.index, table.index)
    pd.testing.assert_series_equal(updated["note"], table["note"])
    assert updated.attrs == table.attrs
    pd.testing.assert_frame_equal(table, before)
    assert updated["channel"].dtype == "Int64"


def test_partial_update_adds_nullable_columns():
    table = pd.DataFrame({"source": ["a", "b", "c"]}, index=[7, 7, 2])
    result = ModelTableAccessor(table).update_from_models(
        {1: SimpleNamespace(value=2**60 + 1)},
        columns={"counter": ModelColumn("value", dtype="Int64")},
    )
    assert pd.isna(result["counter"].iloc[0])
    assert result["counter"].iloc[1] == 2**60 + 1
    assert pd.isna(result["counter"].iloc[2])


@pytest.mark.parametrize("positions", [[-1, 1], [8], [-3]])
@pytest.mark.parametrize("inplace", [False, True])
def test_invalid_updates_leave_input_unchanged(positions, inplace):
    table = pd.DataFrame({"channel": [1, 2]})
    original = table.copy()
    with pytest.raises((ValueError, IndexError)):
        Measurements(table).update_from_models(
            dict.fromkeys(positions, SimpleNamespace(channel=5)),
            columns=Measurements.columns,
            inplace=inplace,
        )
    pd.testing.assert_frame_equal(table, original)


def test_empty_update_and_empty_table():
    table = Measurements.from_models([], columns=Measurements.columns)
    pd.testing.assert_frame_equal(
        Measurements(table).update_from_models([], columns=Measurements.columns), table
    )
    table = pd.DataFrame({"channel": [1]})
    pd.testing.assert_frame_equal(
        Measurements(table).update_from_models({}, columns=Measurements.columns),
        table,
    )


def test_no_large_integer_rounding():
    table = ModelTableAccessor.from_models(
        [SimpleNamespace(value=2**63 + 1), SimpleNamespace(value=None)],
        columns={"value": ModelColumn("value")},
    )
    assert table["value"].tolist() == [2**63 + 1, None]


def test_model_tables_do_not_import_file_or_instrument_modules():
    script = """
import sys
from tolteca_datamodels.model_table import ModelTableAccessor
assert 'tolteca_datamodels.file' not in sys.modules
assert not any(name.startswith(('tolteca_datamodels.lmt', 'tolteca_datamodels.toltec'))
               for name in sys.modules)
"""
    subprocess.run(  # noqa: S603 -- isolated import of fixed local modules.
        [sys.executable, "-c", script], check=True, capture_output=True
    )


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("as_iterator", [False, True])
def test_sequence_length_is_checked_before_projection(count, inplace, as_iterator):
    table = pd.DataFrame({"channel": [1, 2]})
    before = table.copy()
    models = [object()] * count
    with pytest.raises(ValueError, match="one object"):
        Measurements(table).update_from_models(
            iter(models) if as_iterator else models,
            columns=Measurements.columns,
            inplace=inplace,
        )
    pd.testing.assert_frame_equal(table, before)


def test_full_sequence_and_position_mapping_agree():
    table = pd.DataFrame({"channel": pd.array([1, 2], dtype="Int64")}, index=[8, 8])
    models = [SimpleNamespace(channel=5), SimpleNamespace(channel=7)]
    accessor = Measurements(table)
    expected = accessor.update_from_models(iter(models), columns=Measurements.columns)
    actual = accessor.update_from_models(
        {1: models[1], 0: models[0]}, columns=Measurements.columns
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert expected["channel"].tolist() == [5, 7]
    assert expected["channel"].dtype == "Int64"


@pytest.mark.parametrize("positions", [[-1, 2], [2, -1], [-3, 0]])
@pytest.mark.parametrize("inplace", [False, True])
def test_mapping_rejects_aliases_before_reading_models(positions, inplace):
    table = pd.DataFrame({"channel": [1, 2, 3]})
    before = table.copy()
    with pytest.raises(ValueError, match="distinct rows"):
        Measurements(table).update_from_models(
            dict.fromkeys(positions, object()),
            columns=Measurements.columns,
            inplace=inplace,
        )
    pd.testing.assert_frame_equal(table, before)


@pytest.mark.parametrize("as_mapping", [False, True])
def test_update_does_not_replace_existing_column_dtype(as_mapping):
    table = pd.DataFrame({"channel": pd.array([1, 2], dtype="Float64")})
    models = [SimpleNamespace(channel=5), SimpleNamespace(channel=7)]
    result = Measurements(table).update_from_models(
        dict(enumerate(models)) if as_mapping else models,
        columns=Measurements.columns,
    )
    assert result["channel"].tolist() == [5, 7]
    assert result["channel"].dtype == table["channel"].dtype


@pytest.mark.parametrize("as_mapping", [False, True])
def test_full_update_rejects_incompatible_existing_dtype(as_mapping):
    table = pd.DataFrame({"channel": ["old", "values"]})
    models = [SimpleNamespace(channel=5), SimpleNamespace(channel=7)]
    with pytest.raises(TypeError):
        Measurements(table).update_from_models(
            dict(enumerate(models)) if as_mapping else models,
            columns=Measurements.columns,
        )
    assert table["channel"].tolist() == ["old", "values"]


def test_positional_assignment_preserves_mixed_columns_and_large_integers():
    table = pd.DataFrame(
        {
            "label": pd.array(["a", "b", "c"], dtype="string"),
            "value": pd.array([10, 20, 30], dtype="Float64"),
            "counter": pd.array([1, 2, 3], dtype="Int64"),
        },
        index=["same", "same", "last"],
    )
    result = ModelTableAccessor(table).update_from_models(
        {
            -1: SimpleNamespace(counter=2**60 + 1, value=1.5, label="first"),
            0: SimpleNamespace(counter=2**60 + 3, value=None, label="second"),
        },
        columns={
            "counter": ModelColumn("counter", dtype="Int64"),
            "value": ModelColumn("value", dtype="Float64"),
            "label": ModelColumn("label", dtype="string"),
        },
    )
    assert result["counter"].tolist() == [2**60 + 3, 2, 2**60 + 1]
    assert result["label"].tolist() == ["second", "b", "first"]
    assert pd.isna(result["value"].iloc[0])
    assert result["value"].iloc[2] == 1.5
    pd.testing.assert_series_equal(result.dtypes, table.dtypes)
    pd.testing.assert_index_equal(result.index, table.index)


def test_mapping_uses_positions_not_labels_or_mapping_order():
    table = pd.DataFrame({"channel": [1, 2, 3]}, index=[1, 0, 0])
    result = Measurements(table).update_from_models(
        {np.int64(-1): SimpleNamespace(channel=30), 0: SimpleNamespace(channel=10)},
        columns=Measurements.columns,
    )
    assert result["channel"].tolist() == [10, 2, 30]
    pd.testing.assert_index_equal(result.index, table.index)


@pytest.mark.parametrize("position", [":", "1", slice(None), (0, 1), 1.0, True, None])
def test_mapping_rejects_non_integer_positions(position):
    table = pd.DataFrame({"channel": [1, 2]})
    before = table.copy()
    with pytest.raises(TypeError, match="integer row positions"):
        Measurements(table).update_from_models(
            {position: SimpleNamespace(channel=5)},
            columns=Measurements.columns,
            inplace=True,
        )
    pd.testing.assert_frame_equal(table, before)


def test_mapping_values_are_single_objects_not_batches():
    table = pd.DataFrame({"channel": [1, 2]})
    with pytest.raises(AttributeError, match="channel"):
        Measurements(table).update_from_models(
            {0: [SimpleNamespace(channel=5), SimpleNamespace(channel=7)]},
            columns=Measurements.columns,
        )
    assert table["channel"].tolist() == [1, 2]


@pytest.mark.parametrize("as_mapping", [False, True])
def test_inplace_updates_without_copying_table(monkeypatch, as_mapping):
    table = pd.DataFrame({"channel": pd.array([1, 2], dtype="Int64")}, index=[8, 8])
    table.attrs["batch"] = "test"
    models = [SimpleNamespace(channel=5), SimpleNamespace(channel=7)]

    def fail_copy(*args, **kwargs):
        pytest.fail("In-place update copied the input DataFrame")

    monkeypatch.setattr(table, "copy", fail_copy)
    result = Measurements(table).update_from_models(
        dict(enumerate(models)) if as_mapping else models,
        columns=Measurements.columns,
        inplace=True,
    )
    assert result is None
    assert table["channel"].tolist() == [5, 7]
    assert table["channel"].dtype == "Int64"
    assert table["amplitude"].isna().all()
    assert table.index.tolist() == [8, 8]
    assert table.attrs == {"batch": "test"}


def test_inplace_projection_failure_leaves_table_unchanged():
    table = pd.DataFrame({"channel": [1, 2]})
    with pytest.raises(AttributeError, match="channel"):
        Measurements(table).update_from_models(
            {0: SimpleNamespace(channel=5), 1: object()},
            columns=Measurements.columns,
            inplace=True,
        )
    assert table["channel"].tolist() == [1, 2]


def test_inplace_partial_update_overwrites_nulls_and_keeps_other_rows():
    table = Measurements.from_models(
        [
            SimpleNamespace(channel=1, amplitude=2),
            SimpleNamespace(channel=2, amplitude=3),
        ],
        columns=Measurements.columns,
    )
    result = Measurements(table).update_from_models(
        {1: SimpleNamespace(channel=20, amplitude=None)},
        columns=Measurements.columns,
        inplace=True,
    )
    assert result is None
    assert table["channel"].tolist() == [1, 20]
    assert table["amplitude"].iloc[0] == 2
    assert pd.isna(table["amplitude"].iloc[1])
    assert table["amplitude"].dtype == "Float64"


def test_default_update_preserves_input_on_assignment_failure():
    table = pd.DataFrame(
        {
            "channel": pd.array([1, 2], dtype="Int64"),
            "amplitude": pd.array([2, 3], dtype="Int64"),
        }
    )
    before = table.copy()
    with pytest.raises(TypeError):
        Measurements(table).update_from_models(
            {1: SimpleNamespace(channel=20, amplitude=2.5)},
            columns=Measurements.columns,
        )
    pd.testing.assert_frame_equal(table, before)


def test_empty_mapping_is_noop_inplace():
    table = pd.DataFrame({"channel": [1, 2]})
    assert (
        Measurements(table).update_from_models(
            {}, columns=Measurements.columns, inplace=True
        )
        is None
    )
    assert list(table) == ["channel"]
    assert table["channel"].tolist() == [1, 2]
