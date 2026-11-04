"""Observation selectors resolve label interpretations before applying column masks."""

import pandas as pd
import pytest

from tolteca_datamodels.core.obs_spec import ColumnSelector, ObsSpecContext
from tolteca_datamodels.lmt._obs_spec import _obs_spec_context as lmt_obs_spec_context
from tolteca_datamodels.lmt.types import LmtInfo
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec
from tolteca_datamodels.toltec._obs_spec import (
    _obs_spec_context as toltec_obs_spec_context,
)
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.types import ToltecInfo


@pytest.mark.parametrize("dtype", ["object", "string", "string[pyarrow]", "category"])
@pytest.mark.parametrize("text", ["1/(1,5)", "1/(1,5)[-1]"])
def test_roach_bounds_resolve_one_declared_context(monkeypatch, dtype, text):
    values = ["toltec1", "toltec3", "tel", None] * 1000
    table = pd.DataFrame(
        {"obsnum": [1] * len(values), "interface": pd.array(values, dtype=dtype)}
    )
    calls = []
    resolve = ColumnSelector._resolve_numbers

    def checked_resolve(self, values):
        calls.append(values.tolist())
        return resolve(self, values)

    monkeypatch.setattr(ColumnSelector, "_resolve_numbers", checked_resolve)
    result = ObsSpec.parse(text).select(
        table, context=lmt_obs_spec_context | toltec_obs_spec_context
    )
    resolved = [label for group in calls for label in group]
    assert len(resolved) <= len(set(LmtInfo.interfaces) | set(ToltecInfo.interfaces))
    expected = ["toltec3"] * 1000 if "[-1]" in text else ["toltec1", "toltec3"] * 1000
    assert result.interface.tolist() == expected


@pytest.mark.parametrize("dtype", ["object", "string", "string[pyarrow]", "category"])
def test_explicit_pattern_resolves_distinct_labels_once(monkeypatch, dtype):
    values = ["foo010bar9", "foo011bar9", "foo020bar9", None] * 1000
    table = pd.DataFrame(
        {
            "obsnum": [1, 1, 2, 1] * 1000,
            "interface": pd.array(values, dtype=dtype),
        }
    )
    calls = []
    resolve = ColumnSelector._resolve_numbers

    def checked_resolve(self, values):
        calls.append(values.tolist())
        return resolve(self, values)

    monkeypatch.setattr(ColumnSelector, "_resolve_numbers", checked_resolve)
    result = ObsSpec.parse("1/foo0(10,21)[-1]bar9").select(table)
    assert sum(map(len, calls)) <= table.interface.nunique()
    assert result.index.tolist() == list(range(1, 4000, 4))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1/{tel,hwpr}", [0, 1]),
        ("{tcs,ics}-1", [0, 1]),
        ("(1,3)[-1]", [2]),
        ("1/toltec1", []),
    ],
)
def test_direct_predicates_do_not_resolve_numbers(monkeypatch, text, expected):
    def unexpected(*args):
        pytest.fail("Direct numeric or label predicates do not need label resolution")

    monkeypatch.setattr(ColumnSelector, "_resolve_numbers", unexpected)
    table = pd.DataFrame(
        {
            "master": ["tcs", "ics", "tcs"],
            "obsnum": pd.array([1, 1, 2], dtype="Int64"),
            "interface": ["tel", "hwpr", "toltec1"],
        }
    )
    assert ObsSpec.parse(text).select(table).index.tolist() == expected


def test_position_mask_preserves_duplicate_index_and_input():
    table = pd.DataFrame(
        {"obsnum": [1, 2, 1, 1], "interface": ["toltec1", "toltec4", "toltec3", "tel"]},
        index=[8, 8, 3, 3],
    )
    original = table.copy(deep=True)
    result = ObsSpec.parse("1/(1,5)[-1]").select(
        table, context=lmt_obs_spec_context | toltec_obs_spec_context
    )
    pd.testing.assert_frame_equal(result, table.iloc[[2]])
    pd.testing.assert_frame_equal(table, original)


@pytest.mark.parametrize("dtype", ["object", "string", "string[pyarrow]"])
def test_large_numeric_components_remain_exact(dtype):
    lower = 2**70
    table = pd.DataFrame(
        {
            "obsnum": [1] * 4,
            "interface": pd.array(
                [f"foo{lower}", f"foo{lower + 1}", None, f"foo{lower + 2}"],
                dtype=dtype,
            ),
        }
    )
    spec = ObsSpec.parse(f"1/foo({lower},{lower + 2})[-1]")
    assert spec.select(table).index.tolist() == [1]


def test_empty_and_unordered_values_keep_their_semantics():
    table = pd.DataFrame({"obsnum": [1, 1], "interface": ["toltec1", "tel"]})
    assert ObsSpec.parse("1/(1,5)[-1]").select(
        table, context=lmt_obs_spec_context | toltec_obs_spec_context
    ).index.tolist() == [0]
    assert (
        ObsSpec.parse("1/(1,5)[10]")
        .select(table, context=lmt_obs_spec_context | toltec_obs_spec_context)
        .empty
    )
    assert ObsSpec.parse("1/{tel,1}[-1]").select(
        table, context=lmt_obs_spec_context | toltec_obs_spec_context
    ).index.tolist() == [0, 1]


def test_contexts_can_share_numeric_components():
    table = pd.DataFrame(
        {"obsnum": [1, 1, 1, 2], "interface": ["toltec1", "bar1", "bar3", "bar4"]}
    )
    context = toltec_obs_spec_context | ObsSpecContext(
        field_label_numbers={"interface": {"bar1": 1, "bar3": 3, "bar4": 4}},
    )
    spec = ObsSpec.parse("1/(1,5)[0]")
    assert spec.select(table, context=context).index.tolist() == [
        0,
        1,
    ]


@pytest.mark.parametrize("values", [[True, 1, False, 0], [1, True, 0, False]])
def test_mixed_object_column_does_not_treat_bools_as_numbers(values):
    table = pd.DataFrame(
        {"obsnum": [1] * 4, "interface": pd.Series(values, dtype=object)}
    )
    expected = [
        i for i, value in enumerate(values) if type(value) is int and value == 1
    ]
    assert ObsSpec.parse("1/(1,5)").select(table).index.tolist() == expected
    assert ObsSpec.parse("1/()[-1]").select(table).index.tolist() == expected


def test_bounded_roaches_and_positions_match_recorded_db(toltecdb_records):
    table = ToltecFileAccessor.from_sources(
        [record["FileName"] for record in toltecdb_records], mode="guess"
    ).table
    candidates = {
        i: record
        for i, record in enumerate(toltecdb_records)
        if record["master_label"].lower() == "tcs"
        and 145606 <= record["ObsNum"] < 145612
        and 1 <= record["RoachIndex"] < 5
    }
    largest = max(record["ObsNum"] for record in candidates.values())
    expected = [i for i, record in candidates.items() if record["ObsNum"] == largest]
    assert len(expected) == 8
    selected = ObsSpec.parse("tcs-(145606,145612)[-1]/(1,5)").select(
        table, context=lmt_obs_spec_context | toltec_obs_spec_context
    )
    assert selected.index.tolist() == expected


def test_reusing_spec_does_not_cache_table_values():
    table = pd.DataFrame({"obsnum": [1, 1], "interface": ["foo1", "foo2"]})
    spec = ObsSpec.parse("1/foo()[-1]")
    assert spec.select(table).index.tolist() == [1]
    table.loc[0, "interface"] = "foo3"
    assert spec.select(table).index.tolist() == [0]


@pytest.mark.parametrize(
    ("dtype", "text", "positions"),
    [
        ("Int64", "[0]", [1]),
        ("Int64", "[-1]", [0, 2]),
        ("Int64", "[:1]", [1]),
        *(
            (dtype, "[-1]", [0, 2])
            for dtype in (
                "int64",
                "uint64",
                "UInt64",
                "float64",
                "Float64",
                "int64[pyarrow]",
            )
        ),
    ],
)
def test_numeric_extrema_keep_duplicate_rows_and_indexes(dtype, text, positions):
    table = pd.DataFrame(
        {
            "obsnum": pd.array([9, 3, 9, 100], dtype=dtype),
            "interface": ["tel"] * 3 + ["hwpr"],
        },
        index=[2, 2, 1, 1],
    )
    original = table.copy(deep=True)
    selected = ObsSpec.parse(f"{text}/tel").select(table)
    pd.testing.assert_frame_equal(selected, table.iloc[positions])
    pd.testing.assert_frame_equal(table, original)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "int64[pyarrow]"])
@pytest.mark.parametrize("values", [[], [None, None], [None, 9, None, 3]])
@pytest.mark.parametrize("text", ["[0]", "[-1]"])
def test_numeric_extrema_exclude_nulls_and_handle_empty_values(dtype, values, text):
    table = pd.DataFrame({"obsnum": pd.array(values, dtype=dtype)})
    expected = (
        table.iloc[[3 if text == "[0]" else 1]] if len(values) == 4 else table.iloc[:0]
    )
    pd.testing.assert_frame_equal(ObsSpec.parse(text).select(table), expected)
    assert ObsSpec.parse(f"(100,){text}").select(table).empty


@pytest.mark.parametrize(("dtype", "lower"), [("UInt64", 2**63), ("object", 2**70)])
@pytest.mark.parametrize(("text", "position"), [("[0]", 0), ("[-1]", 1)])
def test_numeric_extrema_preserve_large_integers(dtype, lower, text, position):
    table = pd.DataFrame({"obsnum": pd.array([lower, lower + 1, None], dtype=dtype)})
    pd.testing.assert_frame_equal(
        ObsSpec.parse(text).select(table), table.iloc[[position]]
    )


@pytest.mark.parametrize(
    "number_dtype", ["object", "Int64", "UInt64", "Float64", "int64[pyarrow]"]
)
@pytest.mark.parametrize("label_dtype", ["object", "string[python]", "string[pyarrow]"])
@pytest.mark.parametrize(
    ("positions", "expected"),
    [
        ("[::2]", [3, 5]),
        ("[-3:-1]", [1, 3]),
        ("[-1,1]", [0, 1, 4]),
        ("[::-2]", [0, 1, 4]),
    ],
)
def test_positions_after_mixed_dtype_filters(
    number_dtype, label_dtype, positions, expected
):
    table = pd.DataFrame(
        {
            "obsnum": pd.array([9, 3, None, 7, 9, 1, 100, 200], dtype=number_dtype),
            "master": pd.array(["tcs"] * 6 + ["ics", None], dtype=label_dtype),
        },
        index=pd.MultiIndex.from_tuples([(2, 0), (2, 0), (1, 1), (1, 1)] * 2),
    )
    original = table.copy(deep=True)
    selected = ObsSpec.parse(f"tcs-{positions}").select(table)
    pd.testing.assert_frame_equal(selected, table.iloc[expected])
    pd.testing.assert_frame_equal(table, original)


@pytest.mark.parametrize(
    ("dtype", "lower"),
    [("UInt64", 2**63), ("uint64[pyarrow]", 2**63), ("object", 2**70)],
)
@pytest.mark.parametrize(
    ("text", "expected"), [("[-2:]", [0, 2, 4]), ("[::2]", [1, 2])]
)
def test_sorted_positions_preserve_large_integer_precision(
    dtype, lower, text, expected
):
    table = pd.DataFrame(
        {
            "obsnum": pd.array(
                [lower + 1, lower, lower + 2, None, lower + 1], dtype=dtype
            )
        }
    )
    pd.testing.assert_frame_equal(
        ObsSpec.parse(text).select(table), table.iloc[expected]
    )
