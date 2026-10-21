"""Value constraints and distinct-value positions have separate semantics."""

import re
from dataclasses import dataclass
from typing import ClassVar

import pandas as pd
import pytest

from tolteca_datamodels.core.obs_spec import (
    ColumnSelector,
    FieldConstraint,
    FieldSelection,
    ObsSpecBase,
    ObsSpecContext,
    Selector,
    ValueBounds,
)
from tolteca_datamodels.core.obs_spec import selectors as obs_spec


@dataclass(frozen=True)
class PairSpec(ObsSpecBase):
    first: Selector = None
    second: Selector = None

    _field_constraints: ClassVar = {
        name: FieldConstraint(kind="integer") for name in ("first", "second")
    }
    _primary_field: ClassVar = "first"


@dataclass(frozen=True)
class NumericPrimarySpec(ObsSpecBase):
    batch: Selector = None
    first: Selector = None
    second: Selector = None

    _field_constraints: ClassVar = {
        name: FieldConstraint(kind="integer") for name in ("batch", "first", "second")
    }
    _primary_field: ClassVar = "first"
    # Batch numbers occupy 2000-2099; primary tokens exclude that range.
    _primary_pattern: ClassVar = re.compile(r"(?!20[0-9]{2}$)[^a-zA-Z_]+")


@dataclass(frozen=True)
class LabelPrimarySpec(ObsSpecBase):
    phase: str | None = None
    name: str | None = None
    visit: Selector = None

    _field_constraints: ClassVar = {
        "phase": FieldConstraint(kind="label"),
        "name": FieldConstraint(kind="label"),
        "visit": FieldConstraint(kind="integer"),
    }
    _primary_field: ClassVar = "name"
    _primary_pattern: ClassVar = re.compile(r"science|\*")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (1, {"first": 1}),
        ("1", {"first": 1}),
        ("1-2", {"first": 1, "second": 2}),
        ("2026-1", {"batch": 2026, "first": 1}),
        ("2026-1/2", {"batch": 2026, "first": 1, "second": 2}),
        ("2026-*-2", {"batch": 2026, "second": 2}),
        ("*-2", {"second": 2}),
        ("2026-[-1]", {"batch": 2026, "first": FieldSelection(None, slice(-1, None))}),
    ],
)
def test_primary_recognition_is_declared_not_alphabetic(text, expected):
    spec = NumericPrimarySpec.parse(text)
    assert dict(spec.constraints()) == expected
    assert NumericPrimarySpec.parse(str(spec)) == spec


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("science-2", {"name": "science", "visit": 2}),
        ("warm-science-2", {"phase": "warm", "name": "science", "visit": 2}),
        ("cold-science/2", {"phase": "cold", "name": "science", "visit": 2}),
        ("*-2", {"visit": 2}),
        ("warm-*-2", {"phase": "warm", "visit": 2}),
    ],
)
def test_primary_field_can_have_a_label_kind(text, expected):
    spec = LabelPrimarySpec.parse(text)
    assert dict(spec.constraints()) == expected
    assert LabelPrimarySpec.parse(str(spec)) == spec


@pytest.mark.parametrize("text", ["2026-", "2026-1-2-3", "warm", "warm-1"])
def test_primary_recognition_does_not_bypass_field_validation(text):
    with pytest.raises(ValueError):
        NumericPrimarySpec.parse(text)


@pytest.mark.parametrize("text", ["warm-science_extra-2", "warm-prescience-2"])
def test_primary_pattern_matches_whole_tokens(text):
    with pytest.raises(ValueError, match="primary field"):
        LabelPrimarySpec.parse(text)


@pytest.mark.parametrize("text", ["warm/science-2", "warm-cold-science-2"])
def test_preceding_tokens_must_fit_the_declared_layout(text):
    with pytest.raises(ValueError):
        LabelPrimarySpec.parse(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1", {"first": 1}),
        ("1/2", {"first": 1, "second": 2}),
        ("1-2", {"first": 1, "second": 2}),
        ("(1,9)/{4,2,4}", {"first": ValueBounds(1, 9), "second": (2, 4)}),
        ("*/[2]", {"second": FieldSelection(None, slice(2, 3))}),
        (
            "(1,9)[-1]/{2,4}",
            {
                "first": FieldSelection(ValueBounds(1, 9), slice(-1, None)),
                "second": (2, 4),
            },
        ),
    ],
)
def test_parse_and_roundtrip(text, expected):
    spec = PairSpec.parse(text)
    assert dict(spec.constraints()) == expected
    assert PairSpec.parse(str(spec)) == spec
    assert hash(PairSpec.parse(str(spec))) == hash(spec)


@pytest.mark.parametrize(
    "text",
    [
        "1/2/3",
        "1-2-3",
        "1/2-3",
        "1//2",
        "1-",
        "",
        "-1",
        "{-1,-3}",
        "(1:2)",
        "(1,2,3)",
        "(1,2]",
        "((1,2))",
        "[-1",
        "[1:2:0]",
        "[1:2:3:4]",
        "[1,]",
        "[,]",
        "[1,2:3]",
        "1 2",
        "[1 2]",
        "{}{}",
        "(,)[0][1]",
        "[-1]-[-1]",
        "[-1]/[0]",
    ],
)
def test_invalid_syntax_is_rejected(text):
    with pytest.raises(ValueError):
        PairSpec.parse(text)


@pytest.mark.parametrize("text", ["(1,9)[-1]/{2,4}", "*-1"])
@pytest.mark.parametrize("whitespace", [" ", "\t", "\n", "\r", "\u00a0"])
def test_whitespace_is_rejected_at_every_position(text, whitespace):
    for position in range(len(text) + 1):
        with pytest.raises(ValueError):
            PairSpec.parse(text[:position] + whitespace + text[position:])


@pytest.mark.parametrize(
    ("token", "positions"),
    [
        ("1:9:2", slice(1, 9, 2)),
        ("-1", slice(-1, None)),
        ("-10:", slice(-10, None)),
        (":5", slice(None, 5)),
        ("::2", slice(None, None, 2)),
        ("::-1", slice(None, None, -1)),
        ("+1:3", slice(1, 3)),
    ],
)
def test_positions_delegate_to_tollan(monkeypatch, token, positions):
    calls = []
    parse_slice = obs_spec.parse_slice

    def parse(value):
        calls.append(value)
        return parse_slice(value)

    monkeypatch.setattr(obs_spec, "parse_slice", parse)
    assert PairSpec.parse(f"[{token}]").first == FieldSelection(None, positions)
    assert calls == [token]


@pytest.mark.parametrize("value", ["*", "{}", "{,}"])
@pytest.mark.parametrize("position", ["", "[]", "[:]", "[::]", "[::1]", "[0:]"])
def test_unconstrained_parts_preserve_nulls_and_need_no_column(value, position):
    spec = PairSpec.parse(value + position)
    assert spec == PairSpec()
    table = pd.DataFrame({"unrelated": [None, 3]})
    pd.testing.assert_frame_equal(spec.select(table), table)


@pytest.mark.parametrize("text", ["[]", "[:]", "[::]", "[::1]", "[0:]"])
def test_noop_positions_do_not_count_toward_the_limit(text):
    assert PairSpec.parse(f"{text}/[-1]") == PairSpec.parse("*/[-1]")


@pytest.fixture
def table():
    return pd.DataFrame(
        {
            "first": pd.array([199, 150, 199, 100, 198, 200, None], dtype="Int64"),
            "second": pd.array([1, 2, 1, 2, 4, 3, 1], dtype="Int64"),
        },
        index=[5, 5, 0, 2, 3, 4, 1],
    )


@pytest.mark.parametrize(
    ("text", "positions"),
    [
        ("(100,200)", [0, 1, 2, 3, 4]),
        ("(,150)", [3]),
        ("(199,)", [0, 2, 5]),
        ("(100,200)[:2]", [1, 3]),
        ("(100,200)[-1]/(1,5)", [0, 2]),
        ("(100,200)[-1]/(1,2)", [0, 2]),
        ("(100,200)/(1,5)[-1]", [4]),
        ("(100,200)[-1]/4", [4]),
        ("[0]", [3]),
        ("[-1]", [5]),
        ("*[-1]", [5]),
        ("()[1]", [1]),
        ("{100,198,199}[-1]", [0, 2]),
        ("199[-1]", [0, 2]),
        ("[0,-1]", [3, 5]),
        ("[-1,-3]", [4, 5]),
        ("[-10:]", [0, 1, 2, 3, 4, 5]),
        ("[-100:100]", [0, 1, 2, 3, 4, 5]),
        ("[::2]", [3, 4, 5]),
        ("[::-2]", [3, 4, 5]),
        ("[100]", []),
        ("[-100]", []),
        ("[-100,100]", []),
        ("[0,-100,100]", [3]),
        ("(200,100)[-1]", []),
        ("(201,)[-1]", []),
    ],
)
def test_selection_uses_distinct_values_and_preserves_rows(table, text, positions):
    original = table.copy()
    selected = PairSpec.parse(text).select(table)
    pd.testing.assert_frame_equal(selected, table.iloc[positions])
    pd.testing.assert_frame_equal(table, original)


def test_all_value_filters_precede_positions_in_either_field_order(table):
    assert PairSpec.parse("[-1]/4").select(table)["first"].tolist() == [198]
    # Largest second value among the first=199 rows is 1, not the global maximum 4.
    assert PairSpec.parse("199/[-1]").select(table)["second"].tolist() == [1, 1]


def test_empty_or_null_values_has_no_positions():
    for values in ([], [None, None]):
        table = pd.DataFrame({"first": pd.array(values, dtype="Int64")})
        for text in ("[-1]", "[:10]", "[-1,-3]"):
            assert PairSpec.parse(text).select(table).empty


def test_constructor_enforces_the_same_contract():
    with pytest.raises(ValueError, match="Only one positional selector"):
        PairSpec(
            first=FieldSelection(None, slice(-1, None)),
            second=FieldSelection(None, slice(0, 1)),
        )
    assert PairSpec(first=FieldSelection(ValueBounds(1, 5), slice(None))) == PairSpec(
        first=ValueBounds(1, 5)
    )
    for value in (True, -1, (), (1, True)):
        with pytest.raises(ValueError):
            PairSpec(first=value)
    for positions in (True, (True,), slice(None, None, 0), slice(True)):  # noqa: FBT003
        with pytest.raises(ValueError):
            FieldSelection(positions=positions)  # ty: ignore[invalid-argument-type]
    for lower, upper in ((-1, None), (True, None), (0, -1)):
        with pytest.raises(ValueError):
            ValueBounds(lower, upper)


def test_formatting_collapses_only_the_rightmost_middle_gap():
    @dataclass(frozen=True)
    class FiveFieldSpec(PairSpec):
        third: Selector = None
        fourth: Selector = None
        fifth: Selector = None

        _field_constraints: ClassVar = PairSpec._field_constraints | {
            name: FieldConstraint(kind="integer")
            for name in ("third", "fourth", "fifth")
        }

    for data, text in (
        ({"first": 1, "third": 2, "fifth": 3}, "1-*-2/3"),
        ({"first": 1, "third": 2, "fourth": 3, "fifth": 4}, "1/2/3/4"),
        ({"first": 1, "third": 2, "fourth": 3}, "1-*-2-3"),
    ):
        spec = FiveFieldSpec(**data)
        assert str(spec) == text
        assert FiveFieldSpec.parse(text) == spec


def test_missing_constrained_column_is_rejected():
    with pytest.raises(ValueError, match="lacks selector field"):
        PairSpec.parse("[-1]").select(pd.DataFrame({"second": [1]}))


def test_layout_declarations_fail_before_parsing():
    with pytest.raises(ValueError, match="no declared constraint"):

        class MissingPrimary(PairSpec):
            _primary_field = "missing"


def test_constructor_rejects_a_leading_constraint_that_looks_like_primary():
    from tolteca_datamodels.lmt_base.obs_spec import ObsSpec

    with pytest.raises(
        ValueError, match="cannot be represented.*primary field 'obsnum'"
    ):
        ObsSpec(master=FieldSelection(positions=slice(-1, None)), obsnum=1)
    with pytest.raises(
        ValueError, match="cannot be represented.*primary field 'first'"
    ):
        NumericPrimarySpec(batch=1, first=2)
    with pytest.raises(
        ValueError, match="cannot be represented.*primary field 'first'"
    ):
        NumericPrimarySpec(first=2026)
    # Explicit unordered labels are still valid, including ignored positions.
    spec = ObsSpec(master=FieldSelection(("tcs", "ics"), slice(-1, None)), obsnum=1)
    assert ObsSpec.parse(str(spec)) == spec


def test_formatting_preserves_internal_gaps_before_primary():
    @dataclass(frozen=True)
    class TwoLeadingFieldsSpec(ObsSpecBase):
        phase: Selector = None
        batch: Selector = None
        visit: Selector = 1

        _field_constraints: ClassVar = {
            "phase": FieldConstraint(kind="label"),
            "batch": FieldConstraint(kind="label"),
            "visit": FieldConstraint(kind="integer"),
        }
        _primary_field: ClassVar = "visit"
        _primary_pattern: ClassVar = re.compile(r"[0-9]+")

    for data, text in (
        ({"phase": "warm"}, "warm-*-1"),
        ({"phase": "warm", "batch": "a"}, "warm-a-1"),
        ({"batch": "a"}, "a-1"),
    ):
        spec = TwoLeadingFieldsSpec(**data)
        assert str(spec) == text
        assert TwoLeadingFieldsSpec.parse(text) == spec


def test_column_constraints_bind_each_field_once_with_context():
    from tolteca_datamodels.lmt_base.obs_spec import ObsSpec

    numbers = {"bar1": 1, "bar2": 2}
    context = ObsSpecContext(field_label_numbers={"interface": numbers})
    spec = ObsSpec.parse("tcs-1/{tel,1}")
    assert list(spec.column_constraints(context=context)) == [
        ColumnSelector("master", "tcs"),
        ColumnSelector("obsnum", 1),
        ColumnSelector("interface", (1, "tel"), label_numbers=numbers),
    ]


@pytest.mark.parametrize(
    "text", ["*", "1", "tcs-145609-0-0/toltec1", "1/{tel,1,3}", "1/(1,5)"]
)
def test_value_only_parsing_has_no_positional_constraint(text):
    from tolteca_datamodels.lmt_base.obs_spec import ObsSpec

    spec = ObsSpec.parse(text)
    assert spec.positional_constraint is None
    assert all(not isinstance(value, FieldSelection) for _, value in spec.constraints())
    assert ObsSpec.parse(str(spec)) == spec
