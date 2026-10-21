"""Label and sequence interpretation uses the same field-selector grammar."""

from dataclasses import dataclass, fields
from typing import ClassVar

import pandas as pd
import pytest
from pydantic import ValidationError

from tolteca_datamodels.core.obs_spec import (
    FieldConstraint,
    FieldPattern,
    FieldSelection,
    ObsSpecBase,
    ObsSpecContext,
    Selector,
    ValueBounds,
)
from tolteca_datamodels.lmt._obs_spec import _obs_spec_context as lmt_obs_spec_context
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec
from tolteca_datamodels.toltec._obs_spec import (
    _obs_spec_context as toltec_obs_spec_context,
)
from tolteca_datamodels.toltec.types import ToltecInfo


def test_parsed_selector_carries_no_instrument_context():
    spec = ObsSpec.parse("1/(1,5)[-1]")
    normalized, hash_before = str(spec), hash(spec)
    table = pd.DataFrame(
        {"obsnum": [1, 1, 1, 1], "interface": ["toltec1", "toltec3", "bar1", "bar2"]}
    )
    assert spec.select(table).empty
    for numbers, expected in (
        ({"toltec1": 1, "toltec3": 3}, ["toltec3"]),
        ({"bar1": 1, "bar2": 2}, ["bar2"]),
    ):
        context = ObsSpecContext(field_label_numbers={"interface": numbers})
        assert spec.select(table, context=context).interface.tolist() == expected
        assert spec.resolve_values(
            "interface", table.interface.tolist(), context=context
        ) == list(numbers)
    assert spec.select(table).empty
    assert (str(spec), hash(spec)) == (normalized, hash_before)
    assert ObsSpec.parse(normalized) == spec


@pytest.mark.parametrize("interface", ["tel", "hwpr", "toltec_hk", "wyatt"])
def test_literal_interfaces_select_without_a_roach_column(interface):
    table = pd.DataFrame(
        {"obsnum": [1, 1, 2], "interface": [interface, "unrelated", interface]}
    )
    spec = ObsSpec.parse(f"1/{interface}")
    assert spec == ObsSpec(obsnum=1, interface=interface)
    assert spec.select(table).index.tolist() == [0]
    assert ObsSpec.parse(str(spec)) == spec


@pytest.mark.parametrize("roach", ToltecInfo.roaches)
def test_numeric_and_named_forms_resolve_equally_in_toltec_context(roach):
    interface = ToltecInfo.roach_interface[roach]
    named, numeric = ObsSpec.parse(f"1/{interface}"), ObsSpec.parse(f"1/{roach}")
    assert named.interface == interface
    assert numeric.interface == roach
    assert named != numeric
    assert named.resolve_values("interface", ToltecInfo.interfaces) == [interface]
    assert numeric.resolve_values(
        "interface",
        ToltecInfo.interfaces,
        context=lmt_obs_spec_context | toltec_obs_spec_context,
    ) == [interface]


@pytest.mark.parametrize("interface", ["toltec13", "toltec01", "toltec99"])
def test_unknown_labels_parse_but_do_not_resolve_in_toltec_context(interface):
    spec = ObsSpec.parse(f"1/{interface}")
    assert spec.interface == interface
    assert spec.resolve_values("interface", ToltecInfo.interfaces) == []


def test_numeric_bounds_and_positions_need_only_the_interface_column():
    spec = ObsSpec.parse("1/(1,5)[-1]")
    assert spec.interface == FieldSelection(ValueBounds(1, 5), slice(-1, None))
    table = pd.DataFrame(
        {
            "obsnum": [1, 1, 1, 2],
            "interface": ["toltec0", "toltec1", "toltec3", "toltec4"],
        }
    )
    assert spec.select(
        table, context=lmt_obs_spec_context | toltec_obs_spec_context
    ).index.tolist() == [2]


def test_no_housekeeping_alias():
    table = pd.DataFrame({"obsnum": [2], "interface": ["toltec_hk"]})
    assert ObsSpec.parse("2/hk").select(table).empty
    assert ObsSpec.parse("2/toltec_hk").select(table).index.tolist() == [0]


@pytest.fixture
def table():
    return pd.DataFrame(
        {
            "master": ["tcs", "ics", "tcs", "tcs", "tcs", "clip", "tcs", "tcs"],
            "obsnum": [1, 1, 1, 1, 1, 1, 2, 1],
            "interface": [
                "tel",
                "toltec1",
                "toltec3",
                "hwpr",
                "toltec2",
                "tel",
                "toltec1",
                None,
            ],
            "roach": pd.array([None, 1, 3, None, 2, None, 1, None], dtype="Int64"),
        }
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1/{tel,1,3}", [0, 1, 2, 5]),
        ("tcs-1/{tel,1,3}", [0, 2]),
        ("{tcs,ics}-1/{tel,1,3}", [0, 1, 2]),
        ("1/{tel,hwpr}", [0, 3, 5]),
        ("1/{toltec1,1,toltec3}", [1, 2]),
        ("[-1]/{tel,1,3}", [6]),
        ("1/(1,4)[-1]", [2]),
        ("1/{}", [0, 1, 2, 3, 4, 5, 7]),
    ],
)
def test_alternatives_within_a_field_are_or_and_fields_are_and(table, text, expected):
    spec = ObsSpec.parse(text)
    assert (
        spec.select(
            table, context=lmt_obs_spec_context | toltec_obs_spec_context
        ).index.tolist()
        == expected
    )
    assert ObsSpec.parse(str(spec)) == spec
    assert hash(ObsSpec.parse(str(spec))) == hash(spec)


def test_unordered_set_ignores_positions_and_keeps_its_values(table):
    assert ObsSpec.parse("1/{tel,1}[-1]").select(
        table, context=lmt_obs_spec_context | toltec_obs_spec_context
    ).index.tolist() == [0, 1, 5]


def test_label_only_field_ignores_positions(table):
    assert ObsSpec.parse("{tcs,ics}[-1]-1").select(table).index.tolist() == [
        0,
        1,
        2,
        3,
        4,
        7,
    ]


def test_positional_slot_without_matches_is_not_invalid(table):
    spec = ObsSpec.parse("1/tel()[-1]")
    assert spec.interface == FieldPattern(
        "tel", FieldSelection(ValueBounds(), slice(-1, None)), ""
    )
    assert spec.select(table).empty


def test_duplicate_literals_normalize_without_instrument_interpretation():
    assert ObsSpec.parse("1/{toltec1,1,tel,tel}").interface == (1, "tel", "toltec1")
    assert str(ObsSpec.parse("1/tel()[:]")) == "1/tel(,)"


def test_logical_interface_has_no_parallel_roach_field():
    assert [item.name for item in fields(ObsSpec)] == [
        "master",
        "obsnum",
        "subobsnum",
        "scannum",
        "interface",
    ]
    with pytest.raises(TypeError):
        ObsSpec(roach=1)  # ty: ignore[unknown-argument] -- Test rejection of the removed field.
    assert ObsSpec.parse("1/toltec1").interface == "toltec1"
    assert ObsSpec.parse("1/1").interface == 1


@dataclass(frozen=True)
class MixedSpec(ObsSpecBase):
    value: Selector = None

    _primary_field: ClassVar = "value"
    _field_constraints: ClassVar = {"value": FieldConstraint(kind="mixed")}


def test_generic_mixed_field_needs_no_instrument_parser():
    table = pd.DataFrame({"value": [1, "red", 3, "blue", None]})
    spec = MixedSpec.parse("{red,1}")
    assert spec.select(table).index.tolist() == [0, 1]
    assert MixedSpec.parse(str(spec)) == spec


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("(1,4)", [0, 2]),
        ("()[-1]", [2]),
        ("(1,4)[0]", [0]),
        ("*", [0, 1, 2, 3, 4]),
        ("[-1]", [0, 1, 2, 3, 4]),
    ],
)
def test_mixed_column_numeric_bounds_and_unrestricted_values(text, expected):
    table = pd.DataFrame({"value": [1, "red", 3, "inf", None]})
    assert MixedSpec.parse(text).select(table).index.tolist() == expected


def test_field_kind_is_required():
    with pytest.raises(ValidationError, match="Field required"):
        FieldConstraint()  # ty: ignore[missing-argument] -- Test required-field validation.


@pytest.mark.parametrize("kind", ["", "number", "labels", None])
def test_invalid_field_kind_raises(kind):
    with pytest.raises(ValidationError, match="literal_error"):
        FieldConstraint(kind=kind)


def test_prefixed_matching_is_generic_and_not_interface_specific():
    @dataclass(frozen=True)
    class DeviceSpec(ObsSpecBase):
        device: Selector = None

        _primary_field: ClassVar = "device"
        _field_constraints: ClassVar = {"device": FieldConstraint(kind="mixed")}

    table = pd.DataFrame(
        {"device": ["camera", "detector2", "detector3", "sensor3", None]}
    )
    numbers = {"device": {"detector2": 2, "detector3": 3, "sensor3": 3}}
    context = ObsSpecContext(field_label_numbers=numbers)
    assert DeviceSpec.parse("{camera,2}").select(
        table, context=context
    ).index.tolist() == [0, 1]
    assert DeviceSpec.parse("(1,4)[-1]").select(
        table, context=context
    ).index.tolist() == [2, 3]
    assert DeviceSpec.parse("detector(1,4)[-1]").select(table).index.tolist() == [2]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("(1,5)", ["toltec1", "toltec4", "bar1", "bar3"]),
        ("toltec(1,5)", ["toltec1", "toltec4"]),
        ("bar(1,5)", ["bar1", "bar3"]),
        ("toltec(,)", ["toltec0", "toltec1", "toltec4", "toltec10"]),
        ("1", ["toltec1", "bar1"]),
        ("toltec1", ["toltec1"]),
        ("{tel,1}", ["tel", "toltec1", "bar1"]),
    ],
)
def test_one_selector_resolves_against_available_values(text, expected):
    available = ["tel", "toltec0", "toltec1", "toltec4", "toltec10", "bar1", "bar3"]
    spec = ObsSpec.parse(f"1/{text}")
    context = toltec_obs_spec_context | ObsSpecContext(
        field_label_numbers={"interface": {"bar1": 1, "bar3": 3}},
    )
    assert spec.resolve_values("interface", available, context=context) == expected
    table = pd.DataFrame({"obsnum": [1] * len(available), "interface": available})
    assert spec.select(table, context=context).interface.tolist() == expected
    assert ObsSpec.parse(str(spec)) == spec
    assert hash(ObsSpec.parse(str(spec))) == hash(spec)


def test_prefix_is_preserved_and_positions_are_deferred():
    spec = ObsSpec.parse("1/toltec(1,5)[-1]")
    assert str(spec) == "1/toltec(1,5)[-1:]"
    assert spec.interface == FieldPattern(
        "toltec", FieldSelection(ValueBounds(1, 5), slice(-1, None)), ""
    )
    assert spec.resolve_values("interface", ToltecInfo.interfaces) == [
        "toltec1",
        "toltec2",
        "toltec3",
        "toltec4",
    ]
    table = pd.DataFrame(
        {"obsnum": [1, 2, 1], "interface": ["toltec1", "toltec4", "bar3"]}
    )
    assert spec.select(table).interface.tolist() == ["toltec1"]


def test_positions_order_numeric_components_not_labels():
    table = pd.DataFrame(
        {"obsnum": [1] * 4, "interface": ["toltec2", "toltec10", "bar10", "tel"]}
    )
    context = toltec_obs_spec_context | ObsSpecContext(
        field_label_numbers={"interface": {"bar10": 10}}
    )
    assert ObsSpec.parse("1/()[-1]").select(table, context=context).index.tolist() == [
        1,
        2,
    ]
    assert ObsSpec.parse("1/toltec(,)[-1]").select(table).index.tolist() == [1]


@pytest.mark.parametrize(
    "text", ["toltec(1, 5)", "toltec(-1,5)", "toltec(1,5", "foo(1,5)bar[-1]"]
)
def test_invalid_prefixed_bounds_are_rejected(text):
    with pytest.raises(ValueError):
        ObsSpec.parse(f"1/{text}")


@pytest.mark.parametrize("dtype", ["object", "string", "string[pyarrow]", "category"])
def test_numbered_label_matching_retains_nullable_indexed_columns(dtype):
    table = pd.DataFrame(
        {
            "obsnum": [1] * 4,
            "interface": pd.array(["tel", None, "toltec2", "toltec10"], dtype=dtype),
        },
        index=[5, 5, 3, 2],
    )
    selected = ObsSpec.parse("1/toltec(,)[-1]").select(table)
    assert selected.index.tolist() == [2]
    assert selected.interface.tolist() == ["toltec10"]


def test_numeric_components_are_not_rounded_through_float():
    values = ["bar9007199254740992", None, "bar9007199254740993"]
    table = pd.DataFrame({"obsnum": [1] * 3, "interface": values})
    assert ObsSpec.parse("1/bar(,)[-1]").select(table).index.tolist() == [2]


@pytest.mark.parametrize(
    ("text", "before", "bounds", "positions", "after"),
    [
        ("foo(100,200)[-1]bar", "foo", ValueBounds(100, 200), slice(-1, None), "bar"),
        ("foo()[-1]bar", "foo", ValueBounds(), slice(-1, None), "bar"),
        ("foo0(1,2)", "foo0", ValueBounds(1, 2), None, ""),
        ("(1,2)bar", "", ValueBounds(1, 2), None, "bar"),
        ("foo()[:]bar", "foo", ValueBounds(), None, "bar"),
        ("100(,)[-1]", "100", ValueBounds(), slice(-1, None), ""),
    ],
)
def test_numeric_selection_between_literal_parts(
    text, before, bounds, positions, after
):
    spec = MixedSpec.parse(text)
    assert spec.value == FieldPattern(before, FieldSelection(bounds, positions), after)
    assert MixedSpec.parse(str(spec)) == spec
    assert hash(MixedSpec.parse(str(spec))) == hash(spec)


@pytest.mark.parametrize(
    "constraint", [FieldConstraint(kind="label"), FieldConstraint(kind="integer")]
)
@pytest.mark.parametrize(
    "text",
    ["foo(100,200)[-1]bar", "foo()[-1]bar", "foo0(1,2)", "(1,2)bar", "foo()[:]bar"],
)
def test_field_constraints_raise_for_otherwise_valid_grammar(constraint, text):
    class RestrictedSpec(MixedSpec):
        _field_constraints: ClassVar = {"value": constraint}

    valid = MixedSpec.parse(text)
    with pytest.raises(ValueError, match="require a mixed field"):
        RestrictedSpec.parse(text)
    with pytest.raises(ValueError, match="require a mixed field"):
        RestrictedSpec(value=valid.value)


def test_numeric_only_field_accepts_selection_without_literal_parts():
    class NumericSpec(MixedSpec):
        _field_constraints: ClassVar = {"value": FieldConstraint(kind="integer")}

    spec = NumericSpec.parse("(100,200)[-1]")
    table = pd.DataFrame({"value": [199, 100, 200]})
    assert spec.select(table).value.tolist() == [199]
    for text in ("foo", "{1,foo}", "foo(100,200)[-1]bar"):
        with pytest.raises(ValueError):
            NumericSpec.parse(text)


def test_digit_ending_literal_is_not_split_into_an_inferred_number():
    values = ["foo00", "foo01", "foo10", "foo11", "foo1"]
    table = pd.DataFrame({"value": values})
    for value in values:
        assert MixedSpec.parse(value).select(table).value.tolist() == [value]
    assert MixedSpec.parse("1").select(table).empty
    numbers = {"foo00": 0, "foo01": 1, "foo10": 0, "foo11": 1}
    context = ObsSpecContext(field_label_numbers={"value": numbers})
    assert MixedSpec.parse("1").select(table, context=context).value.tolist() == [
        "foo01",
        "foo11",
    ]
    assert MixedSpec.parse("foo0(1,2)").select(table).value.tolist() == ["foo01"]
    assert MixedSpec.parse("foo1(0,2)").select(table).value.tolist() == [
        "foo10",
        "foo11",
    ]


def test_pattern_slot_not_suffix_digits_controls_numeric_order():
    table = pd.DataFrame({"value": ["foo010bar9", "foo011bar9", "foo012bar8", None]})
    selected = MixedSpec.parse("foo0(10,20)[-1]bar9").select(table)
    assert selected.value.tolist() == ["foo011bar9"]
    assert MixedSpec.parse("foo0(10,20)[-1]bar7").select(table).empty


def test_literal_patterns_keep_positions_after_all_value_constraints():
    spec = ObsSpec.parse("(100,200)/foo(100,200)[-1]bar")
    table = pd.DataFrame(
        {"obsnum": [100, 99, 199], "interface": ["foo100bar", "foo199bar", "foo110bar"]}
    )
    assert spec.select(table).interface.tolist() == ["foo110bar"]
    assert (
        spec.resolve_values("interface", table.interface.tolist())
        == table.interface.tolist()
    )
    with pytest.raises(ValueError, match="Only one positional"):
        ObsSpec.parse("[-1]/foo(100,200)[-1]bar")


@pytest.mark.parametrize(
    "text",
    ["foo(1,2)bar(3,4)", "foo*[-1]bar*[-2]", "foo(1,2)bar[-1]", "foo(1,2)[-1] bar"],
)
def test_only_one_contiguous_numeric_slot_is_supported(text):
    with pytest.raises(ValueError):
        MixedSpec.parse(text)


def test_exact_label_is_distinct_from_a_numeric_slot():
    spec = MixedSpec.parse("foo01")
    assert str(spec) == "foo01"
    assert MixedSpec.parse(str(spec)) == spec
    table = pd.DataFrame({"value": ["foo01", "foo011"]})
    assert spec.select(table).value.tolist() == ["foo01"]
    assert MixedSpec.parse("foo01()[-1]").select(table).value.tolist() == ["foo011"]
