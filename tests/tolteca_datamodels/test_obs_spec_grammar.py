"""Whole-field values, embedded numeric slots, and positional resolution."""

from dataclasses import dataclass
from typing import ClassVar

import pandas as pd
import pytest
from tollan.utils.log import logger

from tolteca_datamodels.lmt.obs_spec import obs_spec_context as lmt_obs_spec_context
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec
from tolteca_datamodels.obs_spec import (
    FieldConstraint,
    FieldPattern,
    FieldSelection,
    ObsSpecBase,
    ObsSpecContext,
    Selector,
    ValueBounds,
)
from tolteca_datamodels.toltec.obs_spec import (
    obs_spec_context as toltec_obs_spec_context,
)


@dataclass(frozen=True)
class MixedSpec(ObsSpecBase):
    value: Selector = None

    _primary_field: ClassVar = "value"
    _field_constraints: ClassVar = {"value": FieldConstraint(kind="mixed")}


@pytest.mark.parametrize(
    ("text", "canonical"),
    [
        ("*", "*"),
        ("{}", "*"),
        ("{,}", "*"),
        ("()", "(,)"),
        ("(,)", "(,)"),
        ("[]", "*"),
        ("[:]", "*"),
        ("[-1]", "*[-1:]"),
        ("*[-1]", "*[-1:]"),
        ("()[-1]", "(,)[-1:]"),
        ("(,)[-1]", "(,)[-1:]"),
        ("{}[-1]", "*[-1:]"),
        ("{,}[-1]", "*[-1:]"),
        ("foo()bar", "foo(,)bar"),
        ("foo(,)bar", "foo(,)bar"),
        ("foo()[-1]bar", "foo(,)[-1:]bar"),
        ("foo(,)[-1]bar", "foo(,)[-1:]bar"),
        ("foo()[:]bar", "foo(,)bar"),
        ("foo(,)[]bar", "foo(,)bar"),
        ("foo(100,)bar", "foo(100,)bar"),
        ("foo(,200)bar", "foo(,200)bar"),
        ("foo(100,200)[-1]bar", "foo(100,200)[-1:]bar"),
        ("100(,)[-1]", "100(,)[-1:]"),
        ("()[-1]bar", "(,)[-1:]bar"),
        ("foo()", "foo(,)"),
        ("(100,200)[:]", "(100,200)"),
        ("{tcs,ics}[-1]", "{ics,tcs}[-1:]"),
        ("{toltec5,toltec1}[-1]", "{toltec1,toltec5}[-1:]"),
    ],
)
def test_canonical_text_is_idempotent(text, canonical):
    spec = MixedSpec.parse(text)
    assert str(spec) == canonical
    reparsed = MixedSpec.parse(canonical)
    assert reparsed == spec
    assert hash(reparsed) == hash(spec)
    assert str(reparsed) == canonical


@pytest.mark.parametrize("value", ["100", "tel", "{1,3}", "{tcs,ics}", "{tel,1}"])
@pytest.mark.parametrize("position", ["[]", "[:]", "[::1]", "[0:]", "[-1]", "[-1,-3]"])
def test_position_syntax_is_independent_of_value_sortability(value, position):
    spec = MixedSpec.parse(value + position)
    assert MixedSpec.parse(str(spec)) == spec


@pytest.mark.parametrize("value", [100, "tel", (1, 3), ("tcs", "ics"), ("tel", 1)])
@pytest.mark.parametrize("positions", [None, slice(None), slice(-1, None), (-1, -3)])
def test_selection_constructor_and_text_have_the_same_contract(value, positions):
    spec = MixedSpec(FieldSelection(value, positions))
    assert MixedSpec.parse(str(spec)) == spec


@pytest.mark.parametrize(
    "text",
    [
        "foo[-1]bar",
        "foo[:]bar",
        "foo[]bar",
        "[-1]bar",
        "foo*bar",
        "foo*",
        "*bar",
        "foo*[-1]bar",
        "foo(,)bar[-1]",
        "foo(1,2)bar(3,4)",
        "foo{}bar",
        "toltec{}[-1]",
        "{1,2}bar",
        "foo{1,2}",
    ],
)
def test_only_parentheses_can_appear_between_literal_parts(text):
    with pytest.raises(ValueError):
        MixedSpec.parse(text)


def test_empty_bounds_retain_numeric_meaning():
    assert str(ValueBounds()) == "(,)"
    assert str(FieldSelection(ValueBounds())) == "(,)"
    assert str(FieldSelection(ValueBounds(), slice(-1, None))) == "(,)[-1:]"
    table = pd.DataFrame({"value": [0, "label", 3, None]})
    assert MixedSpec.parse("{}").select(table).index.tolist() == [0, 1, 2, 3]
    for text in ("()", "(,)", "()[:]", "(,)[]"):
        assert MixedSpec.parse(text).select(table).index.tolist() == [0, 2]


@pytest.mark.parametrize("text", ["()", "(,)", "()[-1]"])
def test_label_only_field_rejects_numeric_bounds(text):
    class LabelSpec(MixedSpec):
        _field_constraints: ClassVar = {"value": FieldConstraint(kind="label")}

    with pytest.raises(ValueError, match="integer or mixed field"):
        LabelSpec.parse(text)


@pytest.mark.parametrize("value", [1, "label", (1, 2), ("a", "b"), None])
def test_python_pattern_requires_numeric_bounds(value):
    with pytest.raises(ValueError, match="numeric bounds"):
        FieldPattern("foo", FieldSelection(value), "bar")


@pytest.mark.parametrize("text", ["foo0()[-1]bar9", "foo0(,)[-1]bar9"])
def test_explicit_numeric_slot_preserves_digit_ending_literals(text):
    table = pd.DataFrame(
        {"value": ["foo010bar9", "foo011bar9", "foo012bar8", "foo0textbar9", None]}
    )
    assert MixedSpec.parse(text).select(table).value.tolist() == ["foo011bar9"]


@pytest.mark.parametrize("text", ["{tcs,ics}[:]-1", "{tcs,ics}[]-1", "{tcs,ics}[-1]-1"])
def test_master_positions_preserve_label_filter(text):
    table = pd.DataFrame({"master": ["tcs", "ics", "clip"], "obsnum": [1, 1, 1]})
    assert ObsSpec.parse(text).select(table).master.tolist() == ["tcs", "ics"]


@pytest.mark.parametrize("with_tel", [True, False])
def test_unsortable_declared_set_does_not_depend_on_available_files(with_tel):
    interfaces = ["toltec1", "toltec5"] + (["tel"] if with_tel else [])
    table = pd.DataFrame({"obsnum": [1] * len(interfaces), "interface": interfaces})
    messages = []
    sink = logger.add(messages.append, level="DEBUG")
    try:
        spec = ObsSpec.parse("1/{tel,toltec1,toltec5}[-1]")
        assert (
            spec.select(
                table, context=lmt_obs_spec_context | toltec_obs_spec_context
            ).interface.tolist()
            == interfaces
        )
    finally:
        logger.remove(sink)
    assert any("Ignoring positional selection" in message for message in messages)


def test_mixed_wildcard_checks_full_context():
    table = pd.DataFrame({"obsnum": [1, 1], "interface": ["toltec1", "toltec5"]})
    # Even with only ROACH files present, the declared context also contains tel/HK.
    assert ObsSpec.parse("1/*[-1]").select(
        table, context=lmt_obs_spec_context | toltec_obs_spec_context
    ).interface.tolist() == [
        "toltec1",
        "toltec5",
    ]
    assert ObsSpec.parse("1/()[-1]").select(
        table, context=lmt_obs_spec_context | toltec_obs_spec_context
    ).interface.tolist() == ["toltec5"]


def test_numeric_set_uses_observed_values_after_other_filters():
    spec = ObsSpec.parse("1/{toltec1,toltec5,toltec12}[-1]")
    table = pd.DataFrame(
        {"obsnum": [1, 1, 2], "interface": ["toltec1", "toltec5", "toltec12"]}
    )
    assert spec.select(
        table, context=lmt_obs_spec_context | toltec_obs_spec_context
    ).interface.tolist() == ["toltec5"]
    assert (
        spec.resolve_values("interface", table.interface.tolist())
        == table.interface.tolist()
    )


def test_contexts_control_sortability_without_reparsing():
    spec = MixedSpec.parse("{a,b}[-1]")
    table = pd.DataFrame({"value": ["a", "b"]})
    ordered = ObsSpecContext(labels={"value": {"a": 5, "b": 1}})
    unordered = ObsSpecContext(labels={"value": {"a": 5, "b": None}})
    assert spec.select(table, context=ordered).value.tolist() == ["a"]
    assert spec.select(table, context=unordered).value.tolist() == ["a", "b"]


def test_numeric_positions_wait_for_all_value_constraints():
    spec = ObsSpec.parse("(100,200)/foo()[-1]bar")
    table = pd.DataFrame(
        {"obsnum": [100, 99, 199], "interface": ["foo100bar", "foo199bar", "foo110bar"]}
    )
    assert spec.select(table).interface.tolist() == ["foo110bar"]
    assert (
        spec.resolve_values("interface", table.interface.tolist())
        == table.interface.tolist()
    )
