"""Explicit, immutable label interpretations for observation selectors."""

import pandas as pd
import pytest
from pydantic import ValidationError

from tolteca_datamodels.core.obs_spec import ObsSpecContext
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec


def test_contexts_are_independent_and_do_not_change_parsing():
    spec = ObsSpec.parse("1/()[-1]")
    table = pd.DataFrame({"obsnum": [1, 1], "interface": ["bar1", "bar2"]})
    context = ObsSpecContext(field_label_numbers={"interface": {"bar1": 1, "bar2": 2}})
    other = ObsSpecContext(field_label_numbers={"interface": {"bar1": 99, "bar2": 2}})
    assert spec.select(table, context=context).interface.tolist() == ["bar2"]
    assert spec.select(table, context=other).interface.tolist() == ["bar1"]
    assert spec.select(table).empty
    assert ObsSpec.parse(str(spec)) == spec


def test_context_copies_declarations_and_is_deeply_read_only():
    labels = {"interface": {"bar1": 1}}
    context = ObsSpecContext(field_label_numbers=labels)
    labels["interface"]["bar1"] = 2
    labels["other"] = {"new": 3}
    assert context.field_label_numbers == {"interface": {"bar1": 1}}
    with pytest.raises(TypeError):
        context.field_label_numbers["interface"]["bar1"] = 4  # ty: ignore[invalid-assignment] -- Verify runtime immutability.
    with pytest.raises(TypeError):
        context.field_label_numbers["other"] = {}  # ty: ignore[invalid-assignment] -- Verify runtime immutability.
    with pytest.raises(ValidationError, match="frozen_instance"):
        setattr(context, "field_label_numbers", {})  # noqa: B010 -- Test frozen validation.


def test_empty_context_is_read_only():
    context = ObsSpecContext()
    assert context.field_label_numbers == {}
    with pytest.raises(TypeError):
        context.field_label_numbers["interface"] = {}  # ty: ignore[invalid-assignment] -- Verify runtime immutability.


def test_context_model_validation_and_dump():
    data = {"field_label_numbers": {"interface": {"tel": None, "bar0": 0, "bar1": 1}}}
    context = ObsSpecContext.model_validate(data)
    assert context == ObsSpecContext(**data)
    assert context.model_dump(warnings="error") == data
    assert ObsSpecContext.model_validate_json(context.model_dump_json()) == context
    with pytest.raises(TypeError):
        context.field_label_numbers["interface"]["bar1"] = 2  # ty: ignore[invalid-assignment] -- Verify runtime immutability.


@pytest.mark.parametrize("number", [2, None])
@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_composition_raises(number, reverse):
    first = ObsSpecContext(field_label_numbers={"interface": {"bar1": 1}})
    second = ObsSpecContext(field_label_numbers={"interface": {"bar1": number}})
    with pytest.raises(ValueError, match="Conflicting numeric interpretation"):
        _ = second | first if reverse else first | second
    assert first.field_label_numbers["interface"]["bar1"] == 1
    assert second.field_label_numbers["interface"]["bar1"] == number


def test_composition_keeps_shared_labels_and_independent_fields():
    first = ObsSpecContext(field_label_numbers={"interface": {"bar1": 1, "tel": None}})
    second = ObsSpecContext(
        field_label_numbers={
            "interface": {"bar1": 1, "bar2": 2},
            "other": {"bar1": 99},
        }
    )
    context = first | second | first
    assert context.field_label_numbers == {
        "interface": {"bar1": 1, "tel": None, "bar2": 2},
        "other": {"bar1": 99},
    }
    assert (
        context | ObsSpecContext()
    ).field_label_numbers == context.field_label_numbers
    assert (
        ObsSpecContext() | context
    ).field_label_numbers == context.field_label_numbers
    assert ObsSpec.parse("1/()").resolve_values(
        "interface", ["bar1", "bar2", "tel"], context=context
    ) == ["bar1", "bar2"]


def test_context_is_not_an_allowlist_for_literal_matching():
    context = ObsSpecContext(field_label_numbers={"interface": {"bar1": 1}})
    table = pd.DataFrame({"obsnum": [1], "interface": ["other"]})
    pd.testing.assert_frame_equal(
        ObsSpec.parse("1/other").select(table, context=context),
        table,
    )


def test_column_constraints_reuse_immutable_declarations():
    context = ObsSpecContext(field_label_numbers={"interface": {"bar1": 1}})
    spec = ObsSpec.parse("1/1")
    first = list(spec.column_constraints(context=context))[-1].label_numbers
    second = list(spec.column_constraints(context=context))[-1].label_numbers
    assert first is second is context.field_label_numbers["interface"]


@pytest.mark.parametrize("validate", [ObsSpecContext, ObsSpecContext.model_validate])
@pytest.mark.parametrize(
    ("number", "error_type"),
    [
        (-1, "greater_than_equal"),
        (True, "int_type"),
        (1.5, "int_type"),
        ("1", "int_type"),
    ],
)
def test_invalid_numeric_interpretation_raises(validate, number, error_type):
    data = {"field_label_numbers": {"interface": {"bar1": number}}}
    with pytest.raises(ValidationError) as exc:
        validate(**data) if validate is ObsSpecContext else validate(data)
    error = exc.value.errors()[0]
    assert error["loc"] == ("field_label_numbers", "interface", "bar1")
    assert error["type"] == error_type


@pytest.mark.parametrize(
    ("label", "error_type"),
    [
        (1, "string_type"),
        ("", "string_pattern_mismatch"),
        ("two words", "string_pattern_mismatch"),
        ("bar1\n", "string_pattern_mismatch"),
    ],
)
def test_invalid_label_raises(label, error_type):
    with pytest.raises(ValidationError) as exc:
        ObsSpecContext(field_label_numbers={"interface": {label: 1}})
    error = exc.value.errors()[0]
    assert error["loc"] == ("field_label_numbers", "interface", label, "[key]")
    assert error["type"] == error_type


@pytest.mark.parametrize("name", ["label", "labels", "field_label_transforms"])
def test_context_rejects_extra_fields(name):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ObsSpecContext.model_validate({name: {"interface": {"bar1": 1}}})
