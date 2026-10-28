"""Explicit label interpretations for observation selectors."""

import pandas as pd
import pytest

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


def test_column_constraints_reuse_declarations():
    context = ObsSpecContext(field_label_numbers={"interface": {"bar1": 1}})
    spec = ObsSpec.parse("1/1")
    first = list(spec.column_constraints(context=context))[-1].label_numbers
    second = list(spec.column_constraints(context=context))[-1].label_numbers
    assert first is second is context.field_label_numbers["interface"]
