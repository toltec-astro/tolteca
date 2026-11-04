"""Selector values validate their declared contracts before normalization."""

from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from tolteca_datamodels.core.obs_spec import (
    FieldConstraint,
    FieldPattern,
    FieldSelection,
    ValueBounds,
)


@pytest.mark.parametrize(
    ("cls", "data", "field"),
    [
        (ValueBounds, {"lower": True}, "lower"),
        (ValueBounds, {"upper": "2"}, "upper"),
        (ValueBounds, {"lower": -1}, "lower"),
        (FieldConstraint, {"kind": "other"}, "kind"),
        (FieldSelection, {"value_constraint": (1, True)}, "value_constraint"),
        (FieldSelection, {"value_constraint": "two words"}, "value_constraint"),
        (FieldSelection, {"value_constraint": "foo\n"}, "value_constraint"),
        (FieldSelection, {"value_constraint": ()}, "value_constraint"),
        (FieldSelection, {"positions": (True,)}, "positions"),
        (FieldSelection, {"positions": ()}, "positions"),
        (FieldSelection, {"positions": slice(True)}, "positions"),  # noqa: FBT003 -- invalid input.
        (FieldSelection, {"positions": slice(None, None, 0)}, "positions"),
        (
            FieldPattern,
            {"before": "!foo", "selection": FieldSelection(ValueBounds()), "after": ""},
            "before",
        ),
        (
            FieldPattern,
            {"before": "foo", "selection": object(), "after": ""},
            "selection",
        ),
    ],
)
def test_invalid_values_raise_located_validation_errors(cls, data, field):
    with pytest.raises(ValidationError) as exc:
        cls(**data)
    assert exc.value.errors()[0]["loc"][0] == field


def test_validated_selectors_retain_positional_construction_and_normalization():
    bounds = ValueBounds(1, 5)
    selection = FieldSelection(bounds, slice(-1, None))
    assert str(FieldPattern("toltec", selection, "")) == "toltec(1,5)[-1:]"
    assert FieldSelection(("tel", 2, 1, 2)).value_constraint == (1, 2, "tel")
    with pytest.raises(FrozenInstanceError):
        bounds.lower = 2  # ty: ignore[invalid-assignment] -- frozen-field rejection.


@pytest.mark.parametrize(
    ("positions", "expected"),
    [
        (None, None),
        (slice(None), None),
        (slice(0, None), None),
        (slice(None, None, 1), None),
        (slice(0, None, 1), None),
        ((2, 2), slice(2, 3)),
        ((-1, -1), slice(-1, None)),
        ((-3,), slice(-3, -2)),
        ((2, -1, 2, -3), (-3, -1, 2)),
        (slice(None, None, -1), slice(None, None, -1)),
        (slice(0, None, 2), slice(0, None, 2)),
        (slice(0, 0), slice(0, 0)),
    ],
)
def test_positions_normalize_without_changing_selection(positions, expected):
    selection = FieldSelection(positions=positions)
    assert selection.positions == expected
    values = list(range(6))
    if expected is None:
        assert selection.select_values(values) == values
    elif isinstance(expected, slice):
        assert selection.select_values(values) == values[expected]
    else:
        assert selection.select_values(values) == [values[index] for index in expected]


def test_omitted_positions_remain_unconstrained():
    selection = FieldSelection()
    assert selection.positions is None
    assert selection.select_values([1, 3]) == [1, 3]
