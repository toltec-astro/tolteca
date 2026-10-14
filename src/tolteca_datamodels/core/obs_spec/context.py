"""Label interpretations for observation selectors."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import (
    ConfigDict,
    Field,
    NonNegativeInt,
    field_serializer,
    field_validator,
)

from ..types import FrozenDataModel
from .selectors import _SelectorLabel

type _FieldLabelNumbers = Mapping[
    str,
    Mapping[_SelectorLabel, NonNegativeInt | None],
]


class ObsSpecContext(FrozenDataModel):
    """Label interpretations for observation selectors.

    Parameters
    ----------
    field_label_numbers : mapping, optional
        Field names mapped to known labels and their nonnegative integer values.
        None marks an unnumbered label. Default is an empty mapping. Declarations
        are copied and exposed as read-only mappings.
        Labels start with a letter and contain letters, digits, or underscores.

    Raises
    ------
    pydantic.ValidationError
        If a label is invalid or its numeric value is not a nonnegative integer
        or None.

    Notes
    -----
    Combine contexts with ``|``. Identical declarations are accepted; conflicting
    numeric values for the same field and label raise ValueError. A context
    supplies numeric interpretations, not an allowlist for literal matching.

    Examples
    --------
    >>> context = ObsSpecContext(
    ...     field_label_numbers={"interface": {"tel": None, "board1": 1}},
    ... )
    >>> context.field_label_numbers["interface"]["board1"]
    1
    """

    model_config = ConfigDict(extra="forbid")

    field_label_numbers: _FieldLabelNumbers = Field(default_factory=dict)

    @field_validator("field_label_numbers")
    @classmethod
    def _freeze_field_label_numbers(
        cls, field_label_numbers: _FieldLabelNumbers
    ) -> _FieldLabelNumbers:
        return MappingProxyType(
            {
                name: MappingProxyType(values)
                for name, values in field_label_numbers.items()
            }
        )

    @field_serializer("field_label_numbers")
    def _serialize_field_label_numbers(
        self, field_label_numbers: _FieldLabelNumbers
    ) -> dict[str, dict[str, int | None]]:
        return {name: dict(values) for name, values in field_label_numbers.items()}

    def __or__(self, other: ObsSpecContext) -> ObsSpecContext:
        if not isinstance(other, ObsSpecContext):
            return NotImplemented
        combined = {
            name: dict(labels) for name, labels in self.field_label_numbers.items()
        }
        for name, labels in other.field_label_numbers.items():
            target = combined.setdefault(name, {})
            for label, number in labels.items():
                if label in target and target[label] != number:
                    msg = f"Conflicting numeric interpretation for {name}={label!r}"
                    raise ValueError(msg)
                target[label] = number
        return ObsSpecContext(field_label_numbers=combined)


_CONTEXT_EMPTY = ObsSpecContext()
