"""Numeric interpretations of observation-selector labels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

type _FieldLabelNumbers = Mapping[str, Mapping[str, int | None]]


@dataclass(frozen=True, slots=True)
class ObsSpecContext:
    """Assign numbers to labels for numeric observation selection.

    Parameters
    ----------
    field_label_numbers : mapping of str to mappings of str to int or None, optional
        For each selector field, map labels to nonnegative integers or None
        for unnumbered labels. Defaults to an empty mapping. Supplied maps
        are shared declarations; keep them unchanged while the context is used.

    Notes
    -----
    Numeric constraints use these interpretations; literal matching does not
    require a label to be declared here. Combine contexts with ``|`` to create
    a new context without changing either input. Shared labels must have the
    same interpretation, including None; conflicts raise `ValueError`.

    Examples
    --------
    >>> context = ObsSpecContext(
    ...     field_label_numbers={"interface": {"tel": None, "board1": 1}},
    ... )
    >>> context.field_label_numbers["interface"]["board1"]
    1
    """

    field_label_numbers: _FieldLabelNumbers = field(default_factory=dict)

    def __or__(self, other: ObsSpecContext) -> ObsSpecContext:
        """Combine label interpretations, raising ValueError on conflicts."""
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
