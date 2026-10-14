"""Observation-selector grammar and table selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
import pandas as pd
from tollan.utils.log import logger

from .context import _CONTEXT_EMPTY, ObsSpecContext
from .selectors import (
    ColumnSelector,
    FieldConstraint,
    FieldSelection,
    FieldValue,
    Selector,
    _ColumnMatch,
    _format_selector,
    _get_field_selection,
    _parse_selector,
)

if TYPE_CHECKING:
    import re
    from collections.abc import Iterator, Mapping, Sequence

_LEFT_SEPARATOR = "-"
_RIGHT_SEPARATOR = "/"
_BRACKETS = {"(": ")", "[": "]", "{": "}"}
_CLOSING_BRACKETS = frozenset(_BRACKETS.values())


def _split_field_tokens(text: str) -> tuple[list[str], list[str]]:
    """Split field separators outside value and positional brackets."""
    tokens, separators = [], []
    closing = None
    start = 0
    for index, char in enumerate(text):
        if char in _BRACKETS:
            if closing is not None:
                msg = f"Nested brackets in observation selector: {text!r}"
                raise ValueError(msg)
            closing = _BRACKETS[char]
        elif char in _CLOSING_BRACKETS:
            if char != closing:
                msg = f"Mismatched brackets in observation selector: {text!r}"
                raise ValueError(msg)
            closing = None
        elif closing is None and char in (_LEFT_SEPARATOR, _RIGHT_SEPARATOR):
            tokens.append(text[start:index])
            separators.append(char)
            start = index + 1
    tokens.append(text[start:])
    if closing is not None or any(not token for token in tokens):
        msg = f"Invalid observation selector: {text!r}"
        raise ValueError(msg)
    return tokens, separators


@dataclass(frozen=True)
class ObsSpecBase:
    """Base for immutable observation selectors.

    Subclasses declare dataclass fields and an ordered mapping of
    ``_field_constraints``. ``_primary_field`` anchors abbreviated expressions.
    Dash-separated tokens fill fields from the primary onward; slash-separated
    tokens fill remaining fields from the right.

    ``_primary_pattern`` identifies the first complete token belonging to the
    primary field. Preceding tokens fill the immediately preceding fields and
    must use dashes. Without a pattern, the first token is the primary field.
    All tokens are parsed and validated with the shared field grammar.

    Raises
    ------
    ValueError
        If a field violates its constraint, more than one positional selection is
        supplied, or normalized tokens cannot identify the primary field
        unambiguously.
    """

    _field_constraints: ClassVar[Mapping[str, FieldConstraint]] = {}
    _primary_field: ClassVar[str]
    _primary_pattern: ClassVar[re.Pattern[str] | None] = None
    _field_order: ClassVar[tuple[str, ...]] = ()
    _primary_index: ClassVar[int]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._field_order = tuple(cls._field_constraints)
        if cls._primary_field not in cls._field_order:
            msg = f"Primary field {cls._primary_field!r} has no declared constraint"
            raise ValueError(msg)
        cls._primary_index = cls._field_order.index(cls._primary_field)

    def __post_init__(self) -> None:
        positional_fields = []
        for name, constraint in self._field_constraints.items():
            value = getattr(self, name)
            if value is None:
                continue
            value_normalized = constraint.normalize(value)
            if value_normalized is not value:
                object.__setattr__(self, name, value_normalized)
            selection = _get_field_selection(value_normalized)
            if selection is not None and selection.positions is not None:
                positional_fields.append(name)
        if len(positional_fields) > 1:
            msg = (
                "Only one positional selector is allowed per ObsSpec; found "
                + ", ".join(positional_fields)
            )
            raise ValueError(msg)
        # Leading tokens must not be mistaken for the primary field when dumped.
        prefix = self._prefix_tokens()
        tokens = [*prefix, _format_selector(getattr(self, self._primary_field))]
        if self._find_primary_token(tokens) != len(prefix):
            msg = (
                "Selector fields cannot be represented around primary field "
                f"{self._primary_field!r}: {tokens!r}"
            )
            raise ValueError(msg)

    @classmethod
    def parse(cls, value: str | int | Self) -> Self:
        """Parse an observation selector.

        Parameters
        ----------
        value : str, int or Self
            Selector text without whitespace, a nonnegative integer for the
            primary field, or an existing instance. Existing instances are
            returned unchanged.

        Returns
        -------
        Self
            Validated selector with literal sets sorted and deduplicated.

        Raises
        ------
        ValueError
            If syntax, a field value or the number of positional selectors is
            invalid. Empty text is invalid.
        TypeError
            If the input is not a string, integer or instance of this class.
        """
        if isinstance(value, cls):
            return value
        if type(value) is int:
            return cls(**{cls._primary_field: value})
        if not isinstance(value, str):
            msg = f"Expected an ObsSpec string or integer, got {value!r}"
            raise TypeError(msg)
        return cls(
            **{
                name: _parse_selector(token)
                for name, token in cls._parse_layout(value).items()
            }
        )

    @classmethod
    def _find_primary_token(cls, tokens: Sequence[str]) -> int | None:
        pattern = cls._primary_pattern
        if pattern is None:
            return 0
        for index, token in enumerate(tokens):
            if pattern.fullmatch(token):
                return index
        return None

    @classmethod
    def _parse_layout(cls, text: str) -> dict[str, str]:
        tokens, separators = _split_field_tokens(text)
        primary_token_index = cls._find_primary_token(tokens)
        if primary_token_index is None:
            msg = f"Missing primary field {cls._primary_field!r}: {text!r}"
            raise ValueError(msg)
        primary_index = cls._primary_index
        if (
            primary_token_index > primary_index
            or _RIGHT_SEPARATOR in separators[:primary_token_index]
        ):
            msg = (
                f"Invalid layout before primary field {cls._primary_field!r}: {text!r}"
            )
            raise ValueError(msg)
        available_fields = cls._field_order[primary_index - primary_token_index :]
        first_right = (
            separators.index(_RIGHT_SEPARATOR)
            if _RIGHT_SEPARATOR in separators
            else len(separators)
        )
        if _LEFT_SEPARATOR in separators[first_right:] or len(tokens) > len(
            available_fields
        ):
            msg = f"Invalid observation selector: {text!r}"
            raise ValueError(msg)
        left, right = tokens[: first_right + 1], tokens[first_right + 1 :]
        assignments = dict(zip(available_fields, left, strict=False))
        if right:
            assignments.update(zip(available_fields[-len(right) :], right, strict=True))
        return assignments

    def _prefix_tokens(self) -> list[str]:
        tokens = []
        for name in self._field_order[: self._primary_index]:
            value = getattr(self, name)
            if value is not None or tokens:
                tokens.append(_format_selector(value))
        return tokens

    def __str__(self) -> str:
        values = [
            _format_selector(getattr(self, name))
            for name in self._field_order[self._primary_index :]
        ]
        suffix = []
        if values[-1] != "*":
            # Keep the primary field; slash tokens align to the actual final field.
            for i in range(len(values) - 2, 0, -1):
                if values[i] == "*":
                    suffix = values[i + 1 :]
                    values = values[: i + 1]
                    break
        while len(values) > 1 and values[-1] == "*":
            values.pop()
        return _RIGHT_SEPARATOR.join(
            (_LEFT_SEPARATOR.join((*self._prefix_tokens(), *values)), *suffix)
        )

    def constraints(self) -> Iterator[tuple[str, Selector]]:
        """Yield constrained logical fields.

        Yields
        ------
        tuple of str and Selector
            Field name and its normalized, non-None selector.
        """
        for name in self._field_constraints:
            value = getattr(self, name)
            if value is not None:
                yield name, value

    def column_constraints(
        self, *, context: ObsSpecContext = _CONTEXT_EMPTY
    ) -> Iterator[ColumnSelector]:
        """Yield one column selector per constrained field.

        Parameters
        ----------
        context : ObsSpecContext, optional
            Label interpretations. Default is an empty context, giving labels
            no numeric interpretation.

        Yields
        ------
        ColumnSelector
            Column constraints combined with AND. Literal alternatives within
            each selector combine with OR.
        """
        for name, selector in self.constraints():
            yield ColumnSelector(
                name, selector, label_numbers=context.field_label_numbers.get(name)
            )

    def resolve_values(
        self,
        name: str,
        values: Sequence[FieldValue],
        *,
        context: ObsSpecContext = _CONTEXT_EMPTY,
    ) -> list[FieldValue]:
        """Resolve a field's value constraints against available values.

        Parameters
        ----------
        name : str
            Selector field to resolve.
        values : sequence of int or str
            Values declared by a store or present in a table.
        context : ObsSpecContext, optional
            Label interpretations. Default is an empty context.
            Labels are never split to infer a number;
            explicit patterns specify their own numeric slot.

        Returns
        -------
        list of int or str
            Matching values in supplied order. Positional selection is deferred
            until all table value constraints have been applied.
        """
        selector = ColumnSelector(
            name,
            getattr(self, name),
            label_numbers=context.field_label_numbers.get(name),
        )
        constraint = selector.value_constraint
        if constraint is None:
            return list(values)
        available = pd.Index(list(values), dtype=object)
        if isinstance(constraint, str):
            return available[available == constraint].tolist()
        labels, constraint = selector._value_parts
        matched = available.isin(labels)
        if constraint is None:
            return available[matched].tolist()
        numbers = selector._resolve_numbers(available)
        matched |= _ColumnMatch._numeric_mask(numbers, constraint).to_numpy(dtype=bool)
        return available[matched].tolist()

    @property
    def positional_constraint(self) -> tuple[str, FieldSelection] | None:
        """The ordered column and its positional selection, if specified."""
        return next(
            (
                (name, selection)
                for name, value in self.constraints()
                if (selection := _get_field_selection(value)) is not None
                and selection.positions is not None
            ),
            None,
        )

    def select(
        self,
        table: pd.DataFrame,
        *,
        context: ObsSpecContext = _CONTEXT_EMPTY,
    ) -> pd.DataFrame:
        """Filter values, then select distinct-value positions.

        Parameters
        ----------
        table : pandas.DataFrame
            Table with columns required by the selector. All value constraints
            are applied before the positional selector, regardless of field order.
        context : ObsSpecContext, optional
            Label interpretations. Default is an empty context. Unmapped labels
            do not match bare numeric constraints. Explicit patterns identify
            the numeric slot from their literal parts.

        Returns
        -------
        pandas.DataFrame
            All matching rows, retaining their index and order. Constrained
            values cannot be null; unconstrained columns retain nulls. Positions
            refer to sorted distinct non-null values, not to rows. The input
            table is unchanged. If the selected values cannot be ordered
            numerically, positions are ignored with a debug message; value
            constraints still apply.

        Raises
        ------
        ValueError
            If a constrained field has no column.
        """
        # All predicates follow table row order. Combine their masks without
        # converting between pandas' nullable and Arrow boolean representations.
        mask = np.ones(len(table), dtype=bool)
        positional_match = None
        positional_constraint = self.positional_constraint
        for item in self.column_constraints(context=context):
            if item.name not in table:
                msg = f"Table lacks selector field {item.name!r}"
                raise ValueError(msg)
            match = _ColumnMatch(item, table[item.name])
            mask &= match.value_mask().to_numpy(dtype=bool, na_value=False)
            if (
                positional_constraint is not None
                and item.name == positional_constraint[0]
            ):
                positional_match = match
        if positional_match is not None and positional_constraint is not None:
            name, selection = positional_constraint
            if not self._field_constraints[name]._supports_positions(
                positional_match.selector
            ):
                logger.debug(
                    "Ignoring positional selection for {}={!s}: the selected "
                    "values have no common numeric ordering; "
                    "retaining value constraints",
                    name,
                    getattr(self, name),
                )
            else:
                mask &= positional_match.position_mask(mask, selection).to_numpy(
                    dtype=bool, na_value=False
                )
        return table.loc[mask]
