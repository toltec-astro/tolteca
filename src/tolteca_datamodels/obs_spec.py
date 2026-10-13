"""Observation-selector grammar and table selection."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self

import numpy as np
import pandas as pd
from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    NegativeInt,
    NonNegativeInt,
    PositiveInt,
    TypeAdapter,
    field_serializer,
    field_validator,
)
from pydantic.dataclasses import dataclass as validated_dataclass
from tollan.utils.log import logger
from tollan.utils.slice import parse_slice

from .types import FrozenDataModel

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy.typing as npt

__all__ = [
    "ColumnSelector",
    "FieldConstraint",
    "FieldPattern",
    "FieldSelection",
    "FieldValue",
    "ObsSpecBase",
    "ObsSpecContext",
    "Selector",
    "ValueBounds",
]

_LABEL_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*")
_LITERAL_PART_PATTERN = re.compile(r"[a-zA-Z0-9_]*")
_FIELD_PATTERN = re.compile(
    r"(?P<before>[a-zA-Z0-9_]*)"
    r"\((?:(?P<lower>[0-9]*),(?P<upper>[0-9]*))?\)"
    r"(?:\[(?P<positions>[^\[\]]*)\])?"
    r"(?P<after>[a-zA-Z0-9_]*)"
)
_SELECTION_PATTERN = re.compile(r"(?P<values>[^\[\]]*)(?:\[(?P<positions>[^\[\]]*)\])?")
_POSITION_LIST_PATTERN = re.compile(r"[+-]?\d+(?:,[+-]?\d+)+")
_LEFT_SEPARATOR = "-"
_RIGHT_SEPARATOR = "/"
_BRACKETS = {"(": ")", "[": "]", "{": "}"}
_CLOSING_BRACKETS = frozenset(_BRACKETS.values())
_UNCONSTRAINED = {"", "*", "{}", "{,}"}

_SELECTOR_CONFIG = ConfigDict(
    strict=True, validate_default=True, arbitrary_types_allowed=True, extra="forbid"
)
type _SelectorLabel = Annotated[str, Field(pattern=rf"^{_LABEL_PATTERN.pattern}\z")]

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


@validated_dataclass(frozen=True, config=_SELECTOR_CONFIG)
class ValueBounds:
    """Half-open bounds on nonnegative integer values.

    Parameters
    ----------
    lower : int, optional
        Inclusive lower bound. Default is None, leaving the lower side unbounded.
    upper : int, optional
        Exclusive upper bound. Default is None, leaving the upper side unbounded.

    Raises
    ------
    pydantic.ValidationError
        If a bound is neither a nonnegative integer nor None.
    """

    lower: NonNegativeInt | None = None
    upper: NonNegativeInt | None = None

    def __str__(self) -> str:
        lower = "" if self.lower is None else str(self.lower)
        upper = "" if self.upper is None else str(self.upper)
        return f"({lower},{upper})"


type FieldValue = NonNegativeInt | _SelectorLabel


def _normalize_value_set(values: tuple[FieldValue, ...]) -> tuple[FieldValue, ...]:
    return tuple(sorted(set(values), key=lambda item: (isinstance(item, str), item)))


type _ValueSet[ValueT] = Annotated[
    tuple[ValueT, ...], Field(min_length=1), AfterValidator(_normalize_value_set)
]
type FieldValues = FieldValue | _ValueSet[FieldValue] | ValueBounds | None
_field_value_adapters = {
    "label": TypeAdapter(
        _SelectorLabel | _ValueSet[_SelectorLabel] | None, config=_SELECTOR_CONFIG
    ),
    "integer": TypeAdapter(
        NonNegativeInt | _ValueSet[NonNegativeInt] | ValueBounds | None,
        config=_SELECTOR_CONFIG,
    ),
    "mixed": TypeAdapter(FieldValues, config=_SELECTOR_CONFIG),
}
_slice_parts_adapter = TypeAdapter(
    tuple[int | None, int | None, PositiveInt | NegativeInt | None],
    config=ConfigDict(strict=True),
)


def _validate_position_slice(value: slice) -> slice:
    _slice_parts_adapter.validate_python((value.start, value.stop, value.step))
    return value


type _PositionSlice = Annotated[slice, AfterValidator(_validate_position_slice)]
type _Positions = (
    _PositionSlice | Annotated[tuple[int, ...], Field(min_length=1)] | None
)


@validated_dataclass(frozen=True, config=_SELECTOR_CONFIG)
class FieldSelection:
    """Value constraints with optional selection of distinct-value positions.

    Parameters
    ----------
    value_constraint : int, str, tuple or ValueBounds, optional
        Allowed values. None leaves values unconstrained.
    positions : slice or tuple of int, optional
        Positions in sorted, distinct, non-null values after all value filters.
        Negative positions count from the end. Slice bounds are clipped; list
        positions outside the available sequence select nothing. None or a full
        forward slice leaves positions unconstrained.
        Positions are ignored when the selected values have no common numeric
        ordering; the value constraint still applies.

    Raises
    ------
    pydantic.ValidationError
        If values or positions are invalid, including a zero slice step.
    """

    value_constraint: FieldValues = None
    positions: _Positions = None

    @field_validator("positions")
    @classmethod
    def _normalize_positions(cls, positions: _Positions) -> _Positions:
        if positions is None:
            return None
        if isinstance(positions, slice):
            if (
                positions.start in (None, 0)
                and positions.stop is None
                and positions.step in (None, 1)
            ):
                return None
        else:
            positions = tuple(sorted(set(positions)))
            if len(positions) == 1:
                return parse_slice(str(positions[0]))
        return positions

    def select_values(self, values: Sequence[int]) -> list[int]:
        """Select positions from ordered values.

        Parameters
        ----------
        values : sequence of int
            Sorted, distinct, non-null values surviving all value constraints.

        Returns
        -------
        list of int
            Values at the requested positions. Duplicate positions select the
            same value when the result is used as a membership filter.
        """
        if self.positions is None:
            return list(values)
        if isinstance(self.positions, slice):
            return list(values[self.positions])
        return [
            values[position]
            for position in self.positions
            if -len(values) <= position < len(values)
        ]

    def __str__(self) -> str:
        if self.positions is None:
            return _format_selector(self.value_constraint)
        value_text = _format_selector(self.value_constraint)
        if isinstance(self.positions, tuple):
            position_text = ",".join(map(str, self.positions))
        else:
            start, stop, step = (
                "" if value is None else str(value)
                for value in (
                    self.positions.start,
                    self.positions.stop,
                    self.positions.step,
                )
            )
            position_text = f"{start}:{stop}" + (f":{step}" if step else "")
        return f"{value_text}[{position_text}]"


@validated_dataclass(frozen=True, config=_SELECTOR_CONFIG)
class FieldPattern:
    """A numeric selection between literal parts of a label.

    Parameters
    ----------
    before : str
        Literal text preceding the numeric component; may be empty.
    selection : FieldSelection
        Numeric bounds with optional positions for the numeric component.
    after : str
        Literal text following the numeric component; may be empty.

    Raises
    ------
    pydantic.ValidationError
        If literal parts are invalid or absent, or selection does not use
        numeric bounds.
    """

    before: Annotated[str, Field(pattern=rf"^{_LITERAL_PART_PATTERN.pattern}\z")]
    selection: FieldSelection
    after: Annotated[str, Field(pattern=rf"^{_LITERAL_PART_PATTERN.pattern}\z")]

    def __post_init__(self) -> None:
        if not (self.before or self.after):
            msg = (
                "A field pattern requires valid literal text around its "
                "numeric selection"
            )
            raise ValueError(msg)
        if not isinstance(self.selection.value_constraint, ValueBounds):
            msg = "A field pattern requires numeric bounds"
            # The FieldSelection itself is valid, but not for a numeric slot.
            raise ValueError(msg)  # noqa: TRY004

    def __str__(self) -> str:
        return f"{self.before}{self.selection}{self.after}"


type Selector = FieldValues | FieldSelection | FieldPattern


def _get_field_selection(selector: Selector) -> FieldSelection | None:
    if isinstance(selector, FieldPattern):
        return selector.selection
    return selector if isinstance(selector, FieldSelection) else None


def _parse_atom(value: str) -> FieldValue:
    if value.isascii() and value.isdigit():
        return int(value)
    if _LABEL_PATTERN.fullmatch(value):
        return value
    msg = f"Invalid selector value: {value!r}"
    raise ValueError(msg)


def _parse_values(value: str) -> FieldValues:
    if value in _UNCONSTRAINED:
        return None
    if value.startswith("{") and value.endswith("}"):
        return tuple(_parse_atom(item) for item in value[1:-1].split(","))
    return _parse_atom(value)


def _parse_selector(value: str) -> Selector:
    if pattern := _FIELD_PATTERN.fullmatch(value):
        bounds = ValueBounds(
            int(pattern["lower"]) if pattern["lower"] else None,
            int(pattern["upper"]) if pattern["upper"] else None,
        )
        positions = _parse_positions(pattern["positions"])
        if pattern["before"] or pattern["after"]:
            return FieldPattern(
                pattern["before"], FieldSelection(bounds, positions), pattern["after"]
            )
        return bounds if positions is None else FieldSelection(bounds, positions)
    match = _SELECTION_PATTERN.fullmatch(value)
    if match is None:
        msg = f"Invalid field selector: {value!r}"
        raise ValueError(msg)
    values = _parse_values(match["values"])
    if match["positions"] is None:
        return values
    return FieldSelection(values, _parse_positions(match["positions"]))


def _parse_positions(text: str | None) -> slice | tuple[int, ...] | None:
    if text is None:
        return None
    if "," in text:
        if not _POSITION_LIST_PATTERN.fullmatch(text):
            msg = f"Invalid positional list: {text!r}"
            raise ValueError(msg)
        return tuple(int(item) for item in text.split(","))
    return parse_slice(text)


def _format_selector(value: Selector) -> str:
    if value is None:
        return "*"
    if isinstance(value, tuple):
        return "{" + ",".join(map(str, value)) + "}"
    return str(value)


@dataclass(frozen=True)
class ColumnSelector:
    """A selector bound to a data column.

    Parameters
    ----------
    name : str
        Column name.
    selector : Selector
        Value constraint and optional positional selection.
    label_numbers : mapping of str to int or None, optional
        Declared labels and their numeric values. None marks an unordered label.
        Labels absent from the mapping have no numeric value. Explicit field
        patterns identify their numeric component independently of this mapping.
    """

    name: str
    selector: Selector
    label_numbers: Mapping[str, int | None] | None = None

    @property
    def value_constraint(self) -> FieldValues:
        """The column's value constraint without positional selection."""
        if isinstance(self.selector, FieldPattern):
            return self.selector.selection.value_constraint
        if isinstance(self.selector, FieldSelection):
            return self.selector.value_constraint
        return self.selector

    @property
    def _value_parts(
        self,
    ) -> tuple[tuple[str, ...], int | tuple[int, ...] | ValueBounds | None]:
        """Literal labels and numeric constraints in a value selection."""
        constraint = self.value_constraint
        if isinstance(constraint, str):
            return (constraint,), None
        if isinstance(constraint, tuple):
            labels = tuple(item for item in constraint if isinstance(item, str))
            numbers = tuple(item for item in constraint if isinstance(item, int))
            return labels, numbers or None
        return (), constraint

    def _resolve_numbers(self, values: pd.Index) -> pd.Series:
        """Numeric interpretations indexed by the supplied values."""
        pattern = (
            re.compile(
                re.escape(self.selector.before)
                + r"([0-9]+)"
                + re.escape(self.selector.after)
            )
            if isinstance(self.selector, FieldPattern)
            else None
        )
        label_numbers = self.label_numbers or {}

        def get_number(value):
            if isinstance(value, str):
                if pattern is None:
                    return label_numbers.get(value)
                match = pattern.fullmatch(value)
                return int(match[1]) if match else None
            if (
                pattern is None
                and isinstance(value, Real)
                and not isinstance(value, bool)
            ):
                return value
            return None

        return pd.Series(
            [get_number(value) for value in values], index=values, dtype=object
        )

    def match_values(self, values: pd.Series) -> pd.Series:
        """Match value constraints without applying positional selection.

        Parameters
        ----------
        values : pandas.Series
            Available values in a store or table.

        Returns
        -------
        pandas.Series
            Boolean mask with the original index. Nulls match only an
            unrestricted value selector.
        """
        return _ColumnMatch(self, values).value_mask()


@dataclass(frozen=True)
class _ColumnMatch:
    """Column masks sharing one resolution of distinct numeric labels."""

    selector: ColumnSelector
    values: pd.Series

    @cached_property
    def _is_numeric(self) -> bool:
        return (
            not isinstance(self.selector.selector, FieldPattern)
            and pd.api.types.is_numeric_dtype(self.values.dtype)
            and not pd.api.types.is_bool_dtype(self.values.dtype)
        )

    @cached_property
    def _is_string(self) -> bool:
        return pd.api.types.is_string_dtype(self.values)

    @cached_property
    def _values_without_bools(self) -> pd.Series:
        if pd.api.types.is_object_dtype(self.values.dtype) and not self._is_string:
            # Object-column membership equates True with 1 and False with 0.
            # Exclude booleans without also excluding numeric values or labels.
            booleans = self.values.isin([True, False]) & self.values.astype(
                "string"
            ).isin(["True", "False"])
            return self.values.mask(booleans)
        return self.values

    @cached_property
    def _numbered_values(self) -> pd.Series:
        if not isinstance(self.selector.selector, FieldPattern) and self._is_string:
            values = pd.Index(self.selector.label_numbers or {})
        else:
            values = pd.Index(
                self._values_without_bools.dropna().unique(), dtype=object
            )
        return self.selector._resolve_numbers(values)

    @staticmethod
    def _numeric_mask(
        values: pd.Series, constraint: int | tuple[int, ...] | ValueBounds
    ) -> pd.Series:
        if isinstance(constraint, ValueBounds):
            matched = values.notna()
            if constraint.lower is not None:
                matched &= values >= constraint.lower
            if constraint.upper is not None:
                matched &= values < constraint.upper
            return matched.fillna(value=False)
        if isinstance(constraint, tuple):
            return values.isin(constraint)
        return values.eq(constraint).fillna(value=False)

    def value_mask(self) -> pd.Series:
        values = self.values
        constraint = self.selector.value_constraint
        if constraint is None:
            return pd.Series(data=True, index=values.index, dtype=bool)
        if isinstance(constraint, str):
            return values.eq(constraint).fillna(value=False)
        labels, constraint = self.selector._value_parts
        if constraint is None:
            return values.isin(labels)
        if self._is_numeric:
            return self._numeric_mask(values, constraint)
        numbers = self._numbered_values
        numeric_values = numbers.index[self._numeric_mask(numbers, constraint)]
        return self._values_without_bools.isin([*labels, *numeric_values])

    def position_mask(
        self, mask: npt.NDArray[np.bool_], selection: FieldSelection
    ) -> pd.Series:
        if self._is_numeric:
            surviving = self.values.loc[mask]
            positions = selection.positions
            if isinstance(positions, slice) and positions.step in (None, 1):
                # First/last distinct values need no sorting. Equality excludes
                # null extrema too, when the surviving values are empty or null.
                if positions.start in (None, 0) and positions.stop == 1:
                    return self.values.eq(surviving.min()).fillna(value=False)
                if positions.start == -1 and positions.stop is None:
                    return self.values.eq(surviving.max()).fillna(value=False)
            values_available = np.sort(surviving.dropna().unique())
            return self.values.isin(selection.select_values(values_available))
        numbers = self._numbered_values
        present = numbers.index.isin(
            self._values_without_bools.loc[mask].dropna().unique()
        )
        numbers_available = np.sort(numbers.loc[present].dropna().unique())
        selected = numbers.index[
            numbers.isin(selection.select_values(numbers_available))
        ]
        return self._values_without_bools.isin(selected)


@validated_dataclass(frozen=True, config=_SELECTOR_CONFIG)
class FieldConstraint:
    """Additional value constraints for a selector field.

    Parameters
    ----------
    kind : {"label", "integer", "mixed"}
        Accepted selector values. Integer values are nonnegative. Mixed fields
        also accept literal parts around a numeric selection.

    Raises
    ------
    pydantic.ValidationError
        If kind is missing or is not one of the supported choices.

    Notes
    -----
    Positions are accepted syntactically for all kinds. Selection ignores
    positions when the chosen values have no common numeric ordering.
    """

    kind: Literal["label", "integer", "mixed"]

    def normalize(self, selector: Selector) -> Selector:
        """Validate and normalize a field selector.

        Parameters
        ----------
        selector : Selector
            A literal, set, bounds, positional selection, or None.

        Returns
        -------
        Selector
            Normalized values with no-op positional selection removed.

        Raises
        ------
        ValueError
            If the expression contains labels or numeric selections that the
            field does not accept.
        """
        if isinstance(selector, FieldPattern):
            if self.kind != "mixed":
                msg = "Literal parts around a numeric selection require a mixed field"
                raise ValueError(msg)
            return selector
        values = _field_value_adapters[self.kind].validate_python(
            selector.value_constraint
            if isinstance(selector, FieldSelection)
            else selector
        )
        if isinstance(selector, FieldSelection) and selector.positions is not None:
            return selector
        return values

    def _supports_positions(self, selector: ColumnSelector) -> bool:
        """Whether the selected values have a common numeric ordering."""
        if self.kind == "label":
            return False
        if self.kind == "integer" or isinstance(selector.selector, FieldPattern):
            return True
        constraint = selector.value_constraint
        if isinstance(constraint, (int, ValueBounds)):
            return True
        numbers = selector.label_numbers or {}
        if constraint is None:
            return bool(numbers) and all(
                value is not None for value in numbers.values()
            )
        members = constraint if isinstance(constraint, tuple) else (constraint,)
        return all(
            type(value) is int or numbers.get(value) is not None for value in members
        )


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
