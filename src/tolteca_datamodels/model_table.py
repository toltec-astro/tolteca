"""Attribute projection and column-wise selection on pandas DataFrames."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import MISSING
from datetime import datetime
from enum import Enum, Flag
from functools import partial
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Literal,
    NamedTuple,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
    overload,
)

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.time.formats import TimeNumeric
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from tollan.config.types.quantity import get_physical_type_from_quantity_type
from tollan.utils.py import rgetattr
from tollan.utils.typing import get_literal_values

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "ColumnTypeRule",
    "ModelColumn",
    "ModelColumnMapper",
    "ModelTableAccessor",
    "QuantityColumnRule",
    "ScalarColumnRule",
    "SkyCoordColumnRule",
    "TimeColumnRule",
]


class ModelColumn(NamedTuple):
    """A table column read from a dotted model attribute.

    Parameters
    ----------
    attr : str
        Attribute path to read, with dots separating nested attributes.
    default : object, optional
        Value used when the attribute is missing. If omitted, a missing
        attribute raises AttributeError.
    dtype : pandas dtype, optional
        Output column dtype. Default is object dtype.
    converter : callable, optional
        Function applied to non-None values, including a non-None default.
        None is preserved. If omitted, values are used unchanged.
    """

    attr: str
    default: Any = MISSING
    dtype: Any = None
    converter: Callable[[Any], Any] | None = None

    def read(self, model: object) -> Any:
        """Read and convert a column value.

        Parameters
        ----------
        model : object
            Object from which to read the configured attribute path.

        Returns
        -------
        object
            Converted attribute value, or the configured fallback.

        Raises
        ------
        AttributeError
            If the attribute is missing and no default is configured.
        """
        value = (
            rgetattr(model, self.attr)
            if self.default is MISSING
            else rgetattr(model, self.attr, self.default)
        )
        if value is None or self.converter is None:
            return value
        return self.converter(value)


def _annotation_types(
    annotation: Any, aliases: frozenset[TypeAliasType] = frozenset()
) -> set[Any]:
    """Collect value types from annotations, aliases, literals and unions."""
    if isinstance(annotation, TypeAliasType):
        if annotation in aliases:
            return {Any}
        return _annotation_types(annotation.__value__, aliases | {annotation})
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        return _annotation_types(args[0], aliases)
    if origin in (Union, UnionType):
        return set().union(*(_annotation_types(arg, aliases) for arg in args))
    if origin is Literal:
        return {type(value) for value in get_literal_values(annotation)}
    return {annotation}


def _enum_value(value):
    return value.value if isinstance(value, Enum) else value


def _quantity_value(value, *, unit):
    if not value.isscalar:
        msg = "A quantity column requires scalar values"
        raise ValueError(msg)
    return value.to_value(unit)


def _time_value(value, *, format):
    if not value.isscalar:
        msg = "A time column requires scalar values"
        raise ValueError(msg)
    return value.to_value(format)


def _coordinate_value(value, *, attr, unit):
    return _quantity_value(getattr(value, attr), unit=unit)


class ColumnTypeRule(BaseModel, ABC):
    """Base class for rules that declare columns from field annotations."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", frozen=True, validate_default=True
    )

    @abstractmethod
    def __call__(
        self, name: str, annotation: Any, column_name: str
    ) -> dict[str, ModelColumn] | None:
        """Declare columns for a matching annotation.

        Parameters
        ----------
        name : str
            Model attribute name.
        annotation : object
            Field's type annotation, including any Annotated metadata.
        column_name : str
            Column name supplied by a name rule.

        Returns
        -------
        dict of str to ModelColumn or None
            Column declarations, or None if the annotation does not match.
            An empty dict matches without producing columns. No model values
            are read.
        """


class TimeColumnRule(ColumnTypeRule):
    """Project scalar Time values in the selected format, preserving time scale.

    Parameters
    ----------
    format : str
        Astropy Time output format. Appended to the column name as a suffix,
        for example ``_mjd``.
    """

    format: str

    @field_validator("format")
    @classmethod
    def _validate_format(cls, value):
        if value not in Time.FORMATS:
            msg = f"Unknown time format {value!r}"
            raise ValueError(msg)
        return value

    def __call__(self, name, annotation, column_name):
        if _annotation_types(annotation) - {type(None)} != {Time}:
            return None
        dtype = (
            "Float64" if issubclass(Time.FORMATS[self.format], TimeNumeric) else None
        )
        return {
            f"{column_name}_{self.format}": ModelColumn(
                name, dtype=dtype, converter=partial(_time_value, format=self.format)
            )
        }


class SkyCoordColumnRule(ColumnTypeRule):
    """Project selected angular coordinate attributes without changing frame.

    Parameters
    ----------
    attrs : sequence of str
        Distinct angular attributes to read, for example ``["ra", "dec"]``.
        At least one is required. A missing attribute raises AttributeError
        when reading a coordinate.
    unit : str or astropy.units.UnitBase
        Angular output unit. Column names gain attribute and unit suffixes;
        ``attrs=["l", "b"], unit="deg"`` produces ``_l_deg`` and ``_b_deg``.
    """

    attrs: tuple[Annotated[str, Field(min_length=1)], ...] = Field(min_length=1)
    unit: u.UnitBase

    @field_validator("attrs")
    @classmethod
    def _validate_attrs(cls, value):
        if len(value) != len(set(value)):
            msg = "Coordinate attributes must be unique"
            raise ValueError(msg)
        return value

    @field_validator("unit", mode="before")
    @classmethod
    def _parse_unit(cls, value):
        unit = u.Unit(value)
        if unit.physical_type != u.get_physical_type("angle"):
            msg = f"Coordinate unit {unit!s} is not an angle"
            raise ValueError(msg)
        return unit

    def __call__(self, name, annotation, column_name):
        if _annotation_types(annotation) - {type(None)} != {SkyCoord}:
            return None
        return {
            f"{column_name}_{attr}_{self.unit}": ModelColumn(
                name,
                dtype="Float64",
                converter=partial(_coordinate_value, attr=attr, unit=self.unit),
            )
            for attr in self.attrs
        }


class QuantityColumnRule(ColumnTypeRule):
    """Project quantities annotated with one physical type into a selected unit.

    Parameters
    ----------
    physical_type : str or astropy.units.PhysicalType
        Physical type required by the quantity annotation. Unconstrained
        quantities and annotations allowing several physical types do not match.
    unit : str or astropy.units.UnitBase
        Compatible output unit. Appended to the column name as a suffix,
        for example ``_s`` or ``_Hz``.
    """

    physical_type: u.PhysicalType
    unit: u.UnitBase

    @field_validator("physical_type", mode="before")
    @classmethod
    def _parse_physical_type(cls, value):
        return u.get_physical_type(value)

    @field_validator("unit", mode="before")
    @classmethod
    def _parse_unit(cls, value):
        return u.Unit(value)

    @model_validator(mode="after")
    def _validate_unit(self):
        if self.unit.physical_type != self.physical_type:
            msg = (
                f"Unit {self.unit!s} does not have physical type {self.physical_type!s}"
            )
            raise ValueError(msg)
        return self

    def __call__(self, name, annotation, column_name):
        if _annotation_types(annotation) - {type(None)} != {u.Quantity}:
            return None
        if get_physical_type_from_quantity_type(annotation) != self.physical_type:
            return None
        return {
            f"{column_name}_{self.unit}": ModelColumn(
                name,
                dtype="Float64",
                converter=partial(_quantity_value, unit=self.unit),
            )
        }


class ScalarColumnRule(ColumnTypeRule):
    """Project primitive scalars and enum values without changing column names.

    Nullable homogeneous types use their pandas extension dtype. Mixed scalar
    unions use object dtype; containers and nested models do not match.
    """

    def __call__(self, name, annotation, column_name):
        types = _annotation_types(annotation) - {type(None)}
        enums = {
            item for item in types if isinstance(item, type) and issubclass(item, Enum)
        }
        value_types = (types - enums) | {
            type(member.value) for enum in enums for member in enum.__members__.values()
        }
        dtypes = {bool: "boolean", int: "Int64", float: "Float64", str: "string"}
        if value_types == {AwareDatetime}:
            dtype = "datetime64[ns, UTC]"
        elif value_types <= {bool, int, float, str, bytes, datetime}:
            dtype = (
                dtypes.get(next(iter(value_types))) if len(value_types) == 1 else None
            )
        else:
            return None
        return {
            column_name: ModelColumn(
                name, dtype=dtype, converter=_enum_value if enums else None
            )
        }


class ModelColumnMapper(BaseModel):
    r"""Declare table columns using ordered name and type rules.

    Parameters
    ----------
    name_rules : sequence of (regex, replacement) pairs
        Full-match patterns for model field names and their column-name
        replacements. All matching rules apply; duplicate aliases are removed
        in declaration order. Replacements accept regular-expression
        backreferences, such as ``r"file_\g<0>"``. Unmatched fields are skipped.
    type_rules : sequence of callable
        Rules accepting ``(name, annotation, column_name)``; see `ColumnTypeRule`.
        For each alias, the first matching rule supplies its final columns.
        Repeated callables and identically configured rule models are rejected.
        An empty name-rule or type-rule list produces no columns.
    exclude : str or compiled regex, optional
        Full-match pattern for field names to omit, before applying name rules.
        Default is no exclusion.
    include_computed : bool, optional
        Whether to include Pydantic computed fields. Default is False.
    model_path : str, optional
        Dotted attribute path locating the model in an input object, for example
        ``"metadata"``. Does not affect field-name matching. Default is None,
        meaning attributes are read directly from the input object.
    default : object, optional
        Fallback for missing attributes in the generated columns. If omitted,
        missing attributes raise AttributeError when reading values.

    Examples
    --------
    >>> class Reading(BaseModel):
    ...     value: int
    >>> columns = ModelColumnMapper(
    ...     name_rules=[(r".+", r"sample_\g<0>")],
    ...     type_rules=[ScalarColumnRule()],
    ... )(Reading)
    >>> columns["sample_value"].read(Reading(value=3))
    3
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", frozen=True, validate_default=True
    )

    name_rules: tuple[tuple[re.Pattern[str], str], ...]
    type_rules: tuple[Callable[[str, Any, str], dict[str, ModelColumn] | None], ...]
    exclude: re.Pattern[str] | None = None
    include_computed: bool = False
    model_path: Annotated[str, Field(pattern=r"^[^.]+(?:\.[^.]+)*$")] | None = None
    default: Any = Field(default_factory=lambda: MISSING)

    @field_validator("name_rules")
    @classmethod
    def _validate_name_rules(cls, value):
        for pattern, replacement in value:
            try:
                # re.sub validates backreferences even without a matching name.
                pattern.sub(replacement, "")
            except (re.error, IndexError) as error:
                raise ValueError(str(error)) from error
        return value

    @field_validator("type_rules")
    @classmethod
    def _validate_type_rules(cls, value):
        for index, rule in enumerate(value):
            for previous_index, previous in enumerate(value[:index]):
                if rule is previous or (
                    isinstance(rule, ColumnTypeRule)
                    and isinstance(previous, ColumnTypeRule)
                    and rule == previous
                ):
                    msg = (
                        f"Duplicate type rules at indices {previous_index} and {index}"
                    )
                    raise ValueError(msg)
        return value

    def _map_name(self, name: str) -> tuple[str, ...]:
        if self.exclude is not None and self.exclude.fullmatch(name) is not None:
            return ()
        return tuple(
            dict.fromkeys(
                match.expand(replacement)
                for pattern, replacement in self.name_rules
                if (match := pattern.fullmatch(name)) is not None
            )
        )

    def _map_type(
        self, name: str, annotation: Any, column_name: str
    ) -> dict[str, ModelColumn] | None:
        for rule in self.type_rules:
            mapped = rule(name, annotation, column_name)
            if mapped is not None:
                return mapped
        return None

    def __call__(
        self,
        model_cls: type[BaseModel],
        **updates: Any,
    ) -> dict[str, ModelColumn]:
        """Create columns without evaluating model values.

        Parameters
        ----------
        model_cls : type of pydantic.BaseModel
            Model whose field annotations determine the columns. Serialization
            aliases and exclusions do not affect projection.
        **updates
            Validated replacements for mapper settings, for this call only.
            This mapper is not modified.

        Returns
        -------
        dict of str to ModelColumn
            Column declarations. Explicit entries can be added or overridden
            with the dict ``|`` operator.

        Raises
        ------
        ValueError
            If multiple projections produce the same final column name.
        """
        if updates:
            # model_copy(update=...) does not validate replacement settings.
            # dict(self) also preserves an arbitrary default object as-is.
            mapper = type(self).model_validate(dict(self) | updates)
            return mapper(model_cls)

        # Pydantic separates outer Annotated metadata from the field annotation.
        annotations = {
            name: Annotated[info.annotation, *info.metadata]
            if info.metadata
            else info.annotation
            for name, info in model_cls.model_fields.items()
        }
        if self.include_computed:
            annotations |= {
                name: info.return_type
                for name, info in model_cls.model_computed_fields.items()
            }
        columns = {}
        for name, annotation in annotations.items():
            for column_name_base in self._map_name(name):
                mapped = self._map_type(name, annotation, column_name_base)
                if mapped is None:
                    continue
                for column_name, column in mapped.items():
                    if column_name in columns:
                        msg = (
                            f"Column name {column_name!r} is produced by "
                            "multiple projections"
                        )
                        raise ValueError(msg)
                    columns[column_name] = column._replace(
                        attr=column.attr
                        if self.model_path is None
                        else f"{self.model_path}.{column.attr}",
                        default=self.default,
                    )
        return columns


def _bits_any_set(values: pd.Series, mask: int | Flag) -> pd.Series:
    """Test whether any bits in the mask are set in each value."""
    mask = mask.value if isinstance(mask, Flag) else mask
    return (values & mask) != 0


def _bits_all_set(values: pd.Series, mask: int | Flag) -> pd.Series:
    """Test whether all bits in the mask are set in each value."""
    mask = mask.value if isinstance(mask, Flag) else mask
    return (values & mask) == mask


class ModelTableAccessor:
    """Project objects into flat tables for selection and enrichment.

    Parameters
    ----------
    data : pandas.DataFrame
        Table to access. Construction neither copies nor validates the frame.

    Examples
    --------
    >>> class Measurement(BaseModel):
    ...     channel: int
    ...     amplitude: float
    >>> table = ModelTableAccessor.from_models([
    ...     Measurement(channel=1, amplitude=2.5),
    ...     Measurement(channel=2, amplitude=0.5),
    ... ], columns={
    ...     "channel": ModelColumn("channel", dtype="Int64"),
    ...     "amplitude": ModelColumn("amplitude", dtype="Float64"),
    ... })
    >>> selected = ModelTableAccessor(table).query("amplitude > 1")
    >>> next(selected.itertuples(index=False)).channel
    np.int64(1)
    """

    _query_namespace: ClassVar[dict[str, object]] = {
        "bits_any_set": _bits_any_set,
        "bits_all_set": _bits_all_set,
    }

    def __init__(self, data: pd.DataFrame) -> None:
        self._obj = data

    @classmethod
    def from_models(
        cls,
        models: Iterable[object],
        *,
        columns: Mapping[str, ModelColumn],
    ) -> pd.DataFrame:
        """Build a flat table from declared attribute paths.

        Parameters
        ----------
        models : iterable of object
            Objects to project, one per output row. They need not share a type;
            only the declared attribute paths are read.
        columns : mapping of str to ModelColumn
            Output column names and their attribute paths, defaults, converters
            and dtypes.

        Returns
        -------
        pandas.DataFrame
            Projected values in input order. Empty input retains the declared
            columns and dtypes.

        Raises
        ------
        AttributeError
            If a required attribute is missing and no fallback is declared.
        """
        readers = tuple(column.read for column in columns.values())
        rows = ([read(model) for read in readers] for model in models)
        # Preserve native values and large nullable integers before dtype conversion.
        table = pd.DataFrame(rows, columns=list(columns), dtype=object)
        dtypes = {
            name: column.dtype
            for name, column in columns.items()
            if column.dtype is not None
        }
        return table.astype(dtypes) if dtypes else table

    @overload
    def update_from_models(
        self,
        models: Iterable[object] | Mapping[int, object],
        *,
        columns: Mapping[str, ModelColumn],
        inplace: Literal[False] = False,
    ) -> pd.DataFrame: ...

    @overload
    def update_from_models(
        self,
        models: Iterable[object] | Mapping[int, object],
        *,
        columns: Mapping[str, ModelColumn],
        inplace: Literal[True],
    ) -> None: ...

    @overload
    def update_from_models(
        self,
        models: Iterable[object] | Mapping[int, object],
        *,
        columns: Mapping[str, ModelColumn],
        inplace: bool,
    ) -> pd.DataFrame | None: ...

    def update_from_models(
        self,
        models: Iterable[object] | Mapping[int, object],
        *,
        columns: Mapping[str, ModelColumn],
        inplace: bool = False,
    ) -> pd.DataFrame | None:
        """Project objects into existing table rows.

        Parameters
        ----------
        models : iterable of object or mapping of int to object
            An iterable with exactly one object per table row, or a mapping
            from integer row positions to objects. Positions follow ``iloc``
            conventions, including negative positions. Slices and broadcasting
            are not supported.
        columns : mapping of str to ModelColumn
            Columns to update and their projections. Null values overwrite
            existing values. Existing column dtypes are retained; new columns
            use the declared dtypes, which must support missing values.
        inplace : bool, optional
            Whether to update this table. Default is False.

        Returns
        -------
        pandas.DataFrame or None
            Updated table, or None if ``inplace=True``. Other rows and columns,
            the index and table attributes are preserved.

        Raises
        ------
        ValueError
            If an iterable has the wrong length, or mapping keys identify the
            same row after resolving negative positions.
        TypeError
            If a mapping key is not an integer row position.
        IndexError
            If a row position is out of bounds.

        Notes
        -----
        Duplicate index labels are supported because updates use row positions.
        In-place updates are not transactional: an assignment error may leave
        earlier columns updated.
        """
        row_range = range(len(self._obj))
        if isinstance(models, Mapping):
            models_resolved = {}
            for position, model in models.items():
                if not pd.api.types.is_integer(position):
                    msg = "Update mapping keys must be integer row positions"
                    raise TypeError(msg)
                models_resolved[row_range[position]] = model
            if len(models_resolved) != len(models):
                msg = "Updated row positions must identify distinct rows"
                raise ValueError(msg)
            if not models_resolved:
                return None if inplace else self._obj.copy()
            rows = list(models_resolved)
            models = models_resolved.values()
        else:
            models = list(models)
            if len(models) != len(row_range):
                msg = (
                    "Provide one object per table row, or a position-to-object mapping"
                )
                raise ValueError(msg)
            rows = list(row_range)
        projected = self.from_models(models, columns=columns)
        result = self._obj if inplace else self._obj.copy()
        for name in projected:
            if name not in result:
                result[name] = pd.Series(
                    None, index=result.index, dtype=projected[name].dtype
                )
        cols = result.columns.get_indexer(projected.columns)
        result.iloc[rows, cols] = projected
        return None if inplace else result

    def query(self, expression: str) -> pd.DataFrame:
        """Select rows using pandas expressions on the table columns.

        Parameters
        ----------
        expression : str
            Pandas query expression evaluated with the Python engine.
            Only use trusted expressions. Column names take precedence over
            query functions and constants.

        Returns
        -------
        pandas.DataFrame
            Selected rows, preserving their index labels.

        Notes
        -----
        Comparisons, membership and nulls follow pandas query semantics.
        ``bits_any_set(column, mask)`` tests for any matching bit;
        ``bits_all_set(column, mask)`` requires every bit in the mask. Masks
        accept integers or enum flags. Use integer columns; nullable integer
        columns preserve nulls in the Boolean result. With a zero mask, any is
        false and all is true for non-null values.

        """
        # Pandas mutates resolvers; protect the declaration and let columns win.
        namespace = {
            name: value
            for name, value in self._query_namespace.items()
            if name not in self._obj
        }
        return self._obj.query(expression, engine="python", resolvers=[namespace])
