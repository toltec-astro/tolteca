"""Validated capabilities of selected file accessors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, cast

import pandas as pd
from pydantic import AfterValidator, BaseModel, ConfigDict, PrivateAttr, model_validator

from ..obs_spec import ObsSpecBase, ObsSpecContext
from ._resolution import _SourceOpenerResolver, _SourceReaderResolver

if TYPE_CHECKING:
    from . import (
        FileAccessor,  # noqa: TC004 -- PrivateAttr annotations are not validated.
    )


def _concrete_accessor_types[AccessorT](
    accessor_types: tuple[type[AccessorT], ...],
) -> tuple[type[AccessorT], ...]:
    if any(not hasattr(item, "data_model_cls") for item in accessor_types):
        msg = "Expected concrete FileAccessor classes with data-model bindings"
        raise ValueError(msg)
    return tuple(dict.fromkeys(accessor_types))


type _AccessorTypes[AccessorT] = Annotated[
    tuple[type[AccessorT], ...], AfterValidator(_concrete_accessor_types)
]


class _FileCapabilities[AccessorT](BaseModel):
    """Capabilities supplied by a collection of file accessors."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    accessor_types: _AccessorTypes[AccessorT]
    default_accessor: type[AccessorT]

    _data_models: frozenset[str] = PrivateAttr()
    _opener_resolver: _SourceOpenerResolver[type[FileAccessor]] = PrivateAttr()
    _reader_resolver: _SourceReaderResolver[type[FileAccessor]] = PrivateAttr()
    _obs_spec_cls: type[ObsSpecBase] | None = PrivateAttr()
    _obs_spec_context: ObsSpecContext = PrivateAttr()
    _query_namespace: dict[str, object] = PrivateAttr()
    _data_kind_accessors: dict[str, type[FileAccessor]] = PrivateAttr()

    @model_validator(mode="after")
    def _configure(self) -> _FileCapabilities[AccessorT]:
        """Combine shared capabilities without widening concrete accessors."""
        classes = cast("tuple[type[FileAccessor], ...]", self.accessor_types)
        spec_classes = {
            cls._obs_spec_cls for cls in classes if cls._obs_spec_cls is not None
        }
        if len(spec_classes) > 1:
            msg = "Registered accessors have incompatible observation selectors"
            raise ValueError(msg)
        obs_spec_cls = next(iter(spec_classes), None)
        namespace: dict[str, object] = {}
        context = ObsSpecContext()
        data_kind_accessors = {}
        for cls in classes:
            context = context | cls._obs_spec_context
            kind_type = cls._data_kind_type
            if kind_type is not None:
                if (
                    kind_type.__name__ in namespace
                    and namespace[kind_type.__name__] is not kind_type
                ):
                    msg = f"Conflicting flag class name: {kind_type.__name__}"
                    raise ValueError(msg)
                namespace[kind_type.__name__] = kind_type
                name = cls._data_kind_column
                if name in data_kind_accessors:
                    previous = data_kind_accessors[name]
                    if (
                        previous._data_kind_type is not kind_type
                        or previous._data_kind_mapper is not cls._data_kind_mapper
                    ):
                        msg = f"Conflicting classification column: {name!r}"
                        raise ValueError(msg)
                data_kind_accessors[name] = cls
        self._opener_resolver = _SourceOpenerResolver(
            handlers={
                cls: cls._source_opener
                for cls in classes
                or (cast("type[FileAccessor]", self.default_accessor),)
            }
        )
        self._data_models = frozenset(
            name for cls in classes for name in cls._data_models
        )
        self._reader_resolver = _SourceReaderResolver(
            handlers={cls: cls._reader_rules for cls in classes}
        )
        self._obs_spec_cls = obs_spec_cls
        self._query_namespace = namespace
        self._obs_spec_context = context
        self._data_kind_accessors = data_kind_accessors
        return self

    def _guess_derived_columns(
        self, table: pd.DataFrame, groups: dict[frozenset[str], list[int]]
    ) -> pd.DataFrame:
        """Collect scoped computations without modifying the input table."""
        result = pd.DataFrame(index=table.index)
        for cls in cast("tuple[type[FileAccessor], ...]", self.accessor_types):
            positions = sorted(
                position
                for data_models, rows in groups.items()
                if not cls._data_models.isdisjoint(data_models)
                for position in rows
            )
            scoped = table if len(positions) == len(table) else table.iloc[positions]
            computed = cls._guess_derived_columns(scoped)
            if not computed.index.equals(scoped.index):
                msg = "Computed columns must preserve the selected index and row order"
                raise ValueError(msg)
            for name, values in computed.items():
                if name not in result:
                    result[name] = (
                        table[name].array.copy()
                        if name in table
                        else pd.Series(index=table.index, dtype=values.dtype).array
                    )
                column_position = cast("int", result.columns.get_loc(name))
                result.iloc[positions, column_position] = values.array
        return result
