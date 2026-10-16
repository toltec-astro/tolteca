"""Reader declarations and identification of opened objects."""

from __future__ import annotations

from collections.abc import (
    Callable,  # noqa: TC003 -- Pydantic resolves this field at runtime.
)
from functools import cached_property
from typing import Protocol

from tollan.accessor.pandas.model_table import RecordRuleMapper

__all__ = [
    "SourceReader",
    "SourceReaderRules",
]


class SourceReader(Protocol):
    """An identified data source with optional metadata."""

    @property
    def meta(self) -> object:
        """Source metadata, or None when unavailable."""
        ...


class SourceReaderRules(RecordRuleMapper[type]):
    """Reader selection within one data model.

    Parameters
    ----------
    rules : iterable of (mapping, type)
        Ordered pairs of field constraints and accessor classes. The first
        matching rule selects a class. Use ``Contains`` for collection fields
        such as ``data_models``. The selected class is constructed with the
        opened object; its ``meta`` property supplies metadata, or None when
        unavailable.
    prepare_record : callable, optional
        Extract routing values from an opened object, or return None if
        unsupported. Do not open or close the object, construct an accessor,
        read full metadata, or load signal arrays. Omit if only existing records
        will be matched.

    Notes
    -----
    Call the policy for one record or use ``map_table`` for a table. Both return
    accessor classes without accessing payloads. Use ``identify`` to match the
    same rules against routing values extracted from an opened object.
    """

    prepare_record: Callable[[object], object | None] | None = None
    default: None = None

    @cached_property
    def accessor_classes(self) -> frozenset[type]:
        """Accessor classes selectable by these rules."""
        return frozenset(cls for _, cls in self.rules)

    def identify(self, data_source: object) -> type | None:
        """Select an accessor class from an opened object's routing values.

        Parameters
        ----------
        data_source : object
            Opened payload, owned by the caller.

        Returns
        -------
        type or None
            First matching accessor class, or None if unsupported or no
            ``prepare_record`` callback is configured. No accessor is created.

        Notes
        -----
        Errors extracting routing values propagate to the caller.
        """
        if self.prepare_record is None:
            return None
        record = self.prepare_record(data_source)
        return None if record is None else self(record)
