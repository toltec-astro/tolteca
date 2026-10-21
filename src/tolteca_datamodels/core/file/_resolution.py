"""Resolve file-table opener and reader choices across data models."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from pydantic import BaseModel, ConfigDict, RootModel, ValidationInfo, model_validator

from ..source import SourceOpener, SourceReader, SourceReaderRules

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd
    from tollan.accessor.pandas.model_table import RecordRuleMapper


def _match_rule_tables[HandlerT, ResultT](
    policies: Mapping[HandlerT, RecordRuleMapper[ResultT]],
    table: pd.DataFrame,
    *,
    column_names: Mapping[str, str] | None,
) -> dict[HandlerT, list[ResultT | None]]:
    """Match each shared policy once, retaining all supporting handler keys."""
    matches = {}
    result = {}
    for handler, policy in policies.items():
        if id(policy) not in matches:
            matches[id(policy)] = policy.map_table(table, column_names=column_names)
        result[handler] = matches[id(policy)]
    return result


def _callable_name(func: Callable) -> str:
    return getattr(func, "__qualname__", type(func).__qualname__)


def _group_candidates[HandlerT, CandidateT](
    choices: Iterable[tuple[HandlerT, CandidateT | None]],
    *,
    key: Callable[[CandidateT], Hashable],
) -> dict[Hashable, tuple[CandidateT, list[HandlerT]]]:
    groups = {}
    for handler, candidate in choices:
        if candidate is not None:
            groups.setdefault(key(candidate), (candidate, []))[1].append(handler)
    return groups


def _resolve_unique[HandlerT, CandidateT](
    groups: Mapping[Hashable, tuple[CandidateT, list[HandlerT]]],
    *,
    label: str,
    describe: Callable[[CandidateT], str],
) -> tuple[CandidateT, tuple[HandlerT, ...]] | None:
    if len(groups) > 1:
        names = ", ".join(describe(candidate) for candidate, _ in groups.values())
        msg = f"Ambiguous {label}: {names}"
        raise RuntimeError(msg)
    if groups:
        candidate, handlers = next(iter(groups.values()))
        return candidate, tuple(handlers)
    return None


class _SourceOpenerResolver[HandlerT: Hashable](BaseModel):
    """Resolve selected data models' policies to one format opener."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    handlers: dict[HandlerT, SourceOpener]

    def match_table(
        self, table: pd.DataFrame, *, column_names: Mapping[str, str] | None = None
    ) -> dict[HandlerT, list[Callable[..., Any] | None]]:
        return _match_rule_tables(self.handlers, table, column_names=column_names)

    def __call__(
        self, choices: Mapping[HandlerT, Callable[..., Any] | None]
    ) -> tuple[Callable[..., Any], tuple[HandlerT, ...]] | None:
        return _resolve_unique(
            _group_candidates(choices.items(), key=id),
            label="source openers",
            describe=_callable_name,
        )


class _IdentifiedSource(NamedTuple):
    """A bound reader and the data-model handlers accepting it."""

    reader: SourceReader
    handlers: tuple[Hashable, ...]


class _ReaderChoices(RootModel[dict[Hashable, type | None]]):
    @model_validator(mode="after")
    def _check_declared_readers(self, info: ValidationInfo) -> _ReaderChoices:
        handlers = cast("Mapping[Hashable, SourceReaderRules]", info.context)
        for key, accessor_cls in self.root.items():
            if key not in handlers:
                msg = f"Unknown reader handler: {key!r}"
                raise ValueError(msg)
            if (
                accessor_cls is not None
                and accessor_cls not in handlers[key].accessor_classes
            ):
                msg = f"Reader choice is not declared by handler {key!r}"
                raise ValueError(msg)
        return self


class _SourceReaderResolver[HandlerT: Hashable](BaseModel):
    """Resolve per-data-model choices to one physical reader.

    Parameters
    ----------
    handlers : mapping
        Data-model handler keys mapped to SourceReaderRules. Keys identify
        the handlers accepting a resolved reader; they have no routing semantics.

    Notes
    -----
    Reuse the resolver across opened objects. Call ``match_table()`` to select
    readers from table values before identifying individual objects. The resolver
    does not open, close, or read metadata from the supplied objects.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    handlers: dict[HandlerT, SourceReaderRules]

    def __bool__(self) -> bool:
        return any(policy.rules for policy in self.handlers.values())

    def match_table(
        self, table: pd.DataFrame, *, column_names: Mapping[str, str] | None = None
    ) -> dict[HandlerT, list[type | None]]:
        """Match reader rules to current table values.

        Parameters
        ----------
        table : pandas.DataFrame
            Current query values, with one source per row.
        column_names : mapping, optional
            Logical rule attributes mapped to table columns. Omit for identical
            names. Missing columns do not satisfy constraints.

        Returns
        -------
        dict
            One ordered list of rule results per handler. None requests content
            identification. Row scope filtering belongs to the caller.
        """
        return _match_rule_tables(self.handlers, table, column_names=column_names)

    def __call__(
        self,
        data_source: object,
        *,
        choices: Mapping[HandlerT, type | None] | None = None,
    ) -> _IdentifiedSource | None:
        """Find a reader for an opened data source.

        Parameters
        ----------
        data_source : object
            Opened payload. Its owner must keep it open while using the reader.
        choices : mapping, optional
            Eligible handlers and their rule results for this row. None values
            request content identification. Omit to inspect content with all
            handlers. An empty mapping accepts nothing.

        Returns
        -------
        _IdentifiedSource or None
            Bound reader and accepting handler keys, or None if
            no handler accepts the payload. Metadata has not yet been read.

        Raises
        ------
        ValueError
            If a choice refers to an unknown handler or a reader not declared
            by that handler, or a selected reader rejects the payload.
        RuntimeError
            If content inspection cannot resolve distinct reader choices.

        Notes
        -----
        A unique class selected by current values binds directly. With no unique
        choice, each eligible policy matches its rules against content-derived
        routing values. A shared class is constructed once. Failed construction
        never falls back to identification.
        Errors propagate. This function does not open or close resources.
        """
        if choices is None:
            choices = dict.fromkeys(self.handlers)
        else:
            choices = cast(
                "dict[HandlerT, type | None]",
                _ReaderChoices.model_validate(choices, context=self.handlers).root,
            )
        selected = _group_candidates(choices.items(), key=id)
        if len(selected) != 1:
            selected = _group_candidates(
                ((key, self.handlers[key].identify(data_source)) for key in choices),
                key=id,
            )
        resolved = _resolve_unique(
            selected,
            label="source readers",
            describe=_callable_name,
        )
        if resolved is not None:
            accessor_cls, handlers = resolved
            return _IdentifiedSource(accessor_cls(data_source), handlers)
        return None
