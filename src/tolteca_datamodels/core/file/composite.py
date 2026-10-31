"""Composite workflows and selections over a shared file table."""

from __future__ import annotations

from contextlib import contextmanager
from functools import cache
from graphlib import TopologicalSorter
from typing import TYPE_CHECKING, ClassVar, Self

from tollan.accessor.pandas.model_table import RecordRuleMapper
from tollan.utils.pandas import query

from ..obs_spec import ObsSpecBase, ObsSpecContext
from ._metadata import _read_source_metadata
from ._open import _open_workflow_sources
from .accessor import FileAccessorBase, _FileWorkflowProxy
from .workflow import (
    FileAccessorWorkflow,
    _apply_derived_column_updates,
    _group_matched_rows_and_record_issues,
    _select_obs_spec,
    _validate_file_table_mode,
    _validate_source_error_policy,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import pandas as pd

    from .trace import SourceErrorPolicyT
    from .workflow import FileTableModeT

__all__ = ["CompositeFileAccessor", "CompositeFileWorkflow", "FileAccessorView"]


class CompositeFileWorkflow(FileAccessorWorkflow):
    """Run selected concrete workflows on disjoint row subsets.

    Parameters
    ----------
    file_accessor : FileAccessorBase
        Accessor over the shared table.
    accessor_types : tuple of FileAccessorBase subclasses, optional
        Fixed selection of concrete or composite accessors. When omitted,
        use the workflow's declared or dynamically supplied components.

    Notes
    -----
    Extensions declare ``_component_accessor_types`` or override
    ``_get_component_accessor_types()`` for dynamically available components.
    """

    _component_accessor_types: ClassVar[tuple[type[FileAccessorBase], ...]] = ()

    def __init__(
        self,
        file_accessor: FileAccessorBase,
        accessor_types: tuple[type[FileAccessorBase], ...] | None = None,
    ) -> None:
        super().__init__(file_accessor)
        self._selected_accessor_types = (
            None
            if accessor_types is None
            else self._expand_component_accessor_types(accessor_types)
        )

    @classmethod
    def _get_component_accessor_types(cls) -> tuple[type[FileAccessorBase], ...]:
        return cls._component_accessor_types

    @staticmethod
    def _expand_component_accessor_types(
        accessor_types: tuple[type[FileAccessorBase], ...],
    ) -> tuple[type[FileAccessorBase], ...]:
        leaves: dict[type[FileAccessorBase], None] = {}

        def visit(component, ancestors):
            workflow_cls = component._workflow_cls
            if not issubclass(workflow_cls, CompositeFileWorkflow):
                leaves[component] = None
                return
            if component in ancestors:
                names = " -> ".join(cls.__name__ for cls in (*ancestors, component))
                msg = f"Cyclic file-accessor composition: {names}"
                raise ValueError(msg)
            for child in workflow_cls._get_component_accessor_types():
                visit(child, (*ancestors, component))

        for component in accessor_types:
            visit(component, ())
        return tuple(leaves)

    @property
    def accessor_types(self) -> tuple[type[FileAccessorBase], ...]:
        """Concrete accessor types available to this workflow."""
        if self._selected_accessor_types is not None:
            return self._selected_accessor_types
        return self._expand_component_accessor_types(
            self._get_component_accessor_types()
        )

    def _concrete_workflows(self) -> tuple[FileAccessorWorkflow, ...]:
        owner = self.file_accessor
        return tuple(
            (owner if type(owner) is cls else cls(self.table))._workflow
            for cls in self.accessor_types
        )

    @staticmethod
    @cache
    def _order_accessor_types(
        accessor_types: tuple[type[FileAccessorBase], ...],
    ) -> tuple[type[FileAccessorBase], ...]:
        predecessors = {cls: [] for cls in accessor_types}
        for cls in predecessors:
            workflow_cls = cls._workflow_cls
            for superseded in workflow_cls._supersedes_workflow_types:
                for other, dependencies in predecessors.items():
                    if other._workflow_cls is superseded:
                        dependencies.append(cls)
        return tuple(TopologicalSorter(predecessors).static_order())

    @staticmethod
    @cache
    def _compile_match_rules(
        accessor_types: tuple[type[FileAccessorBase], ...],
    ) -> RecordRuleMapper[type[FileAccessorBase]]:
        return RecordRuleMapper(
            rules=[
                (constraints, cls)
                for cls in CompositeFileWorkflow._order_accessor_types(accessor_types)
                for constraints in cls._workflow_cls._file_match_rules
            ],
            default=None,
        )

    def _match_workflows(
        self, workflows: tuple[FileAccessorWorkflow, ...]
    ) -> Iterator[tuple[list[int], FileAccessorWorkflow]]:
        if not any(workflow._file_match_rules for workflow in workflows):
            return
        accessor_types = tuple(type(workflow.file_accessor) for workflow in workflows)
        selected = self._compile_match_rules(accessor_types).map_table(self.table)
        groups = _group_matched_rows_and_record_issues(
            self.file_accessor, selected, accessor_types
        )
        by_accessor_type = {
            type(workflow.file_accessor): workflow for workflow in workflows
        }
        for cls, positions in groups.items():
            yield positions, by_accessor_type[cls]

    @staticmethod
    def _workflows_in_precedence_order(
        workflows: tuple[FileAccessorWorkflow, ...],
    ) -> tuple[FileAccessorWorkflow, ...]:
        by_type = {
            type(workflow.file_accessor): workflow
            for workflow in workflows
            if workflow._file_match_rules
        }
        return tuple(
            by_type[cls]
            for cls in CompositeFileWorkflow._order_accessor_types(tuple(by_type))
        )

    def prepare(self, *, mode: FileTableModeT, on_error: SourceErrorPolicyT) -> None:
        """Derive initial columns and read metadata requested by the factory.

        Parameters
        ----------
        mode : {"guess", "guess_or_read", "read"}
            Whether to read no sources, unidentified sources, or matching and
            unidentified sources after deriving filename-based information.
        on_error : {"record", "raise"}
            Source-error handling for metadata reads.
        """
        workflows = self._concrete_workflows()
        if not workflows:
            return
        groups = list(self._match_workflows(workflows))
        defaults = {}
        for workflow in workflows:
            if workflow._file_match_rules:
                defaults.update(workflow._prepare_derived_column_defaults())
        _apply_derived_column_updates(self.table, groups, defaults=defaults)
        if mode != "guess":
            _read_source_metadata(
                self.file_accessor,
                groups,
                candidates=self._workflows_in_precedence_order(workflows),
                unidentified_only=mode == "guess_or_read",
                on_error=_validate_source_error_policy(on_error),
            )

    @contextmanager
    def open(
        self,
        *,
        on_error: SourceErrorPolicyT = "record",
    ) -> Iterator[pd.DataFrame]:
        """Open sources through the selected concrete workflows.

        Parameters
        ----------
        on_error : {"record", "raise"}, optional
            Record source failures by default, or propagate them.

        Yields
        ------
        pandas.DataFrame
            Shared table with live resource and data-accessor columns. Owned
            resources close and temporary columns are removed on exit.
        """
        on_error = _validate_source_error_policy(on_error)
        workflows = self._concrete_workflows()
        with _open_workflow_sources(
            self.file_accessor,
            self._match_workflows(workflows),
            candidates=self._workflows_in_precedence_order(workflows),
            on_error=on_error,
        ):
            yield self.table
        self.file_accessor._log_traces()

    def read_metadata(self, *, on_error: SourceErrorPolicyT = "record") -> None:
        """Read source metadata and update the shared table.

        Each call reads metadata, reusing live readers when available.

        Parameters
        ----------
        on_error : {"record", "raise"}, optional
            Record source failures by default, or propagate them. Projection
            failures always raise before changing destination columns.
        """
        on_error = _validate_source_error_policy(on_error)
        workflows = self._concrete_workflows()
        _read_source_metadata(
            self.file_accessor,
            self._match_workflows(workflows),
            candidates=self._workflows_in_precedence_order(workflows),
            on_error=on_error,
        )
        self.file_accessor._log_traces()

    def query_obs_spec(self, spec: ObsSpecBase | str | int) -> pd.DataFrame:
        """Select rows using the workflows' combined observation vocabulary.

        Parameters
        ----------
        spec : ObsSpecBase, str or int
            Selector accepted by the selected workflows' common selector class.
            When they have no common class, supply a parsed selector.

        Returns
        -------
        pandas.DataFrame
            Matching rows from the full table. Positional constraints are
            evaluated once over the combined vocabulary.

        Raises
        ------
        ValueError
            No common selector layout, or conflicting interface numbering.
        TypeError
            A parsed selector is incompatible with the common selector class.
        """
        workflows = self._concrete_workflows()
        classes = {
            type(w)._obs_spec_cls
            for w in workflows
            if type(w)._obs_spec_cls is not None
        }
        spec_cls = next(iter(classes)) if len(classes) == 1 else None
        context = workflows[0]._obs_spec_context if workflows else ObsSpecContext()
        for workflow in workflows[1:]:
            context = context | workflow._obs_spec_context
        return _select_obs_spec(self.table, spec, spec_cls, context)

    def _get_query_namespace(self) -> dict[str, object]:
        workflows = self._concrete_workflows()
        namespace = {}
        for workflow in workflows:
            for name, value in workflow._query_symbols.items():
                if name in namespace and namespace[name] is not value:
                    msg = f"Conflicting query symbol: {name!r}"
                    raise ValueError(msg)
                namespace[name] = value
        shorthand = dict(workflows[0]._query_namespace) if workflows else {}
        for workflow in workflows[1:]:
            shorthand = {
                name: value
                for name, value in shorthand.items()
                if name in workflow._query_namespace
                and workflow._query_namespace[name] is value
            }
        return shorthand | namespace

    def _iter_matched_accessors(self) -> Iterator[tuple[list[int], FileAccessorBase]]:
        """Yield disjoint matched positions and accessors over the shared table."""
        for positions, workflow in self._match_workflows(self._concrete_workflows()):
            yield positions, workflow.file_accessor


class CompositeFileAccessor[WorkflowT: CompositeFileWorkflow](
    FileAccessorBase[WorkflowT]
):
    """Select capabilities and iterate matched groups in a composite workflow.

    Parameters
    ----------
    data : pandas.DataFrame
        Table to access without copying or modifying it. Define subclasses as
        ``CompositeFileAccessor[YourCompositeWorkflow]``.
    """

    @property
    def view(self) -> FileAccessorView:
        """Snapshot of available capabilities over the same table."""
        return FileAccessorView(self, self._workflow.accessor_types)

    def partial(
        self, *, accessor_types: tuple[type[FileAccessorBase], ...]
    ) -> FileAccessorView:
        """Narrow available capabilities without filtering or modifying rows.

        Parameters
        ----------
        accessor_types : tuple of FileAccessorBase subclasses
            Groups or concrete policies to retain. Concrete selections also
            retain available subclasses. Empty selects no capabilities.

        Returns
        -------
        FileAccessorView
            Restricted view sharing the original table. Repeated narrowing
            cannot enable policies excluded by an earlier selection.
        """
        return self.view.partial(accessor_types=accessor_types)

    def iter_accessors(self) -> Iterator[tuple[list[int], FileAccessorBase]]:
        """Yield disjoint row positions and their concrete accessors.

        Yields
        ------
        positions : list of int
            Positions matched from current values. Groups follow their first
            matching row. Invalid and unmatched rows are excluded; matching
            records issues without modifying row values.
        file_accessor : FileAccessorBase
            Accessor over the full original table. Apply updates only to the
            accompanying positions to leave other rows unchanged.

        Notes
        -----
        Matching runs when the iterator is consumed. Keep row positions stable
        while applying the groups. Iteration does not update derived columns.
        """
        return self._workflow._iter_matched_accessors()


class FileAccessorView(_FileWorkflowProxy[CompositeFileWorkflow]):
    """A fixed selection of workflows over an existing accessor's table.

    Parameters
    ----------
    file_accessor : FileAccessorBase
        Originating accessor, retained without copying its table.
    accessor_types : tuple of FileAccessorBase subclasses
        Concrete or composite capabilities to retain. Construction does not
        process rows; the selection remains fixed if registrations later change.
    """

    def __init__(
        self,
        file_accessor: FileAccessorBase,
        accessor_types: tuple[type[FileAccessorBase], ...],
    ) -> None:
        self.file_accessor = file_accessor
        self._workflow = CompositeFileWorkflow(file_accessor, accessor_types)

    @property
    def table(self) -> pd.DataFrame:
        """Underlying table, including rows outside this selection."""
        return self.file_accessor.table

    @property
    def accessor_types(self) -> tuple[type[FileAccessorBase], ...]:
        """Concrete accessor types retained by this selection."""
        return self._workflow.accessor_types

    @classmethod
    def from_sources(
        cls,
        sources: Iterable[object],
        *,
        accessor_types: tuple[type[FileAccessorBase], ...],
        mode: FileTableModeT = "guess",
        on_error: SourceErrorPolicyT = "record",
    ) -> Self:
        """Build a table using an explicit selection of file policies.

        Parameters
        ----------
        sources : finite iterable
            Resources or source/location dicts; see `FileAccessorBase.from_sources`.
        accessor_types : tuple of FileAccessorBase subclasses
            Concrete policies or composite groups to use.
        mode : {"guess", "guess_or_read", "read"}, optional
            Default "guess" uses filenames. "guess_or_read" reads unidentified
            rows; "read" also reads matching rows. Foreign rows stay unopened.
        on_error : {"record", "raise"}, optional
            Record source failures by default, or propagate them.

        Returns
        -------
        Self
            View over a new table, retaining every input row.
        """
        mode = _validate_file_table_mode(mode)
        file_accessor = FileAccessorBase.from_sources(sources, on_error=on_error)
        view = cls(file_accessor, accessor_types)
        view._workflow.prepare(mode=mode, on_error=on_error)
        file_accessor._log_traces()
        return view

    def partial(self, *, accessor_types: tuple[type[FileAccessorBase], ...]) -> Self:
        """Narrow this view without changing its table or earlier selection.

        Parameters
        ----------
        accessor_types : tuple of FileAccessorBase subclasses
            Groups or concrete policies to retain. Concrete selections also
            retain available subclasses. Empty selects no capabilities.

        Returns
        -------
        Self
            A new view over the same originating accessor. Only currently
            available policies can be retained.
        """
        allowed = CompositeFileWorkflow._expand_component_accessor_types(accessor_types)
        return type(self)(
            self.file_accessor,
            accessor_types=tuple(
                cls for cls in self.accessor_types if issubclass(cls, allowed)
            ),
        )

    def iter_accessors(self) -> Iterator[tuple[list[int], FileAccessorBase]]:
        """Yield disjoint row positions and concrete accessors over this table.

        Yields
        ------
        positions : list of int
            Zero-based positions assigned to one selected policy. Groups follow
            their first matching row. Invalid sources and unmatched rows are
            not yielded.
        accessor : FileAccessorBase
            Concrete policy bound to the full original table. Apply changes
            to ``positions`` to leave other rows unchanged.

        Notes
        -----
        Each operation matches current values and records compatibility issues.
        Keep row positions stable while applying the groups. No derived columns
        are updated by iteration itself.
        """
        return self._workflow._iter_matched_accessors()

    def query(self, expression: str) -> pd.DataFrame:
        """Select rows using query symbols from the selected policies.

        Parameters
        ----------
        expression : str
            Trusted pandas expression. Columns take precedence over query
            symbols. Shorthand is available only when all selected policies
            define the same object under that name.

        Returns
        -------
        pandas.DataFrame
            Selected rows, retaining their index labels and order. The query
            applies to all rows, not just rows matched by the selected policies.

        Raises
        ------
        ValueError
            Selected policies export different objects under the same name.
        """
        return query(
            self.table, expression, namespace=self._workflow._get_query_namespace()
        )
