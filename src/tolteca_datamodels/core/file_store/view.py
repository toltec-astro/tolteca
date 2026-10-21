"""Selections over declared file-store locations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from inspect import getattr_static
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Self

from tollan.utils.log import logger

from ..obs_spec import ObsSpecBase

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from tollan.config.types import SubPath

    from .base import FileStoreBase

__all__ = ["StorePaths", "StoreView", "Substore"]


@dataclass(frozen=True)
class _Location:
    chain: tuple[type[FileStoreBase], ...]
    subpath: PurePosixPath


def _locations(
    store: FileStoreBase | type[FileStoreBase],
    chain: tuple[type[FileStoreBase], ...] = (),
) -> Iterator[_Location]:
    for subpath in store._subpaths:
        yield _Location(chain, subpath)
    children = (
        ((child, child) for child in store._substore_paths)
        if isinstance(store, type)
        else store.substores.items()
    )
    for child_cls, child in children:
        yield from _locations(child, (*chain, child_cls))


@dataclass(frozen=True, kw_only=True)
class StoreView:
    """An immutable selection of declared store locations.

    Obtain views through a store's descriptors or `partial`.
    Class descriptors produce unbound selections; instance descriptors bind
    those selections to a store.
    Bound views select from the store's constructed tree. Create a new store
    to include subsequently registered children.
    """

    _root_cls: type[FileStoreBase]
    _root: FileStoreBase | None
    _locations: tuple[_Location, ...]
    _chain: tuple[type[FileStoreBase], ...] = ()

    @classmethod
    def _from_store(
        cls, owner: type[FileStoreBase], store: FileStoreBase | None = None
    ) -> Self:
        return cls(
            _root_cls=owner,
            _root=store,
            _locations=tuple(_locations(owner if store is None else store)),
        )

    @property
    def _store(self) -> FileStoreBase | None:
        store = self._root
        if store is not None:
            for child in self._chain:
                store = store.substores[child]
        return store

    @property
    def _substore_classes(self) -> Iterable[type[FileStoreBase]]:
        store = self._store
        return self.store_cls._substore_paths if store is None else store.substores

    @property
    def store_cls(self) -> type[FileStoreBase]:
        """Concrete store class at this view's branch."""
        return self._chain[-1] if self._chain else self._root_cls

    @property
    def root_path(self) -> Path:
        """Root relative to which this view's glob patterns are expressed."""
        if self._root is None:
            msg = "An unbound selection has no filesystem root"
            raise ValueError(msg)
        return self._root.path

    @property
    def path(self) -> Path:
        """Concrete directory selected by the view, when there is one."""
        root = self.root_path
        directory = self._common_directory()
        if directory is None:
            msg = "This selection does not identify one complete directory"
            raise ValueError(msg)
        return root / directory

    def _relative_path(self, location: _Location) -> PurePosixPath:
        store = self._root
        assert store is not None
        for child in location.chain:
            store = store.substores[child]
        return PurePosixPath(store.path.relative_to(self.root_path)) / location.subpath

    def _common_directory(self) -> PurePosixPath | None:
        assert self._root is not None
        directory = None
        if self._locations:
            paths = [self._relative_path(item) for item in self._locations]
            directory = paths[0]
            while not all(path.is_relative_to(directory) for path in paths):
                directory = directory.parent
            covered = {
                item
                for item in _locations(self._root)
                if self._relative_path(item).is_relative_to(directory)
            }
            if covered != set(self._locations):
                directory = None
        return directory

    def _narrow(self, locations: tuple[_Location, ...]) -> Self:
        return replace(self, _locations=locations)

    def _select_paths(self, paths: tuple[PurePosixPath, ...]) -> Self:
        accepted = set(paths)
        return self._narrow(
            tuple(
                item
                for item in self._locations
                if item.chain == self._chain and item.subpath in accepted
            )
        )

    @property
    def local(self) -> Self:
        """Only the locations managed by this store, excluding substores."""
        return self._select_paths(self.store_cls._subpaths)

    def _substore(self, child: type[FileStoreBase]) -> StoreView:
        if child not in self._substore_classes:
            raise KeyError(child)
        chain = (*self._chain, child)
        locations = tuple(
            item for item in self._locations if item.chain[: len(chain)] == chain
        )
        return StoreView(
            _root_cls=self._root_cls,
            _root=self._root,
            _chain=chain,
            _locations=locations,
        )

    def _selection_locations(self, selection: StoreView | str) -> set[_Location]:
        if isinstance(selection, str):
            descriptor = getattr_static(self.store_cls, selection, None)
            if not isinstance(descriptor, (StorePaths, Substore)):
                msg = f"Unknown store selection: {selection!r}"
                raise ValueError(msg)  # noqa: TRY004 -- invalid selection name
            selection = descriptor._select(self)
        if not isinstance(selection, StoreView):
            msg = "Expected a store selection or declared selection name"
            raise TypeError(msg)
        if not issubclass(self._root_cls, selection._root_cls):
            msg = "Selection belongs to an incompatible store"
            raise ValueError(msg)  # noqa: TRY004 -- valid view, incompatible root
        if selection._root is not None and selection._root is not self._root:
            msg = "Selection is bound to another store instance"
            raise ValueError(msg)
        return set(selection._locations)

    def __getattr__(self, name: str) -> StoreView:
        # Only explicitly registered descriptors are available through views.
        if name.startswith("_"):
            raise AttributeError(name)
        descriptor = getattr_static(self.store_cls, name, None)
        if isinstance(descriptor, (StorePaths, Substore)):
            return descriptor._select(self)
        raise AttributeError(name)

    def partial(
        self,
        *selections: StoreView | str,
        exclude: Iterable[StoreView | str] | None = None,
        store_types: tuple[type[FileStoreBase], ...] | None = None,
    ) -> Self:
        """Select declared locations within the current view.

        Parameters
        ----------
        *selections : StoreView or str
            Compatible selections, or declared descriptor names. Their union
            is kept within this view. Cannot be combined with ``exclude``.
        exclude : iterable of StoreView or str, optional
            Selections to remove instead. An empty iterable removes nothing.
        store_types : tuple of FileStoreBase subclasses, optional
            Keep locations managed by instances of these classes, including
            subclasses. Each store is considered independently; selecting a
            parent does not select its children. An empty tuple selects nothing.
            Provide exactly one of selections, exclude, or store_types.

        Returns
        -------
        StoreView
            A new view; neither this view nor the original store changes.
            Existing restrictions remain in effect.

        Raises
        ------
        ValueError
            If zero or multiple selection forms are supplied, a name is
            undeclared, or a selection belongs to an incompatible root.
        """
        if sum((bool(selections), exclude is not None, store_types is not None)) != 1:
            msg = "Provide positional selections, exclude, or store_types, exactly one"
            raise ValueError(msg)
        if store_types is not None:
            return self._narrow(
                tuple(
                    item
                    for item in self._locations
                    if issubclass(
                        item.chain[-1] if item.chain else self._root_cls, store_types
                    )
                )
            )
        choices = selections if exclude is None else exclude
        selected = set().union(*(self._selection_locations(s) for s in choices))
        return self._narrow(
            tuple(
                item
                for item in self._locations
                if (item in selected) == (exclude is None)
            )
        )

    def iter_store_views(self) -> Iterator[StoreView]:
        """Yield this branch and its registered descendants, in tree order."""
        yield self
        for child in self._substore_classes:
            yield from self._substore(child).iter_store_views()

    def _iter_searches(self, spec: ObsSpecBase | str) -> Iterator[tuple[Path, str]]:
        if self._root is None:
            msg = "Bind the selection to a store before searching"
            raise ValueError(msg)
        if not isinstance(spec, ObsSpecBase):
            spec = self.store_cls._obs_spec_cls.parse(spec)
        for view in self.iter_store_views():
            if not isinstance(spec, view.store_cls._obs_spec_cls):
                continue
            subpaths = tuple(
                item.subpath for item in view._locations if item.chain == view._chain
            )
            if not subpaths:
                continue
            store = view._store
            assert store is not None
            for subpath in subpaths:
                root = store.path / subpath
                for pattern in store._iter_local_glob_patterns(spec, subpath):
                    yield root, pattern

    def iter_glob_patterns(self, spec: ObsSpecBase | str) -> Iterator[str]:
        """Yield candidate globs relative to `root_path`, without filesystem I/O.

        Parameters
        ----------
        spec : ObsSpecBase or str
            Observation constraints. Text uses this branch's selector type.
            Positions and constraints not expressible as globs are deferred.

        Yields
        ------
        str
            Unique relative patterns in declared location order. Structural
            includes and excludes are applied before patterns are generated.
            Parsing and pattern generation occur during iteration.
        """
        seen = set()
        for root, pattern in self._iter_searches(spec):
            relative = (root.relative_to(self.root_path) / pattern).as_posix()
            if relative not in seen:
                seen.add(relative)
                yield relative

    def glob_obs_spec(self, spec: ObsSpecBase | str) -> list[Path]:
        """Collect candidate paths from the selected store locations.

        Parameters
        ----------
        spec : ObsSpecBase or str
            Observation constraints. Apply the complete selector to the
            populated file table afterward, including positional selection.

        Returns
        -------
        list of pathlib.Path
            Unique paths in declaration and glob order. Missing directories
            and unmatched patterns contribute no paths. No headers are read.
        """
        paths = {}
        for root, pattern in self._iter_searches(spec):
            candidates = list(root.glob(pattern))
            logger.debug(
                "ObsSpec {} under {} using {!r}: {} candidates",
                spec,
                root,
                pattern,
                len(candidates),
            )
            paths.update(dict.fromkeys(candidates))
        return list(paths)


class StorePaths:
    """Declare a named selection of managed paths.

    Parameters
    ----------
    paths : tuple of PurePosixPath
        Declared managed paths included by this descriptor.
    """

    def __init__(self, paths: tuple[PurePosixPath, ...]) -> None:
        self._paths = paths

    def _select(self, view: StoreView) -> StoreView:
        return view._select_paths(self._paths)

    def __get__(
        self,
        instance: FileStoreBase | None,
        owner: type[FileStoreBase],
    ) -> StoreView:
        return self._select(StoreView._from_store(owner, instance))


class Substore:
    """Declare a named child store and its selection descriptor.

    Parameters
    ----------
    store_cls : type of FileStoreBase
        Child class to register when the parent class is created.
    subpath : str or SubPath
        Child directory relative to the parent, validated during registration.

    Notes
    -----
    Access through a class returns an unbound selection; access through an
    instance returns a bound selection. Use ``store.substores[ChildStore]``
    for the concrete child instance. Extensions adding children after class
    creation must use ``register_substore()``.
    """

    def __init__(
        self, store_cls: type[FileStoreBase], *, subpath: str | SubPath
    ) -> None:
        self._store_cls = store_cls
        self._subpath = subpath

    def _select(self, view: StoreView) -> StoreView:
        return view._substore(self._store_cls)

    def __get__(
        self,
        instance: FileStoreBase | None,
        owner: type[FileStoreBase],
    ) -> StoreView:
        return self._select(StoreView._from_store(owner, instance))
