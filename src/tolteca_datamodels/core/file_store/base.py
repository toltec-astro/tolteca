"""Directory-backed file discovery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from inspect import getattr_static
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from pydantic import ConfigDict, TypeAdapter
from tollan.config.types import SubPath, SubPathOrRoot
from tollan.utils.registry import Registry
from tollan.utils.typing import (
    GenericBindingRequirement,
    resolve_concrete_generic_bindings,
)

from ..obs_spec import ObsSpecBase
from .view import StoreView, Substore

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = ["FileStoreBase", "SubstoreRegistryT"]

type SubstoreRegistryT = Mapping[type[FileStoreBase], SubPath]


class FileStoreBase[SpecT: ObsSpecBase](ABC):
    """Base class for discovering files in a directory tree.

    Parameters
    ----------
    path : str or pathlib.Path
        Root directory to search. The directory need not exist.

    Notes
    -----
    To define a store, subclass ``FileStoreBase[YourObsSpec]`` and implement
    `_iter_local_glob_patterns`. Declare directly managed locations in
    ``_subpaths`` as a tuple of SubPathOrRoot values; use
    ``PurePosixPath(".")`` to include the root itself. The default is empty.
    These paths are searched using the patterns supplied by your implementation,
    not recursively by default. Use `StorePaths` to give selections public names.

    Declare named children with
    ``child = Substore(ChildStore, subpath="child")``. For children without
    public names, use ``_substore_paths``, a mapping of child classes to SubPath
    values. Both path declarations are inherited unless replaced. Adding a
    substore to a subclass leaves its parents and siblings unchanged.

    Local paths are searched in declaration order, followed by substores in
    registration order. Register children before constructing a store; later
    registrations affect new instances only. Construction creates the child
    instances without searching directories or opening files.
    """

    _substore_paths: ClassVar[SubstoreRegistryT] = Registry()
    _subpaths: ClassVar[tuple[SubPathOrRoot, ...]] = ()
    _obs_spec_cls: ClassVar

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        substore_paths = cls._substore_paths
        cls._substore_paths = Registry()
        bindings = resolve_concrete_generic_bindings(
            cls,
            FileStoreBase,
            SpecT=GenericBindingRequirement("_obs_spec_cls", ObsSpecBase),
        )
        if bindings is not None:
            cls._obs_spec_cls = bindings["SpecT"]
        for store_cls, subpath in substore_paths.items():
            cls.register_substore(store_cls, subpath)
        for name, descriptor in tuple(vars(cls).items()):
            if isinstance(descriptor, Substore):
                cls.register_substore(
                    descriptor._store_cls, descriptor._subpath, name=name
                )

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._substores = _Substores(
            {
                store_cls: store_cls(self.path / subpath)
                for store_cls, subpath in self._substore_paths.items()
            }
        )

    @property
    def path(self) -> Path:
        """Read-only root path of the store."""
        return self._path

    @classmethod
    def register_substore(
        cls,
        store_cls: type[FileStoreBase],
        subpath: str | SubPath,
        *,
        name: str | None = None,
    ) -> None:
        """Register a store beneath a relative subpath.

        Parameters
        ----------
        store_cls : type of FileStoreBase
            Store constructed with the subpath as its root.
        subpath : str or SubPath
            Child directory relative to this store's root; see `SubPath`.
        name : str, optional
            Public name for selecting this child. Omit to access the child
            only through `substores`. Existing attributes cannot be replaced.

        Raises
        ------
        pydantic.ValidationError
            If subpath or store_cls is invalid.
        ValueError
            If the class is already registered, even at the same subpath;
            the registration would introduce a cycle; or name is invalid or
            already used.

        Notes
        -----
        New registrations apply to subsequently constructed stores only.
        """
        ((store_cls, subpath),) = _substore_adapter.validate_python(
            {store_cls: subpath}
        ).items()
        pending: list[type[FileStoreBase]] = [store_cls]
        seen = set()
        while pending:
            current = pending.pop()
            if current is cls:
                msg = f"Substore registration would create a cycle: {subpath!r}"
                raise ValueError(msg)
            if current not in seen:
                seen.add(current)
                pending.extend(current._substore_paths)
        if name is not None:
            existing = getattr_static(cls, name, None)
            if (
                not name.isidentifier()
                or name.startswith("_")
                or (
                    existing is not None
                    and not (
                        isinstance(existing, Substore)
                        and existing._store_cls is store_cls
                    )
                )
            ):
                msg = f"Invalid or occupied substore descriptor name: {name!r}"
                raise ValueError(msg)
        # Class creation replaces declarations with a per-class registry.
        registry = cast("Registry[type[FileStoreBase], SubPath]", cls._substore_paths)
        registry.register(store_cls, subpath)
        if name is not None:
            setattr(cls, name, Substore(store_cls, subpath=subpath))

    @property
    def substores(self) -> _Substores:
        """Read-only mapping of child classes to their concrete store instances.

        Use ``store.substores[ChildStore]`` to retrieve the existing child with
        its concrete return type. An unregistered class raises `KeyError`.
        """
        return self._substores

    @property
    def view(self) -> StoreView:
        """View of all managed locations in this store and its substores."""
        return StoreView._from_store(type(self), self)

    def partial(
        self,
        *selections: StoreView | str,
        exclude: Iterable[StoreView | str] | None = None,
        store_types: tuple[type[FileStoreBase], ...] | None = None,
    ) -> StoreView:
        """Select declared paths and substores without changing this store.

        Provide exactly one of positional selections, ``exclude``, or
        ``store_types``.

        Parameters
        ----------
        *selections : StoreView or str
            Class selections, bound views, or declared descriptor names.
            Include the union of these selections.
        exclude : iterable of StoreView or str, optional
            Selections to remove instead. An empty iterable removes nothing.
        store_types : tuple of FileStoreBase subclasses, optional
            Keep locations managed by these store classes, including subclasses.
            Parent selection does not include its children automatically.
            An empty tuple selects nothing.

        Returns
        -------
        StoreView
            Immutable selection bound to this store.

        Raises
        ------
        ValueError
            If zero or multiple selection forms are supplied, a name is
            undeclared, or a selection belongs to an incompatible root.
        """
        return self.view.partial(*selections, exclude=exclude, store_types=store_types)

    def iter_glob_patterns(self, spec: ObsSpecBase | str) -> Iterable[str]:
        """Yield candidate patterns relative to this store's path.

        Parameters
        ----------
        spec : ObsSpecBase or str
            Observation constraints; text uses this store's selector type.
            Positions and constraints not expressible as globs are deferred
            to selection on the populated file table.

        Yields
        ------
        str
            Unique relative patterns across local locations and substores,
            without filesystem I/O. Parsing and pattern generation occur
            during iteration.
        """
        return self.view.iter_glob_patterns(spec)

    def glob_obs_spec(self, spec: ObsSpecBase | str) -> list[Path]:
        """Collect candidate paths across declared locations and substores.

        Parameters
        ----------
        spec : ObsSpecBase or str
            Observation constraints; text uses this store's selector type.
            Apply the complete selector to the populated file table afterward,
            including positional selection.

        Returns
        -------
        list of pathlib.Path
            Unique paths in location and glob order. No headers are read;
            missing directories and unmatched globs contribute no paths.
        """
        return self.view.glob_obs_spec(spec)

    @abstractmethod
    def _iter_local_glob_patterns(
        self, spec: SpecT, subpath: SubPathOrRoot
    ) -> Iterable[str]:
        """Yield candidate globs for one managed location.

        Parameters
        ----------
        spec : SpecT
            Parsed observation constraints. Defer positional selection until
            the candidates have been collected into a file table.
        subpath : SubPathOrRoot
            Managed location from ``_subpaths``, relative to this store's root.

        Yields
        ------
        str
            Glob relative to the managed location. Patterns must exclude files
            owned by substores. Prefer fixed-depth patterns without ``**``.
        """


class _Substores(Mapping[type[FileStoreBase], FileStoreBase]):
    """Read-only child instances, indexed by their concrete classes."""

    def __init__(self, stores: dict[type[FileStoreBase], FileStoreBase]) -> None:
        self._stores = stores

    def __getitem__[StoreT: FileStoreBase](self, store_cls: type[StoreT]) -> StoreT:
        return cast("StoreT", self._stores[store_cls])

    def __iter__(self) -> Iterator[type[FileStoreBase]]:
        return iter(self._stores)

    def __len__(self) -> int:
        return len(self._stores)


_substore_adapter = TypeAdapter(SubstoreRegistryT, config=ConfigDict(strict=True))
