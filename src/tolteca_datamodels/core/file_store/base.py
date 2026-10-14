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
    """Candidate discovery within a directory tree.

    Parameters
    ----------
    path : str or pathlib.Path
        Read-only root of the directory tree. The directory need not exist.

    Notes
    -----
    Subclasses bind the selector type through
    ``FileStoreBase[SpecT]``. Declare directly managed locations with
    ``_subpaths``: PurePosixPath values satisfying SubPathOrRoot, with
    ``PurePosixPath(".")`` for the store's own root. These declarations are trusted,
    not validated. The default is empty: no local searching. Subclasses may
    inherit or replace these declarations.
    Implement ``_iter_local_glob_patterns(spec, subpath)`` to yield glob strings
    relative to each managed location, excluding files owned by substores.
    Prefer fixed-depth patterns without ``**``. Managing the root does not
    imply recursive ownership of its contents. Declare named children with
    ``child = Substore(ChildStore, subpath="child")``. A ``_substore_paths``
    mapping of store classes to PurePosixPath values declares children without
    descriptor names. Subclasses inherit the parent's mapping unless they replace
    it. Class creation registers map entries and newly declared descriptors
    through `register_substore`; extensions can call it after class creation.
    Registered substores are searched after local files, in registration
    order. Subclass registrations do not modify their parents or siblings.
    Register children before constructing a store. Construction creates its
    substore instances without searching directories or opening files. Later
    registrations affect new instances only.
    Declare `StorePaths` descriptors for named local selections.
    Views expose the same descriptor names as their stores.
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
        """Root of the directory tree."""
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
            Subpath relative to this store's root, validated as SubPath.
        name : str, optional
            Public descriptor name for this child. Omit for class-based lookup
            only. Existing attributes cannot be replaced.

        Raises
        ------
        pydantic.ValidationError
            If subpath or store_cls is invalid.
        ValueError
            If the class is already registered, including at the same subpath, or the
            registration would introduce a cycle, or name is invalid or occupied.

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

        Children are created with this store, in registration order. Indexing
        preserves the concrete return type and raises `KeyError` for absent
        classes. Repeated access returns the same mapping and child instances.
        """
        return self._substores

    @property
    def view(self) -> StoreView:
        """All locations in this store's constructed tree."""
        return StoreView._from_store(type(self), self)

    def partial(
        self,
        *selections: StoreView | str,
        exclude: Iterable[StoreView | str] | None = None,
    ) -> StoreView:
        """Select declared paths and substores without changing this store.

        Parameters
        ----------
        *selections : StoreView or str
            Class selections, bound views, or declared descriptor names.
            Their union is selected. Cannot be combined with ``exclude``.
        exclude : iterable of StoreView or str, optional
            Selections to remove instead. An empty iterable removes nothing.
            Provide at least one positional selection or this argument.

        Returns
        -------
        StoreView
            Immutable selection bound to this store.

        Raises
        ------
        ValueError
            If neither or both selection forms are supplied, a name is
            undeclared, or a selection belongs to an incompatible root.
        """
        return self.view.partial(*selections, exclude=exclude)

    def iter_glob_patterns(self, spec: ObsSpecBase | str) -> Iterable[str]:
        """Yield candidate patterns relative to this store's path.

        Parameters
        ----------
        spec : ObsSpecBase or str
            Observation constraints; text uses this store's selector type.

        Yields
        ------
        str
            Unique globs across declared local locations and substores, without
            filesystem I/O. Complete ObsSpec selection requires a file table.
        """
        return self.view.iter_glob_patterns(spec)

    def glob_obs_spec(self, spec: ObsSpecBase | str) -> list[Path]:
        """Collect candidate paths across declared locations and substores.

        Parameters
        ----------
        spec : ObsSpecBase or str
            Observation constraints. Apply the complete selector to the
            populated file table afterward.

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
        """Yield local globs, excluding delegated files and deferring positions.

        Patterns are relative to the selected managed location. Use fixed-depth
        patterns where possible; even a nonrecursive wildcard directory must
        not overlap a registered substore's files.
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
