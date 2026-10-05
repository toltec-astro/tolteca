"""Directory-backed file discovery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from tollan.utils.log import logger
from tollan.utils.typing import (
    GenericBindingRequirement,
    resolve_concrete_generic_bindings,
)

from ._input_discovery import InputDiscoveryEntry
from .obs_spec import ObsSpecBase, ObsSpecContext

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import pandas as pd

    from .file import SourceInfoErrorPolicyT

__all__ = ["FileStoreBase", "FileStoreInputDiscovery"]


class FileStoreBase[SpecT: ObsSpecBase](ABC):
    """Candidate discovery within a directory tree.

    Parameters
    ----------
    path : str or pathlib.Path
        Read-only root of the directory tree. The directory need not exist.

    Notes
    -----
    Subclasses supply local glob patterns and bind the selector type through
    ``FileStoreBase[SpecT]``. Declare built-in substores with a
    ``_substore_classes`` mapping of directory names to store classes, or inherit
    the parent's mapping. Extensions can use `register_substore` after class
    creation. Registered substores are searched after local files, in registration
    order. Subclass registrations do not modify their parents or siblings.
    """

    _obs_spec_context: ClassVar[ObsSpecContext] = ObsSpecContext()
    _substore_classes: ClassVar[dict[str, type[FileStoreBase]]] = {}
    _obs_spec_cls: ClassVar

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        substore_classes = cls._substore_classes
        cls._substore_classes = {}
        bindings = resolve_concrete_generic_bindings(
            cls,
            FileStoreBase,
            SpecT=GenericBindingRequirement("_obs_spec_cls", ObsSpecBase),
        )
        if bindings is not None:
            cls._obs_spec_cls = bindings["SpecT"]
        for name, store_cls in substore_classes.items():
            cls.register_substore(name, store_cls)

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._substores: dict[str, FileStoreBase[SpecT]] = {}

    @property
    def path(self) -> Path:
        """Root of the directory tree."""
        return self._path

    @classmethod
    def register_substore(
        cls, name: str, store_cls: type[FileStoreBase[SpecT]]
    ) -> None:
        """Register a store beneath an immediate subdirectory.

        Parameters
        ----------
        name : str
            Subdirectory name relative to this store's root.
        store_cls : type of FileStoreBase
            Store constructed with the subdirectory as its root.

        Raises
        ------
        ValueError
            If the name is not a directory component, is registered for another
            class, or would introduce a cycle. Repeating the same registration
            is allowed.
        TypeError
            If store_cls is not a FileStoreBase subclass.
        """
        if not name or name in {".", ".."} or any(char in name for char in "/\\"):
            msg = f"Invalid substore name: {name!r}"
            raise ValueError(msg)
        if not isinstance(store_cls, type) or not issubclass(store_cls, FileStoreBase):
            msg = "A substore must be a FileStoreBase subclass"
            raise TypeError(msg)
        previous = cls._substore_classes.get(name)
        if previous is not None and previous is not store_cls:
            msg = f"Substore already registered: {name!r}"
            raise ValueError(msg)
        pending, seen = [store_cls], set()
        while pending:
            current = pending.pop()
            if current is cls:
                msg = f"Substore registration would create a cycle: {name!r}"
                raise ValueError(msg)
            if current not in seen:
                seen.add(current)
                pending.extend(current._substore_classes.values())
        cls._substore_classes[name] = store_cls

    def get_substore(self, name: str) -> FileStoreBase[SpecT]:
        """Get a registered substore.

        Parameters
        ----------
        name : str
            Registered subdirectory name.

        Returns
        -------
        FileStoreBase
            Substore rooted beneath this store. Repeated calls return the same
            instance. Registrations made after this store was created are visible.

        Raises
        ------
        KeyError
            If the name is not registered.
        """
        store_cls = self._substore_classes[name]
        if name not in self._substores:
            self._substores[name] = store_cls(self.path / name)
        return self._substores[name]

    @property
    def substores(self) -> Mapping[str, FileStoreBase[SpecT]]:
        """Registered names and their substores, in registration order."""
        return {name: self.get_substore(name) for name in self._substore_classes}

    def find_candidates(self, spec: SpecT | str) -> list[Path]:
        """Find candidate paths without reading file contents.

        Parameters
        ----------
        spec : selector or str
            Search constraints supported by the concrete store. Positional
            selection is deferred until the candidate table is populated.

        Returns
        -------
        list of pathlib.Path
            Unique candidate paths: local matches first, then substores in
            registration order. Glob order and the first occurrence of each
            path are retained. Candidates may not satisfy every constraint;
            apply the complete selector after table population. No matches or
            a nonexistent directory produce an empty list.
        """
        return self._find_candidates(spec, self.substores.values())

    def _find_candidates(
        self, spec: SpecT | str, substores: Iterable[FileStoreBase[SpecT]]
    ) -> list[Path]:
        spec = self._obs_spec_cls.parse(spec)
        paths = []
        for root, pattern in self._iter_candidate_patterns(spec):
            candidates = list(root.glob(pattern))
            logger.debug(
                "ObsSpec {} under {} using {!r}: {} candidates",
                spec,
                root,
                pattern,
                len(candidates),
            )
            paths.extend(candidates)
        for store in substores:
            paths.extend(store.find_candidates(spec))
        result = list(dict.fromkeys(paths))
        logger.debug("ObsSpec {} under {}: {} candidates", spec, self.path, len(result))
        return result

    @abstractmethod
    def _iter_candidate_patterns(self, spec: SpecT) -> Iterable[tuple[Path, str]]:
        """Yield local glob roots and patterns without applying positions."""

    def get_obs_spec_context(self) -> ObsSpecContext:
        """Get selector interpretations for this store and its substores.

        Returns
        -------
        ObsSpecContext
            Combined declarations for the complete store hierarchy, independent
            of which files exist.

        Raises
        ------
        ValueError
            If stores declare conflicting numeric values for the same label.
        """
        return self._get_obs_spec_context(self.substores.values())

    def _get_obs_spec_context(
        self, substores: Iterable[FileStoreBase[SpecT]]
    ) -> ObsSpecContext:
        context = self._obs_spec_context
        for store in substores:
            context = context | store.get_obs_spec_context()
        return context


class FileStoreInputDiscovery(InputDiscoveryEntry):
    """Build a selected file table from a directory-backed store.

    Parameters
    ----------
    obs_spec : ObsSpecBase
        Observation selector used for candidate discovery and table selection.
    mode : {"guess", "guess_or_read", "read"}, optional
        How to populate candidate information. Default ``"guess"`` uses
        filenames only. ``"guess_or_read"`` reads supported, openable files
        whose filenames are unrecognized; ``"read"`` reads all supported,
        openable files.
    """

    obs_spec: ObsSpecBase
    mode: Literal["guess", "guess_or_read", "read"] = "guess"

    @abstractmethod
    def _find_candidates(self) -> list[Path]:
        """Find candidate paths in discovery order without opening files."""

    @abstractmethod
    def _populate_candidates(
        self, paths: list[Path], *, on_error: SourceInfoErrorPolicyT
    ) -> pd.DataFrame:
        """Populate candidate information using the configured mode."""

    @property
    def _obs_spec_context(self) -> ObsSpecContext:
        """Label interpretations for the selected stores."""
        return ObsSpecContext()

    def discover(self, *, on_error: SourceInfoErrorPolicyT = "raise") -> pd.DataFrame:
        """Build and select the candidate file table.

        Parameters
        ----------
        on_error : {"raise", "record"}, optional
            Whether source-information failures raise (default) or remain in
            the table's ``diagnostics`` column before ObsSpec selection.

        Returns
        -------
        pandas.DataFrame
            Candidates populated using ``mode`` and filtered with ``obs_spec``,
            retaining discovery order. No matches yields an empty table.
        """
        table = self._populate_candidates(self._find_candidates(), on_error=on_error)
        selected = self.obs_spec.select(table, context=self._obs_spec_context)
        logger.debug(
            "Candidate selection: mode={}, obs_spec={}; {} candidates -> {} selected",
            self.mode,
            self.obs_spec,
            len(table),
            len(selected),
        )
        return selected
