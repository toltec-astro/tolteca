"""Prepare and open a single resource using its filename information."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

from pydantic import (
    ConfigDict,
    computed_field,
    field_validator,
    model_validator,
)
from tollan.utils.fileloc import FileLoc

from ..filepath import FilepathInfo
from ..source import SourceOpener, source_opener
from ..types import FrozenDataModel

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["SourceSourceLoc"]


class _SourceLocation(FrozenDataModel):
    """A resource and its validated identification location."""

    model_config = ConfigDict(extra="forbid")

    source: object
    source_loc: FileLoc

    @model_validator(mode="before")
    @classmethod
    def _normalize_source(cls, data: Any) -> Any:
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            data = {"source": data}
        if data.get("source_loc") is None and "source" in data:
            data = data | {"source_loc": data["source"]}
        return data

    @field_validator("source")
    @classmethod
    def _validate_source(cls, source: object) -> object:
        if source is None:
            msg = "A source is required"
            raise ValueError(msg)
        return source

    @computed_field
    @property
    def source_loc_is_local(self) -> bool:
        """Whether the filename location is local."""
        return self.source_loc.is_local()

    @computed_field
    @property
    def source_loc_host(self) -> str:
        """Filename location's hostname, empty for local files."""
        return self.source_loc.host

    @computed_field
    @property
    def source_loc_scheme(self) -> str:
        """Filename location's URL scheme."""
        return self.source_loc.scheme


class SourceSourceLoc(_SourceLocation):
    """A resource with filename information and context-managed access.

    Parameters
    ----------
    source : object
        Path string, Path, FileLoc, or already-open object, retained unchanged. Use
        ``model_validate(source)`` to prepare a single identifier directly.
    source_loc : FileLoc, optional
        Filename location used to identify the resource, not to open it.
        Defaults to source when omitted or None. Supply it explicitly for an
        already-open object or a resource with a different original filename.
        The location need not exist.

    Notes
    -----
    Validation parses the filename without opening the resource. Invalid input
    or filename values raise. Unfamiliar names provide an extension and an empty
    ``data_models`` list. Import from ``tolteca_datamodels.source`` to enable
    built-in filename conventions.

    Examples
    --------
    >>> item = SourceSourceLoc.model_validate("archive:/renamed.nc")
    >>> item.source_loc_host
    'archive'
    >>> item.filepath_info.ext
    'nc'
    """

    _source_opener: ClassVar[SourceOpener] = source_opener

    @model_validator(mode="after")
    def _prepare_filepath(self) -> Self:
        _ = self.filepath_info
        return self

    @computed_field(repr=False)
    @cached_property
    def filepath_info(self) -> FilepathInfo | None:
        """Parsed filename information, or None when unavailable."""
        return FilepathInfo.from_source(self.source_loc, raise_on_error=False)

    @contextmanager
    def open(self, **kwargs) -> Iterator[Any]:
        """Open the resource in a context, or borrow an already-open object.

        Parameters
        ----------
        **kwargs
            Arguments passed to the opener; ignored for already-open objects.

        Yields
        ------
        object
            Resource available inside the ``with`` block. Closable resources
            opened here are released on exit, including on exceptions. Supplied
            objects are returned unchanged and remain caller-owned.

        Raises
        ------
        ValueError
            If no opener matches, or a built-in opener receives a remote source.

        Notes
        -----
        Opening occurs on context entry and uses ``source``, not ``source_loc``.
        The opener is selected using ``filepath_info``. To customize opening,
        set ``_source_opener`` to a SourceOpener policy in a subclass. Opened
        resources are not cached, and opening failures propagate.
        """
        if not isinstance(self.source, (str, Path, FileLoc)):
            yield self.source
            return
        with ExitStack() as stack:
            resource = self._source_opener.open(
                self.filepath_info, source=self.source, **kwargs
            )
            if hasattr(resource, "__enter__"):
                resource = stack.enter_context(resource)
            elif callable(close := getattr(resource, "close", None)):
                stack.callback(close)
            yield resource
