"""Identify and open data sources with explicitly chosen readers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Literal, Protocol, overload

from tollan.utils.fileloc import fileloc

if TYPE_CHECKING:
    from pathlib import Path

    from tollan.utils.fileloc import FileLoc

__all__ = [
    "SourceReader",
    "SourceReaderFactory",
    "identify_reader",
    "identify_source",
    "open_source",
]


class SourceReader(Protocol):
    """An identified data source with readable metadata."""

    def read_metadata(self) -> object:
        """Read current source metadata.

        Returns
        -------
        object
            Fresh metadata from the open source.
        """
        ...


type SourceReaderFactory = Callable[[object], SourceReader | None]


@overload
def identify_reader[PayloadT, ReaderT: SourceReader](
    data_source: object,
    *,
    source_types: tuple[type[PayloadT], ...],
    reader_factory: Callable[[PayloadT], ReaderT],
    raise_on_error: Literal[True],
) -> ReaderT: ...


@overload
def identify_reader[PayloadT, ReaderT: SourceReader](
    data_source: object,
    *,
    source_types: tuple[type[PayloadT], ...],
    reader_factory: Callable[[PayloadT], ReaderT],
    raise_on_error: Literal[False] = False,
) -> ReaderT | None: ...


@overload
def identify_reader[PayloadT, ReaderT: SourceReader](
    data_source: object,
    *,
    source_types: tuple[type[PayloadT], ...],
    reader_factory: Callable[[PayloadT], ReaderT],
    raise_on_error: bool,
) -> ReaderT | None: ...


def identify_reader[PayloadT, ReaderT: SourceReader](
    data_source: object,
    *,
    source_types: tuple[type[PayloadT], ...],
    reader_factory: Callable[[PayloadT], ReaderT],
    raise_on_error: bool = False,
) -> ReaderT | None:
    """Obtain a reader for a supported payload.

    Parameters
    ----------
    data_source : object
        Opened payload, owned by the caller.
    source_types : tuple of type
        Payload types accepted by the reader factory.
    reader_factory : callable
        Obtain and validate a reader. Raise ValueError for incompatible data.
        Do not open or close the payload or read full observation metadata.
    raise_on_error : bool, optional
        Raise for unsupported types or incompatible data instead of returning
        None. Default is False. Other exceptions always propagate.

    Returns
    -------
    SourceReader or None
        Reader for the payload, or None if its type or contents are unsupported.

    Raises
    ------
    TypeError
        If the payload type is unsupported and ``raise_on_error=True``.
    ValueError
        If the factory rejects the payload and ``raise_on_error=True``.
    """
    if not isinstance(data_source, source_types):
        if raise_on_error:
            expected = " or ".join(cls.__name__ for cls in source_types)
            msg = f"Expected {expected}, got {type(data_source).__name__}"
            raise TypeError(msg)
        return None
    try:
        return reader_factory(data_source)
    except ValueError:
        if raise_on_error:
            raise
        return None


class _StrictSourceIdentifier(Protocol):
    def __call__(
        self, data_source: object, *, raise_on_error: Literal[True]
    ) -> SourceReader: ...


class _Closeable(Protocol):
    def close(self) -> None: ...


def open_source[ResourceT: _Closeable](
    source: str | Path | FileLoc,
    *,
    opener: Callable[..., ResourceT],
    identifier: _StrictSourceIdentifier | None = None,
    **kwargs,
) -> ResourceT:
    """Open a local source with optional reader identification.

    Parameters
    ----------
    source : str, pathlib.Path or FileLoc
        Local file location. Remote locations must be staged locally before
        opening. An existing FileLoc retains its resolved location.
    opener : callable
        Accept a resolved local `pathlib.Path` and forwarded keyword arguments;
        return an opened resource with a ``close()`` method.
    identifier : callable, optional
        Identify the opened resource. Must accept ``raise_on_error=True`` and
        raise if the resource is not recognized. If omitted, only open the source.
    **kwargs
        Additional arguments passed to the opener.

    Returns
    -------
    object
        Resource returned by the opener. The caller owns it and must close it
        after use. Identification is performed only when requested.

    Raises
    ------
    ValueError
        If the location is invalid or remote, or identification rejects the data.

    Notes
    -----
    Identification failures close the newly opened resource before propagating.
    This function does not request metadata or sample arrays from the reader.
    """
    source_loc = fileloc(source)
    if source_loc.is_remote():
        msg = "Remote sources must be staged locally before opening"
        raise ValueError(msg)
    resource = opener(source_loc.path, **kwargs)
    if identifier is not None:
        try:
            identifier(resource, raise_on_error=True)
        except BaseException:
            resource.close()
            raise
    return resource


def identify_source(
    data_source: object, *, readers: Iterable[SourceReaderFactory]
) -> SourceReader | None:
    """Find a reader for an opened data source.

    Parameters
    ----------
    data_source : object
        Opened payload. Its owner must keep it open while using the reader.
    readers : iterable of callable
        Ordered identification functions. Each accepts the payload and returns
        a reader with a ``read_metadata()`` method, or None for an unrecognized
        payload. Identification must not read sample arrays or full metadata.

    Returns
    -------
    SourceReader or None
        First matching reader, or None if no reader recognizes the payload.
        Call ``reader.read_metadata()`` to read metadata; identification alone
        does not establish that all required headers are valid.

    Notes
    -----
    This function does not open or close resources. Reader exceptions propagate;
    only a None result continues to the next reader.
    """
    for identify in readers:
        reader = identify(data_source)
        if reader is not None:
            return reader
    return None
