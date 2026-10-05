"""Information encoded in local and remote filepaths."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, NamedTuple

from pydantic import BaseModel, ConfigDict
from tollan.utils.dict import dict_from_regex_match
from tollan.utils.fileloc import FileLoc, fileloc

if TYPE_CHECKING:
    from typing import Self


__all__ = ["FilepathInfo"]


class _FilepathInfoParser(NamedTuple):
    """Filepath record type and ordered parsers for one filename pattern."""

    filepath_info_cls: type[FilepathInfo]
    parsers: tuple[Callable[[dict, Path], dict[str, object]], ...]


_filename_pattern_parser: dict[re.Pattern[str], _FilepathInfoParser] = {}


class FilepathInfo(BaseModel):
    """Information encoded in a filepath.

    Use `from_source` to identify the filename and obtain its information model.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", polymorphic_serialization=True
    )

    _filename_patterns: ClassVar[
        dict[re.Pattern[str], tuple[Callable[[dict, Path], dict[str, object]], ...]]
    ] = {}
    # Subclasses associate filename patterns with ordered static parsers.
    # Each parser receives the captures updated by preceding parsers, together
    # with the source path, and returns a dict of additional or converted values.

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        # An inherited pattern still belongs to the class that declared it.
        for pattern, parsers in cls.__dict__.get("_filename_patterns", {}).items():
            _filename_pattern_parser[pattern] = _FilepathInfoParser(
                filepath_info_cls=cls, parsers=parsers
            )

    @classmethod
    def from_source(
        cls,
        source: str | Path | FileLoc,
        *,
        accepted_types: tuple[type[FilepathInfo], ...] | None = None,
        raise_on_error: bool = True,
    ) -> Self | None:
        """Parse a local or remote path without opening the source.

        Parameters
        ----------
        source : str, pathlib.Path or FileLoc
            Location whose filename is to be parsed. The source need not exist.
        accepted_types : tuple of FilepathInfo classes or None, optional
            Restrict results to these classes and their subclasses. Default is
            None, accepting the calling class and its subclasses. An explicit
            tuple further narrows that selection; an empty tuple accepts none.
        raise_on_error : bool, optional
            Whether an unrecognized filename raises an error. Default is True.
            This does not suppress errors in filenames matching an accepted type.

        Returns
        -------
        FilepathInfo or None
            An instance of the accepted subclass identified by the filename,
            or None if no accepted type matches and ``raise_on_error=False``.

        Raises
        ------
        ValueError
            If no accepted type matches and ``raise_on_error=True``, or if
            values in a matching filename are invalid.
        KeyError
            If a filename matching an accepted type names an unsupported interface.

        Examples
        --------
        >>> from tolteca_datamodels.lmt import LmtFilepathInfo
        >>> from tolteca_datamodels.toltec import ToltecFilepathInfo
        >>> info = FilepathInfo.from_source(
        ...     "tel_toltec_2025-10-31_145609_00_0002.nc",
        ...     accepted_types=(LmtFilepathInfo, ToltecFilepathInfo),
        ... )
        >>> type(info).__name__
        'LmtFilepathInfo'
        """
        source = fileloc(source)
        path = source.path
        for pattern, parser in _filename_pattern_parser.items():
            model_cls = parser.filepath_info_cls
            if not issubclass(model_cls, cls):
                continue
            if accepted_types is not None and not issubclass(model_cls, accepted_types):
                continue
            captures = dict_from_regex_match(pattern, path.name)
            if captures is None:
                continue
            for parse in parser.parsers:
                captures.update(parse(captures, path))
            return model_cls.model_validate(captures)
        if not raise_on_error:
            return None
        msg = f"Unrecognized acquisition filename for {cls.__name__}: {source}"
        raise ValueError(msg)
