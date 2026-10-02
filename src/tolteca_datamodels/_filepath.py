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
        cls, source: str | Path | FileLoc, *, raise_on_error: bool = True
    ) -> Self | None:
        """Parse a local or remote path without opening the source.

        Parameters
        ----------
        source : str, pathlib.Path or FileLoc
            Location whose filename is to be parsed. The source need not exist.
        raise_on_error : bool, optional
            Whether an unrecognized filename raises an error. Default is True.
            This does not suppress errors in recognized filenames.

        Returns
        -------
        FilepathInfo or None
            An instance of the subclass identified by the filename. When called
            on a subclass, only instances of that subclass are accepted. Return
            None for a nonmatching filename if ``raise_on_error=False``.

        Raises
        ------
        ValueError
            If the filename does not identify the requested class and
            ``raise_on_error=True``, or if recognized values are invalid.
        KeyError
            If a recognized filename names an unsupported interface.
        """
        source = fileloc(source)
        path = source.path
        for pattern, parser in _filename_pattern_parser.items():
            captures = dict_from_regex_match(pattern, path.name)
            if captures is None:
                continue
            for parse in parser.parsers:
                captures.update(parse(captures, path))
            parsed = parser.filepath_info_cls.model_validate(captures)
            if isinstance(parsed, cls):
                return parsed
            if not raise_on_error:
                return None
            msg = f"Acquisition filename does not identify {cls.__name__}: {source}"
            raise ValueError(msg)
        if not raise_on_error:
            return None
        msg = f"Unrecognized acquisition filename: {source}"
        raise ValueError(msg)
