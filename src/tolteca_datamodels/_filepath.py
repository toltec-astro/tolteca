"""Information encoded in local and remote filepaths."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar, NamedTuple

from pydantic import ConfigDict
from tollan.accessor.pandas.model_table import (
    ModelColumn,
    ModelColumnMapper,
    ScalarColumnRule,
)
from tollan.utils.fileloc import FileLoc, fileloc

from .core import FrozenDataModel

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

__all__ = ["FilepathInfo", "register_filepath_parser"]


class _FilepathInfoParser(NamedTuple):
    """Filepath record type and ordered parsers for one filename pattern."""

    pattern: re.Pattern[str]
    filepath_info_cls: type[FilepathInfo]
    parsers: tuple[Callable[[dict, Path], dict[str, object]], ...]


_filepath_parsers: list[_FilepathInfoParser] = []
_filepath_column_mapper = ModelColumnMapper(
    name_rules=[(r".+", r"file_\g<0>")],
    type_rules=[ScalarColumnRule()],
    default=None,
)


def register_filepath_parser(
    *,
    pattern: str | re.Pattern[str],
    parsers: tuple[Callable[[dict, Path], dict[str, object]], ...],
    filepath_info_cls: type[FilepathInfo],
) -> None:
    """Register a filename pattern, field parsers, and result model.

    Parameters
    ----------
    pattern : str or re.Pattern
        Regular expression with named captures describing the filename.
        Include anchors to match the complete filename.
    parsers : tuple of callable
        Ordered field converters. Each receives the captures updated by previous
        converters and the source path, and returns a dict to merge into the
        captures. An empty tuple uses the captures unchanged.
    filepath_info_cls : type of FilepathInfo
        Model used to validate the converted captures.

    Raises
    ------
    TypeError
        If filepath_info_cls is not a FilepathInfo subclass.

    Notes
    -----
    Registering an identical entry again has no effect. When patterns overlap,
    a result subclass takes precedence over its base class. Other ambiguities
    raise when parsing the matching filename, before field conversion.
    """
    if not isinstance(filepath_info_cls, type) or not issubclass(
        filepath_info_cls, FilepathInfo
    ):
        msg = "Expected a FilepathInfo subclass"
        raise TypeError(msg)
    parser = _FilepathInfoParser(re.compile(pattern), filepath_info_cls, parsers)
    if parser in _filepath_parsers:
        return
    _filepath_parsers.append(parser)


class FilepathInfo(FrozenDataModel):
    """Information encoded in a filepath.

    Use `from_source` to identify the filename and obtain its information model.
    """

    model_config = ConfigDict(extra="forbid", polymorphic_serialization=True)

    _columns: ClassVar[dict[str, ModelColumn]]

    @classmethod
    def __pydantic_on_complete__(cls) -> None:
        super().__pydantic_on_complete__()
        if "_columns" not in cls.__dict__:
            cls._columns = _filepath_column_mapper(cls)

    @staticmethod
    def from_source(
        source: str | Path | FileLoc,
        *,
        raise_on_error: bool = True,
    ) -> FilepathInfo | None:
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
            Parsed information for the most specialized registered filename
            layout, or None for an unrecognized name with ``raise_on_error=False``.

        Raises
        ------
        ValueError
            If no filename pattern matches and ``raise_on_error=True``, or if
            values in a matching filename are invalid, or matching registrations
            do not resolve to one most-specific result type.
        KeyError
            If a recognized filename names an unsupported interface.

        Examples
        --------
        >>> from tolteca_datamodels.file import FilepathInfo
        >>> info = FilepathInfo.from_source(
        ...     "tel_toltec_2025-10-31_145609_00_0002.nc",
        ... )
        >>> info.prefix, info.interface, info.obsnum
        ('tel_toltec', 'tel', 145609)
        """
        source = fileloc(source)
        path = source.path
        matches = []
        for parser in _filepath_parsers:
            match = parser.pattern.match(path.name)
            if match is not None:
                matches.append((parser, match))
        if len(matches) > 1:
            matches = [
                (parser, match)
                for parser, match in matches
                if not any(
                    other.filepath_info_cls is not parser.filepath_info_cls
                    and issubclass(other.filepath_info_cls, parser.filepath_info_cls)
                    for other, _ in matches
                )
            ]
            if len(matches) != 1:
                names = ", ".join(p.filepath_info_cls.__name__ for p, _ in matches)
                msg = f"Ambiguous acquisition filename: {source} ({names})"
                raise ValueError(msg)
        if matches:
            parser, match = matches[0]
            captures = match.groupdict()
            for parse in parser.parsers:
                captures.update(parse(captures, path))
            # Regex captures are text; the filename-ingestion boundary converts
            # them to the declared types before returning a strict data model.
            return parser.filepath_info_cls.model_validate(captures, strict=False)
        if not raise_on_error:
            return None
        msg = f"Unrecognized acquisition filename: {source}"
        raise ValueError(msg)
