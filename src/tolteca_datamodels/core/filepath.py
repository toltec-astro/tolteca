"""Information encoded in local and remote filepaths.

This module uses available parser registrations without loading instrument
parsers. Import from ``tolteca_datamodels.filepath`` for built-in LMT and TolTEC
support.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated, Final, NamedTuple, cast

from pydantic import ConfigDict, Field
from tollan.utils.fileloc import FileLoc, fileloc

from .types import DataModelsT, FrozenDataModel

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Self

__all__ = ["FilepathInfo", "MetadataField", "register_filepath_parser"]


class _MetadataField:
    pass


MetadataField: Final = _MetadataField()
"""Annotation marker for metadata encoded in a filename.

Use ``Annotated[T, MetadataField]`` without parentheses. File tables retain the
``file_*`` snapshot and initialize the corresponding unprefixed metadata column.
The marker does not change validation or make filename information authoritative.
"""


class _FilepathInfoParser(NamedTuple):
    """Filepath record type and ordered parsers for one filename pattern."""

    pattern: re.Pattern[str]
    filepath_info_cls: type[FilepathInfo]
    parsers: tuple[Callable[[dict, Path], dict[str, object]], ...]


_filepath_parsers: list[_FilepathInfoParser] = []


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
        captures. An empty tuple uses the captures unchanged. The resulting
        values must provide the model's required fields, including data_models.
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

    ext: str = Field(
        description="Filename extension without the dot; empty for extensionless names"
    )
    data_models: Annotated[DataModelsT, MetadataField] = Field(
        description="Data-model memberships identified from the filename; may be empty"
    )

    @classmethod
    def from_source(
        cls,
        source: str | Path | FileLoc,
        *,
        raise_on_error: bool = True,
    ) -> Self | None:
        """Parse a local or remote path without opening the source.

        Parameters
        ----------
        source : str, pathlib.Path or FileLoc
            Location whose filename is to be parsed. The source need not exist.
        raise_on_error : bool, optional
            Whether a filename without a matching parser raises an error.
            Default is True. This does not suppress errors in matching layouts.

        Returns
        -------
        FilepathInfo or None
            An instance of the calling class or its most specialized matching
            subclass. Parsers for ancestors or unrelated classes are excluded.
            FilepathInfo itself also accepts unfamiliar names, using the final
            suffix as ext and an empty data_models set. If no eligible parser
            matches, return None when ``raise_on_error=False``.

        Raises
        ------
        ValueError
            If no eligible parser matches and ``raise_on_error=True``, or if
            values in a registered filename layout are invalid.
        KeyError
            If a recognized filename names an unsupported interface.
        RuntimeError
            If matching registrations do not resolve to one most-specific result
            type, regardless of raise_on_error.

        Examples
        --------
        >>> from tolteca_datamodels.filepath import FilepathInfo
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
            if not issubclass(parser.filepath_info_cls, cls):
                continue
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
                raise RuntimeError(msg)
        if matches:
            parser, match = matches[0]
            captures = match.groupdict()
            for parse in parser.parsers:
                captures.update(parse(captures, path))
            # Regex captures are text; the filename-ingestion boundary converts
            # them to the declared types before returning a strict data model.
            return cast(
                "Self", parser.filepath_info_cls.model_validate(captures, strict=False)
            )
        if not raise_on_error:
            return None
        msg = f"No filename parser matches {cls.__name__}: {source}"
        raise ValueError(msg)


def _parse_basic_filename(captures: dict, path: Path) -> dict[str, object]:
    return {"ext": path.suffix.removeprefix("."), "data_models": []}


register_filepath_parser(
    pattern=re.compile(r"^.+$", re.DOTALL),
    parsers=(_parse_basic_filename,),
    filepath_info_cls=FilepathInfo,
)
