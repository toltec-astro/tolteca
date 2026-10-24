"""Parse information from local and remote filenames."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple, cast

from pydantic import ConfigDict, Field
from tollan.utils.fileloc import FileLoc, fileloc

from .types import DataModelsT, FrozenDataModel

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Self

__all__ = ["FilepathInfo", "register_filepath_parser"]


class _FilepathInfoParser(NamedTuple):
    """A filename pattern, its converters, and the resulting model type."""

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
    """Add a filename convention for a FilepathInfo subclass.

    Parameters
    ----------
    pattern : str or re.Pattern
        Regular expression with named captures. Use anchors to match the
        complete filename.
    parsers : tuple of callable
        Converters applied in order. Each receives the current capture dict and
        source path, and returns values to merge into the dict. Use an empty
        tuple when captures already supply all required model fields.
    filepath_info_cls : type of FilepathInfo
        Model used to validate the converted captures.

    Raises
    ------
    TypeError
        If filepath_info_cls is not a FilepathInfo subclass.

    Notes
    -----
    Identical registrations have no effect. For overlapping patterns, a subclass
    takes precedence over its base; otherwise, parsing raises an ambiguity error.
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
    """Information inferred from a filename.

    Use `from_source` to obtain the matching registered subclass.
    """

    model_config = ConfigDict(extra="forbid", polymorphic_serialization=True)

    ext: str = Field(
        description="File extension without the leading dot, or an empty string"
    )
    data_models: DataModelsT = Field(
        description="Data models supported by this filename, or an empty list"
    )

    @classmethod
    def from_source(
        cls,
        source: str | Path | FileLoc,
        *,
        raise_on_error: bool = True,
    ) -> Self | None:
        """Return filename information without opening the source.

        Parameters
        ----------
        source : str, pathlib.Path or FileLoc
            Location to parse. The source need not exist.
        raise_on_error : bool, optional
            Raise if no supported filename matches. Default is True; False
            returns None for a nonmatch. Parsing and validation errors still raise.

        Returns
        -------
        FilepathInfo or None
            Parsed information in the calling class or its most specific matching
            subclass. FilepathInfo itself also accepts unfamiliar filenames,
            returning their final suffix as ``ext`` and ``data_models=[]``.

        Raises
        ------
        ValueError
            If no filename matches and ``raise_on_error=True``, or filename
            values are invalid.
        KeyError
            If a field converter rejects a captured value.
        RuntimeError
            If matching parsers give ambiguous result types.

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
            # Convert text captures to the declared field types.
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
