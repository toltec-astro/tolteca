"""Default format-opening policy for Tolteca data sources."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import xarray as xr
from astropy.table import QTable
from tollan.accessor.pandas.model_table import RecordRuleMapper
from tollan.utils.fileloc import fileloc

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    import pandas as pd
    from tollan.utils.fileloc import FileLoc

__all__ = ["SourceOpener", "source_opener"]


class SourceOpener(RecordRuleMapper[Callable[..., Any]]):
    """Open explicit sources using ordered record rules.

    Parameters
    ----------
    rules : tuple
        Ordered pairs of field constraints and opening callables. Constraints
        follow `RecordRuleMapper`; the first matching rule wins. Each callable
        accepts a source as its first argument, followed by opening keywords.
        Sources are passed unchanged, independently of the matching record.

    Notes
    -----
    Compose policies by placing overrides before another policy's ``rules``.
    Matching does not open sources. Opened objects belong to the caller.

    Examples
    --------
    >>> from io import StringIO
    >>> opener = SourceOpener(rules=(({"format": "text"}, StringIO),))
    >>> with opener.open({"format": "text"}, source="measurement") as stream:
    ...     stream.read()
    'measurement'
    """

    default: None = None

    def open(self, record: object, *, source: object, **kwargs) -> Any:
        """Open a source with the callable selected by a record.

        Parameters
        ----------
        record : mapping or object
            Rule values read from mapping keys or object attributes.
        source : object
            Resource passed unchanged to the selected callable.
        **kwargs
            Additional arguments passed to the selected callable.

        Returns
        -------
        object
            Opened object owned by the caller. Close it after use if closable.

        Raises
        ------
        ValueError
            If no rule matches. Exceptions from the selected callable propagate;
            another rule is not tried after failure.
        """
        opener = self(record)
        if opener is None:
            msg = "No source opener matches record"
            raise ValueError(msg)
        return opener(source, **kwargs)

    def iter_open_table(
        self,
        table: pd.DataFrame,
        *,
        source_col: str,
        column_names: Mapping[str, str] | None = None,
        **kwargs,
    ) -> Iterator[Any]:
        """Yield opened sources in table row order.

        Parameters
        ----------
        table : pandas.DataFrame
            Rule values and sources, one per row. The table is not modified.
        source_col : str
            Column containing sources to pass unchanged to opening callables.
            This name is not translated by column_names.
        column_names : mapping, optional
            Rule names mapped to table columns. Unmapped names are unchanged.
        **kwargs
            Additional arguments passed to every selected opening callable.

        Yields
        ------
        object
            One opened object per row, in positional order. The caller owns
            each yielded object and must close it after use if closable.

        Raises
        ------
        KeyError
            If source_col is absent.
        ValueError
            During iteration, if a row has no matching opener. Exceptions from
            opening callables also stop iteration; no later rule or row is tried.

        Notes
        -----
        Sources open only as iteration advances. Previously yielded objects
        remain owned by the caller on failure or early termination; no objects
        are closed here.
        """
        sources = table[source_col].tolist()
        openers = self.map_table(table, column_names=column_names)

        def iterate():
            for position, (source, opener) in enumerate(
                zip(sources, openers, strict=True)
            ):
                if opener is None:
                    msg = f"No source opener matches row {position}"
                    raise ValueError(msg)
                yield opener(source, **kwargs)

        return iterate()


def _local_path(source: str | Path | FileLoc) -> Path:
    location = fileloc(source)
    if location.is_remote():
        msg = "Remote sources must be staged locally before opening"
        raise ValueError(msg)
    return location.path


def _open_netcdf(source: str | Path | FileLoc, **kwargs) -> xr.Dataset:
    return xr.open_dataset(_local_path(source), **kwargs)


def _open_ecsv(source: str | Path | FileLoc, **kwargs) -> QTable:
    return QTable.read(_local_path(source), format="ascii.ecsv", **kwargs)


source_opener = SourceOpener(
    rules=(({"ext": "nc"}, _open_netcdf), ({"ext": "ecsv"}, _open_ecsv))
)
"""Default policy opening local NetCDF as xarray datasets and ECSV as QTables."""
