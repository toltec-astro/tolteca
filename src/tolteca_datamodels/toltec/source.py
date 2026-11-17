"""Open TolTEC acquisition sources using their recorded contents."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

import xarray as xr
from tollan.accessor.xarray import unwrap_data_source

from ..lmt.tel import LmtTelAccessor, _get_tel_reader_record
from ..source import SourceSourceLoc
from .hwpr import ToltecHwprAccessor, ToltecHwprIOMapper
from .kids.core import ToltecKidsAccessor, ToltecKidsIOMapper
from .types import ToltecDataKind

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..core.source import SourceReader

__all__ = ["open_toltec_source"]


def _acquisition_reader(data: object) -> SourceReader:
    if not isinstance(data, (xr.Dataset, xr.DataTree)):
        msg = "A TolTEC acquisition requires a native dataset"
        raise TypeError(msg)
    telescope = _get_tel_reader_record(data)
    if telescope is not None:
        if "toltec" not in telescope.data_models:
            msg = "Telescope acquisition does not identify TolTEC as its receiver"
            raise ValueError(msg)
        return LmtTelAccessor(data)
    ds = cast("xr.Dataset", unwrap_data_source(data))
    hwpr = ToltecHwprIOMapper.from_data_source(ds)
    if all(
        hwpr.has_arr(ds, field)
        for field in (
            hwpr.schema.enc_ts,
            hwpr.schema.enc_val,
            hwpr.schema.enc_uts,
        )
    ):
        return ToltecHwprAccessor(ds, mapper=hwpr)
    kids = ToltecKidsIOMapper.from_data_source(ds)
    if kids._get_data_kind(ds) & ToltecDataKind.raw_kids_data:
        return ToltecKidsAccessor(ds, mapper=kids)
    msg = "Source is not a supported raw TolTEC acquisition"
    raise ValueError(msg)


@contextmanager
def open_toltec_source(source: object) -> Iterator[SourceReader]:
    """Open a raw KIDs, HWPR or TolTEC telescope source.

    Parameters
    ----------
    source : path-like or SourceSourceLoc
        Local file, or a prepared source with an explicit location. Reader
        selection uses the opened contents, independently of the filename.

    Yields
    ------
    SourceReader
        Identified acquisition accessor. Metadata and lazy arrays are available
        within the context. Owned files close on exit; borrowed objects stay open.

    Raises
    ------
    TypeError
        The opened source is not a native dataset.
    ValueError
        The contents do not identify a supported raw TolTEC acquisition.
    OSError
        The source cannot be opened.
    """
    item = SourceSourceLoc.model_validate(source)
    with item.open() as data:
        yield _acquisition_reader(data)
