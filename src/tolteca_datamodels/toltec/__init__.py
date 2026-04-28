"""TolTEC-specific data models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import xarray as xr

from .kids import (
    ReducedSweepView,
    SweepReducer,
    ToltecKidsAccessor,
    ToltecKidsIOMapper,
    ToltecKidsIOSchema,
)
from .metadata import (
    ToltecRawObsMetadata,
    ToltecSweepMetadata,
    ToltecTimeStreamMetadata,
)
from .types import (
    ToltecArrayNameT,
    ToltecArrayType,
    ToltecDataKind,
    ToltecInfo,
    ToltecMasterNameT,
    ToltecMasterType,
)

if TYPE_CHECKING:
    from pathlib import Path


def open_toltec(
    filepath: str | Path,
    *,
    decode_coords: Literal["coordinates", "all"] | bool | None = "all",
    decode_times: bool = True,
    **kwargs,
) -> xr.Dataset:
    """Open a TolTEC netCDF file with proper coordinate handling.

    This function opens a TolTEC netCDF file using xarray and ensures the
    toltec_kids accessor is registered. It handles coordinate decoding and
    provides sensible defaults for TolTEC data files.

    Parameters
    ----------
    filepath : str or Path
        Path to the TolTEC netCDF file
    decode_coords : Literal["coordinates", "all"] or bool or None, default "all"
        Controls coordinate variable handling:
        - "coordinates" or True: Set variables referred to in attributes
          as coordinate variables
        - "all": Set variables referred to in CF attributes (bounds, etc.)
          as coordinates
        - False: Don't decode coordinates
    decode_times : bool, default True
        If True, decode times encoded in the standard NetCDF datetime format
    **kwargs
        Additional keyword arguments passed to xarray.open_dataset

    Returns
    -------
    xr.Dataset
        Dataset with toltec_kids accessor available

    Examples
    --------
    >>> from tolteca_datamodels.toltec import open_toltec
    >>> ds = open_toltec("toltec_sweep.nc")
    >>> ds.toltec_kids.data_kind
    >>> ds.toltec_kids.array_name

    See Also
    --------
    reduce_raw_sweep : Reduce raw sweep data to mean/std per sweep step
    SweepReducer : Reducer class for sweep data reduction
    """
    # Accessor is already imported above, ensuring registration
    return xr.open_dataset(
        filepath,
        decode_coords=decode_coords,
        decode_times=decode_times,
        **kwargs,
    )


# Backward compatibility: provide reduce_raw_sweep as wrapper
def reduce_raw_sweep(ds, **kwargs):
    """Reduce raw sweep data (backward compatibility wrapper).

    This is a compatibility wrapper for the old reduce_raw_sweep function.
    New code should use SweepReducer directly.

    Parameters
    ----------
    ds : xr.Dataset
        Raw sweep dataset
    **kwargs
        Passed to SweepReducer constructor

    Returns
    -------
    xr.Dataset
        Reduced sweep dataset with namespaced variables
    """
    reducer = SweepReducer(**kwargs)
    return reducer(ds)


__all__ = [
    "ReducedSweepView",
    "SweepReducer",
    "ToltecArrayNameT",
    "ToltecArrayType",
    "ToltecDataKind",
    "ToltecInfo",
    "ToltecKidsAccessor",
    "ToltecKidsIOMapper",
    "ToltecKidsIOSchema",
    "ToltecMasterNameT",
    "ToltecMasterType",
    "ToltecRawObsMetadata",
    "ToltecSweepMetadata",
    "ToltecTimeStreamMetadata",
    "open_toltec",
    "reduce_raw_sweep",
]
