"""TolTEC KIDs zarr store reader.

This module provides the read side of the zarr store produced by
``tolteca_db.export.zarr_exporter``.  It lets downstream code open a
TolTEC zarr store as a standard :class:`xarray.Dataset` with no knowledge
of the export internals.

The on-disk schema written by the exporter is:

* ``I(chan, sample)``       — float32, in-phase detector signal
* ``Q(chan, sample)``       — float32, quadrature detector signal
* ``tone_freq(chan)``       — float64, tone Hz offsets from LO centre
* coord ``lo_freq(sample)`` — float64, absolute LO Hz per sweep step
* attr ``lo_center_freq_hz`` — float64 scalar, nominal LO centre Hz
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

__all__ = ["open_zarr_dataset"]

_EXPECTED_VARS = frozenset({"I", "Q", "tone_freq"})
_EXPECTED_COORDS = frozenset({"lo_freq"})
_EXPECTED_ATTRS = frozenset({"lo_center_freq_hz"})


def open_zarr_dataset(path: str | Path) -> xr.Dataset:
    """Open a TolTEC KIDs zarr store as an :class:`xarray.Dataset`.

    Loads a zarr store written by ``tolteca_db.export.ZarrExporter`` and
    validates that it contains the expected variables, coordinates, and
    attributes.

    Parameters
    ----------
    path : str | Path
        Path to the zarr store directory.

    Returns
    -------
    xr.Dataset
        Dataset with the zarr schema:

        * ``I(chan, sample)`` — float32
        * ``Q(chan, sample)`` — float32
        * ``tone_freq(chan)`` — float64, Hz offsets from LO centre
        * coord ``lo_freq(sample)`` — float64, absolute LO Hz per step
        * attrs[``"lo_center_freq_hz"``] — float64 scalar

    Raises
    ------
    ValueError
        If the zarr store is missing expected variables, coordinates, or
        attributes.
    """
    ds = xr.open_zarr(Path(path))

    missing_vars = _EXPECTED_VARS - set(ds.data_vars)
    if missing_vars:
        raise ValueError(
            f"zarr store at {path!r} is missing expected variables: {missing_vars}"
        )
    missing_coords = _EXPECTED_COORDS - set(ds.coords)
    if missing_coords:
        raise ValueError(
            f"zarr store at {path!r} is missing expected coordinates: {missing_coords}"
        )
    missing_attrs = _EXPECTED_ATTRS - set(ds.attrs)
    if missing_attrs:
        raise ValueError(
            f"zarr store at {path!r} is missing expected attributes: {missing_attrs}"
        )

    return ds
