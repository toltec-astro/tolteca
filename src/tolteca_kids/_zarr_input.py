"""Reader for flat TolTEC KIDs Zarr interface stores.

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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr
from tollan.accessor import NameMapping, Schema
from tollan.accessor.xarray import XarrayMapper

if TYPE_CHECKING:
    from tollan.accessor import Mapping

__all__ = ["ToltecKidsZarrMapper", "ToltecKidsZarrSchema", "open_zarr_dataset"]


@dataclass
class ToltecKidsZarrSchema(Schema):
    """Detector signals and frequency axes in an exported KIDs Zarr dataset."""

    I: Mapping = NameMapping.var("I")  # noqa: E741 -- physical signal component
    Q: Mapping = NameMapping.var("Q")
    f_tone: Mapping = NameMapping.var("tone_freq")
    f_lo: Mapping = NameMapping.var("lo_freq")
    f_lo_center: Mapping = NameMapping.attr("lo_center_freq_hz")


class ToltecKidsZarrMapper(XarrayMapper[ToltecKidsZarrSchema]):
    """Mapper for exported KIDs Zarr datasets."""

    def validate(self, data_source: xr.Dataset) -> None:
        """Check required arrays, coordinates and attributes without reading values.

        Parameters
        ----------
        data_source : xarray.Dataset
            Open exported KIDs dataset.

        Raises
        ------
        ValueError
            If a required variable, coordinate or attribute is missing.
        """
        schema = self.schema
        for label, mappings, collection in (
            ("variables", (schema.I, schema.Q, schema.f_tone), data_source.data_vars),
            ("coordinates", (schema.f_lo,), data_source.coords),
            ("attributes", (schema.f_lo_center,), data_source.attrs),
        ):
            missing = {
                self.get_name(mapping)
                for mapping in mappings
                if self.get_name(mapping) not in collection
            }
            if missing:
                msg = f"Zarr dataset is missing expected {label}: {missing}"
                raise ValueError(msg)


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
        Lazily opened dataset. The caller must close it after use. Its schema is:

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

    try:
        ToltecKidsZarrMapper.from_data_source(ds).validate(ds)
    except BaseException:
        ds.close()
        raise

    return ds
