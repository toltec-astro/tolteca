"""Explicit structural inspection of a native KIDs acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

    from ..metadata import (
        ToltecRawObsBlocksMetadata,
        ToltecRawObsMetadata,
    )

__all__ = ["ToltecBlockIndex"]


@dataclass(frozen=True)
class ToltecBlockIndex:
    """Recorded block bounds, channel/probe axes and acquisition metadata.

    Attributes
    ----------
    index : xarray.Dataset
        One row per block with sample and axis-row bounds (exclusive ends),
        data kind and, for sweeps, the programmed step count.
    chan, probe : xarray.Dataset
        Columnar channel and probe axes shared by the recorded blocks.
    meta : tolteca_datamodels.toltec.metadata.ToltecRawObsBlocksMetadata
        Acquisition metadata, discovered block count, block index and axis tables.
    acquisition : ToltecRawObsMetadata or None
        Acquisition metadata for a native unpartitioned source.
    """

    index: xr.Dataset
    chan: xr.Dataset
    probe: xr.Dataset
    meta: ToltecRawObsBlocksMetadata
    acquisition: ToltecRawObsMetadata | None


def _find_lo_block_slices(los, *, split_resets):
    """Locate current-format sweep resets without interpreting mixed-file dwells."""
    if los.ndim != 1 or not np.all(np.isfinite(los) & (los > 0)):
        msg = "LO frequencies must be a positive finite 1-D array"
        raise ValueError(msg)
    # TODO: mixed sweep/timestream boundaries. Expected acquisitions move the
    # timestream LO away from the final sweep LO; confirm with future files.
    resets = np.flatnonzero(np.diff(los) < 0) + 1 if split_resets else []
    edges = [0, *resets, len(los)]
    return [slice(a, b) for a, b in pairwise(edges) if a < b]
