"""Synthetic datasets for integration tests and examples."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .metadata import DummyInstruMetadata

__all__ = ["make_dataset"]


def make_dataset(metadata: DummyInstruMetadata, counts: Sequence[int]) -> xr.Dataset:
    """Create an in-memory dummy dataset with explicit metadata and samples.

    Parameters
    ----------
    metadata : DummyInstruMetadata
        Observation headers to store.
    counts : sequence of int
        Counter samples, in acquisition order.

    Returns
    -------
    xarray.Dataset
        Synthetic dataset. This function does not write files.
    """
    return xr.Dataset(
        {"counts": ("sample", list(counts))},
        attrs={
            "instru": metadata.instru,
            "interface": metadata.interface,
            "master": metadata.master,
            "obsnum": metadata.obsnum,
            "subobsnum": metadata.subobsnum,
            "scannum": metadata.scannum,
            "filepath_orig": metadata.filepath_orig,
            "t0_unix": metadata.t0.unix,
            "t1_unix": metadata.t1.unix,
            "t_int_s": metadata.t_int.to_value("s"),
        },
    )
