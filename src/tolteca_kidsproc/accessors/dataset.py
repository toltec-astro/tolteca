"""Utilities for create and open KIDs dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from tolteca_kidsproc.accessors.kids import KidsMapper

__all__ = [
    "KidsDataTree",
    "KidsDataset",
    "make_kids_dataset",
    "open_datatree",
]

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from astropy.units import Quantity

    from tolteca_kidsproc.accessors.kids import KidsAccessor

    class KidsDataset(xr.Dataset):
        """Typed wrapper for xarray Dataset with KIDs accessor."""

        kids: KidsAccessor

    class KidsDataTree(xr.DataTree):
        """Typed wrapper for xarray DataTree with KIDs accessor."""

        kids: KidsAccessor


def open_datatree(
    *args,
    **kwargs,
) -> KidsDataTree:
    """Open a KIDs data file with accessor support.

    Wraps xarray.open_datatree with proper typing for KIDs accessor.
    Uses lazy loading so there's no overhead - loads only the tree structure,
    not the actual data arrays.

    Works for both simple datasets and hierarchical trees with analysis results
    stored as child nodes. Always returns a DataTree for consistency.

    Returns
    -------
    KidsDataTree
        DataTree with kids accessor registered on root and children.
        If file contains only a dataset, it becomes the root of the tree.

    Examples
    --------
    Open simple sweep data:
        >>> dt = open_datatree("sweep.nc")
        >>> dt.kids.sweep.S21              # Access sweep data

    Open DataTree with D21 analysis results:
        >>> dt = open_datatree("sweep_with_d21.nc")
        >>> dt.kids.sweep.S21              # Access root sweep data
        >>> dt.kids.d21.matched            # Access D21 child node
        >>> list(dt.children.keys())       # Show child nodes
        ['tolteca_kidsproc.analysis.d21']
    """
    return xr.open_datatree(*args, **kwargs)  # type: ignore[return-value]


def make_kids_dataset(  # noqa: C901
    i_data: npt.NDArray[np.floating],
    q_data: npt.NDArray[np.floating],
    frequency: Quantity | None = None,
    time: Quantity | None = None,
    *,
    mapper: KidsMapper | None = None,
) -> KidsDataset:
    """Create a KIDs dataset with I, Q, and optional frequency and/or time data.

    Parameters
    ----------
    i_data : ndarray
        In-phase component data. Can be 1-D (single channel) or 2-D (multi-channel).
    q_data : ndarray
        Quadrature component data. Must match i_data shape.
    frequency : Quantity, optional
        Frequency data as astropy Quantity with any frequency unit (Hz, MHz, GHz, etc.).
        Must match i_data shape.
    time : Quantity, optional
        Time data as astropy Quantity with any time unit (s, ms, etc.).
        Must match i_data shape.
    mapper : KidsMapper, optional
        Mapper instance to use for populating dataset fields.
        If not provided, uses KidsMapper.from_defaults().

    Returns
    -------
    KidsDataset
        Dataset with I, Q data variables and optional coordinates.
        - 1-D timestream: dims=["time"], coords=["time"]
        - 2-D timestream: dims=["chan", "time"], coords=["time"]
        - 1-D time + frequency: dims=["time"], coords=["time", "frequency"]
        - 2-D time + frequency: dims=["chan", "time"], coords=["time", "frequency"]
        - 1-D sweep: dims=["sweep"], coords=["frequency"]
        - 2-D sweep: dims=["chan", "sweep"], coords=["frequency"]
        - 1-D S21-only: dims=["sample"], no coords
        - 2-D S21-only: dims=["chan", "sample"], no coords
    """
    # Validate inputs
    if i_data.shape != q_data.shape:
        msg = f"I and Q data must have same shape: {i_data.shape} != {q_data.shape}"
        raise ValueError(msg)

    # Use provided mapper or create default
    if mapper is None:
        mapper = KidsMapper.from_defaults()

    schema = mapper.schema
    i_key = mapper.get_name(schema.I)
    q_key = mapper.get_name(schema.Q)
    f_key = mapper.get_name(schema.frequency)
    t_key = mapper.get_name(schema.time)

    ndim = i_data.ndim

    def _get_dims(dim_name: str) -> list[str]:
        if ndim == 1:
            return [dim_name]
        if ndim == 2:  # noqa: PLR2004
            return ["chan", dim_name]
        msg = f"Data must be 1-D or 2-D, got {ndim}-D"
        raise ValueError(msg)

    def _make_coord(coord: Quantity, dim_name: str):  # noqa: ANN202
        if coord.shape != i_data.shape:
            msg = f"Frequency must match I/Q shape: {coord.shape} != {i_data.shape}"
            raise ValueError(msg)
        dims = _get_dims(dim_name)
        return (dims, coord.value, {"units": str(coord.unit)}), dims

    # Determine coordinate and dimensions based on what's provided
    coords = {}
    dims = None

    if time is not None:
        if time.unit is not None and time.unit.physical_type != "time":
            msg = "time must have time units"
            raise ValueError(msg)
        # reuse the frequency dims if provided
        coord, dims = _make_coord(time, "time")
        coords[t_key] = coord

    if frequency is not None:
        if frequency.unit is not None and frequency.unit.physical_type != "frequency":
            msg = "frequency must have frequency units"
            raise ValueError(msg)

        # reuse the time dims if provided
        coord, dims = _make_coord(frequency, "sweep" if dims is None else dims[-1])
        coords[f_key] = coord

    if not dims:
        # S21-only dataset (no frequency or time)
        dims = _get_dims("sample")

    return xr.Dataset(  # type: ignore[return-value]
        {
            i_key: (dims, i_data),
            q_key: (dims, q_data),
        },
        coords=coords,
    )
