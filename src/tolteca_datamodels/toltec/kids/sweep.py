"""TolTEC sweep data schema, reducer, and views.

This module provides the schema for reduced sweep data with proper namespacing
to avoid conflicts, following the D21Schema pattern. The schema uses __name__
for namespacing reduced data fields.

Schema fields are namespaced as: tolteca_datamodels.toltec.kids.sweep.*
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import xarray as xr
from pydantic import Field
from pydantic.dataclasses import dataclass
from tollan.accessor import Mapping, Schema
from tollan.accessor.xarray import XarrayAccessorBase, XarrayMapper, ensure_dataset
from tollan.config import FrozenBaseModel

if TYPE_CHECKING:
    from typing import Self

__all__ = [
    "ReducedSweepView",
    "SweepReducer",
    "ToltecSweepMapper",
    "ToltecSweepSchema",
]


# Schema for Reduced Sweep Data
# ==============================


@dataclass
class ToltecSweepSchema(Schema):
    """Schema for reduced TolTEC KIDs sweep data.

    This schema maps to reduced sweep data variables with proper namespacing
    to avoid conflicts. Uses __name__ prefix for data variables following
    the D21Schema pattern.

    **Reduced Data Variables (Namespaced):**
    - I, Q: Reduced in-phase/quadrature (mean per sweep step)
    - unc_I, unc_Q: Uncertainties (std per sweep step)
    - r, x: Solved timestream components

    **Coordinates:**
    - sweep: Frequency offset from LO center
    - block: Block index for multi-block data
    - chan: Channel index

    **LO/Frequency:**
    - f_lo: LO frequency array
    - f_center: Center frequency

    **Metadata:**
    - reducer_config: SweepReducer configuration

    Notes
    -----
    Fields use __name__ for namespacing, resulting in:
    `tolteca_datamodels.toltec.kids.sweep.I`, etc.

    Coordinates (sweep, block, chan, f_lo, f_center) do NOT use namespacing
    as they are shared/standard coordinates.
    """

    # Reduced data variables (namespaced)
    I: Mapping = Mapping(f"{__name__}.I")
    Q: Mapping = Mapping(f"{__name__}.Q")
    unc_I: Mapping = Mapping(f"{__name__}.unc_I")
    unc_Q: Mapping = Mapping(f"{__name__}.unc_Q")

    # Solved timestream (namespaced)
    r: Mapping = Mapping(f"{__name__}.r")
    x: Mapping = Mapping(f"{__name__}.x")

    # Coordinates (not namespaced - shared)
    sweep: Mapping = Mapping("sweep")
    block: Mapping = Mapping("block")
    chan: Mapping = Mapping(("chan", "channel", "n_chans"))

    # LO frequency data (not namespaced)
    f_lo: Mapping = Mapping("f_lo")
    f_center: Mapping = Mapping("f_center")

    # Metadata (namespaced)
    reducer_config: Mapping = Mapping(f"{__name__}.reducer_config")


class ToltecSweepMapper(XarrayMapper[ToltecSweepSchema]):
    """Mapper for reduced TolTEC KIDs sweep data."""



# Module-level namespace constant (matches schema field prefix)
_SWEEP_NAMESPACE: str = __name__  # "tolteca_datamodels.toltec.kids.sweep"


# View for Reduced Sweep Data
# ============================


class ReducedSweepView(XarrayAccessorBase[ToltecSweepMapper]):
    """View for reduced TolTEC sweep data.

    Provides property-based access to reduced sweep data with blocks
    and uncertainties. Transparently resolves DataTree child nodes —
    pass either a flat Dataset or a DataTree returned by SweepReducer.

    Examples
    --------
    >>> dt = SweepReducer()(ds_raw)  # returns DataTree
    >>> view = ReducedSweepView(dt)  # auto-resolves child node
    >>> I_data = view.I  # Returns namespaced DataArray
    >>> unc_I_data = view.unc_I
    """

    @functools.cached_property
    def data_source(self) -> xr.Dataset:
        """Resolve to reduced sweep Dataset.

        If given a DataTree (from SweepReducer), resolves to the sweep
        child node at ``_SWEEP_NAMESPACE``. Falls back to root dataset
        or wraps a plain Dataset.
        """
        if isinstance(self._data_source, xr.DataTree):
            if _SWEEP_NAMESPACE in self._data_source.children:
                return self._data_source.children[_SWEEP_NAMESPACE].dataset
            return self._data_source.dataset
        return ensure_dataset(self._data_source)

    @property
    def I(self) -> xr.DataArray | None:
        """Reduced in-phase data (mean per sweep step)."""
        if self.mapper.schema.I not in self.mapper:
            return None
        return self.mapper.get_arr(self.data_source, self.mapper.schema.I)

    @property
    def Q(self) -> xr.DataArray | None:
        """Reduced quadrature data (mean per sweep step)."""
        if self.mapper.schema.Q not in self.mapper:
            return None
        return self.mapper.get_arr(self.data_source, self.mapper.schema.Q)

    @property
    def unc_I(self) -> xr.DataArray | None:
        """In-phase uncertainty (std per sweep step)."""
        if self.mapper.schema.unc_I not in self.mapper:
            return None
        return self.mapper.get_arr(self.data_source, self.mapper.schema.unc_I)

    @property
    def unc_Q(self) -> xr.DataArray | None:
        """Quadrature uncertainty (std per sweep step)."""
        if self.mapper.schema.unc_Q not in self.mapper:
            return None
        return self.mapper.get_arr(self.data_source, self.mapper.schema.unc_Q)

    @property
    def sweep(self) -> xr.DataArray | None:
        """Sweep frequency coordinate."""
        if self.mapper.schema.sweep not in self.mapper:
            return None
        return self.mapper.get_arr(self.data_source, self.mapper.schema.sweep)

    @property
    def f_lo(self) -> xr.DataArray | None:
        """LO frequency array."""
        if self.mapper.schema.f_lo not in self.mapper:
            return None
        return self.mapper.get_arr(self.data_source, self.mapper.schema.f_lo)

    @property
    def has_blocks(self) -> bool:
        """Check if data has block dimension (multi-block)."""
        return "block" in self.data_source.dims

    @property
    def n_blocks(self) -> int:
        """Number of blocks (1 if single block)."""
        if not self.has_blocks:
            return 1
        return self.data_source.sizes["block"]


class SweepReducer(FrozenBaseModel):
    """Reduce raw TolTEC sweep data to mean/std per sweep step.

    Similar to D21Analysis - configurable with pydantic, cacheable.
    Processes raw sweep data (multiple samples per frequency step) into
    reduced data with mean and uncertainty per sweep step.

    Automatically detects and handles multi-block data (e.g., tune files
    with multiple sweeps).

    Parameters
    ----------
    detect_blocks : bool, default True
        Auto-detect multi-block data by finding breaks in LO frequency
    compute_uncertainty : bool, default True
        Compute standard deviation as uncertainty
    sweep_axis : str or None, default None
        Name of sweep axis variable (auto-detect from LO freq if None)
    time_axis : str, default "time"
        Name of time/sample axis dimension

    Examples
    --------
    Basic reduction with defaults:

    >>> import xarray as xr
    >>> ds_raw = xr.open_dataset("vnasweep.nc")
    >>> reducer = SweepReducer()
    >>> dt = reducer(ds_raw)
    >>> dt.children  # {'tolteca_datamodels.toltec.kids.sweep': DataTree}

    Multi-block data (tune file):

    >>> ds_raw = xr.open_dataset("tune.nc")
    >>> reducer = SweepReducer(detect_blocks=True)
    >>> dt = reducer(ds_raw)
    >>> ds_reduced = dt.children["tolteca_datamodels.toltec.kids.sweep"].dataset
    >>> ds_reduced.dims  # {'block': 3, 'chan': 1000, 'sweep': 491}

    Access via view (pass DataTree directly):

    >>> view = ReducedSweepView(dt)
    >>> view.I  # auto-resolves child node

    Skip re-reduction if already processed:

    >>> dt2 = reducer(dt, force=False)
    >>> # Returns immediately without reprocessing
    """

    detect_blocks: bool = Field(
        default=True,
        description="Auto-detect multi-block data (tune files)",
    )

    compute_uncertainty: bool = Field(
        default=True,
        description="Compute std dev as uncertainty",
    )

    sweep_axis: str | None = Field(
        default=None,
        description="Name of sweep axis variable (auto-detect if None)",
    )

    time_axis: str = Field(
        default="time",
        description="Name of time/sample axis dimension",
    )

    _mapper: ClassVar[ToltecSweepMapper] = ToltecSweepMapper.from_defaults()

    def _is_already_reduced(self, ds: xr.Dataset) -> bool:
        """Check if dataset already contains reduced data with same config."""
        config_key = self._mapper.get_name(self._mapper.schema.reducer_config)
        if config_key not in ds.attrs:
            return False

        # Compare configurations
        stored_config = self.model_validate_json(ds.attrs[config_key])
        return stored_config == self

    def _dump_to_attr(self, ds: xr.Dataset) -> None:
        """Save reducer configuration to dataset attributes."""
        config_key = self._mapper.get_name(self._mapper.schema.reducer_config)
        ds.attrs[config_key] = self.model_dump_json()

    @classmethod
    def _load_from_attr(cls, ds: xr.Dataset) -> Self | None:
        """Load reducer configuration from dataset attributes."""
        mapper = cls._mapper
        config_key = mapper.get_name(mapper.schema.reducer_config)
        if config_key not in ds.attrs:
            return None
        return cls.model_validate_json(ds.attrs[config_key])

    def __call__(
        self,
        ds: xr.Dataset | xr.DataTree,
        *,
        force: bool = False,
    ) -> xr.DataTree:
        """Reduce raw sweep data, returning a DataTree.

        The returned DataTree has the raw dataset at the root and the
        reduced sweep dataset as a child at ``_SWEEP_NAMESPACE``.

        Parameters
        ----------
        ds : xr.Dataset or xr.DataTree
            Raw sweep dataset (or DataTree from a previous call)
        force : bool, default False
            If True, reprocess even if already reduced with same config

        Returns
        -------
        xr.DataTree
            DataTree with root = raw data, child at
            ``tolteca_datamodels.toltec.kids.sweep`` = reduced data

        Raises
        ------
        ValueError
            If sweep axis cannot be found or data is invalid
        """
        raw_ds = ds.dataset if isinstance(ds, xr.DataTree) else ds
        existing_dt = ds if isinstance(ds, xr.DataTree) else xr.DataTree(dataset=raw_ds)

        # Return early if already reduced with same config
        if not force and _SWEEP_NAMESPACE in existing_dt.children:
            child_ds = existing_dt.children[_SWEEP_NAMESPACE].dataset
            if self._is_already_reduced(child_ds):
                return existing_dt

        # Perform reduction
        ds_reduced = self._reduce(raw_ds)

        # Store config in attrs of reduced dataset
        self._dump_to_attr(ds_reduced)

        return xr.DataTree(
            dataset=raw_ds,
            children={_SWEEP_NAMESPACE: xr.DataTree(dataset=ds_reduced)},
        )

    def _reduce(self, ds: xr.Dataset) -> xr.Dataset:
        """Perform the actual reduction logic.

        Migrated from io.py reduce_raw_sweep() and adapted to use
        namespaced schema.

        Returns reduced dataset with namespaced variable names.
        """
        # Determine sweep axis variable
        if self.sweep_axis is None:
            # Try to find LO frequency variable
            if "Data.Toltec.LoFreq" in ds:
                f_lo_var = "Data.Toltec.LoFreq"
            elif "f_lo" in ds:
                f_lo_var = "f_lo"
            else:
                msg = "Cannot find LO frequency variable for sweep reduction"
                raise ValueError(msg)
        else:
            f_lo_var = self.sweep_axis

        # Get LO frequencies
        f_lo = ds[f_lo_var].values
        if f_lo.ndim > 1:
            # Take first row if multi-dimensional
            f_lo = f_lo[0]

        # Validate LO frequencies
        if not np.all(f_lo > 0):
            msg = "Invalid LO frequency found (f_lo <= 0)"
            raise ValueError(msg)

        # Detect blocks if enabled
        if self.detect_blocks:
            # Find breaks in LO frequency (where frequency decreases)
            break_indices = np.where(np.diff(f_lo) < 0)[0] + 1

            if break_indices.size == 0:
                # Single block
                block_slices = [slice(0, len(f_lo))]
            else:
                # Multiple blocks
                block_starts = [0] + break_indices.tolist() + [len(f_lo)]
                block_slices = [
                    slice(block_starts[i], block_starts[i + 1])
                    for i in range(len(block_starts) - 1)
                ]
        else:
            # Treat as single block
            block_slices = [slice(0, len(f_lo))]

        n_blocks = len(block_slices)

        # Process each block independently
        reduced_blocks = []
        for iblock, block_slice in enumerate(block_slices):
            ds_block_reduced = self._reduce_single_block(
                ds=ds,
                block_slice=block_slice,
                f_lo_var=f_lo_var,
                block_index=iblock,
            )
            reduced_blocks.append(ds_block_reduced)

        # Combine blocks
        if n_blocks == 1:
            # Single block - return as-is
            return reduced_blocks[0]

        # Multiple blocks - concatenate along 'block' dimension
        ds_reduced = xr.concat(
            reduced_blocks,
            dim="block",
            coords="minimal",
            compat="override",
        )

        # Add block coordinate
        ds_reduced = ds_reduced.assign_coords(
            block=xr.DataArray(
                np.arange(n_blocks),
                dims=["block"],
                attrs={
                    "long_name": "Block index",
                    "description": "Monotonic sweep block index",
                },
            ),
        )

        # Update attrs to indicate multi-block
        ds_reduced.attrs["n_blocks"] = n_blocks
        ds_reduced.attrs["is_multi_block"] = True

        return ds_reduced

    def _reduce_single_block(
        self,
        ds: xr.Dataset,
        block_slice: slice,
        f_lo_var: str,
        block_index: int,
    ) -> xr.Dataset:
        """Reduce a single block of raw sweep data.

        Parameters
        ----------
        ds : xr.Dataset
            Raw sweep dataset
        block_slice : slice
            Slice for this block in the time dimension
        f_lo_var : str
            Name of LO frequency variable
        block_index : int
            Index of this block

        Returns
        -------
        xr.Dataset
            Reduced dataset for this block with namespaced variables
        """
        # Slice dataset to this block
        ds_block = ds.isel({self.time_axis: block_slice})

        # Get LO frequencies for this block
        f_lo = ds_block[f_lo_var].values
        if f_lo.ndim > 1:
            f_lo = f_lo[0]

        # Get unique frequencies and their sample indices
        unique_f_lo, inverse_indices, counts = np.unique(
            f_lo,
            return_inverse=True,
            return_counts=True,
        )

        # Get center frequency for computing sweep offsets
        if "Header.Toltec.LoCenterFreq" in ds_block.attrs:
            f_lo_center = ds_block.attrs["Header.Toltec.LoCenterFreq"]
        elif "Header.Toltec.LoCenterFreq" in ds_block:
            f_lo_center = ds_block["Header.Toltec.LoCenterFreq"].values.item()
        elif "f_center" in ds_block.attrs:
            f_lo_center = ds_block.attrs["f_center"]
        else:
            # Use median as fallback
            f_lo_center = np.median(unique_f_lo)

        f_sweep = unique_f_lo - f_lo_center
        n_sweeps = len(unique_f_lo)

        # Get I/Q data dimensions
        if "Data.Toltec.Is" in ds_block:
            I_var, Q_var = "Data.Toltec.Is", "Data.Toltec.Qs"
        elif "I" in ds_block:
            I_var, Q_var = "I", "Q"
        else:
            msg = "Cannot find I/Q data variables"
            raise ValueError(msg)

        I_data = ds_block[I_var].values
        Q_data = ds_block[Q_var].values

        # Determine channel dimension (the one that's not time)
        data_dims = ds_block[I_var].dims
        chan_dim = [d for d in data_dims if d != self.time_axis][0]
        n_chans = ds_block.sizes[chan_dim]

        # Allocate output arrays
        I_reduced = np.full((n_chans, n_sweeps), np.nan, dtype=np.float64)
        Q_reduced = np.full((n_chans, n_sweeps), np.nan, dtype=np.float64)

        if self.compute_uncertainty:
            unc_I_reduced = np.full((n_chans, n_sweeps), np.nan, dtype=np.float64)
            unc_Q_reduced = np.full((n_chans, n_sweeps), np.nan, dtype=np.float64)

        # Compute mean and std for each sweep step
        for i_sweep in range(n_sweeps):
            # Find samples belonging to this sweep step
            mask = inverse_indices == i_sweep

            # Get samples for this sweep step (axis ordering: time, chan or chan, time)
            if data_dims[0] == self.time_axis:
                # (time, chan) ordering
                I_samples = I_data[mask, :]
                Q_samples = Q_data[mask, :]
                # Mean over time axis (axis 0)
                I_reduced[:, i_sweep] = np.mean(I_samples, axis=0)
                Q_reduced[:, i_sweep] = np.mean(Q_samples, axis=0)
                if self.compute_uncertainty:
                    unc_I_reduced[:, i_sweep] = np.std(I_samples, axis=0)
                    unc_Q_reduced[:, i_sweep] = np.std(Q_samples, axis=0)
            else:
                # (chan, time) ordering
                I_samples = I_data[:, mask]
                Q_samples = Q_data[:, mask]
                # Mean over time axis (axis 1)
                I_reduced[:, i_sweep] = np.mean(I_samples, axis=1)
                Q_reduced[:, i_sweep] = np.mean(Q_samples, axis=1)
                if self.compute_uncertainty:
                    unc_I_reduced[:, i_sweep] = np.std(I_samples, axis=1)
                    unc_Q_reduced[:, i_sweep] = np.std(Q_samples, axis=1)

        # Create new dataset with reduced data using namespaced names
        sweep_coord = xr.DataArray(
            f_sweep,
            dims=["sweep"],
            attrs={
                "long_name": "Sweep frequency offset",
                "units": "Hz",
                "description": "Frequency offset from LO center frequency",
            },
        )

        # Preserve channel coordinate
        if chan_dim in ds_block.coords:
            chan_coord = ds_block.coords[chan_dim]
        else:
            chan_coord = xr.DataArray(
                np.arange(n_chans),
                dims=[chan_dim],
                attrs={"long_name": "Channel index"},
            )

        # Create data variables with namespaced names
        I_key = self._mapper.get_name(self._mapper.schema.I)
        Q_key = self._mapper.get_name(self._mapper.schema.Q)

        data_vars = {
            I_key: xr.DataArray(
                I_reduced,
                dims=[chan_dim, "sweep"],
                attrs={"long_name": "In-phase component (mean)", "units": "ADU"},
            ),
            Q_key: xr.DataArray(
                Q_reduced,
                dims=[chan_dim, "sweep"],
                attrs={"long_name": "Quadrature component (mean)", "units": "ADU"},
            ),
            "f_lo": xr.DataArray(
                unique_f_lo,
                dims=["sweep"],
                attrs={"long_name": "LO frequency", "units": "Hz"},
            ),
        }

        # Add uncertainty if computed
        if self.compute_uncertainty:
            unc_I_key = self._mapper.get_name(self._mapper.schema.unc_I)
            unc_Q_key = self._mapper.get_name(self._mapper.schema.unc_Q)
            data_vars[unc_I_key] = xr.DataArray(
                unc_I_reduced,
                dims=[chan_dim, "sweep"],
                attrs={"long_name": "In-phase uncertainty (std)", "units": "ADU"},
            )
            data_vars[unc_Q_key] = xr.DataArray(
                unc_Q_reduced,
                dims=[chan_dim, "sweep"],
                attrs={"long_name": "Quadrature uncertainty (std)", "units": "ADU"},
            )

        # Create coordinates
        coords = {chan_dim: chan_coord, "sweep": sweep_coord}

        # Copy relevant attributes from original dataset
        attrs = dict(ds_block.attrs)
        attrs["processing"] = "reduced_raw_sweep"
        attrs["n_sweeps"] = n_sweeps
        attrs["f_center"] = f_lo_center
        attrs["block_index"] = block_index

        # Store raw data info in attrs
        attrs["raw_I_shape"] = I_data.shape
        attrs["raw_Q_shape"] = Q_data.shape
        attrs["samples_per_sweep"] = counts.tolist()

        # Create new dataset
        ds_reduced = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        # Copy over metadata variables that don't have time dimension
        for var_name, var_data in ds_block.data_vars.items():
            if self.time_axis not in var_data.dims and var_name not in ds_reduced:
                # Check if this variable has a numSweeps-like dimension
                if "numSweeps" in var_data.dims:
                    # This is block-dependent metadata - select the appropriate block
                    ds_reduced[var_name] = var_data.isel(numSweeps=block_index)
                else:
                    # Regular metadata - copy as-is
                    ds_reduced[var_name] = var_data

        return ds_reduced
