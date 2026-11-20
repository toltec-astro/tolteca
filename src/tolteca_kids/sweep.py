"""TolTEC sweep reduction and data views."""

# ruff: noqa: E741, E743, N802, N815 -- Preserve physical I/Q names.

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from tollan.accessor import NameMapping, Schema
from tollan.accessor.xarray import (
    XarrayAccessorBase,
    XarrayMapper,
    dataset_to_qtable,
    unwrap_data_source,
)

from tolteca_datamodels.toltec.metadata import ToltecSweepMetadata

from .blocks import NODE_TYPE

if TYPE_CHECKING:
    from astropy.table import QTable
    from tollan.accessor import Mapping

__all__ = [
    "ReducedSweepView",
    "ToltecSweepMapper",
    "ToltecSweepSchema",
]


# Schema for Reduced Sweep Data
# ==============================


@dataclass
class ToltecSweepSchema(Schema):
    """Schema for reduced TolTEC KIDs sweep data."""

    # Reduced data variables
    I: Mapping = NameMapping.var("I")
    Q: Mapping = NameMapping.var("Q")
    unc_I: Mapping = NameMapping.var("unc_I")
    unc_Q: Mapping = NameMapping.var("unc_Q")

    # Frequency and channel coordinates
    sweep: Mapping = NameMapping.var("sweep")
    chan: Mapping = NameMapping.var(("chan", "channel", "n_chans", "iqlen"))
    f_chan: Mapping = NameMapping.var("f_chan")

    # LO frequencies
    f_lo: Mapping = NameMapping.var("f_lo")
    f_center: Mapping = NameMapping.attr("f_center")

    # Metadata and producing configuration
    meta: Mapping = NameMapping.attr("meta")
    reduction_config: Mapping = NameMapping.attr("tolteca_kids.sweep_reduce.config")


class ToltecSweepMapper(XarrayMapper[ToltecSweepSchema]):
    """Mapper for reduced TolTEC KIDs sweep data."""


# View for Reduced Sweep Data
# ============================


class ReducedSweepView(XarrayAccessorBase[ToltecSweepMapper]):
    """View of reduced TolTEC sweep data.

    Parameters
    ----------
    data_source : xarray.Dataset or xarray.DataTree
        Reduced dataset or tree containing reduced sweep results.
    mapper : ToltecSweepMapper, optional
        Field mapper. If omitted, resolve fields against the reduced dataset.

    Examples
    --------
    >>> from tollan.accessor.xarray import ensure_datatree
    >>> from tolteca_kids import SweepReduce
    >>> dt = ensure_datatree(ds_raw)
    >>> _ = SweepReduce()(dt)
    >>> view = dt.toltec_kids.blocks[0].sweep
    >>> I_data = view.I
    >>> unc_I_data = view.unc_I

    Select channels before constructing the frequency grid:

    >>> chan_dim = view.f_chan.dims[-1]
    >>> subset = ReducedSweepView(view.data_source.isel({chan_dim: [0, 1]}))
    >>> frequency = subset.frequency
    """

    @functools.cached_property
    def data_source(self) -> xr.Dataset | xr.DataArray:
        """Reduced sweep dataset."""
        if isinstance(self._data_source, xr.DataTree):
            node = self._data_source
            if node.attrs.get("kids_node_type") == "sweep":
                return node.to_dataset()
            if "sweep" in node.children:
                return node["sweep"].to_dataset()
            if node.attrs.get(NODE_TYPE) in {"acquisition", "blocks"}:
                msg = "Select a block before viewing its reduced sweep"
                raise ValueError(msg)
            return node.to_dataset()
        return unwrap_data_source(self._data_source)

    @functools.cached_property
    def meta(self) -> ToltecSweepMetadata | None:
        """Complete source block metadata, when present on the reduced Dataset."""
        node = self._data_source
        if isinstance(node, xr.DataTree):
            if node.attrs.get("kids_node_type") == "sweep":
                node = node.parent
            if node is not None and node.attrs.get("kids_node_type") == "block":
                return node.toltec_kids.block.meta
        if not self.mapper.has_attr(self.data_source, self.mapper.schema.meta):
            return None
        return ToltecSweepMetadata.model_validate_json(
            self.mapper.get_value(self.data_source, self.mapper.schema.meta)
        )

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
    def f_chan(self) -> xr.DataArray | None:
        """Channel frequencies in Hz, aligned with the reduced I/Q channel axis."""
        if self.mapper.schema.f_chan not in self.mapper:
            return None
        return self.mapper.get_arr(self.data_source, self.mapper.schema.f_chan)

    @property
    def chan_axis_table(self) -> QTable:
        """Recorded probe properties aligned with this view's channel selection.

        Native and reopened block views use their channel axis. Standalone
        reduced datasets must supply ``f_chan``, ``tone_amp`` and ``tone_mask``
        on the signal's channel dimension. Missing probe properties raise
        ``ValueError``; they are not replaced with unit amplitudes.
        """
        signal = self.I
        if signal is None or self.f_chan is None:
            msg = "Probe properties require reduced signals and channel frequencies"
            raise ValueError(msg)
        dim = self.f_chan.dims[0]
        assert isinstance(dim, str)
        ids = np.asarray(signal.coords[dim])
        meta = self.meta
        if meta is not None and meta.chan_axis_table is not None:
            axis = meta.chan_axis_table
            positions = {int(chan): i for i, chan in enumerate(axis["chan"])}
            return axis[[positions[int(chan)] for chan in ids]]
        required = {"f_chan", "tone_amp", "tone_mask"}
        missing = required.difference(self.data_source.variables)
        if missing:
            msg = f"Missing recorded probe properties: {sorted(missing)}"
            raise ValueError(msg)
        return dataset_to_qtable(
            self.data_source,
            dim=dim,
            columns={"chan": dim, **{name: name for name in sorted(required)}},
        )

    @property
    def frequency(self) -> xr.DataArray | None:
        """Per-channel sweep frequencies in Hz, in reduced I/Q dimension order."""
        f_chan = self.f_chan
        sweep = self.sweep
        signal = self.I
        if f_chan is None or sweep is None or signal is None:
            return None
        frequency = (f_chan + sweep).transpose(*signal.dims)
        frequency.attrs["units"] = "Hz"
        return frequency
