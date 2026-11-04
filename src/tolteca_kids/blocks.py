"""Acquisition block index, lazy subtree views, and compact persistence."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from tollan.accessor import NameMapping, Schema
from tollan.accessor.xarray import XarrayAccessorBase, XarrayMapper, dataset_to_qtable

from tolteca_datamodels.toltec.axes import ToltecAxisView
from tolteca_datamodels.toltec.metadata import (
    ToltecRawObsBlocksMetadata,
    ToltecSweepMetadata,
    ToltecTimestreamMetadata,
)
from tolteca_datamodels.toltec.types import ToltecDataKind

if TYPE_CHECKING:
    from typing import Any

    from tollan.accessor import Mapping

__all__ = [
    "ToltecBlockView",
    "ToltecBlocksMapper",
    "ToltecBlocksSchema",
    "ToltecBlocksView",
]

NODE_TYPE = "kids_node_type"


def _drop_timestream_projections(tree):
    """Remove reconstructible time samples while retaining reduction metadata."""
    for node in tree.subtree:
        if node.attrs.get(NODE_TYPE) == "block":
            for child in node.children.values():
                if child.attrs.get(NODE_TYPE) == "timestream":
                    child.dataset = xr.Dataset(attrs=child.attrs)


@dataclass
class ToltecBlocksSchema(Schema):
    """Small persisted index locating block payloads and axis rows."""

    block: Mapping = NameMapping.var("block")
    n_blocks: Mapping = NameMapping.dim("block")
    data_kind: Mapping = NameMapping.var("data_kind")
    n_sweepsteps: Mapping = NameMapping.var("n_sweepsteps")
    sample_start: Mapping = NameMapping.var("sample_start")
    sample_end: Mapping = NameMapping.var("sample_end")
    chan_start: Mapping = NameMapping.var("chan_start")
    chan_end: Mapping = NameMapping.var("chan_end")
    probe_start: Mapping = NameMapping.var("probe_start")
    probe_end: Mapping = NameMapping.var("probe_end")


class ToltecBlocksMapper(XarrayMapper[ToltecBlocksSchema]):
    """Mapper for the acquisition block index."""


class _AxisTables:
    tree: xr.DataTree
    meta: Any

    @cached_property
    def chan_axis(self):
        """Columnar channel-axis view."""
        return ToltecAxisView(self.tree["chan"])

    @cached_property
    def probe_axis(self):
        """Columnar probe-axis view."""
        return ToltecAxisView(self.tree["probe"])

    @cached_property
    def chan_axis_table(self):
        """Quantity-aware channel table, materialized on first access."""
        return self._table(self.chan_axis)

    @cached_property
    def probe_axis_table(self):
        """Quantity-aware probe table, materialized on first access."""
        return self._table(self.probe_axis)

    def _table(self, axis):
        table = axis.axis_table
        if hasattr(self.meta, "block"):
            table.meta.update(
                roach=self.meta.roach,
                array_name=self.meta.array_name,
                block=self.meta.block,
                f_lo_center=self.meta.f_lo_center,
            )
        return table


class ToltecBlocksView(_AxisTables, XarrayAccessorBase[ToltecBlocksMapper]):
    """Collection of lazily selected acquisition blocks.

    Parameters
    ----------
    data_source : xarray.DataTree
        Acquisition tree or its ``blocks`` node. Only the small index is loaded
        on construction. Shared axes and detector data remain lazy on reopen.
    mapper : ToltecBlocksMapper, optional
        Existing index mapping.
    """

    def __init__(self, data_source, mapper=None):
        if data_source.attrs.get(NODE_TYPE) == "acquisition":
            data_source = data_source["blocks"]
        if data_source.attrs.get(NODE_TYPE) != "blocks":
            msg = "Expected an acquisition or blocks tree"
            raise ValueError(msg)
        super().__init__(data_source, mapper)
        self._views = {}
        # A structural index is small and explicitly eager, including remotely.
        self.data_source.load()
        _ = self.meta
        ids = self.data_source["block"].values.tolist()
        if len(ids) != len(set(ids)) or len(ids) != self.meta.n_blocks:
            msg = "Block index identities/count disagree with collection meta"
            raise ValueError(msg)
        self._positions = {int(block): pos for pos, block in enumerate(ids)}
        for axis, dim in (("chan", "chan_entry"), ("probe", "probe")):
            starts = self.data_source[f"{axis}_start"].values
            ends = self.data_source[f"{axis}_end"].values
            size = self.tree[axis].sizes[dim]
            if np.any(starts < 0) or np.any(ends < starts) or np.any(ends > size):
                msg = f"Block index contains invalid {axis} row bounds"
                raise ValueError(msg)

    @property
    def tree(self):
        """Collection node, holding metadata, shared axes and block children."""
        return self._data_source

    @cached_property
    def data_source(self):
        """Eager block-index Dataset."""
        return self.tree["index"].dataset.copy(deep=False)

    @cached_property
    def meta(self) -> ToltecRawObsBlocksMetadata:
        """Acquisition metadata, block index and shared axes."""
        meta = ToltecRawObsBlocksMetadata.model_validate_json(self.tree.attrs["meta"])
        meta._block_index = self.data_source
        meta._chan_axis = self.chan_axis
        meta._probe_axis = self.probe_axis
        return meta

    @property
    def n_blocks(self):
        """Number of discovered blocks."""
        return self.meta.n_blocks

    @property
    def block_index_table(self):
        """One row per block, with kinds and exclusive selection bounds."""
        return self.meta.block_index_table

    def __len__(self):
        return self.n_blocks

    def __iter__(self):
        for block in self._positions:
            yield self[block]

    def __getitem__(self, block):
        if block not in self._positions:
            msg = f"Block {block} out of range for this collection"
            raise ValueError(msg)
        if block not in self._views:
            view = ToltecBlockView(self.tree[str(block)], parent=self)
            if view.meta.n_chans != self.meta.n_chans:
                msg = "Readout channel count cannot change across blocks"
                raise ValueError(msg)
            row = self._row(block)
            if (
                view.meta.block != block
                or view.meta.n_blocks != self.n_blocks
                or view.data_kind.value != int(row["data_kind"])
                or view.meta.n_times != int(row["sample_end"] - row["sample_start"])
            ):
                msg = "Block metadata disagrees with its index row"
                raise ValueError(msg)
            self._views[block] = view
        return self._views[block]

    def _row(self, block):
        return self.data_source.isel(block=self._positions[block])

    def detach(self, *, include_raw=True):
        """Detach the collection with the data needed by its blocks.

        Parameters
        ----------
        include_raw : bool, optional
            Include each block's raw selection (default True). False retains
            metadata, shared axes, model parameters and results without raw samples.

        Returns
        -------
        xarray.DataTree
            Independent collection subtree. Its blocks share the collection's
            axes and carry local raw leaves, requiring no acquisition parent.
            The original tree is unchanged. Arrays remain lazy and borrowed
            source files must stay open until the arrays are loaded or exported.
        """
        result = self.tree.copy(deep=False)
        for block in self._positions:
            node = result[str(block)]
            if not include_raw:
                params = self[block].model_params
                if params is not None:
                    node["model_params"] = xr.DataTree(dataset=params)
            if include_raw:
                node["raw"] = xr.DataTree(dataset=self[block].raw)
            elif "raw" in node.children:
                del node["raw"]
        if not include_raw:
            _drop_timestream_projections(result)
        return result


class ToltecBlockView(_AxisTables):
    """One block with local metadata and lazy payload/axis selections.

    Parameters
    ----------
    data_source : xarray.DataTree
        Block node, either attached to a collection or independently exported.
    parent : ToltecBlocksView, optional
        Collection providing shared arrays and index bounds. A detached block
        needs its own ``raw``, ``chan`` and ``probe`` leaves for those accesses.
    """

    def __init__(self, data_source, *, parent=None):
        if data_source.attrs.get(NODE_TYPE) != "block":
            msg = "Expected a block tree"
            raise ValueError(msg)
        self.tree = data_source
        self.parent = parent
        _ = self.meta

    @property
    def data_source(self):
        """Attributes-only block Dataset."""
        return self.tree.dataset

    @cached_property
    def meta(self) -> ToltecSweepMetadata | ToltecTimestreamMetadata:
        """Complete block metadata, including inherited acquisition fields."""
        kind = ToltecDataKind(self.tree.attrs["data_kind"])
        model = (
            ToltecSweepMetadata
            if kind & ToltecDataKind.sweep
            else ToltecTimestreamMetadata
        )
        meta = model.model_validate_json(self.tree.attrs["meta"])
        meta._block_view = self
        if self.parent is not None:
            meta._block_index = self.parent.data_source
        return meta

    @property
    def data_kind(self):
        """Scientific data kind of this block."""
        return ToltecDataKind(self.tree.attrs["data_kind"])

    def _axis(self, name):
        if name in self.tree.children:
            return ToltecAxisView(self.tree[name])
        if self.parent is None:
            msg = f"Block {name} axis is not present in this subtree"
            raise ValueError(msg)
        row = self.parent._row(self.meta.block)
        dim = "chan_entry" if name == "chan" else "probe"
        start, end = (int(row[f"{name}_{bound}"]) for bound in ("start", "end"))
        return ToltecAxisView(
            self.parent.tree[name].dataset.isel({dim: slice(start, end)})
        )

    @cached_property
    def chan_axis(self):
        """Lazy channel rows for this block."""
        return self._axis("chan")

    @cached_property
    def probe_axis(self):
        """Lazy probe rows for this block."""
        return self._axis("probe")

    @cached_property
    def model_params(self):
        """Lazy recorded parameter columns with original channel IDs, or None."""
        if "model_params" in self.tree.children:
            return self.tree["model_params"].dataset.copy(deep=False)
        acquisition = self.parent.tree.parent if self.parent is not None else None
        if "raw" not in self.tree.children and (
            acquisition is None or "raw" not in acquisition.children
        ):
            return None
        source = self.raw
        return source.toltec_kids.mapper.get_model_params(source)

    @cached_property
    def model_params_table(self):
        """Materialized QTable with ``chan`` then recorded columns, or None."""
        params = self.model_params
        if params is None:
            return None
        return dataset_to_qtable(
            params,
            dim="chan",
            columns={name: name for name in ("chan", *params.attrs["parameter_names"])},
        )

    @property
    def raw(self):
        """Current lazy native payload; no detector arrays are loaded."""
        if "raw" in self.tree.children:
            return self.tree["raw"].dataset
        acquisition = self.parent.tree.parent if self.parent is not None else None
        if acquisition is None or "raw" not in acquisition.children:
            msg = "Block source data is not present in this tree"
            raise ValueError(msg)
        assert self.parent is not None
        ds = acquisition["raw"].dataset
        mapper = ds.toltec_kids.mapper
        schema = mapper.schema
        row = self.parent._row(self.meta.block)
        indexers = {}
        if mapper.has_dim(ds, schema.n_blocks):
            indexers[mapper.get_name(schema.n_blocks)] = self.parent._positions[
                self.meta.block
            ]
        elif mapper.has_dim(ds, schema.n_times):
            indexers[mapper.get_name(schema.n_times)] = slice(
                int(row["sample_start"]), int(row["sample_end"])
            )
        if mapper.has_dim(ds, schema.n_blocks_max):
            indexers[mapper.get_name(schema.n_blocks_max)] = self.parent._positions[
                self.meta.block
            ]
        return ds.isel(indexers)

    @property
    def sweep_check(self):
        """Saved quality flags and diagnostics for this sweep block."""
        return self.tree.toltec_kids.sweep_check

    @property
    def sweep(self):
        """Reduced sweep results beneath this block."""
        return self.tree.toltec_kids.sweep

    @property
    def timestream(self):
        """Time-domain signals and available spectra for this block."""
        return self.tree.toltec_kids.timestream

    def detach(self, *, include_raw=True):
        """Detach this block as a self-contained tree.

        Parameters
        ----------
        include_raw : bool, optional
            Include this block's native payload (default True). False allows
            axes, model parameters and results to be exported without raw data.

        Returns
        -------
        xarray.DataTree
            Detached subtree with complete metadata and selected axis leaves.
            The original tree is unchanged. Arrays remain lazy and share buffers
            where possible; keep borrowed source files open until loaded or
            exported. Detaching does not compact or discard analysis results.
        """
        result = self.tree.copy(deep=False)
        result["chan"] = xr.DataTree(dataset=self.chan_axis.data_source)
        result["probe"] = xr.DataTree(dataset=self.probe_axis.data_source)
        if not include_raw:
            params = self.model_params
            if params is not None:
                result["model_params"] = xr.DataTree(dataset=params)
        if include_raw:
            result["raw"] = xr.DataTree(dataset=self.raw)
        elif "raw" in result.children:
            del result["raw"]
        if not include_raw:
            _drop_timestream_projections(result)
        return result
