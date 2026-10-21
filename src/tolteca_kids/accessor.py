"""Block and result access on TolTEC KIDs processing trees."""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astropy.table import QTable
    from tollan.accessor import Mapping

import numpy as np
import xarray as xr
from tollan.accessor import NameMapping, Schema
from tollan.accessor.xarray import XarrayMapper
from tollan.pipeline import PIPELINE_CONTEXT_KEY

from tolteca_datamodels.toltec.kids.core import ToltecKidsAccessor
from tolteca_datamodels.toltec.metadata import (
    ToltecRawObsMetadata,
    ToltecSweepMetadata,
    ToltecTimestreamMetadata,
)
from tolteca_datamodels.toltec.types import ToltecDataKind

from .blocks import (
    NODE_TYPE,
    ToltecBlocksView,
    ToltecBlockView,
    _drop_timestream_projections,
)


@dataclass
class ToltecKidsTreeSchema(Schema):
    """Acquisition payload within a KIDs working tree."""

    acquisition: Mapping = NameMapping.path(("raw", "."))


class ToltecKidsTreeMapper(XarrayMapper[ToltecKidsTreeSchema]):
    """Mapper for the acquisition payload of a KIDs working tree."""


@xr.register_datatree_accessor("toltec_kids")
class ToltecKidsTreeAccessor(ToltecKidsAccessor):
    """Access blocks and scientific results in a TolTEC KIDs processing tree.

    Parameters
    ----------
    data_source : xarray.DataTree
        Prepared acquisition, selected block, or reopened product tree.
    mapper : ToltecKidsIOMapper, optional
        Existing native field mapping for the selected acquisition payload.

    Examples
    --------
    >>> from tollan.accessor.xarray import ensure_datatree
    >>> from tolteca_kids import SweepReduce
    >>> tree = ensure_datatree(ds)
    >>> _ = SweepReduce()(tree)
    >>> sweep = tree.toltec_kids.blocks[0].sweep
    """

    _data_source: xr.DataTree

    @functools.cached_property
    def data_source(self) -> xr.Dataset:
        """Native acquisition dataset retained by this working tree."""
        mapper = ToltecKidsTreeMapper.from_data_source(self.tree)
        return mapper.get_node(self.tree, mapper.schema.acquisition).dataset

    @functools.cached_property
    def meta(self):
        """Primary metadata for the supplied acquisition, collection or block."""
        if isinstance(self.tree, xr.DataTree):
            kind = self.tree.attrs.get(NODE_TYPE)
            if kind == "block":
                return self.block.meta
            if kind == "blocks":
                return self.blocks.meta
            if "meta" in self.tree.attrs:
                if kind in {"sweep", "timestream"}:
                    model = (
                        ToltecSweepMetadata
                        if "n_sweepsteps" in json.loads(self.tree.attrs["meta"])
                        else ToltecTimestreamMetadata
                    )
                    return model.model_validate_json(self.tree.attrs["meta"])
                return ToltecRawObsMetadata.model_validate_json(self.tree.attrs["meta"])
        return self.mapper.get_metadata(self.data_source)

    @functools.cached_property
    def blocks(self) -> ToltecBlocksView:
        """Block collection at the supplied acquisition or collection scope."""
        return ToltecBlocksView(self.tree)

    @functools.cached_property
    def block(self) -> ToltecBlockView:
        """View of the supplied block node, including detached block exports."""
        if (
            not isinstance(self.tree, xr.DataTree)
            or self.tree.attrs.get(NODE_TYPE) != "block"
        ):
            msg = "The supplied object is not a block node"
            raise ValueError(msg)
        parent = self.tree.parent
        if parent is not None and parent.attrs.get(NODE_TYPE) == "blocks":
            collection = parent.toltec_kids.blocks
            return collection[json.loads(self.tree.attrs["meta"])["block"]]
        return ToltecBlockView(self.tree)

    @property
    def chan_axis_table(self):
        """Full channel table at the current acquisition/collection/block scope."""
        view = self.block if self.tree.attrs.get(NODE_TYPE) == "block" else self.blocks
        return view.chan_axis_table

    @property
    def probe_axis_table(self):
        """Full probe table at the current acquisition/collection/block scope."""
        view = self.block if self.tree.attrs.get(NODE_TYPE) == "block" else self.blocks
        return view.probe_axis_table

    @property
    def model_params(self):
        """Lazy recorded parameter Dataset at block scope, or None if absent."""
        return self.block.model_params

    @property
    def model_params_table(self):
        """Recorded parameter QTable at block scope, or None if absent."""
        return self.block.model_params_table

    @property
    def tree(self) -> xr.DataTree:
        """The supplied processing tree node."""
        return self._data_source

    def detach(self, *, include_raw=True):
        """Detach the current scope as an independent, self-contained tree.

        Parameters
        ----------
        include_raw : bool, optional
            Retain raw samples (default True). False omits raw leaves from the
            detached scope while retaining metadata, axes, model parameters
            and results.

        Returns
        -------
        xarray.DataTree
            A new tree structure, leaving the original tree untouched. Blocks
            bring their selected raw and axis leaves; acquisitions retain their
            shared storage and existing children. Arrays remain lazy and share
            buffers where possible. Keep borrowed source files open until the
            arrays are loaded or exported. No export compaction is performed.
        """
        tree = self.tree
        kind = tree.attrs.get(NODE_TYPE)
        if kind == "block":
            return self.block.detach(include_raw=include_raw)
        if kind == "blocks":
            return self.blocks.detach(include_raw=include_raw)
        result = tree.copy(deep=False)
        if not include_raw:
            for node in tree.subtree:
                if node.attrs.get(NODE_TYPE) == "block":
                    params = node.toltec_kids.block.model_params
                    if params is not None:
                        result[node.relative_to(tree)]["model_params"] = xr.DataTree(
                            dataset=params
                        )
            for node in result.subtree:
                if (
                    node.attrs.get(NODE_TYPE) in {"acquisition", "block"}
                    and "raw" in node.children
                ):
                    del node["raw"]
            _drop_timestream_projections(result)
        return result

    def to_zarr(self, store, *, include_raw=True, **kwargs):
        """Export the current scope without duplicate reconstructible slices.

        Parameters
        ----------
        store : str, path-like, or Zarr store
            Destination supported by xarray, including remote stores.
        include_raw : bool, optional
            Include raw data once (default True). False exports axes, metadata,
            recorded model parameters and already attached results.
        **kwargs
            Passed to DataTree.to_zarr. Defaults are Zarr format 2, consolidated
            metadata and mode ``w-``; callers may explicitly override them.

        Returns
        -------
        object
            The result of xarray's DataTree.to_zarr writer.
        """
        options = {"zarr_format": 2, "consolidated": True, "mode": "w-", **kwargs}
        return self._get_export_tree(include_raw=include_raw).to_zarr(store, **options)

    def _get_export_tree(self, *, include_raw):
        """Prepare the scientific tree for serialization."""
        tree = self.tree
        kind = tree.attrs.get(NODE_TYPE)
        if not kind:
            msg = (
                "Run BlockCreate on the working tree before exporting blocks; "
                "this tree is outside the block layout"
            )
            raise ValueError(msg)
        result = self.detach(include_raw=include_raw)
        self._compact_block_projections(result, include_raw=include_raw)
        for node in result.subtree:
            node.attrs.pop(PIPELINE_CONTEXT_KEY, None)
        return result

    @staticmethod
    def _compact_block_projections(tree, *, include_raw):
        """Drop only projections reconstructible within the exported scope."""
        kind = tree.attrs.get(NODE_TYPE)
        _drop_timestream_projections(tree)
        for node in tree.subtree:
            if node.attrs.get(NODE_TYPE) != "block":
                continue
            redundant = set()
            if kind == "acquisition":
                redundant.update(("raw", "chan", "probe"))
            elif kind == "blocks":
                redundant.update(("chan", "probe"))
            if include_raw and (
                "raw" in node.children
                or (kind == "acquisition" and "raw" in tree.children)
            ):
                redundant.add("model_params")
            for name in redundant.intersection(node.children):
                del node[name]

    @property
    def sweep_check(self):
        """Saved sweep quality checks for the selected block."""
        from .sweep_check import SweepCheck  # noqa: PLC0415
        from .sweep_check_view import SweepCheckView  # noqa: PLC0415

        return SweepCheckView(self.tree[SweepCheck.result_path()])

    @property
    def d21(self):
        """D21 results at the supplied sweep or D21 node."""
        from tolteca_kidsproc.analysis.d21 import D21View  # noqa: PLC0415

        return D21View(self.tree)

    # Data Kind Identification
    # ========================

    @functools.cached_property
    def data_kind(self) -> ToltecDataKind:
        """KIDs data classification."""
        kind = self.tree.attrs.get(NODE_TYPE)
        if kind == "block":
            return self.block.data_kind
        if kind in {"sweep", "timestream"}:
            return {
                "sweep": ToltecDataKind.reduced_sweep,
                "timestream": ToltecDataKind.timestream,
            }[kind]
        if kind in {"acquisition", "blocks"}:
            values = self.blocks.data_source["data_kind"].values
            return ToltecDataKind(int(np.bitwise_or.reduce(values, initial=0)))
        from tolteca_kidsproc.analysis.d21 import D21Mapper  # noqa: PLC0415

        mapper = D21Mapper.from_defaults()
        if mapper.has_attr(self.tree, mapper.schema.analysis):
            return ToltecDataKind.d21
        if mapper._has_node(self.tree, mapper.namespace_path) and mapper.has_attr(
            mapper._get_node(self.tree, mapper.namespace_path), mapper.schema.analysis
        ):
            return ToltecDataKind.d21
        return self.mapper._get_data_kind(self.data_source)

    # Reduced Data Views
    # ==================

    @property
    def sweep(self):
        """Reduced sweep view.

        Examples
        --------
        >>> dt = ensure_datatree(ds_raw)
        >>> _ = SweepReduce()(dt)
        >>> view = dt.toltec_kids.blocks[0].sweep
        >>> sweep = view.sweep
        >>> n_blocks = dt.toltec_kids.blocks.n_blocks
        """
        from .sweep import ReducedSweepView  # noqa: PLC0415

        if isinstance(self.tree, xr.DataTree):
            scope = self.tree.attrs.get(NODE_TYPE)
            if scope and scope != "sweep" and "sweep" not in self.tree.children:
                msg = "No reduced sweep at this scope; select or reduce a block first"
                raise ValueError(msg)
        return ReducedSweepView(self.tree)

    @property
    def timestream(self):
        """Raw time-domain signals and available power spectral density results.

        Examples
        --------
        >>> dt = ensure_datatree(ds_ts)
        >>> _ = TimestreamReduce()(dt)
        >>> view = dt.toltec_kids.timestream
        >>> frequencies = view.f_psd
        >>> psd = view.I_psd
        >>> median = view.I_psd_median
        """
        from .timestream import TimestreamView  # noqa: PLC0415

        if isinstance(self.tree, xr.DataTree):
            scope = self.tree.attrs.get(NODE_TYPE)
            if scope in {"acquisition", "blocks"}:
                blocks = self.blocks
                if len(blocks) != 1:
                    msg = "Select a block explicitly for the timestream view"
                    raise ValueError(msg)
                return TimestreamView(blocks[0].tree)
            if scope and scope not in {"block", "timestream"}:
                msg = "No timestream at this scope; select a timestream block"
                raise ValueError(msg)
        return TimestreamView(self.tree)

    # Axis Data Methods
    # =================

    def get_chan_axis_table(self, block: int = 0) -> QTable:
        """Get channel axis data as an astropy QTable.

        Parameters
        ----------
        block : int, optional
            Zero-based block index; default is 0.

        Returns
        -------
        QTable
            Table with channel metadata including:
            - chan: Channel index
            - f_tone: Tone frequency
            - f_chan: Channel frequency (tone + LO center)
            - tone_mask: Enabled/disabled mask
            - tone_amp: Tone amplitude (if available)
            - tone_phase: Tone phase (if available)

        Examples
        --------
        >>> chan_data = ds.toltec_kids.get_chan_axis_table()
        >>> chan_data['f_chan']  # Access channel frequencies
        >>> enabled = chan_data[chan_data['tone_mask']]  # Filter enabled
        """
        return self.blocks[block].meta.chan_axis_table

    def get_probe_axis_table(self, block: int = 0) -> QTable:
        """Get probe axis data as an astropy QTable.

        Parameters
        ----------
        block : int, optional
            Zero-based block index; default is 0.

        Returns
        -------
        QTable
            Table with probe step metadata including:
            - step: Block-local probe step index
            - f_sweep: Sweep frequency offset from LO center
            - f_lo: LO frequency at this sweep step
            - n_samples: Actual samples acquired at this step
            - sample_start, sample_end: Acquisition sample bounds (end exclusive)

        Raises
        ------
        ValueError
            If the block is out of range

        Examples
        --------
        >>> sweep_data = ds.toltec_kids.get_probe_axis_table()
        >>> sweep_data['f_sweep']  # Access sweep frequencies
        """
        return self.blocks[block].meta.probe_axis_table
