"""Identification of recorded TolTEC blocks and their shared axes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from astropy import units as u
from pydantic import Field, InstanceOf
from tollan.pipeline import Step, StepConfig, StepContext

from tolteca_datamodels.toltec.axes import ToltecAxisView

# Pydantic needs the index class to validate the live payload at runtime.
from tolteca_datamodels.toltec.kids.blocks import ToltecBlockIndex
from tolteca_datamodels.toltec.metadata import ToltecRawObsBlocksMetadata
from tolteca_datamodels.toltec.types import ToltecDataKind, ToltecInfo

from .blocks import NODE_TYPE

if TYPE_CHECKING:
    from typing import Any

__all__ = ["BlockIdentify", "BlockIdentifyContext"]


class BlockIdentifyContext(StepContext["BlockIdentify", StepConfig]):
    """Recorded block index and axes for a native acquisition."""

    data: InstanceOf[ToltecBlockIndex] | None = Field(
        default=None,
        exclude=True,
        description="Native block index and axes; excluded from JSON export.",
    )


class BlockIdentify(Step[StepConfig, BlockIdentifyContext]):
    """Identify recorded blocks while retaining the native data layout.

    Examples
    --------
    >>> from tollan.accessor.xarray import ensure_datatree
    >>> tree = ensure_datatree(ds)
    >>> tree, context = BlockIdentify()(tree, return_context=True)
    >>> meta = context.data.meta
    >>> first = meta.block_index.isel(block=0)
    >>> index_table = meta.block_index_table
    >>> channels = meta.chan_axis_table
    >>> probes = meta.probe_axis_table
    """

    @classmethod
    def run(cls, data: xr.DataTree, context: BlockIdentifyContext) -> None:
        """Publish the native block index and shared channel/probe axes.

        Parameters
        ----------
        data : xarray.DataTree
            Native acquisition at the root. Use ``ensure_datatree(ds)`` for a
            Dataset. Existing acquisition, collection or block processing scopes
            already identify their blocks and are left unchanged.
        context : BlockIdentifyContext
            Invocation state. Its data holds a native ``ToltecBlockIndex``, or
            None when the supplied tree already contains a block structure.

        Returns
        -------
        None
            LO/tone information is read and the index is published on the context.
            Native variables and children stay in place; detector arrays remain
            lazy. Repeated inspection reuses the native accessor's cached index.

        Raises
        ------
        ValueError
            Native acquisition metadata or recorded block axes are invalid.
        """
        if data.attrs.get(NODE_TYPE) not in {"acquisition", "blocks", "block"}:
            context.data = data.toltec_kids.block_index

    @staticmethod
    def _identify(mapper, data_source, *, acquisition=None):
        """Assemble the acquisition block index and axes."""
        ds = data_source
        schema = mapper.schema
        indexed = mapper.has_dim(ds, schema.n_blocks)
        kind = (
            ToltecInfo.obs_type_data_kind[acquisition.obs_type]
            if acquisition is not None
            else mapper._get_data_kind(ds)
        )
        is_raw = bool(kind & ToltecDataKind.raw_kids_data)
        if acquisition is not None and (
            indexed or schema.kind_str in mapper or not is_raw
        ):
            msg = "Acquisition metadata requires an unpartitioned raw source"
            raise ValueError(msg)
        if is_raw and not indexed:
            if acquisition is None:
                acquisition = mapper.get_metadata(ds)
            values = acquisition.model_dump(exclude_computed_fields=True)
        else:
            values = mapper._get_metadata_fields(ds, is_raw=is_raw)
        is_sweep = bool(kind & ToltecDataKind.sweep)
        if is_sweep:
            values["n_sweepsteps"] = mapper.get_scalar(ds, schema.n_sweepsteps)
        if acquisition is None:
            values.update(mapper._get_sample_metadata_fields(ds, values, is_raw=is_raw))
        f_los = mapper.get_arr(ds, schema.f_los) if schema.f_los in mapper else None
        if is_sweep and f_los is not None:
            f_los = f_los.compute()
        indexers = mapper._get_block_indexers(ds, kind, f_los=f_los)
        n_blocks = len(indexers)
        channels, probes, starts, ends = [], [], [], []
        chan_edges, probe_edges = [0], [0]
        sample_start = 0
        for index, selectors in enumerate(indexers):
            block_ds = ds.isel(selectors)
            block_values: dict[str, Any] = dict(values, n_blocks=n_blocks, block=index)
            block_values["n_times"] = (
                mapper.get_dim(block_ds, schema.n_times)
                if schema.n_times in mapper
                else values["n_times"]
            )
            channels.append(mapper._read_chan_axis(block_ds, block_values))
            block_los = (
                f_los.isel(selectors, missing_dims="ignore")
                if f_los is not None
                else None
            )
            probes.append(
                mapper._read_probe_axis(
                    block_ds,
                    block_values,
                    block_los,
                    sample_start=sample_start,
                    is_sweep=is_sweep,
                )
            )
            chan_edges.append(chan_edges[-1] + channels[-1].sizes["chan_entry"])
            probe_edges.append(probe_edges[-1] + probes[-1].sizes["probe"])
            starts.append(sample_start)
            sample_start += block_values["n_times"]
            ends.append(sample_start)
        block_data = xr.Dataset(
            {
                "data_kind": ("block", np.full(n_blocks, kind.value, dtype=np.int64)),
                "sample_start": ("block", np.asarray(starts, dtype=np.int64)),
                "sample_end": ("block", np.asarray(ends, dtype=np.int64)),
                "chan_start": ("block", np.asarray(chan_edges[:-1], dtype=np.int64)),
                "chan_end": ("block", np.asarray(chan_edges[1:], dtype=np.int64)),
                "probe_start": ("block", np.asarray(probe_edges[:-1], dtype=np.int64)),
                "probe_end": ("block", np.asarray(probe_edges[1:], dtype=np.int64)),
            },
            coords={"block": np.arange(n_blocks)},
        )
        if is_sweep:
            block_data["n_sweepsteps"] = (
                "block",
                np.full(n_blocks, values["n_sweepsteps"]),
            )
        if acquisition is None:
            values.update(
                n_times=sample_start,
                t_int=(sample_start / values["f_smp"]).to(u.s),
            )
        values.pop("n_sweepsteps", None)
        identified = ToltecBlockIndex(
            index=block_data,
            chan=xr.concat(channels, dim="chan_entry")
            if channels
            else mapper._empty_axis("chan_entry"),
            probe=xr.concat(probes, dim="probe")
            if probes
            else mapper._empty_axis("probe"),
            meta=ToltecRawObsBlocksMetadata(**values, n_blocks=n_blocks),
            acquisition=acquisition,
        )
        identified.meta._block_index = identified.index
        identified.meta._chan_axis = ToltecAxisView(identified.chan)
        identified.meta._probe_axis = ToltecAxisView(identified.probe)
        return identified
