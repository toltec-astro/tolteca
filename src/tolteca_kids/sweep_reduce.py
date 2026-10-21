"""Raw sweep reduction step and its data-backed result view."""

from __future__ import annotations

import xarray as xr
from pydantic import ConfigDict, Field
from tollan.pipeline import Step, StepContext

from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_kidsproc.reduction import reduce_sweep_samples

from ._reduction import calculation_settings, ensure_blocks, select_block
from .blocks import NODE_TYPE
from .config import SweepReduceConfig
from .sweep import ReducedSweepView, ToltecSweepMapper

__all__ = ["SweepReduce", "SweepReduceConfig", "SweepReduceContext"]


class SweepReduceContext(StepContext["SweepReduce", SweepReduceConfig]):
    """Sweep-reduction execution context."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: ReducedSweepView | None = Field(
        default=None,
        exclude=True,
        description="Live view of the selected sweep block; excluded from JSON export.",
    )
    block: int = Field(default=0, description="Selected block index after reduction.")


class SweepReduce(Step[SweepReduceConfig, SweepReduceContext]):
    """Raw-sweep reduction step."""

    @classmethod
    def result_path(cls, *, alias: str | None = None) -> str:
        """Get the result path for this step.

        Parameters
        ----------
        alias : str or None, optional
            Invocation label. None selects the primary reduction.

        Returns
        -------
        str
            Result subpath relative to each block scope.
        """
        name = "sweep"
        return f"{name}_{alias}" if alias is not None else name

    @classmethod
    def view(
        cls, data: xr.DataTree, *, block: int, alias: str | None = None
    ) -> ReducedSweepView:
        """Access a reduced block without reading an execution context.

        Parameters
        ----------
        data : xarray.DataTree
            Acquisition, collection or block tree containing reduced results.
        block : int
            Acquisition block ID recorded by the reduction context.
        alias : str or None, optional
            Invocation label. None selects the primary reduction.

        Returns
        -------
        ReducedSweepView
            Scientific view of the selected block's arrays and metadata.
        """
        node = select_block(data, block).tree
        view = ReducedSweepView(node[cls.result_path(alias=alias)])
        # Retain the source block's axis access if a rerun replaces this leaf.
        _ = view.meta
        return view

    @classmethod
    def run(cls, data: xr.DataTree, context: SweepReduceContext) -> None:
        """Reduce sweep blocks and publish the selected block's view.

        Parameters
        ----------
        data : xarray.DataTree
            Acquisition, collection, block, or native data at the root. Native
            data moves to the acquisition raw leaf. Results update in place.
        context : SweepReduceContext
            Execution settings. Its block records the selected acquisition ID;
            data publishes that block's result after successful reduction.

        Returns
        -------
        None
            Each sweep block retains its own reduced Dataset. Unrelated nodes
            and sibling analyses remain unchanged.

        Raises
        ------
        IndexError
            The requested block is outside the collection.
        ValueError
            Acquisition preparation conflicts with existing nodes, or a selected
            block is not a sweep.
        """
        ensure_blocks(data)
        selected = select_block(data, context.config.block)
        context.block = selected.meta.block
        blocks = (
            [selected]
            if data.attrs.get(NODE_TYPE) == "block"
            else data.toltec_kids.blocks
        )
        for block in blocks:
            cls._reduce_block(block, context.config)
        context.data = cls.view(data, block=context.block, alias=context.config.alias)

    @classmethod
    def _reduce_block(cls, block, config):
        """Install a sweep using the block's previously discovered probe bounds."""
        if not block.data_kind & ToltecDataKind.sweep:
            msg = "Cannot run sweep reduction on this block kind"
            raise ValueError(msg)
        scope = block.tree
        path = cls.result_path(alias=config.alias)
        output_mapper = ToltecSweepMapper.from_defaults()
        config_key = output_mapper.get_name(output_mapper.schema.reduction_config)
        if not config.force:
            existing = path if path in scope.children else "sweep"
            if existing in scope.children and scope[existing].attrs.get(
                config_key
            ) == calculation_settings(config):
                if existing != path:
                    scope[path] = scope[existing]
                return
        raw = block.raw
        mapper = raw.toltec_kids.mapper
        schema = mapper.schema
        probe = block.probe_axis
        probe_mapper = probe.mapper
        starts = probe_mapper.get_arr(
            probe.data_source, probe_mapper.schema.sample_start
        ).values
        ends = probe_mapper.get_arr(
            probe.data_source, probe_mapper.schema.sample_end
        ).values
        # Probe bounds are acquisition-relative, while raw is this block's slice.
        sample_slices = [
            slice(int(a - starts[0]), int(b - starts[0]))
            for a, b in zip(starts, ends, strict=True)
        ]
        payload = reduce_sweep_samples(
            mapper.get_arr(raw, schema.I),
            mapper.get_arr(raw, schema.Q),
            sample_dim=mapper.get_name(schema.n_times),
            sample_slices=sample_slices,
            compute_uncertainty=config.compute_uncertainty,
        )
        sweep_dim = output_mapper.get_name(output_mapper.schema.sweep)
        chan_dim = mapper.get_name(schema.n_chans)
        channels = block.chan_axis
        chan_mapper = channels.mapper
        payload = payload.assign_coords(
            {
                chan_dim: chan_mapper.get_arr(
                    channels.data_source, chan_mapper.schema.chan
                ).values,
                sweep_dim: xr.DataArray(
                    probe_mapper.get_arr(
                        probe.data_source, probe_mapper.schema.f_sweep
                    ).values,
                    dims=sweep_dim,
                    attrs={"units": "Hz"},
                ),
                output_mapper.get_name(output_mapper.schema.f_chan): xr.DataArray(
                    chan_mapper.get_arr(
                        channels.data_source, chan_mapper.schema.f_chan
                    ).values,
                    dims=chan_dim,
                    attrs={"units": "Hz"},
                ),
            }
        )
        payload[output_mapper.get_name(output_mapper.schema.f_lo)] = xr.DataArray(
            probe_mapper.get_arr(probe.data_source, probe_mapper.schema.f_lo).values,
            dims=sweep_dim,
            attrs={"units": "Hz"},
        )
        payload = payload.assign_coords(frequency=ReducedSweepView(payload).frequency)
        payload.attrs.update(
            {
                NODE_TYPE: "sweep",
                config_key: calculation_settings(config),
                "meta": scope.attrs["meta"],
                "block": block.meta.block,
                output_mapper.get_name(
                    output_mapper.schema.f_center
                ): block.meta.f_lo_center,
            }
        )
        scope[path] = xr.DataTree(dataset=payload)
