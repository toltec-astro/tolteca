"""Creation of TolTEC block scopes in a working acquisition tree."""

from __future__ import annotations

import xarray as xr
from astropy import units as u
from pydantic import ConfigDict, Field
from tollan.pipeline import PIPELINE_CONTEXT_KEY, Step, StepConfig, StepContext

from tolteca_datamodels.toltec.metadata import (
    ToltecSweepMetadata,
    ToltecTimestreamMetadata,
)
from tolteca_datamodels.toltec.types import ToltecDataKind

from .block_identify import BlockIdentify
from .blocks import NODE_TYPE, ToltecBlocksView, ToltecBlockView

__all__ = ["BlockCreate", "BlockCreateContext"]


class BlockCreateContext(StepContext["BlockCreate", StepConfig]):
    """Live view of the created collection or selected block."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: ToltecBlocksView | ToltecBlockView | None = Field(
        default=None,
        exclude=True,
        description="Live block view; excluded from JSON export.",
    )


class BlockCreate(Step[StepConfig, BlockCreateContext]):
    """Create block scopes and shared axis leaves for processing.

    Examples
    --------
    >>> from tollan.accessor.xarray import ensure_datatree
    >>> tree = ensure_datatree(ds)
    >>> _ = BlockIdentify()(tree)
    >>> tree, context = BlockCreate()(tree, return_context=True)
    >>> blocks = context.data
    >>> first = blocks[0]
    """

    @classmethod
    def run(cls, data: xr.DataTree, context: BlockCreateContext) -> None:
        """Install block scopes and publish their view on the supplied tree.

        Parameters
        ----------
        data : xarray.DataTree
            Native acquisition at the root, or an existing acquisition, block
            collection or block scope. Use ``ensure_datatree(ds)`` for a Dataset.
            Keep its source open while using borrowed arrays.
        context : BlockCreateContext
            Invocation state. Its data exposes the collection, or the supplied
            block when operating on a block scope.

        Returns
        -------
        None
            Native variables move to ``raw``; ``blocks`` holds the index, shared
            axes and block scopes with metadata. An available BlockIdentify result
            is reused; otherwise identification runs first. Detector arrays remain
            lazy. Existing block structures and unrelated children are retained.
            Reopened trees need no previous live context.

        Raises
        ------
        ValueError
            Native acquisition metadata is invalid, or an existing child conflicts
            with ``raw`` or ``blocks``.
        """
        kind = data.attrs.get(NODE_TYPE)
        if kind not in {"acquisition", "blocks", "block"}:
            cls._create(data)
        context.data = (
            data.toltec_kids.block if kind == "block" else data.toltec_kids.blocks
        )

    @staticmethod
    def _create(data):
        collisions = {"raw", "blocks"}.intersection(data.children)
        if collisions:
            msg = f"Acquisition layout conflicts with existing nodes: {collisions}"
            raise ValueError(msg)
        identified = (
            BlockIdentify.get_context(data).data
            if BlockIdentify.has_context(data)
            else None
        )
        if identified is None:
            _, context = BlockIdentify()(data, return_context=True)
            identified = context.data
        if identified is None:
            msg = "BlockCreate requires an identified block index"
            raise ValueError(msg)
        source = data.dataset.copy(deep=False)
        source.attrs.pop(PIPELINE_CONTEXT_KEY, None)
        tree = xr.DataTree.from_dict(
            {
                "raw": source,
                "blocks/index": identified.index,
                "blocks/chan": identified.chan,
                "blocks/probe": identified.probe,
            }
        )
        tree.attrs.update(data.attrs)
        tree.attrs[NODE_TYPE] = "acquisition"
        if identified.acquisition is not None:
            tree.attrs["meta"] = identified.acquisition.model_dump_json(round_trip=True)
        tree["blocks"].attrs.update(
            {
                NODE_TYPE: "blocks",
                "meta": identified.meta.model_dump_json(),
            }
        )
        acquisition_data = identified.meta.model_dump(exclude_computed_fields=True)
        for block in identified.index.block.values:
            row = identified.index.sel(block=block)
            kind = ToltecDataKind(int(row["data_kind"]))
            n_times = int(row["sample_end"] - row["sample_start"])
            block_data = dict(
                acquisition_data,
                block=int(block),
                n_times=n_times,
                t_int=(n_times / identified.meta.f_smp).to(u.s),
            )
            for name in ("atten_drive", "atten_sense", "dac_normalization"):
                if name in row:
                    block_data[name] = row[name].item()
            if kind & ToltecDataKind.sweep:
                block_data["n_sweepsteps"] = row["n_sweepsteps"].item()
                meta = ToltecSweepMetadata.model_validate(block_data)
            else:
                meta = ToltecTimestreamMetadata.model_validate(block_data)
            tree[f"blocks/{block}"] = xr.DataTree(
                dataset=xr.Dataset(
                    attrs={
                        NODE_TYPE: "block",
                        "meta": meta.model_dump_json(round_trip=True),
                        "data_kind": int(row["data_kind"]),
                    }
                )
            )
        data.dataset = xr.Dataset(attrs=tree.attrs)
        data.children = {**data.children, **tree.children}
