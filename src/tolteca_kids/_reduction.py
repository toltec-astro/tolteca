"""Working scopes shared by the configured raw-data reducers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .block_create import BlockCreate
from .blocks import NODE_TYPE

if TYPE_CHECKING:
    import xarray as xr

    from .blocks import ToltecBlockView


def ensure_blocks(data: xr.DataTree) -> None:
    """Create blocks when the working tree has no block structure yet."""
    if data.attrs.get(NODE_TYPE) not in {"acquisition", "blocks", "block"}:
        BlockCreate()(data)


def select_block(data: xr.DataTree, block: int) -> ToltecBlockView:
    """Select the supplied block scope or an acquisition block by ID."""
    if data.attrs.get(NODE_TYPE) == "block":
        return data.toltec_kids.block
    blocks = data.toltec_kids.blocks
    block_ids = blocks.mapper.get_arr(blocks.data_source, blocks.mapper.schema.block)
    if block == -1:
        block = int(block_ids[-1])
    if block in block_ids:
        return blocks[block]
    msg = f"Block {block} out of range for this collection"
    raise IndexError(msg)


def calculation_settings(config):
    """Serialize scientific settings for matching reusable result arrays."""
    return config.model_dump_json(exclude={"block", "force", "alias", "enabled"})
