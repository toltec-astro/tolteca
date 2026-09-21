"""Configuration models for KIDs processing and product output."""

from pathlib import Path

from pydantic import ConfigDict, Field
from tollan.config import FrozenBaseModel
from tollan.config.types import (
    AbsAnyPath,  # noqa: TC002 -- Pydantic resolves the annotation at runtime
)
from tollan.pipeline import StepConfig

__all__ = ["DataProdOutputConfig", "KidsConfig", "SweepReduceConfig"]


class SweepReduceConfig(StepConfig):
    """Control sweep reduction and select a block from its result.

    ``block=-1`` selects the last block; nonnegative indices select an explicit
    block. This stage detects sweep blocks and computes standard deviations.
    """

    model_config = ConfigDict(extra="forbid")
    block: int = Field(default=-1, ge=-1)
    force: bool = Field(default=False, description="Recompute existing reduced data.")


class DataProdOutputConfig(StepConfig):
    """Publish beneath ``path`` using a metadata-formatted directory ``name``.

    Output is opt-in. Names may use source-record fields and ``{stem}``;
    each name must be a single directory component. Existing products are
    never overwritten.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    path: AbsAnyPath = Field(default=Path(), validate_default=True)
    name: str = "{stem}"


class KidsConfig(FrozenBaseModel):
    """Configuration for the supported KIDs processing stages.

    Only raw sweep reduction is currently supported by this pipeline.
    Unsupported stage sections are rejected.
    """

    model_config = ConfigDict(extra="forbid")
    sweep_reduce: SweepReduceConfig = Field(default_factory=SweepReduceConfig)
    output: DataProdOutputConfig = Field(default_factory=DataProdOutputConfig)
