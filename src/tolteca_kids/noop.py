"""A no-op step for exercising input and configuration workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tollan.pipeline import Step, StepConfig, StepContext
from tollan.utils.log import logger

if TYPE_CHECKING:
    import xarray as xr

__all__ = ["NoOp", "NoOpContext"]


class NoOpContext(StepContext["NoOp", StepConfig]):
    """Live execution state for a no-op invocation."""


class NoOp(Step[StepConfig, NoOpContext]):
    """Complete a step without reading arrays or producing results."""

    @classmethod
    def run(cls, data: xr.DataTree, context: NoOpContext) -> bool:
        """Leave the input unchanged apart from the attached step context."""
        logger.debug("No-op: skipping computation and product output")
        return True
