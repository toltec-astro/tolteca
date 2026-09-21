"""Raw sweep reduction step and its data-backed result view."""

from __future__ import annotations

from typing import ClassVar

import xarray as xr
from tollan.pipeline import Step, StepContext

from tolteca_datamodels.toltec.kids import ReducedSweepView, SweepReducer

from .config import SweepReduceConfig

__all__ = ["SweepReduce", "SweepReduceConfig", "SweepReduceContext", "SweepReduceView"]


class SweepReduceContext(StepContext["SweepReduce", SweepReduceConfig]):
    """Serializable execution metadata for a reduced-sweep result."""

    block_index: int = 0


class SweepReduce(Step[SweepReduceConfig, SweepReduceContext]):
    """Add reduced sweep results to an owned working DataTree.

    Numerical arrays live in the reduced node. The selected block and execution
    configuration are stored alongside them as JSON-compatible metadata.
    Direct calls run the step; `SequentialPipeline` handles the enabled flag.
    """

    _reducer: ClassVar[SweepReducer] = SweepReducer()

    @classmethod
    def result_path(cls) -> str:
        """Return the result namespace, distinct for explicitly aliased passes."""
        name = SweepReducer.__module__
        return f"{name}_{cls._alias_name}" if cls._alias_name else name

    @classmethod
    def view(cls, data: xr.Dataset | xr.DataTree) -> SweepReduceView:
        """Resolve this step's result and recorded invocation."""
        if isinstance(data, xr.DataTree):
            data = data[cls.result_path()].to_dataset()
            data = data[list(data.data_vars)]
        return SweepReduceView(data, context_key=cls.context_key)

    def create_context(self, data: xr.DataTree) -> SweepReduceContext:
        """Create invocation metadata before the result node exists."""
        return SweepReduceContext(config=self.config)

    @classmethod
    def run(cls, data: xr.DataTree, context: SweepReduceContext) -> bool:
        """Run the independent reducer and validate the requested block."""
        result = cls._reducer(data, force=context.config.force)
        view = ReducedSweepView(result)
        block = context.config.block
        if block >= view.n_blocks:
            msg = f"Block {block} out of range for {view.n_blocks} block(s)"
            raise IndexError(msg)
        context.block_index = view.n_blocks - 1 if block == -1 else block
        # The numerical object owns its result; the step adds it to the caller's
        # working tree. Assignment copies the node, not its array buffers.
        data[cls.result_path()] = result[SweepReducer.__module__]
        return True

    @classmethod
    def set_context(
        cls, data: xr.Dataset | xr.DataTree, context: SweepReduceContext
    ) -> None:
        """Store the completed invocation on the reduced result node."""
        target = data[cls.result_path()] if isinstance(data, xr.DataTree) else data
        target.attrs[cls.context_key] = context.model_dump(
            mode="json",
            exclude_none=True,
        )

    @classmethod
    def get_context(cls, data: xr.Dataset | xr.DataTree) -> SweepReduceContext:
        """Read typed invocation metadata, including from a reopened product."""
        return cls.view(data).context

    @classmethod
    def has_context(cls, data: xr.Dataset | xr.DataTree) -> bool:
        """Return whether this result carries reduction-step metadata."""
        if isinstance(data, xr.DataTree) and cls.result_path() not in data:
            return False
        return cls.context_key in cls.view(data).data_source.attrs


class SweepReduceView(ReducedSweepView):
    """Reduced arrays and the step metadata that describes their production."""

    def __init__(
        self, data: xr.Dataset | xr.DataTree, *, context_key: str | None = None
    ) -> None:
        super().__init__(data)
        self._context_key = context_key or SweepReduce.context_key

    @property
    def context(self) -> SweepReduceContext:
        """The recorded invocation, reconstructed without reading arrays."""
        return SweepReduceContext.model_validate(
            self.data_source.attrs[self._context_key]
        )

    @property
    def selected_block(self) -> ReducedSweepView:
        """The block selected by this invocation as a reduced-sweep view."""
        ds = self.data_source
        if self.has_blocks:
            block_dim = self.mapper.get_name(self.mapper.schema.n_blocks)
            ds = ds.isel({block_dim: self.context.block_index})
        return ReducedSweepView(ds)
