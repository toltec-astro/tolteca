import dataclasses
import inspect
import os
from enum import Enum, auto
from pathlib import Path
from typing import Any

from pydantic import Field, PrivateAttr
from tollan.config.types import ImmutableBaseModel
from tollan.utils.log import logger

from tolteca_datamodels.toltec.ncfile import NcFileIO
from tolteca_kidsproc.kidsdata import MultiSweep

__all__ = ["prepare_data_context"]


@dataclasses.dataclass
class DataContext:
    """A context object for holding data."""

    filepath: None | Path
    data_io: None | NcFileIO
    data: None | Any
    context: dict = dataclasses.field(repr=False)


class RunState(Enum):
    """The run state."""

    Success = auto()
    Failure = auto()
    Skipped = auto()


@dataclasses.dataclass
class RunContext:
    """The context of a run."""

    parent_workflow: "None | WorkflowStepBase"
    input: None | DataContext
    context: dict = dataclasses.field(repr=False)
    state: None | RunState


class _MissingType(Enum):
    missing = "missing"


class WorkflowStepBase(ImmutableBaseModel):
    """The base class for workflow step."""

    # these are for external flow control, and is not consumed by this step.
    enabled: bool = Field(default=True, description="Whether this step is enabled.")

    _run_context: RunContext = PrivateAttr(
        default_factory=lambda: RunContext(
            parent_workflow=None,
            input=None,
            context={},
            state=None,
        ),
    )

    def prepare_input(self, *args, **kwargs):
        """Prepare input for workflow."""
        if self.run_context.input is None:
            self.run_context.input = prepare_data_context(*args, **kwargs)
        return self.run_context.input

    def ensure_input_data(self, *args, **kwargs):
        """Prepare input data for workflow."""
        kwargs.setdefault("read", True)
        dctx = self.prepare_input(*args, **kwargs)
        if dctx.data is None:
            raise ValueError(f"Unable to load data from {args}")
        return dctx.data

    def _should_skip(self):
        # when invoking with parent workflow, check if this step
        # is enable
        return self.run_context.parent_workflow is not None and not self.enabled

    def _mark_start(self, parent_workflow):
        self.run_context.parent_workflow = parent_workflow

    def _mark_success(self, ctx=None):
        self.run_context.state = RunState.Success
        if ctx is not None:
            self.run_context.context.update(ctx)
        return self.run_context

    def _mark_failure(self, ctx=None):
        self.run_context.state = RunState.Failure
        if ctx is not None:
            self.run_context.context.update(ctx)
        return self.run_context

    def _mark_skipped(self, ctx=None):
        self.run_context.state = RunState.Skipped
        if ctx is not None:
            self.run_context.context.update(ctx)
        return self.run_context

    @staticmethod
    def _make_sub_context_key(depth=1, arg=None):
        if arg is None:
            # infer from stack
            frame = inspect.stack()[depth][0]
            c = frame.f_locals["self"].__class__.__qualname__
            m = frame.f_code.co_name
            return f"{c}.{m}"
        # inter frame arg
        if isinstance(arg, str):
            return arg
        if callable(arg) and arg.__self__ is not None:
            c = arg.__self__.__class__.__qualname__
            m = arg.__name__
            return f"{c}.{m}"
        raise TypeError("uhable to infer sub context key.")

    def _mark_sub_context(self, ctx, key=None):
        # the calling function name
        key = self._make_sub_context_key(depth=2)
        self.run_context.context[key] = ctx
        return self.run_context

    def _get_sub_context(self, func, default=_MissingType.missing):
        key = self._make_sub_context_key(arg=func)
        if isinstance(default, _MissingType):
            if key not in self.run_context.context:
                raise ValueError(f"no context for {key} is found.")
            return self.run_context.context[key]
        return self.run_context.context.get(key, default)

    def _get_or_create_sub_context(self, func, arg):
        key = self._make_sub_context_key(arg=func)
        if key not in self.run_context.context:
            if not callable(func):
                raise TypeError(
                    "cannot create sub context with non-callable func type"
                    f" {type(func)}"
                )
            logger.debug(f"create sub context for {key} with {arg=}")
            func(arg)
        elif arg is not None and (
            (isinstance(arg, RunContext) and arg is not self.run_context)
            or (isinstance(arg, DataContext) and arg is not self.input)
            or (
                self.input is not None
                and isinstance(arg, type(self.input.data))
                and arg is not self.input.data
            )
            or (
                self.input is not None
                and isinstance(arg, type(self.input.data_io))
                and arg is not self.input.data_io
            )
            or (
                self.input is not None
                and isinstance(arg, (str, type(self.input.filepath)))
                and arg != self.input.filepath
            )
        ):
            # make sure the arg is consistent with the context.
            raise ValueError("In-consistent data context")
        else:
            logger.debug(f"found existing sub context for {key} with {arg=}")
        return self.run_context.context[key]

    def reset_run_context(self):
        """Reset the run context."""
        self.run_context.parent_workflow = None
        self.run_context.input = None
        self.run_context.state = None
        self.run_context.context.clear()

    @property
    def run_context(self):
        """The run context object."""
        return self._run_context

    @property
    def input(self):
        """The input data context."""
        return self.run_context.input

    @property
    def context(self):
        """The run context data."""
        return self.run_context.context

    @property
    def run_state(self):
        """The run state."""
        return self.run_context.state


def prepare_data_context(arg, read=False, read_kw=None):
    """Return a context object of data io and data items."""
    if isinstance(arg, dict):
        # data context is in arg.
        for v in arg.values():
            if isinstance(v, DataContext):
                return dataclasses.replace(v, context=arg)
    if isinstance(arg, DataContext):
        # data context is arg
        return arg

    if isinstance(arg, RunContext):
        # run context is arg
        if arg.input is None:
            raise ValueError("no data found in run context.")
        return arg.input

    # try creating new data context
    if isinstance(arg, (str, os.PathLike)):
        filepath = Path(arg)
        data_io = NcFileIO(filepath)
        data = data_io.read(**(read_kw or {})) if read else None
    elif isinstance(arg, NcFileIO):
        data_io = arg
        filepath = data_io.filepath
        data = data_io.read(**(read_kw or {})) if read else None
    elif isinstance(arg, MultiSweep):
        data = arg
        data_io = None
        filepath = data.meta["filepath"]  # type: ignore
    else:
        raise TypeError(f"invalid input arg type: {type(arg)}")
    return DataContext(filepath=filepath, data_io=data_io, data=data, context={})
