from collections import UserDict
from dataclasses import dataclass
from typing import ClassVar, Generic, Literal, TypeVar, overload

from astropy.utils.decorators import classproperty
from pydantic import BaseModel, Field
from tollan.config.types import FieldDefaults, ImmutableBaseModel
from tollan.utils.general import getname
from tollan.utils.log import logger, timeit
from tollan.utils.typing import get_typing_args
from typing_extensions import Self

__all__ = [
    "StepConfig",
    "StepContext",
    "Step",
    "PIPELINE_CONTEXT_KEY",
    "get_pipeline_contexts",
    "Pipeline",
    "SequentialPipeline",
]


T = TypeVar("T")
ConfigT = TypeVar("ConfigT", bound="StepConfig")
ContextT = TypeVar("ContextT", bound="StepContext")
ContextDataT = TypeVar("ContextDataT")
StepT = TypeVar("StepT", bound="Step")


class StepConfig(ImmutableBaseModel):
    """A base model for defining a pipeline step."""

    field_defaults: ClassVar[FieldDefaults] = FieldDefaults()
    enabled: bool = Field(
        default=True,
        description="Set to False to skip this step in a pipeline.",
    )


class StepContext(BaseModel, Generic[StepT, ConfigT]):
    """A base class for arbituary data associated with the step."""

    step_cls: ClassVar[type["StepT"]]
    config_cls: ClassVar[type["ConfigT"]]
    config: ConfigT
    data: None
    completed: bool = False

    def make_step(self):
        """Return a step instance from the context."""
        return self.step_cls(self.config)


class StepContextDict(UserDict[str, StepContext]):
    """A mapping like container to hold step contexts.

    This class provides a mechinism to cache intermediate calcuation
    results on data object when passed through processing steps.
    """

    @classmethod
    def resolve_key(cls, arg):
        """Return the key to step context related to ``arg``."""
        if isinstance(arg, str):
            return arg
        if isinstance(arg, Step) or issubclass(arg, Step):
            return arg.context_key
        if callable(arg):
            return getname(arg)
        raise TypeError("invalid step context key arg.")

    def __setitem__(self, key, value):
        return super().__setitem__(self.resolve_key(key), value)

    def __getitem__(self, key, *args):
        return super().__getitem__(self.resolve_key(key), *args)


class Step(Generic[ConfigT, ContextT]):
    """A base class to define a processing step."""

    config_cls: ClassVar[type[ConfigT]]
    context_cls: ClassVar[type[ContextT]]
    _alias_name: ClassVar[None | str] = None
    _orig_context_key: ClassVar[None | str] = None

    _config: ConfigT

    def __init_subclass__(cls, **kwargs):
        cls.config_cls = get_typing_args(
            cls,
            max_depth=2,
            bound=StepConfig,
            unique=True,
        )
        cls.context_cls: StepContext = get_typing_args(
            cls,
            max_depth=2,
            bound=StepContext,
            unique=True,
        )
        # this allows the context class to rebuld the
        # run.
        cls.context_cls.step_cls = cls
        cls.context_cls.config_cls = cls.config_cls
        return super().__init_subclass__(**kwargs)

    def __init__(self, *args, **kwargs):
        if not args:
            config = self.config_cls.model_validate(kwargs)
        elif len(args) == 1:
            _config = args[0]
            if isinstance(_config, self.config_cls):
                if not kwargs:
                    config = _config
                else:
                    config = _config.model_validate(_config.model_dump() | kwargs)
            else:
                raise ValueError(
                    f"positional arg is not a valid "
                    f"{self.config_cls.__name__} instance.",
                )
        else:
            raise ValueError("too many positional args.")
        self._config = config

    @property
    def config(self):
        """The config object."""
        return self._config

    @classproperty
    def context_key(cls):  # noqa: N805
        """The context key."""  # noqa: D401
        basename = cls._orig_context_key or getname(cls)
        if cls._alias_name:
            return f"{basename}_{cls._alias_name}"
        return basename

    def create_context(self, data) -> ContextT:
        """Create step context for data."""
        pctx = get_pipeline_contexts(data)
        ctx = pctx[self.context_key] = self.context_cls(
            config=self.config,
            completed=False,
        )
        return ctx

    @classmethod
    def has_context(cls, data) -> bool:
        """Return True if context object exists."""
        pctx = get_pipeline_contexts(data)
        return cls.context_key in pctx

    @classmethod
    def get_context(cls, data) -> ContextT:
        """Return the context object for data."""
        pctx = get_pipeline_contexts(data)
        return pctx[cls.context_key]

    @classmethod
    def run(cls, data, context: ContextT) -> bool:
        """Subclass implement the work flow of this step."""
        raise NotImplementedError

    @overload
    def __call__(
        self,
        data: T,
        return_context: Literal[True] = ...,
    ) -> tuple[T, ContextT]: ...

    @overload
    def __call__(
        self,
        data: T,
        return_context: Literal[False] = ...,
    ) -> T: ...

    def __call__(self, data, return_context=False):
        """Run the step."""
        # TODO: revisit this. may do some checks to compare the old
        # and now config.  This will replace the context object
        # with a new one
        context = self.create_context(data)
        if self.run(data, context):
            context.completed = True
        if return_context:
            return data, context
        return data

    @classmethod
    def alias(cls, name) -> type[Self]:
        """Return a step class with an alternative context key."""
        return type(
            cls.__name__,
            (cls,),
            {
                "_alias_name": name,
                "_orig_context_key": cls._orig_context_key or cls.context_key,
            },
        )


PIPELINE_CONTEXT_KEY: Literal["__pipeline_context__"] = "__pipeline_context__"


def get_pipeline_contexts(
    data,
) -> StepContextDict:
    """Return context dict for the pipeline."""
    if not hasattr(data, "meta"):
        raise NotImplementedError
    data.meta.setdefault(PIPELINE_CONTEXT_KEY, StepContextDict())
    return data.meta[PIPELINE_CONTEXT_KEY]


class Pipeline:
    """A class to pipeline."""

    @staticmethod
    def get_contexts(data):
        """Return pipeline contexts."""
        return get_pipeline_contexts(data)

    def __call__(self, *args, **kwargs):
        """Run the pipeline."""
        raise NotImplementedError


@dataclass(kw_only=True)
class SequentialPipeline(Pipeline):
    """A simple pipeline that execute steps sequentially."""

    steps: list[Step]

    @timeit
    def __call__(self, data):
        """Run the pipeline."""
        _data = data
        for step in self.steps:
            if step.config.enabled:
                _data = step(_data)
            else:
                logger.debug(f"step {step.context_key} is not enabled, skip")
        return _data
