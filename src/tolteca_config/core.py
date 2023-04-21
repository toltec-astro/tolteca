from functools import cached_property
from typing import Any, ClassVar

from pydantic import Field
from tollan.config.models.config_snapshot import ConfigSnapshot
from tollan.config.runtime_context import RuntimeConfigBackend as _RuntimeConfigBackend
from tollan.config.runtime_context import RuntimeContext as _RuntimeContext
from tollan.config.runtime_context import RuntimeInfo as _RuntimeInfo
from tollan.config.types import AbsAnyPath, AbsDirectoryPath, ImmutableBaseModel

from . import __version__ as _current_version
from .utils import get_user_data_dir

__all__ = [
    "RuntimeInfo",
    "RuntimeConfigBackend",
    "RuntimeContext",
    "WorkflowConfigBase",
    "WorkflowBase",
]


class RuntimeInfo(_RuntimeInfo):
    """The runtime info for tolteca."""

    config_snapshot: None | ConfigSnapshot = Field(
        default=None,
        description="A persisted config dict to compare runtime info against.",
    )

    version: str = Field(
        default=_current_version,
        description="The software version.",
    )

    bin_dir: None | AbsDirectoryPath = Field(
        default=None,
        description="The directory to look for external routines.",
    )
    cal_dir: AbsAnyPath = Field(
        default=get_user_data_dir(),
        description="The directory to hold calibration data files.",
    )
    log_dir: None | AbsDirectoryPath = Field(
        default=None,
        description="The directory to hold log files.",
    )


class RuntimeConfigBackend(_RuntimeConfigBackend, runtime_info_model_cls=RuntimeInfo):
    """The tolteca config backend."""


class RuntimeContext(_RuntimeContext, runtime_config_backend_cls=RuntimeConfigBackend):
    """The tolteca runtime context."""

    _workflow_classes_registry: ClassVar[set[type["WorkflowBase"]]] = set()
    """A registry to collect workflow classes."""

    @classmethod
    def register_workflow(cls, workflow_cls: type["WorkflowBase"]):
        """Register workflow.

        Workflow classes registered can be constructed via the `__getitem__` interface.
        """
        cls._workflow_classes_registry.add(workflow_cls)

    def __getitem__(
        self,
        workflow_cls: type["WorkflowBase"] | tuple[type["WorkflowBase"], ...],
    ) -> Any:
        """Return workflow instance configured by this runtime context."""
        if isinstance(workflow_cls, tuple):
            return list(map(self.__getitem__, workflow_cls))

        if not issubclass(workflow_cls, WorkflowBase):
            raise TypeError(
                f"{workflow_cls} is not a workflow class.",
            )
        if workflow_cls not in self._workflow_classes_registry:
            raise ValueError(
                f"{workflow_cls} is not registered to the runtime context.",
            )
        return workflow_cls(self)


class WorkflowConfigBase(ImmutableBaseModel):
    """A base class for tolteca workflow config models."""


class WorkflowBase:
    """A base class for tolteca workflows.

    This class provides the mechanism to implement a set of cohenrent
    analysis tasks that consumes the config and runtime info provided
    by `RuntimeContext`.

    This class acts as a proxy of an underlying `RuntimeContext` object,
    providing a unified interface for subclasses to managed
    specialized config objects constructed from
    the config dict of the runtime context and its the runtime info.

    Parameters
    ----------
    *arg, **kwargs :
        Specify the underlying runtime context.
        If only one arg passed and it is of type `RuntimeContext`,
        it is used as as-is, otherwise these arguments are passed
        to the `RuntimeContext` constructor.
    """

    runtime_config_key: ClassVar[str]
    """Subclass implements to specify the runtime config key to consume."""

    config_model_cls: ClassVar[type[WorkflowConfigBase]]
    """Subclass implements to specify the config mdoel class to use."""

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], RuntimeContext):
            rc = args[0]
        else:
            rc = RuntimeContext(*args, **kwargs)
        self._rc = rc

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        RuntimeContext.register_workflow(cls)

    @property
    def rc(self):
        """Return the runtime context."""
        return self._rc

    @property
    def runtime_config(self):
        """The underlying runtime config."""
        return self.rc.config

    @property
    def runtime_info(self):
        """Return the runtime info."""
        return self.rc.runtime_info

    @cached_property
    def config(self) -> Any:
        """The cached workflow config object.

        This calls out to `load_config`.
        """
        return self.load_config()

    def load_config(self):
        """Load the workflow config object.

        This is created by validating the runtime config agains the
        workflow config model.
        The config dict is validated and the constructed object is cached.
        The config object can be updated by using :meth:`RuntimeBase.update`.
        """
        return self.config_model_cls.model_validate(
            getattr(self.rc.config, self.runtime_config_key, {}),
            context=self.runtime_info.model_validation_context,
        )

    def update_config(self, cfg, mode="override"):
        """Update the workflow config with provided config dict.

        This is done by wrapping ``cfg`` under the runtime config
        key and call `update_runtime_config`.
        """
        return self.update_runtime_config(
            {self.runtime_config_key: cfg},
            mode=mode,
        )

    def update_runtime_config(self, cfg, mode="override"):
        """Update the runtime config with provided config dict.

        Note that this dict should have workflow config under the
        `runtime_config_key`.

        Parameters
        ----------
        config : `dict`
            The config dict to apply.
        mode : {"override", "default"}
            Controls how `config` dict is applied, Wether to override the
            config or use as default for unspecified values.
        """
        if mode == "override":
            self.rc.config_backend.update_override_config(cfg)
        elif mode == "default":
            self.rc.config_backend.update_default_config(cfg)
        else:
            raise ValueError("invalid update mode.")
        # update the cache so any error is raised in the validation now.
        self.__dict__["config"] = self.load_config()
