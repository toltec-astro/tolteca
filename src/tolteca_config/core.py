from functools import cached_property
from typing import Any, ClassVar

from pydantic import Field
from tollan.config.models.config_snapshot import ConfigSnapshot
from tollan.config.runtime_context import ConfigBackendBase, RuntimeContextBase
from tollan.config.runtime_context import RuntimeInfo as _RuntimeInfo
from tollan.config.types import AbsAnyPath, AbsDirectoryPath, ImmutableBaseModel

from . import __version__ as _current_version
from .utils import get_user_data_dir

__all__ = [
    "RuntimeInfo",
    "ConfigBackend",
    "RuntimeContext",
    "ConfigModel",
    "ConfigHandler",
]


class RuntimeInfo(_RuntimeInfo):
    """The tolteca runtime info."""

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
        default=get_user_data_dir().joinpath("cal"),
        description="The directory to hold calibration data files.",
    )
    log_dir: None | AbsDirectoryPath = Field(
        default=None,
        description="The directory to hold log files.",
    )


class ConfigBackend(ConfigBackendBase, runtime_info_model_cls=RuntimeInfo):
    """The tolteca config backend."""


class RuntimeContext(RuntimeContextBase, config_backend_cls=ConfigBackend):
    """The tolteca runtime context."""

    config_handler_cls_registry: ClassVar[set[type["ConfigHandler"]]] = set()
    """A registry to collect config handler classes."""

    @classmethod
    def register_config_handler_cls(cls, config_handler_cls: type["ConfigHandler"]):
        """Register config handler class.

        Types registered can have the instances constructed via the `__getitem__`
        interface.
        """
        cls.config_handler_cls_registry.add(config_handler_cls)

    def __getitem__(
        self,
        config_handler_cls: type["ConfigHandler"] | tuple[type["ConfigHandler"], ...],
    ) -> Any:
        """Return config handler instance configured by this runtime context."""
        if isinstance(config_handler_cls, tuple):
            return list(map(self.__getitem__, config_handler_cls))

        if not issubclass(config_handler_cls, ConfigHandler):
            raise TypeError(
                f"{config_handler_cls} is not a config handler class.",
            )
        if config_handler_cls not in self.config_handler_cls_registry:
            raise TypeError(
                f"{config_handler_cls} is not registered to the runtime context.",
            )
        return config_handler_cls(self)


class ConfigModel(ImmutableBaseModel):
    """A base class for tolteca config models."""


class ConfigHandler:
    """A base class for tolteca config handler.

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

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], RuntimeContext):
            rc = args[0]
        else:
            rc = RuntimeContext(*args, **kwargs)
        self._rc = rc

    config_model_cls: ClassVar[type[ConfigModel]]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        RuntimeContext.register_config_handler_cls(cls)

    def __class_getitem__(cls, config_model_cls):
        return type(
            f"ConfigHandler_{config_model_cls.__name__}",
            (ConfigHandler,),
            {"config_model_cls": config_model_cls},
        )

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
        """The cached config object.

        This calls out to `load_config`.
        """
        return self.load_config()

    @classmethod
    def prepare_config_data(cls, _runtime_config: ImmutableBaseModel) -> dict:
        """Subclass implement to prepare config data from runtime config."""
        return NotImplemented

    @classmethod
    def prepare_runtime_config_data(cls, _config_data: dict) -> dict:
        """Subclass implement to prepare runtime config data stub form config data."""
        return NotImplemented

    def load_config(self):
        """Load the config object.

        This is created by validating the runtime config against the config
        model. The config dict is validated and the constructed object is
        cached. The config object can be updated by using
        :meth:`RuntimeBase.update`.
        """
        return self.config_model_cls.model_validate(
            self.prepare_config_data(self.runtime_config),
            context=self.runtime_info.validation_context,
        )

    def update_config(self, cfg, mode="override"):
        """Update the config with provided config dict.

        This is done by wrapping ``cfg`` under the runtime config
        key and call `update_runtime_config`.
        """
        return self.update_runtime_config(
            self.prepare_runtime_config_data(cfg),
            mode=mode,
        )

    def update_runtime_config(self, cfg, mode="override"):
        """Update the runtime config with provided config dict.

        Note that this dict should have config under the
        ``runtime_config_key``.

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


class SubConfigKeyTransformer:
    """A mixin class to handle a subkey config dict in the runtime config."""

    key: str

    def __class_getitem__(cls, key):
        return type(
            f"ConfigTransformer_{key}",
            (SubConfigKeyTransformer,),
            {"key": key},
        )

    @classmethod
    def prepare_config_data(cls, runtime_config: ImmutableBaseModel) -> dict:
        """Return sub config data."""
        return getattr(runtime_config, cls.key, {})

    @classmethod
    def prepare_runtime_config_data(cls, config_data: dict) -> dict:
        """Return runtime config data."""
        return {cls.key: config_data}
