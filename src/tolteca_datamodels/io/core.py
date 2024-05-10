from contextlib import ExitStack, contextmanager
from typing import Any, ClassVar, Generic, Protocol, TypeVar, get_args

from pydantic import BaseModel, Field, ValidationInfo, model_validator
from tollan.utils.fileloc import FileLoc
from tollan.utils.log import logger, logit
from typing_extensions import Self

__all__ = ["FileIODataModelBase", "FileIO"]


io_cls_registry: set["FileIO"] = set()
"""A registry to hold all file IO subclasses."""


def identify_io_cls(source, type=None, **kwargs):
    """Return the IO class from the registry."""
    for io_cls in io_cls_registry:
        logger.debug(f"identify {source=} against {io_cls=}")
        if type is not None and not issubclass(io_cls, type):
            logger.debug(f"skip {io_cls=} as subclass of {type}")
            continue
        if io_cls.identify(source, **kwargs):
            return io_cls
    raise ValueError("unable to identify IO class.")


class FileIODataModelBase(BaseModel):
    """A container class for file IO data."""

    source: None | FileLoc = Field(description="The data source.")
    source_obj: None | Any = Field(
        default=None,
        description="externally passed source obj",
    )
    io_obj: None | Any = Field(default=None, description="currently active io obj.")

    @model_validator(mode="before")
    @classmethod
    def _validate_arg(cls, source, info: ValidationInfo):
        values = (info.context or {}).copy()
        values["source"] = cls.validate_source(source, **values)
        return values

    @classmethod
    def validate_source(cls, source, **kwargs):  # noqa: ARG003
        """Return validated source."""
        return source

    def is_open(self):
        """Return True if the IO is open."""
        return self.io_obj is not None

    def _set_open_state(self, **kwargs):
        """Open the file IO."""
        raise NotImplementedError

    def _set_close_state(self, **kwargs):  # noqa: ARG002
        """Open the file IO."""
        self.io_obj = None

    @contextmanager
    def open(self, **kwargs):
        """Set open state IO object."""
        if not self.is_open():
            with logit(logger.debug, f"open {self.source.url}"):
                self._set_open_state(**kwargs)
        yield self.io_obj
        if self.is_open():
            with logit(logger.debug, f"close {self.source.url}"):
                self._set_close_state(**kwargs)


FileIODataType = TypeVar("FileIODataType", bound=FileIODataModelBase)
FileIOMetadataType = TypeVar("FileIOMetadataType")


class FileIO(ExitStack, Generic[FileIODataType, FileIOMetadataType]):
    """A base class to help access data file contents.

    This class provide a common interface to work with file IO.
    It manages the file IO data in the ``_io_data`` attribute,
    and the open/close state of the underlying object through the
    ``_io_obj`` attribute.

    Subclass should implement ``_resolve_io_data()``, ``identify()``,
    and ``open()``.

    This class is also a subclass of `~contextlib.ExitStack`, which can be
    used to manage context objects such as opened files.
    """

    io_data_cls: ClassVar[type[FileIODataType]]
    io_metadata_cls: ClassVar[type[FileIOMetadataType]]
    _io_data: FileIODataType
    _meta: FileIOMetadataType

    def __init_subclass__(cls, **kwargs):
        io_cls_registry.add(cls)
        io_data_cls, io_metadata_cls = get_args(
            next(
                iter(
                    c
                    for c in cls.__orig_bases__
                    if (hasattr(c, "__origin__") and c.__origin__ is FileIO)
                ),
            ),
        )
        cls.io_data_cls = io_data_cls
        cls.io_metadata_cls = io_metadata_cls
        super().__init_subclass__(**kwargs)

    def __new__(cls, source, **kwargs):
        """Identify and return the file IO."""
        if cls is FileIO:
            io_cls = identify_io_cls(source, **kwargs)
            return super(io_cls, cls).__new__(io_cls)
        return super().__new__(cls)

    def __init__(self, source, **kwargs):
        self._io_data = self.io_data_cls.model_validate(source, context=kwargs)
        self._meta = self.io_metadata_cls()
        self._update_meta_from_io_data()
        super().__init__()

    @property
    def io_data(self) -> FileIODataType:
        """The io data."""
        return self._io_data

    @property
    def source(self):
        """The data source location."""
        return self.io_data.source

    @property
    def filepath(self):
        """The local data file location."""
        if self.source is None:
            return None
        return self.source.path

    @property
    def io_obj(self):
        """The active data io obj."""
        return self.io_data.io_obj

    @property
    def meta(self) -> FileIOMetadataType:
        """The metadata."""
        return self._meta

    @classmethod
    def identify(cls, source) -> bool:
        """Identify if the given source can be opened by this class."""
        try:
            cls.io_data_cls.validate_source(source)
        except ValueError as e:
            logger.debug(f"{cls.io_data_cls} cannot validate {source=}: {e}")
            return False
        return True

    def is_open(self):
        """Return True if the IO is open."""
        return self.io_data.is_open()

    def open(self, **kwargs) -> Self:
        """Return the opened file IO."""
        if not self.is_open():
            self.enter_context(self.io_data.open(**kwargs))
        self._update_meta_from_io_data()
        return self

    def _update_meta_from_io_data(self):
        """Subclass implement to update meta from io data."""

    def __getstate__(self):
        # need to reset the object before pickling
        is_open = self.is_open()
        self.close()
        return {
            "_auto_close_on_pickle_wrapped": self.__dict__,
            "_auto_close_on_pickle_is_open": is_open,
        }

    def __setstate__(self, state):
        state, is_open = (
            state["_auto_close_on_pickle_wrapped"],
            state["_auto_close_on_pickle_is_open"],
        )
        self.__dict__.update(state)
        if is_open:
            self.open()

    def __repr__(self):
        state = (
            "<not initialized>"
            if not hasattr(self, "_io_data")
            else self.io_data.source
        )
        return f"{self.__class__.__name__}({state})"


class FileIOProtocol(Protocol):
    io_data_cls: ClassVar[type[FileIODataModelBase]]

    def is_open(): ...

    def _set_open_state(): ...

    @property
    def filepath(): ...

    def enter_context(): ...

    def close(): ...
