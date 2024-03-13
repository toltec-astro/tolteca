import dataclasses
from contextlib import ExitStack
from typing import Any

from tollan.utils.fileloc import FileLoc
from tollan.utils.log import logger
from typing_extensions import Self

__all__ = ["FileIOError", "FileIOBase"]


class FileIOError(RuntimeError):
    """An exception class related to file IO."""


@dataclasses.dataclass
class FileIOState:
    """A container class for file IO state."""

    io_obj: Any

    def set_open_state(self, io_obj):
        """Set the open state."""
        io_obj_type = type(io_obj)
        logger.debug(f"open data file io obj type {io_obj_type}")
        self.io_obj = io_obj

    def set_close_state(self):
        """Set the close state."""
        io_obj_type = type(self.io_obj)
        logger.debug(f"close data file io obj type {io_obj_type}")
        self.io_obj = None

    def is_open(self):
        """Return True if the IO is open."""
        return self.io_obj is not None


class FileIOBase(ExitStack):
    """A base class to help access data file contents.

    This class provide a common interface to work with data read and write.
    It manages properties  ``file_loc_orig``, ``file_loc``, ``file_obj``, and
    ``io_state``:

    * ``file_loc_orig``. This is ``FileLoc`` instance of the origin of the file.

    * ``file_loc``. This is ``FileLoc`` instance of local file location to be opened.

    * ``file_obj``. This is the file IO object passed externally.

    * ``io_state``. This is managed by this class to hold the open/close state.

    * ``meta``. Arbitury data.

    Subclass should implement ``_file_loc_orig``, ``_file_loc``, ``_file_obj``,
    ``identify()`` and ``open()``.

    This class is also a subclass of `~contextlib.ExitStack`, which can be
    used to manage context objects such as opened files.

    """

    _file_loc_orig: None | FileLoc
    _file_loc: None | FileLoc
    _file_obj: Any
    _io_state: FileIOState

    @classmethod
    def identify(cls, *_args, **_kwargs) -> bool:
        """Identify if the given file can be opened by this class."""
        return NotImplemented

    def open(self, *_args, **_kwargs) -> Self:
        """Open this file for IO."""
        return NotImplemented

    def __init__(self, file_loc_orig=None, file_loc=None, file_obj=None, meta=None):
        self._file_loc_orig = self._validate_file_loc(file_loc_orig)
        self._file_loc = self._validate_file_loc(file_loc)
        self._file_obj = file_obj
        self._io_state = FileIOState(None)
        self._meta = meta or {}
        # init the exit stack
        super().__init__()
        # add the io_state to the callback stack
        self.callback(self._io_state.set_close_state)

    @staticmethod
    def _validate_file_loc(file_loc):
        if file_loc is None or isinstance(file_loc, FileLoc):
            return file_loc
        return FileLoc(file_loc)

    @property
    def file_loc_orig(self):
        """The original file location.

        Fallback to ``file_loc`` if not specified
        """
        if self._file_loc_orig is None:
            return self.file_loc
        return self._file_loc_orig

    @property
    def file_loc(self):
        """The file location."""
        return self._file_loc

    @property
    def file_obj(self):
        """The file object."""
        return self._file_obj

    @property
    def file_url(self):
        """The file URI."""
        if self.file_loc is None:
            return None
        return self.file_loc.url

    @property
    def filepath(self):
        """The file path."""
        if self.file_loc is None:
            return None
        return self.file_loc.path

    @property
    def io_state(self):
        """The data io state."""
        return self._io_state

    @property
    def meta(self):
        """The metadata."""
        return self._meta

    def __repr__(self):
        return f"{self.__class__.__name__}({self.file_loc})"
