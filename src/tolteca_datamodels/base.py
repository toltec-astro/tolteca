import dataclasses
from contextlib import ExitStack
from typing import Any

from tollan.utils.fileloc import FileLoc
from tollan.utils.log import logger

__all__ = ["DataFileIOError", "DataFileIO"]


class DataFileIOError(RuntimeError):
    """An exception class related to data file IO."""


@dataclasses.dataclass
class DataFileIOState:
    """A container class for IOState."""

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


class DataFileIO(ExitStack):
    """A base class to help access data file contents.

    This class provide a common interface to work with data read and write.
    It manages properties  ``file_loc_orig``, ``file_loc``, ``file_obj``, and ``io_state``:

    * ``file_loc_orig``. This is ``FileLoc`` instance of the origin of the file.

    * ``file_loc``. This is ``FileLoc`` instance of local file location to be opened.

    * ``file_obj``. This is the file IO object passed externally.

    * ``io_state``. This is managed by this class to hold the open/close state.

    * ``meta``. Arbitury data.

    Subclass should implement ``_file_loc_orig``, ``_file_loc``, ``_file_obj``,
    and ``open()``.

    This class is also a subclass of `~contextlib.ExitStack`, which can be
    used to manage context objects such as opened files.

    """

    _file_loc_orig: None | FileLoc
    _file_loc: None | FileLoc
    _file_obj: Any
    _io_state: DataFileIOState
    _meta: dict

    def __init__(self, file_loc_orig=None, file_loc=None, file_obj=None, meta=None):
        self._file_loc_orig = self._validate_file_loc(file_loc_orig)
        self._file_loc = self._validate_file_loc(file_loc)
        self._file_obj = file_obj
        self._io_state = DataFileIOState(None)
        self._meta = {}
        if meta is not None:
            self.meta.update(meta)
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
    def file_uri(self):
        """The file URI."""
        if self.file_loc is None:
            return None
        return self.file_loc.uri

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

    def open(self):
        """Open the data IO."""
        return NotImplemented
